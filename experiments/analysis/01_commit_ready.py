"""
命题 A / B 判定性实验（最小版），模型：Dream-org/Dream-v0-Instruct-7B

核心问题：
  一个未揭示位置 i 在部分揭示状态 s 下的 "跨轨迹后验稳定性" Stability(i|s)
    - 命题A：是否与 predicted_prob 分离（若强相关则该概念无独立内容）
    - 命题B：能否只用 hidden state h_i(s) 预测（state-local，不看全局揭示集 R）

定义：
  Stability(i|s) = Pr_{tau}[ argmax p(x_i | s_T) == argmax p(x_i | s) ]
  其中 tau 是从 s 出发、逐步随机揭示其他位置(填 ground-truth token)直到全揭示的轨迹。

流程：
  1) 取若干真实句子，tokenize 得到 ground-truth ids（这是"揭示时填什么"）。
  2) 随机采若干 anchor 状态 s（随机保留一部分位置为 GT，其余为 mask）。
  3) 在 s 处：对每个 masked 位置 i，记录 predicted_prob、argmax、h_i(s)。
  4) 对采样的一批位置 i，蒙特卡洛估计 Stability(i|s)。
  5) 存成表，交给 analyze 阶段做命题 A/B 分析。

运行：
  python exp/01_commit_ready.py --n_sents 40 --n_anchors 3 --K 8 --out exp/out/data.npz
"""

import argparse
import os
import numpy as np
import torch

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666  # <|mask|>  (由 00_probe_model.py 确认)


# 一批与领域无关的真实文本（短句，避免过长导致轨迹估计太贵）
SENTS = [
    "The capital of France is Paris and it is a beautiful city.",
    "Machine learning models require large amounts of training data.",
    "The sun rises in the east and sets in the west every day.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "She opened the door and walked quietly into the dark room.",
    "The stock market fell sharply after the announcement yesterday.",
    "A balanced diet includes fruits, vegetables, and whole grains.",
    "The ancient castle stood silently on top of the green hill.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The children played happily in the park all afternoon long.",
    "Quantum computers may one day solve very hard problems quickly.",
    "He carefully wrote down every word the teacher said today.",
    "The river flows gently through the valley toward the sea.",
    "Regular exercise improves both physical and mental health greatly.",
    "The museum displayed many rare paintings from the last century.",
    "Climate change is one of the biggest challenges of our time.",
    "The engineer fixed the broken bridge in just three days.",
    "Reading books every night helps children develop their imagination.",
    "The chef prepared a delicious meal for all the guests.",
    "Electric cars are becoming more popular around the world now.",
]


def load_model():
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return tok, model


@torch.no_grad()
def forward(model, ids):
    """一次前向，返回 (logits[L,V] float32 cpu 省显存前先取需要的, last_hidden[L,H])。
    这里 ids: LongTensor[1,L] on cuda。"""
    out = model(input_ids=ids, output_hidden_states=True)
    logits = out.logits[0]          # [L, V]
    h = out.hidden_states[-1][0]    # [L, H]
    return logits, h


def make_state(gt_ids, keep_mask):
    """按 keep_mask(bool[L]) 构造部分揭示状态：True=保留GT, False=mask。"""
    s = gt_ids.clone()
    s[~keep_mask] = MASK_ID
    return s


@torch.no_grad()
def estimate_stability(model, gt_ids, keep_mask, target_pos, base_argmax, K, rng):
    """蒙特卡洛估计一批 target_pos 的 Stability。

    语义（修正版）：我们关心的是——在【当前稀疏揭示状态 s】下对位置 i 的 argmax，
    如果继续沿【随机揭示轨迹】把其余位置逐步揭示为 GT，i 的 argmax 会不会被推翻。
    这刻画的是"现在提前 commit 会不会后悔"，因此必须从稀疏状态出发、且沿途多点评估，
    而不是只在接近完成的终态评估（那样几乎必然稳定，标签会饱和）。

    实现：对每条轨迹，从当前 masked(除target外)位置里随机抽一个【揭示子集比例 r】，
    r 在 (0,1] 上均匀采，构造中间状态 s'（target 仍为 mask），
    看 argmax(s') 是否 == base_argmax。对 K 条轨迹取命中率。
    这样 r 的多样性会覆盖"早期(上下文少)->后期(上下文多)"，让 Stability 产生真实方差。
    返回 stability[len(target_pos)] in [0,1]。
    """
    device = gt_ids.device
    target_pos = list(target_pos)
    hit = np.zeros(len(target_pos), dtype=np.float64)

    # 当前已揭示位置（GT）
    kept = torch.where(keep_mask)[0].tolist()
    # 除 target 外仍为 mask 的位置（这些是"未来可能被揭示"的位置）
    other_masked = [p for p in torch.where(~keep_mask)[0].tolist()
                    if p not in target_pos]

    for _ in range(K):
        s = gt_ids.clone()
        # 起点：只保留当前 kept；其余(含 target)先全 mask
        keep2 = set(kept)
        # 沿轨迹随机再揭示 other_masked 的一个随机子集(比例 r)
        if len(other_masked) > 0:
            r = rng.uniform(0.0, 1.0)
            k_reveal = int(round(r * len(other_masked)))
            if k_reveal > 0:
                reveal = rng.choice(other_masked, size=k_reveal, replace=False)
                keep2.update(int(x) for x in reveal)
        # 构造状态：keep2 用 GT，其余(含全部 target)为 mask
        mask_flag = torch.ones(gt_ids.shape[0], dtype=torch.bool, device=device)
        if keep2:
            mask_flag[torch.as_tensor(sorted(keep2), device=device)] = False
        for p in target_pos:
            mask_flag[p] = True  # target 必须是 mask 才能预测
        s[mask_flag] = MASK_ID

        logits, _ = forward(model, s.unsqueeze(0))
        am = logits[target_pos].argmax(-1)
        hit += (am.cpu().numpy() == base_argmax).astype(np.float64)

    return hit / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sents", type=int, default=20)
    ap.add_argument("--n_anchors", type=int, default=3, help="每句采几个 anchor 状态 s")
    ap.add_argument("--keep_lo", type=float, default=0.1, help="anchor 保留GT比例下界")
    ap.add_argument("--keep_hi", type=float, default=0.6, help="anchor 保留GT比例上界")
    ap.add_argument("--max_targets", type=int, default=6, help="每个anchor评估几个位置")
    ap.add_argument("--K", type=int, default=8, help="每个位置蒙特卡洛轨迹数")
    ap.add_argument("--out", type=str, default="exp/out/data.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    tok, model = load_model()

    rows_prob, rows_stab, rows_h = [], [], []
    rows_keep_ratio, rows_margin = [], []
    rows_seqlen, rows_nrev, rows_leftdist, rows_rightdist, rows_relpos = [], [], [], [], []

    sents = SENTS[: args.n_sents]
    for si, sent in enumerate(sents):
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 6:
            continue

        for _a in range(args.n_anchors):
            # 随机构造 anchor 状态 s：keep_ratio 在 [lo, hi] 随机，
            # 覆盖"稀疏揭示(早期)"到"较多揭示(后期)"，让 Stability 产生真实方差。
            keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
            kr = rng.uniform(args.keep_lo, args.keep_hi)
            n_keep = max(1, int(round(kr * L)))
            keep_idx = rng.choice(L, size=n_keep, replace=False)
            keep[torch.as_tensor(keep_idx, device=gt.device)] = True

            s = make_state(gt, keep)
            logits, h = forward(model, s.unsqueeze(0))
            probs = torch.softmax(logits.float(), dim=-1)

            masked_pos = torch.where(~keep)[0].tolist()
            if len(masked_pos) == 0:
                continue
            # 选一批 target 位置
            tp = rng.choice(masked_pos,
                            size=min(args.max_targets, len(masked_pos)),
                            replace=False).tolist()

            base_argmax = logits[tp].argmax(-1).cpu().numpy()  # [n]
            # top1 prob & margin(top1-top2)
            top2 = torch.topk(probs[tp], k=2, dim=-1).values  # [n,2]
            p_top1 = top2[:, 0].cpu().numpy()
            margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()

            stab = estimate_stability(model, gt, keep, tp, base_argmax, args.K, rng)

            kept_positions = torch.where(keep)[0].tolist()
            n_rev = len(kept_positions)
            for j, p in enumerate(tp):
                rows_prob.append(float(p_top1[j]))
                rows_margin.append(float(margin[j]))
                rows_stab.append(float(stab[j]))
                rows_h.append(h[p].float().cpu().numpy())
                rows_keep_ratio.append(float(keep.float().mean().item()))
                # 进度特征（平凡因素）
                rows_seqlen.append(int(L))
                rows_nrev.append(int(n_rev))
                # 局部耦合特征：到最近已揭示位置的左右距离 & 相对位置
                left = [k for k in kept_positions if k < p]
                right = [k for k in kept_positions if k > p]
                rows_leftdist.append(int(p - max(left)) if left else int(p + 1))
                rows_rightdist.append(int(min(right) - p) if right else int(L - p))
                rows_relpos.append(float(p / max(1, L - 1)))

        print(f"[{si+1}/{len(sents)}] done, collected {len(rows_prob)} samples")

    H = np.stack(rows_h).astype(np.float32)
    np.savez_compressed(
        args.out,
        prob=np.array(rows_prob, dtype=np.float32),
        margin=np.array(rows_margin, dtype=np.float32),
        stability=np.array(rows_stab, dtype=np.float32),
        keep_ratio=np.array(rows_keep_ratio, dtype=np.float32),
        seqlen=np.array(rows_seqlen, dtype=np.float32),
        nrev=np.array(rows_nrev, dtype=np.float32),
        leftdist=np.array(rows_leftdist, dtype=np.float32),
        rightdist=np.array(rows_rightdist, dtype=np.float32),
        relpos=np.array(rows_relpos, dtype=np.float32),
        H=H,
    )
    print(f"\nSaved {H.shape[0]} samples to {args.out}  (H dim={H.shape[1]})")


if __name__ == "__main__":
    main()
