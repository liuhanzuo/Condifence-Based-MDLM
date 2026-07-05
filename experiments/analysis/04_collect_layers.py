"""
Exp4 数据收集：一次性收集【所有层】的 hidden state + stability + 特征，
供逐层 probe (05) 与 certifier 实用性 (06) 复用。

与 01 相同的 stability 定义，但额外保存每个样本在【每一层】的 hidden（降采样存 float16 省空间）。
为控制体积：只存偶数层 + 首末层。

运行：
  python exp/04_collect_layers.py --n_sents 20 --n_anchors 10 --K 12 --out exp/out/layers.npz
"""
import argparse
import os
import numpy as np
import torch

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666

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
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def forward_all(model, ids):
    out = model(input_ids=ids, output_hidden_states=True)
    return out.logits[0], out.hidden_states  # logits[L,V], tuple(n_layer+1)[1,L,H]


@torch.no_grad()
def stability(model, gt, keep, tp, base_am, K, rng):
    device = gt.device
    other = [p for p in torch.where(~keep)[0].tolist() if p not in tp]
    kept = torch.where(keep)[0].tolist()
    hit = np.zeros(len(tp))
    for _ in range(K):
        keep2 = set(kept)
        if other:
            r = rng.uniform(0.0, 1.0)
            k = int(round(r * len(other)))
            if k > 0:
                keep2.update(int(x) for x in rng.choice(other, size=k, replace=False))
        mf = torch.ones(gt.shape[0], dtype=torch.bool, device=device)
        if keep2:
            mf[torch.as_tensor(sorted(keep2), device=device)] = False
        for p in tp:
            mf[p] = True
        s = gt.clone()
        s[mf] = MASK_ID
        logits, _ = forward_all(model, s.unsqueeze(0))
        am = logits[tp].argmax(-1)
        hit += (am.cpu().numpy() == base_am).astype(float)
    return hit / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sents", type=int, default=20)
    ap.add_argument("--n_anchors", type=int, default=10)
    ap.add_argument("--keep_lo", type=float, default=0.05)
    ap.add_argument("--keep_hi", type=float, default=0.7)
    ap.add_argument("--max_targets", type=int, default=8)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--out", default="exp/out/layers.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    tok, model = load_model()

    # 探测层数
    demo = tok("hello world", return_tensors="pt").input_ids.to("cuda")
    _, hs = forward_all(model, demo)
    n_layers = len(hs)  # embedding + 每层
    sel_layers = list(range(0, n_layers, 2))
    if (n_layers - 1) not in sel_layers:
        sel_layers.append(n_layers - 1)
    print(f"total hidden states = {n_layers}, selected layers = {sel_layers}")

    per_layer_H = {li: [] for li in sel_layers}
    prob, marg, stab, nrev, leftd, rightd = [], [], [], [], [], []

    sents = SENTS[: args.n_sents]
    for si, sent in enumerate(sents):
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 6:
            continue
        for _a in range(args.n_anchors):
            keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
            kr = rng.uniform(args.keep_lo, args.keep_hi)
            nk = max(1, int(round(kr * L)))
            keep[torch.as_tensor(rng.choice(L, size=nk, replace=False), device=gt.device)] = True
            s = gt.clone(); s[~keep] = MASK_ID
            logits, hs = forward_all(model, s.unsqueeze(0))
            probs = torch.softmax(logits.float(), -1)
            masked = torch.where(~keep)[0].tolist()
            if not masked:
                continue
            tp = rng.choice(masked, size=min(args.max_targets, len(masked)),
                            replace=False).tolist()
            base_am = logits[tp].argmax(-1).cpu().numpy()
            top2 = torch.topk(probs[tp], 2, -1).values
            p1 = top2[:, 0].cpu().numpy()
            mg = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            st = stability(model, gt, keep, tp, base_am, args.K, rng)
            kept = torch.where(keep)[0].tolist()
            for j, p in enumerate(tp):
                prob.append(float(p1[j])); marg.append(float(mg[j])); stab.append(float(st[j]))
                nrev.append(len(kept))
                left = [k for k in kept if k < p]; right = [k for k in kept if k > p]
                leftd.append(p - max(left) if left else p + 1)
                rightd.append(min(right) - p if right else L - p)
                for li in sel_layers:
                    per_layer_H[li].append(hs[li][0, p].float().cpu().numpy().astype(np.float16))
        print(f"[{si+1}/{len(sents)}] n={len(stab)}")

    save = dict(prob=np.array(prob, np.float32), margin=np.array(marg, np.float32),
                stability=np.array(stab, np.float32), nrev=np.array(nrev, np.float32),
                leftdist=np.array(leftd, np.float32), rightdist=np.array(rightd, np.float32),
                sel_layers=np.array(sel_layers))
    for li in sel_layers:
        save[f"H{li}"] = np.stack(per_layer_H[li]).astype(np.float16)
    np.savez_compressed(args.out, **save)
    print(f"Saved {len(stab)} samples, {len(sel_layers)} layers -> {args.out}")


if __name__ == "__main__":
    main()
