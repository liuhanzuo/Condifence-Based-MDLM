"""
Exp6 AR 退化测量：让模型按 confidence-first 自发揭示（标准 MDLM 采样常用策略），
记录实际揭示顺序，测量：
  1) 揭示顺序与 left-to-right 的一致性（Spearman(揭示步序, 位置索引)）
     -> 越接近 +1 越 AR
  2) 每一步被选中揭示的位置的 predicted prob（自发选的通常很高）
  3) 相邻两次揭示的位置间距分布（AR 的话应集中在 +1）

这直接量化"unmask 过程是否退化成 AR"。

运行： python exp/06_ar_degeneracy.py --n_sents 20 --out exp/out
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


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def load_model():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def confidence_first_unmask(model, gt, tok):
    """从全 mask 开始，每步对所有 masked 位置预测，选 top-1 prob 最高的位置揭示为其 argmax。
    返回：reveal_order(位置索引按揭示先后), step_probs(每步选中位置的prob), positions=range(L)。
    注意：这里不填 GT 而填模型自己的 argmax，测的是模型"自发"生成顺序。"""
    L = gt.shape[0]
    cur = torch.full((L,), MASK_ID, device=gt.device)
    # 保留 BOS 等特殊 token 已知？简化：全 mask 从头恢复
    revealed = torch.zeros(L, dtype=torch.bool, device=gt.device)
    order, step_probs = [], []
    for _ in range(L):
        logits = model(input_ids=cur.unsqueeze(0)).logits[0]
        probs = torch.softmax(logits.float(), -1)
        p1, am = probs.max(-1)  # [L]
        p1 = p1.clone()
        p1[revealed] = -1.0  # 已揭示的不再选
        pos = int(torch.argmax(p1).item())
        cur[pos] = am[pos]
        revealed[pos] = True
        order.append(pos)
        step_probs.append(float(probs[pos, am[pos]].item()))
    return order, step_probs, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sents", type=int, default=20)
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok, model = load_model()

    l2r_corrs, gaps_all, first_probs, mid_probs = [], [], [], []
    sents = SENTS[: args.n_sents]
    for si, sent in enumerate(sents):
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 6:
            continue
        order, sp, _ = confidence_first_unmask(model, gt, tok)
        # 揭示步序 step_of_pos[pos] = 第几步被揭示
        step_of_pos = np.zeros(L)
        for step, pos in enumerate(order):
            step_of_pos[pos] = step
        # AR 一致性：位置索引 vs 揭示步序 的 Spearman
        corr = spearman(list(range(L)), step_of_pos.tolist())
        l2r_corrs.append(corr)
        # 相邻揭示间距
        gaps = [order[i + 1] - order[i] for i in range(len(order) - 1)]
        gaps_all.extend(gaps)
        first_probs.append(sp[0])
        mid_probs.append(float(np.mean(sp[len(sp)//4: 3*len(sp)//4])))
        print(f"[{si+1}/{len(sents)}] L={L} AR-corr={corr:+.3f} order={order}")

    l2r = np.array(l2r_corrs)
    gaps = np.array(gaps_all)
    print("\n===== Exp6 AR 退化测量 =====")
    print(f"  揭示顺序 vs left-to-right Spearman: mean={l2r.mean():+.3f} "
          f"std={l2r.std():.3f}  (越接近+1越AR)")
    print(f"  相邻揭示间距 = +1 的比例: {(gaps == 1).mean():.3f}  (AR应接近1)")
    print(f"  相邻揭示间距 mean={gaps.mean():+.2f} median={np.median(gaps):+.1f}")
    print(f"  首个揭示位置的 prob: mean={np.mean(first_probs):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(l2r, bins=15, color="#39c")
        ax[0].axvline(l2r.mean(), color="r", ls="--", label=f"mean={l2r.mean():.2f}")
        ax[0].set_xlabel("Spearman(reveal-order, position)")
        ax[0].set_title("Exp6: how AR is the self-unmasking?")
        ax[0].legend()
        ax[1].hist(gaps, bins=range(-8, 9), color="#3b7")
        ax[1].axvline(1, color="r", ls="--", label="+1 (pure AR)")
        ax[1].set_xlabel("adjacent reveal gap"); ax[1].set_title("reveal step gap distribution")
        ax[1].legend()
        fig.tight_layout()
        p = os.path.join(args.out, "ar_degeneracy.png")
        fig.savefig(p, dpi=130)
        print(f"saved {p}")
    except Exception as e:
        print("plot skip:", e)


if __name__ == "__main__":
    main()
