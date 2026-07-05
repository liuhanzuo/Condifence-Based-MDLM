"""
Exp13 诊断：测量 Dream-7B 里"天然条件独立子集"的规模，判断
   在 DEMASK 等"检测独立性并行"方法之后，"训练增加独立性"还有没有空间。

核心量——pairwise dependency（factorization gap 的成对近似）：
  对部分揭示状态 s，两个 masked 位置 i,j 的依赖定义为：
    dep(i,j) = KL( p(x_i | s)  ||  p(x_i | s + reveal j 的argmax) )
  即"揭示 j 后 i 的预测变化"。dep 小 => i 对 j 近似条件独立 => 可并行 commit。

流程：
  对每个状态 s，取一批 masked 位置，算所有 pair 的 dep（对称化取 max）。
  给定阈值 τ，用贪心找"最大独立子集"(所有内部 pair dep<τ) 的规模。
  统计：
   - 天然可并行子集规模分布（相对该状态 masked 总数的比例）
   - 依赖矩阵的整体强度
   - 对比：按 confidence top-k 选 k 个，这 k 个里有多少 pair 其实强依赖(被 confidence 忽略)

判决：
  - 若天然独立子集 已占 masked 位置很大比例 -> 检测类方法接近上限, 训练增加空间小
  - 若很小 -> 训练增加独立性有真实空间

运行： python exp/13_independence_capacity.py --n_sents 16 --out exp/out
"""
import argparse
import os
import numpy as np
import torch

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_pool import TRAIN_SENTS

SENTS = TRAIN_SENTS[:20]


def load():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def logits_of(model, ids):
    return model(input_ids=ids.unsqueeze(0)).logits[0].float()


def kl(p, q):
    p = p.clamp_min(1e-9); q = q.clamp_min(1e-9)
    return (p * (p.log() - q.log())).sum().item()


@torch.no_grad()
def pairwise_dep(model, gt, keep, targets):
    """对 targets 中的位置，算 pairwise 依赖矩阵 dep[a,b] = KL(p(i_a|s) || p(i_a|s+reveal i_b))。
    对称化：D[a,b]=max(dep[a,b],dep[b,a])。"""
    device = gt.device
    n = len(targets)
    s = gt.clone(); s[~keep] = MASK_ID
    base_logits = logits_of(model, s)
    base_p = torch.softmax(base_logits[targets], -1)  # [n,V]
    base_am = base_logits.argmax(-1)                   # [L]

    D = np.zeros((n, n))
    for b in range(n):
        # 揭示 targets[b] 为其 argmax，重新预测其余 target 的分布
        s2 = s.clone()
        jb = targets[b]
        s2[jb] = base_am[jb]
        lg2 = logits_of(model, s2)
        p2 = torch.softmax(lg2[targets], -1)
        for a in range(n):
            if a == b:
                continue
            D[a, b] = kl(base_p[a], p2[a])
    D = np.maximum(D, D.T)
    return D, base_p


def greedy_independent_set(D, tau):
    """贪心找一个子集，使内部所有 pair 的 D<tau。返回子集大小。
    策略：按'与他人平均依赖'升序加入，冲突则跳过。"""
    n = D.shape[0]
    order = np.argsort(D.sum(1))  # 平均依赖小的优先
    chosen = []
    for i in order:
        if all(D[i, j] < tau for j in chosen):
            chosen.append(i)
    return len(chosen), chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sents", type=int, default=16)
    ap.add_argument("--n_anchors", type=int, default=4)
    ap.add_argument("--keep_lo", type=float, default=0.1)
    ap.add_argument("--keep_hi", type=float, default=0.5)
    ap.add_argument("--max_targets", type=int, default=10)
    ap.add_argument("--taus", type=str, default="0.05,0.1,0.3")
    ap.add_argument("--out", default="exp/out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tok, model = load()
    taus = [float(x) for x in args.taus.split(",")]

    frac_by_tau = {t: [] for t in taus}
    dep_vals = []
    conf_conflict = []   # confidence top-k 中强依赖 pair 的比例

    for si, sent in enumerate(SENTS[: args.n_sents]):
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 8:
            continue
        for _a in range(args.n_anchors):
            keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
            kr = rng.uniform(args.keep_lo, args.keep_hi)
            nk = max(1, int(round(kr * L)))
            keep[torch.as_tensor(rng.choice(L, size=nk, replace=False), device=gt.device)] = True
            masked = torch.where(~keep)[0].tolist()
            if len(masked) < 3:
                continue
            targets = masked if len(masked) <= args.max_targets else \
                list(rng.choice(masked, size=args.max_targets, replace=False))
            D, base_p = pairwise_dep(model, gt, keep, targets)
            n = len(targets)
            iu = np.triu_indices(n, 1)
            dep_vals.extend(D[iu].tolist())

            for t in taus:
                sz, _ = greedy_independent_set(D, t)
                frac_by_tau[t].append(sz / n)  # 独立子集占可选位置的比例

            # confidence top-k：选 top ceil(n/2) 高 conf 的位置，看其中强依赖(>0.3) pair 比例
            s = gt.clone(); s[~keep] = MASK_ID
            lg = logits_of(model, s)
            conf = torch.softmax(lg[targets], -1).max(-1).values.cpu().numpy()
            k = max(2, n // 2)
            topk = np.argsort(-conf)[:k]
            pairs = [(a, b) for ii, a in enumerate(topk) for b in topk[ii+1:]]
            if pairs:
                strong = sum(1 for a, b in pairs if D[a, b] > 0.3) / len(pairs)
                conf_conflict.append(strong)
        print(f"[{si+1}/{args.n_sents}] samples so far: {len(frac_by_tau[taus[0]])}")

    dep_vals = np.array(dep_vals)
    print("\n===== Exp13 天然条件独立子集规模 =====")
    print(f"  pair 数 = {len(dep_vals)}")
    print(f"  pairwise dependency(KL): 中位数={np.median(dep_vals):.3f} "
          f"均值={dep_vals.mean():.3f} 90分位={np.percentile(dep_vals,90):.3f}")
    print(f"  dep<0.05 的 pair 比例(近独立) = {(dep_vals<0.05).mean():.3f}")
    print(f"  dep<0.1  的 pair 比例        = {(dep_vals<0.1).mean():.3f}")
    print(f"  dep>0.3  的 pair 比例(强依赖) = {(dep_vals>0.3).mean():.3f}")

    print("\n  天然可并行独立子集 占可选masked位置的比例（越大=天然越可并行）：")
    for t in taus:
        arr = np.array(frac_by_tau[t])
        print(f"    τ={t:<4}: mean={arr.mean():.3f}  median={np.median(arr):.3f}")

    if conf_conflict:
        print(f"\n  confidence top-k 选中的位置里, 强依赖(dep>0.3) pair 比例 = {np.mean(conf_conflict):.3f}")
        print("   (>0 说明 confidence 会选进强依赖的位置一起并行 -> 会出错, 印证需要独立性判据)")

    print("\n判决：")
    print("  - 若各 τ 下独立子集比例都很高(接近1) -> 天然就几乎全可并行, DEMASK类检测接近上限, 训练增加空间小")
    print("  - 若比例低(如<0.5) -> 天然可并行的少, '训练主动增加条件独立结构'有真实空间")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(dep_vals, bins=40, color="#39c")
        ax[0].axvline(0.1, color="r", ls="--", label="τ=0.1")
        ax[0].set_xlabel("pairwise dependency (KL)")
        ax[0].set_title("How dependent are masked pairs?")
        ax[0].legend(); ax[0].set_yscale("log")
        means = [np.array(frac_by_tau[t]).mean() for t in taus]
        ax[1].bar([str(t) for t in taus], means, color="#3b7")
        ax[1].set_xlabel("independence threshold τ")
        ax[1].set_ylabel("independent-subset fraction")
        ax[1].set_title("Natural parallelizable fraction")
        ax[1].set_ylim(0, 1)
        fig.tight_layout()
        p = os.path.join(args.out, "independence_capacity.png")
        fig.savefig(p, dpi=130); print(f"\nsaved {p}")
    except Exception as e:
        print("plot skip:", e)


if __name__ == "__main__":
    main()
