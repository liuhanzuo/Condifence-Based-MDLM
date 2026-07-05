"""
分析 01_commit_ready.py 收集的数据，判定命题 A / B。

命题A：Stability 与 predicted_prob / margin 是否分离？
   - Spearman 相关：若 |rho| 很高 -> 稳定性 ≈ 概率，certifier 无独立内容
   - 出散点图 prob-vs-stability
命题B：能否只用 hidden state 预测 Stability（state-local）？
   - probe1: 输入仅 H(3584维)          -> R^2_state_local
   - probe2: 输入 H + [prob, margin]    -> 看加了当前分布特征是否更好
   - probe3: 输入 H + keep_ratio(全局R的摘要) -> 看加"全局揭示信息"是否显著提升
   - 若 probe3 相对 probe1 提升很小 -> readiness 是 state-local 的 -> parallel commit 有据

运行：
  python exp/02_analyze.py --data exp/out/data.npz --out exp/out
"""

import argparse
import os
import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def fit_probe(X, y, seed=0):
    """岭回归 + 5折CV 的 R^2（held-out）。返回 mean R^2。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = Ridge(alpha=10.0).fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], m.predict(Xte)))
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="exp/out/data.npz")
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.data)
    prob, margin, stab = d["prob"], d["margin"], d["stability"]
    keep_ratio, H = d["keep_ratio"], d["H"]
    n = len(stab)
    print(f"n samples = {n}, H dim = {H.shape[1]}")
    print(f"stability: mean={stab.mean():.3f} std={stab.std():.3f} "
          f"[min {stab.min():.2f}, max {stab.max():.2f}]")
    print(f"prob     : mean={prob.mean():.3f} std={prob.std():.3f}")

    # ===== 命题 A =====
    rho_prob = spearman(prob, stab)
    rho_marg = spearman(margin, stab)
    print("\n===== 命题 A（分离性）=====")
    print(f"  Spearman(prob,   stability) = {rho_prob:+.3f}")
    print(f"  Spearman(margin, stability) = {rho_marg:+.3f}")
    print("  解读：|rho| 越接近1 -> stability≈现有confidence，certifier无独立价值；")
    print("        |rho| 明显<1 -> stability 是独立的量（命题A成立）。")

    # ===== 命题 B =====
    print("\n===== 命题 B（state-local 可判定性）=====")
    from sklearn.decomposition import PCA

    # H 维度(3584) >> 样本数，直接线性回归必过拟合 -> 先 PCA 降维再 probe
    n_pca = min(32, H.shape[0] // 6, H.shape[1])
    Hp = PCA(n_components=n_pca, random_state=0).fit_transform(
        (H - H.mean(0)) / (H.std(0) + 1e-6)
    )
    print(f"  (hidden 先 PCA 到 {n_pca} 维以缓解 dim>>n 的过拟合)")

    r0, s0 = fit_probe(np.stack([prob, margin], 1), stab)  # baseline: 现有confidence
    r1, s1 = fit_probe(Hp, stab)                            # state-local: 仅 hidden(PCA)
    r2, s2 = fit_probe(np.concatenate([Hp, prob[:, None], margin[:, None]], 1), stab)
    r3, s3 = fit_probe(np.concatenate([Hp, keep_ratio[:, None]], 1), stab)  # + 全局R
    print(f"  probe0 [prob,margin only]         R^2 = {r0:+.3f} ± {s0:.3f}  (现有confidence上限)")
    print(f"  probe1 [hiddenPCA, state-local]   R^2 = {r1:+.3f} ± {s1:.3f}  <- 核心")
    print(f"  probe2 [hiddenPCA + prob,margin]  R^2 = {r2:+.3f} ± {s2:.3f}")
    print(f"  probe3 [hiddenPCA + keep_ratio(R)]R^2 = {r3:+.3f} ± {s3:.3f}")
    print("  解读：")
    print("   - probe1 > probe0  -> hidden state 含 confidence 之外的 readiness 信息（命题B正向）")
    print("   - probe3 ≈ probe1  -> 加全局揭示信息(R)无明显提升 -> readiness 是 state-local -> 支持 parallel commit")
    print("   - probe3 >> probe1 -> readiness 依赖全局 R -> path-dependent -> AR 退化更可能是最优")

    # ===== 出图 =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].scatter(prob, stab, s=14, alpha=0.5)
        ax[0].set_xlabel("predicted top-1 prob")
        ax[0].set_ylabel("Stability (cross-trajectory)")
        ax[0].set_title(f"Prop.A: prob vs stability (rho={rho_prob:+.2f})")
        ax[0].plot([0, 1], [0, 1], "r--", lw=1)

        labels = ["prob,margin\n(P0)", "hidden\n(P1)", "hidden+pm\n(P2)", "hidden+R\n(P3)"]
        vals = [r0, r1, r2, r3]
        errs = [s0, s1, s2, s3]
        ax[1].bar(labels, vals, yerr=errs, color=["#bbb", "#3b7", "#39c", "#c63"])
        ax[1].set_ylabel("held-out R^2 (predict Stability)")
        ax[1].set_title("Prop.B: what predicts commit-readiness?")
        ax[1].axhline(0, color="k", lw=0.5)

        fig.tight_layout()
        p = os.path.join(args.out, "result.png")
        fig.savefig(p, dpi=130)
        print(f"\nSaved figure -> {p}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
