"""
Exp5 逐层 probe：commit-readiness 信息在哪一层的 hidden 里最强？
对每个保存的层 li，用其 hidden (PCA降维) 预测 stability，比较 held-out R²。

运行： python exp/05_layerwise.py --data exp/out/layers.npz --out exp/out
"""
import argparse
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def fit(X, y, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed)
    r = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
        r.append(r2_score(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="exp/out/layers.npz")
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    d = np.load(args.data)
    y = d["stability"]
    n = len(y)
    layers = d["sel_layers"].tolist()
    print(f"n={n}, layers={layers}, stability mean={y.mean():.3f}")

    r0 = fit(np.stack([d["prob"], d["margin"]], 1), y)
    print(f"baseline [prob,margin] R^2={r0:+.3f}\n")

    results = []
    for li in layers:
        H = d[f"H{li}"].astype(np.float32)
        npca = min(32, n // 6, H.shape[1])
        Hp = PCA(npca, random_state=0).fit_transform((H - H.mean(0)) / (H.std(0) + 1e-6))
        r = fit(Hp, y)
        results.append((li, r))
        print(f"  layer {li:2d}  hidden R^2={r:+.3f}")

    best = max(results, key=lambda x: x[1])
    print(f"\nBEST layer = {best[0]} (R^2={best[1]:+.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [a for a, _ in results]; ys = [b for _, b in results]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, "o-", label="hidden probe")
        plt.axhline(r0, color="gray", ls="--", label=f"prob,margin baseline ({r0:.2f})")
        plt.xlabel("layer index"); plt.ylabel("held-out R2 (predict stability)")
        plt.title("Exp5: where is commit-readiness encoded?")
        plt.legend(); plt.tight_layout()
        p = os.path.join(args.out, "layerwise.png")
        plt.savefig(p, dpi=130)
        print(f"saved {p}")
    except Exception as e:
        print("plot skip:", e)


if __name__ == "__main__":
    main()
