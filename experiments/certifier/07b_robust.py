"""
Exp7b 稳健性：多随机种子重复 Exp7 的核心对比，确认 certifier > confidence 稳定成立。
运行： python exp/07b_robust.py --data exp/out/layers.npz
"""
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="exp/out/layers.npz")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    d = np.load(args.data)
    y = d["stability"]; prob = d["prob"]; n = len(y)
    avail = [int(x) for x in d["sel_layers"]]
    layers_use = [li for li in [2, 4, 18, 20] if li in avail]

    mats = []
    for li in layers_use:
        H = d[f"H{li}"].astype(np.float32)
        mats.append((H - H.mean(0)) / (H.std(0) + 1e-6))
    Xall = np.concatenate(mats, 1)
    X = PCA(min(48, n // 6, Xall.shape[1]), random_state=0).fit_transform(Xall)

    ratio = 0.5  # 最激进并行度，优势最明显处
    diffs = []
    for seed in range(args.seeds):
        kf = KFold(5, shuffle=True, random_state=seed)
        c_all, g_all = [], []
        for tr, te in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            cert = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
            pred = cert.predict(sc.transform(X[te]))
            yte, pte = y[te], prob[te]
            k = max(1, int(round(ratio * len(te))))
            c_all.append(yte[np.argsort(-pte)[:k]].mean())
            g_all.append(yte[np.argsort(-pred)[:k]].mean())
        c, g = np.mean(c_all), np.mean(g_all)
        diffs.append(g - c)
        print(f"seed {seed}: confidence={c:.4f}  certifier={g:.4f}  diff={g-c:+.4f}")

    diffs = np.array(diffs)
    print(f"\ncertifier - confidence @ratio={ratio}: "
          f"mean={diffs.mean():+.4f} std={diffs.std():.4f} "
          f"min={diffs.min():+.4f}  (>0 的种子数: {(diffs>0).sum()}/{len(diffs)})")
    if (diffs > 0).all():
        print(">>> 稳健：所有种子下 certifier 均优于 confidence")
    else:
        print(">>> 不稳健：部分种子 certifier 未超过 confidence")


if __name__ == "__main__":
    main()
