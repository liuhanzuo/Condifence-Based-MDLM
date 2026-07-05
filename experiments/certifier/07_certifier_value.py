"""
Exp7 certifier 实用价值验证。

问题：在"并行 commit"场景下，用【学到的 certifier(读hidden)】选择要 commit 的位置，
      是否比用【predicted confidence】选择更可靠（被后续揭示推翻的更少）？

做法（用 04 收集的 layers.npz，已有 hidden/stability/prob/margin）：
  - 把样本按 5 折切分：train 折训 certifier(Ridge on 浅层+后层 hidden PCA)，test 折评估。
  - 在 test 折上模拟"选 top-r 比例的位置来 commit"：
      * 策略C(confidence)：按 prob 降序选
      * 策略G(certifier)：按 certifier 预测的 readiness 降序选
      * 策略O(oracle)：按真实 stability 降序选（上界）
  - 指标：被选中 commit 的位置的【平均真实 stability】（越高=commit越可靠），
          以及【commit 错误率 = 1 - mean stability】。
  - 对多个 commit 比例 r 画曲线：certifier 是否在 confidence 之上、接近 oracle。

运行： python exp/07_certifier_value.py --data exp/out/layers.npz --out exp/out
"""
import argparse
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def build_features(d, layers_use):
    """拼接若干层的 hidden，做联合 PCA。"""
    n = len(d["stability"])
    mats = []
    for li in layers_use:
        key = f"H{li}"
        if key in d.files:
            H = d[key].astype(np.float32)
            mats.append((H - H.mean(0)) / (H.std(0) + 1e-6))
    X = np.concatenate(mats, 1)
    npca = min(48, n // 6, X.shape[1])
    return PCA(npca, random_state=0).fit_transform(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="exp/out/layers.npz")
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    d = np.load(args.data)
    y = d["stability"]
    prob = d["prob"]
    n = len(y)
    avail = [int(x) for x in d["sel_layers"]]
    # 用"浅层最强(2,4) + 后层稳定(18,20)"组合
    layers_use = [li for li in [2, 4, 18, 20] if li in avail] or avail[:2]
    print(f"n={n}, using layers {layers_use} for certifier")

    X = build_features(d, layers_use)

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    # 每折内评估，跨折平均
    kf = KFold(5, shuffle=True, random_state=0)
    res = {r: {"C": [], "G": [], "O": [], "rand": []} for r in ratios}

    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        cert = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
        pred = cert.predict(sc.transform(X[te]))
        yte, pte = y[te], prob[te]
        m = len(te)
        rng = np.random.default_rng(0)
        for r in ratios:
            k = max(1, int(round(r * m)))
            # 各策略选 top-k
            sel_C = np.argsort(-pte)[:k]      # confidence
            sel_G = np.argsort(-pred)[:k]     # certifier
            sel_O = np.argsort(-yte)[:k]      # oracle
            sel_R = rng.permutation(m)[:k]    # random
            res[r]["C"].append(yte[sel_C].mean())
            res[r]["G"].append(yte[sel_G].mean())
            res[r]["O"].append(yte[sel_O].mean())
            res[r]["rand"].append(yte[sel_R].mean())

    print("\n===== 并行 commit 可靠性（被选中位置的平均真实 stability，越高越好）=====")
    print(f"{'ratio':>6} | {'random':>7} | {'confidence':>10} | {'certifier':>9} | {'oracle':>7}")
    C_curve, G_curve, O_curve, R_curve = [], [], [], []
    for r in ratios:
        c = np.mean(res[r]["C"]); g = np.mean(res[r]["G"])
        o = np.mean(res[r]["O"]); rd = np.mean(res[r]["rand"])
        C_curve.append(c); G_curve.append(g); O_curve.append(o); R_curve.append(rd)
        print(f"{r:>6.1f} | {rd:>7.3f} | {c:>10.3f} | {g:>9.3f} | {o:>7.3f}")

    print("\n解读：")
    print("  - certifier > confidence  -> 读hidden选commit比用置信度更可靠（certifier有实用价值）")
    print("  - certifier ≈ confidence  -> 没超过置信度基线")
    print("  - 与 oracle 的差距 = 还有多少可提升空间")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        plt.plot(ratios, O_curve, "k--", label="oracle (upper bound)")
        plt.plot(ratios, G_curve, "o-", color="#3b7", label="certifier (hidden)")
        plt.plot(ratios, C_curve, "s-", color="#39c", label="confidence")
        plt.plot(ratios, R_curve, ":", color="gray", label="random")
        plt.xlabel("parallel commit ratio (top-r positions committed)")
        plt.ylabel("mean true stability of committed (higher=safer)")
        plt.title("Exp7: does a hidden-state certifier beat confidence?")
        plt.legend(); plt.tight_layout()
        p = os.path.join(args.out, "certifier_value.png")
        plt.savefig(p, dpi=130)
        print(f"saved {p}")
    except Exception as e:
        print("plot skip:", e)


if __name__ == "__main__":
    main()
