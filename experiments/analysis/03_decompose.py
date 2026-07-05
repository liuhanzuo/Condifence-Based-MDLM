"""
拆解命题B里 "hidden+R (P3)" 大幅提升的来源：
   平凡的【全局进度效应】(揭示越多越稳)  vs  真正的【位置间耦合/path-dependence】

关键对照（都用 PCA 后的 hidden 作为 state-local 特征）：
  A. hidden                      —— 纯 state-local
  B. hidden + progress           —— progress = [keep_ratio, nrev, seqlen]（平凡进度）
  C. hidden + progress + local   —— local = [leftdist, rightdist, relpos]（局部结构）
  D. progress only               —— 只用进度能预测多少（衡量"平凡上界"）

判定逻辑：
  - 若 B >> A 而 C ≈ B  ->  P3 的提升几乎全来自【进度】(平凡)，
        说明扣掉进度后 hidden 已接近饱和 -> commit-readiness 更接近 state-local（对 certifier 有利）
  - 若 C >> B          ->  存在真正的【局部耦合】-> path-dependent（对纯 state-local certifier 不利）

另做【同进度分层】：在每个 nrev 档位内部比较 A vs (A+local)，
  控制住进度后看局部结构是否还有增量。

运行： python exp/03_decompose.py --data exp/out/data.npz
"""

import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def fit(X, y, seed=0):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    r2s = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
        r2s.append(r2_score(y[te], m.predict(sc.transform(X[te]))))
    return float(np.mean(r2s)), float(np.std(r2s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="exp/out/data.npz")
    args = ap.parse_args()
    d = np.load(args.data)
    y = d["stability"]
    n = len(y)
    print(f"n = {n}, stability mean={y.mean():.3f} std={y.std():.3f}")

    H = d["H"]
    n_pca = min(32, n // 6, H.shape[1])
    Hp = PCA(n_components=n_pca, random_state=0).fit_transform(
        (H - H.mean(0)) / (H.std(0) + 1e-6))

    progress = np.stack([d["keep_ratio"], d["nrev"], d["seqlen"]], 1)
    local = np.stack([d["leftdist"], d["rightdist"], d["relpos"]], 1)

    print(f"\n(hidden PCA -> {n_pca} 维)")
    rA, sA = fit(Hp, y)
    rB, sB = fit(np.concatenate([Hp, progress], 1), y)
    rC, sC = fit(np.concatenate([Hp, progress, local], 1), y)
    rD, sD = fit(progress, y)
    rL, sL = fit(local, y)
    print("===== 来源拆解 =====")
    print(f"  D  progress only              R^2 = {rD:+.3f} ± {sD:.3f}  (平凡进度能解释多少)")
    print(f"  L  local-structure only       R^2 = {rL:+.3f} ± {sL:.3f}")
    print(f"  A  hidden (state-local)       R^2 = {rA:+.3f} ± {sA:.3f}  <- certifier 能拿到的")
    print(f"  B  hidden + progress          R^2 = {rB:+.3f} ± {sB:.3f}")
    print(f"  C  hidden + progress + local  R^2 = {rC:+.3f} ± {sC:.3f}")
    print(f"\n  Δ(B-A) 进度带来的增量 = {rB-rA:+.3f}")
    print(f"  Δ(C-B) 局部结构额外增量 = {rC-rB:+.3f}")

    # ===== 同进度分层：控制 nrev 后，hidden 是否还需要 local =====
    print("\n===== 同进度分层（控制揭示进度）=====")
    nrev = d["nrev"]
    qs = np.quantile(nrev, [0, 0.33, 0.66, 1.0])
    for bi in range(3):
        lo, hi = qs[bi], qs[bi + 1]
        idx = np.where((nrev >= lo) & (nrev <= hi))[0] if bi == 2 \
            else np.where((nrev >= lo) & (nrev < hi))[0]
        if len(idx) < 60:
            print(f"  bin nrev∈[{lo:.0f},{hi:.0f}]  n={len(idx)} 太少，跳过")
            continue
        Hb = PCA(n_components=min(16, len(idx)//6), random_state=0).fit_transform(
            (H[idx] - H[idx].mean(0)) / (H[idx].std(0) + 1e-6))
        rHa, _ = fit(Hb, y[idx])
        rHl, _ = fit(np.concatenate([Hb, local[idx]], 1), y[idx])
        print(f"  bin nrev∈[{lo:.0f},{hi:.0f}] n={len(idx)}: "
              f"hidden R^2={rHa:+.3f} | hidden+local R^2={rHl:+.3f} | Δlocal={rHl-rHa:+.3f}")
    print("\n解读：若各 bin 内 Δlocal 都很小 -> 控制进度后局部结构没啥用 -> readiness 更 state-local")
    print("      若各 bin 内 Δlocal 明显>0 -> 即使同进度,谁被揭示也重要 -> 真 path-dependent")


if __name__ == "__main__":
    main()
