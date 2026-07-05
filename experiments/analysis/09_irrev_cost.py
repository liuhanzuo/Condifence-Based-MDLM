"""
Exp9 不可逆性的代价（正确版，复用 04 的 layers.npz）。

思路修正：
  "看到完整句子后重预测"会与生成结果自洽(循环论证)，测不出后悔。
  真正的不可逆代价 = 在【上下文稀疏的早期】按 confidence 决定 commit 的位置里，
  有多少其 Stability<1（即后续揭示会推翻它）。这些就是"不可逆性造成的必然错误"。

用 layers.npz 里的 prob / stability / nrev：
  - 定义 commit 规则：prob >= 阈值 就 commit。
  - 对不同 nrev 档(早/中/晚)统计：被 commit 的位置里 stability<1 的比例(=错误 commit 率)。
  - 这直接量化"在信息不足时按 confidence 提前 commit 的代价随阶段如何变化"。

运行： python exp/09_irrev_cost.py --data exp/out/layers.npz
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="exp/out/layers.npz")
    ap.add_argument("--thresh", type=float, default=0.9, help="confidence commit 阈值")
    args = ap.parse_args()
    d = np.load(args.data)
    prob, stab, nrev = d["prob"], d["stability"], d["nrev"]
    n = len(stab)

    committed = prob >= args.thresh
    print(f"n={n}, confidence>={args.thresh} 的 commit 位置数 = {committed.sum()} "
          f"({committed.mean():.1%})")
    if committed.sum() == 0:
        print("无 commit，降低阈值再试"); return

    # 错误 commit = 被 commit 但 stability<1（后续会推翻）
    err = (stab < 0.999)
    print(f"\n===== 不可逆代价：按 confidence>={args.thresh} 提前 commit 的错误率 =====")
    print(f"  全部 commit 位置中 会被推翻(stab<1) 的比例 = {err[committed].mean():.3f}")

    # 分阶段（早/中/晚 by nrev）
    qs = np.quantile(nrev, [0, 0.33, 0.66, 1.0])
    print(f"\n  按揭示进度 nrev 分档（qs={[f'{q:.0f}' for q in qs]}）：")
    for bi in range(3):
        lo, hi = qs[bi], qs[bi + 1]
        m = (nrev >= lo) & (nrev <= hi if bi == 2 else nrev < hi)
        cm = committed & m
        tag = ["早(稀疏)", "中", "晚(充分)"][bi]
        if cm.sum() == 0:
            print(f"    {tag} nrev∈[{lo:.0f},{hi:.0f}]: 无commit")
            continue
        print(f"    {tag} nrev∈[{lo:.0f},{hi:.0f}]: commit数={cm.sum():3d} "
              f"错误commit率={err[cm].mean():.3f} "
              f"该档平均stability={stab[m].mean():.3f}")

    print("\n解读：")
    print("  - 早期错误commit率 >> 晚期  -> 不可逆性主要在早期(上下文稀疏)造成代价")
    print("    这正是'必须先建锚点/延后commit'的定量依据，也解释了 AR/block-wise 的保守性。")
    print("  - 若 confidence 阈值很高但早期错误率仍不低 -> confidence 防不住早期勉强commit")
    print("    (呼应 Exp7: certifier 在此更有价值)")


if __name__ == "__main__":
    main()
