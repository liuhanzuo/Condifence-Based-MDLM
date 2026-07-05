"""
Exp12 偏离结构分析：dLLM 的 unmask 顺序偏离 AR，到底是
  (a) 局部小范围乱序 (如 1 2 5 3 4，偏离的token离已填区很近)
  还是
  (b) 长程跳跃 (偏离的token隔很远)

对每句 confidence-first 自发揭示，记录 order。定义每一步 t 揭示位置 pos_t 的：
  - jump = pos_t - pos_{t-1}                    相邻两步的位置跳跃(带符号)
  - dist_to_filled = min |pos_t - 已填任意位置|   到"最近已揭示位置"的距离(该步揭示时)
      * =1 表示紧贴已填区(局部生长)；大 表示跳到远处空白区
  - is_frontier_local: 该步是否落在已填区的紧邻(dist<=1)
统计这些分布 -> 判断偏离属于局部还是长程。

运行： python exp/12_deviation_structure.py --n_sents 20 --out exp/out
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


def load():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def order_of(model, L, device):
    cur = torch.full((L,), MASK_ID, device=device)
    rev = torch.zeros(L, dtype=torch.bool, device=device)
    order = []
    for _ in range(L):
        logits = model(input_ids=cur.unsqueeze(0)).logits[0]
        p1, am = torch.softmax(logits.float(), -1).max(-1)
        p1 = p1.clone(); p1[rev] = -1
        pos = int(torch.argmax(p1).item())
        cur[pos] = am[pos]; rev[pos] = True
        order.append(pos)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sents", type=int, default=20)
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok, model = load()

    jumps = []            # 相邻两步位置差(带符号)
    dist_to_filled = []   # 每步揭示位置到最近已填位置的距离
    step_frac = []        # 该步在生成中的进度(用于看远跳发生在早期还是全程)
    examples = []

    for si, sent in enumerate(SENTS[: args.n_sents]):
        ids = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = ids.shape[0]
        if L < 6:
            continue
        order = order_of(model, L, ids.device)
        examples.append((L, order))
        filled = []
        for t, pos in enumerate(order):
            if t > 0:
                jumps.append(pos - order[t - 1])
                d = min(abs(pos - f) for f in filled)
                dist_to_filled.append(d)
                step_frac.append(t / (L - 1))
            filled.append(pos)

    jumps = np.array(jumps)
    dist = np.array(dist_to_filled)
    sf = np.array(step_frac)

    print("\n===== Exp12 偏离结构 =====")
    print(f"  样本步数 = {len(jumps)}")
    print("\n[A] 相邻两步位置跳跃 |jump| 分布：")
    for thr, name in [(1, "|jump|=1  紧邻(纯局部生长)"),
                      (2, "|jump|=2"), (3, "|jump|=3")]:
        print(f"    {name:28s}: {(np.abs(jumps)==thr).mean():.3f}")
    print(f"    |jump|<=2 (局部乱序)          : {(np.abs(jumps)<=2).mean():.3f}")
    print(f"    |jump|>=5 (远程跳跃)          : {(np.abs(jumps)>=5).mean():.3f}")
    print(f"    |jump| 中位数={np.median(np.abs(jumps)):.1f} 均值={np.abs(jumps).mean():.2f} 最大={np.abs(jumps).max()}")

    print("\n[B] 每步到'最近已填位置'的距离 dist_to_filled 分布：")
    print("   （=1 表示紧贴已填区生长；大表示跳到远处空白）")
    print(f"    dist=1 (贴着已填区)           : {(dist==1).mean():.3f}")
    print(f"    dist<=2                       : {(dist<=2).mean():.3f}")
    print(f"    dist>=4 (跳到远处空白区)      : {(dist>=4).mean():.3f}")
    print(f"    dist 中位数={np.median(dist):.1f} 均值={dist.mean():.2f}")

    print("\n[C] 远跳(dist>=4)发生在生成的什么阶段：")
    far = dist >= 4
    if far.any():
        print(f"    远跳步的平均进度 step_frac = {sf[far].mean():.2f} (0=最早,1=最晚)")
        print(f"    远跳中 {(sf[far]<0.3).mean():.2f} 发生在前30%阶段")
    else:
        print("    无远跳")

    print("\n[D] 典型顺序样例：")
    for L, order in examples[:6]:
        print(f"    L={L:2d}: {order}")

    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(np.abs(jumps), bins=range(0, int(np.abs(jumps).max()) + 2),
                   color="#39c", align="left", rwidth=0.8)
        ax[0].set_xlabel("|jump| between consecutive reveals")
        ax[0].set_ylabel("count")
        ax[0].set_title("Adjacent-step jump size")
        ax[1].hist(dist, bins=range(0, int(dist.max()) + 2),
                   color="#3b7", align="left", rwidth=0.8)
        ax[1].set_xlabel("distance to nearest already-filled position")
        ax[1].set_title("Is reveal local (=1) or a far jump?")
        fig.tight_layout()
        p = os.path.join(args.out, "deviation_structure.png")
        fig.savefig(p, dpi=130)
        print(f"\nsaved {p}")
    except Exception as e:
        print("plot skip:", e)


if __name__ == "__main__":
    main()
