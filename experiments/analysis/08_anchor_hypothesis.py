"""
Exp8 锚点假说检验：模型"锚点优先"揭示，到底是【位置性】(先填首尾)还是【语义性】(先填内容词)？

对每句用 confidence-first 自发揭示(同 Exp6)，得到每个位置的揭示步序 step_of_pos。
对每个位置计算特征：
  - is_edge: 是否靠近句首/句尾(前2或后2个位置)
  - is_content: 是否内容词(非功能词/非标点)——用封闭功能词表启发式
  - abs_center_dist: 到句子中心的距离(位置性)
然后回归/相关：揭示步序 早(小) 更由哪个特征解释？
  - 若 is_edge/中心距离 主导 -> 位置性(平凡)
  - 若 is_content 主导(控制位置后仍显著) -> 语义性(支持 information-unit 假说)

运行： python exp/08_anchor_hypothesis.py --n_sents 20 --out exp/out
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

# 封闭类功能词（冠词/介词/连词/代词/助动词/be动词等）
FUNCTION_WORDS = set("""
a an the of in on at to for and or but is are was were be been being am
it its it's he she they we you i him her them us my your his their our
this that these those with by from as into onto over under above below
do does did have has had will would shall should can could may might must
not no nor so than then there here who whom whose which what when where why how
up out off down about after before between through during
""".split())


def load_model():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def confidence_first_order(model, L, device):
    cur = torch.full((L,), MASK_ID, device=device)
    revealed = torch.zeros(L, dtype=torch.bool, device=device)
    order = []
    for _ in range(L):
        logits = model(input_ids=cur.unsqueeze(0)).logits[0]
        probs = torch.softmax(logits.float(), -1)
        p1, am = probs.max(-1)
        p1 = p1.clone(); p1[revealed] = -1.0
        pos = int(torch.argmax(p1).item())
        cur[pos] = am[pos]; revealed[pos] = True
        order.append(pos)
    return order


def is_content_token(s):
    t = s.strip().lower().replace("Ġ", "").replace("▁", "")
    if not t or not any(c.isalnum() for c in t):
        return 0  # 标点/空 -> 功能性
    if t in FUNCTION_WORDS:
        return 0
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sents", type=int, default=20)
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok, model = load_model()

    step, is_edge, is_content, center_dist = [], [], [], []
    for si, sent in enumerate(SENTS[: args.n_sents]):
        ids = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = ids.shape[0]
        if L < 6:
            continue
        order = confidence_first_order(model, L, ids.device)
        step_of_pos = np.zeros(L)
        for st, pos in enumerate(order):
            step_of_pos[pos] = st / max(1, L - 1)  # 归一化揭示早晚 [0,1]
        toks = tok.convert_ids_to_tokens(ids.tolist())
        for p in range(L):
            step.append(step_of_pos[p])
            is_edge.append(1 if (p < 2 or p >= L - 2) else 0)
            is_content.append(is_content_token(toks[p]))
            center_dist.append(abs(p - (L - 1) / 2) / ((L - 1) / 2))
        print(f"[{si+1}] L={L}")

    step = np.array(step); is_edge = np.array(is_edge)
    is_content = np.array(is_content); center_dist = np.array(center_dist)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = (ra - ra.mean()) / (ra.std() + 1e-9); rb = (rb - rb.mean()) / (rb.std() + 1e-9)
        return float((ra * rb).mean())

    print("\n===== Exp8 锚点：位置性 vs 语义性 =====")
    print(f"  n tokens = {len(step)}")
    print(f"  内容词占比: {is_content.mean():.2f}, 边缘位置占比: {is_edge.mean():.2f}")
    print("  (揭示步序越小=越早揭示)")
    print(f"  corr(揭示步序, is_edge)     = {spearman(step, is_edge):+.3f}  (负=边缘更早)")
    print(f"  corr(揭示步序, center_dist) = {spearman(step, center_dist):+.3f}  (负=越靠边越早)")
    print(f"  corr(揭示步序, is_content)  = {spearman(step, is_content):+.3f}  (负=内容词更早)")

    # 控制位置后，内容词是否仍显著更早？——线性回归偏系数
    from sklearn.linear_model import LinearRegression
    X = np.stack([is_edge, center_dist, is_content], 1).astype(float)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    reg = LinearRegression().fit(Xs, step)
    print("\n  多元回归(标准化)预测揭示步序的偏系数(负=促使更早揭示)：")
    print(f"    is_edge     : {reg.coef_[0]:+.3f}")
    print(f"    center_dist : {reg.coef_[1]:+.3f}")
    print(f"    is_content  : {reg.coef_[2]:+.3f}  <- 控制位置后内容词的独立效应")
    print("\n解读：若控制位置(edge/center)后 is_content 仍有明显负系数 -> 语义性锚点(支持information-unit)")
    print("      若 is_content 系数≈0 而位置项主导 -> 纯位置性锚点(平凡)")

    # 分组均值
    print("\n  分组：内容词 vs 功能词 的平均揭示步序")
    print(f"    内容词: {step[is_content == 1].mean():.3f}")
    print(f"    功能词: {step[is_content == 0].mean():.3f}")


if __name__ == "__main__":
    main()
