"""
Exp10 真实并行生成对比：base(confidence planner) vs certifier planner，
在不同并行预算下生成整句，用【独立裁判模型(GPT-2)】测质量(NLL)。

补掉 Exp7 的漏洞：这里是真的用两种策略各自生成整句，而非只评估现成状态。

流程：
  1) 用少量句子离线训一个 certifier（Ridge: 浅层+后层 hidden PCA -> stability），
     stability 标签用蒙特卡洛在 random 轨迹上估（同 04）。—— 复用 layers.npz 直接训。
  2) 对一批 prompt，从全 mask 起，每步敲定 k 个位置(k 由并行度决定)：
       - confidence 策略：选当前 top-k prob 的位置，填其 argmax
       - certifier 策略：选当前 certifier 分 top-k 的位置，填其 argmax
     直到填满，得到整句。
  3) 用 GPT-2(独立裁判)算生成句 NLL（越低越流畅）。
  4) 扫并行度(每步 k)，画 质量 vs 并行度 曲线。

运行：
  python exp/10_generate_eval.py --out exp/out
"""
import argparse
import os
import numpy as np
import torch

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
JUDGE_ID = "openai-community/gpt2"
MASK_ID = 151666

PROMPTS = [
    "The weather today is",
    "My favorite kind of food is",
    "In the morning I usually",
    "The most important thing in life is",
    "Scientists recently discovered that",
    "When I travel I always",
    "The best way to learn a language is",
    "Yesterday the news reported that",
    "A good friend is someone who",
    "The future of technology will",
    "During the summer many people",
    "Reading a book can help you",
    "The city was crowded because",
    "She decided to leave early since",
    "The experiment failed because",
    "Music has the power to",
]


def load_dream():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DREAM_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def load_judge():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    jt = GPT2TokenizerFast.from_pretrained(JUDGE_ID)
    jm = GPT2LMHeadModel.from_pretrained(JUDGE_ID).to("cuda").eval()
    return jt, jm


def train_certifier(layers_npz):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    d = np.load(layers_npz)
    y = d["stability"]; n = len(y)
    avail = [int(x) for x in d["sel_layers"]]
    use = [li for li in [2, 4, 18, 20] if li in avail]
    mats = []
    for li in use:
        H = d[f"H{li}"].astype(np.float32)
        mats.append((H - H.mean(0)) / (H.std(0) + 1e-6))
    X = np.concatenate(mats, 1)
    pca = PCA(min(48, n // 6, X.shape[1]), random_state=0).fit(X)
    Xp = pca.transform(X)
    sc = StandardScaler().fit(Xp)
    reg = Ridge(alpha=10.0).fit(sc.transform(Xp), y)
    # 返回一个从"多层hidden拼接"到分数的闭包所需组件
    means = {li: (np.load(layers_npz)[f"H{li}"].astype(np.float32).mean(0),
                  np.load(layers_npz)[f"H{li}"].astype(np.float32).std(0) + 1e-6)
             for li in use}
    return {"use": use, "pca": pca, "sc": sc, "reg": reg, "means": means}


def certifier_score(cert, hs_layers):
    """hs_layers: dict li-> hidden[L,H] (numpy). 返回每个位置的分数[L]。"""
    mats = []
    for li in cert["use"]:
        m, s = cert["means"][li]
        mats.append((hs_layers[li] - m) / s)
    X = np.concatenate(mats, 1)
    Xp = cert["pca"].transform(X)
    return cert["reg"].predict(cert["sc"].transform(Xp))


@torch.no_grad()
def dream_forward(model, ids, want_layers):
    out = model(input_ids=ids.unsqueeze(0), output_hidden_states=True)
    logits = out.logits[0]
    hs = {li: out.hidden_states[li][0].float().cpu().numpy() for li in want_layers}
    return logits, hs


@torch.no_grad()
def generate(model, prompt_ids, gen_len, k, strategy, cert=None):
    """prompt_ids: 已知前缀; 之后 gen_len 个位置从 mask 开始按策略并行敲定。"""
    device = prompt_ids.device
    P = prompt_ids.shape[0]
    L = P + gen_len
    cur = torch.full((L,), MASK_ID, device=device)
    cur[:P] = prompt_ids
    revealed = torch.zeros(L, dtype=torch.bool, device=device)
    revealed[:P] = True
    want = cert["use"] if (strategy == "certifier" and cert) else []
    while not revealed.all():
        logits, hs = dream_forward(model, cur, want)
        probs = torch.softmax(logits.float(), -1)
        p1, am = probs.max(-1)
        cand = (~revealed).cpu().numpy()
        if strategy == "confidence":
            score = p1.cpu().numpy()
        else:
            score = certifier_score(cert, hs)
        score = score.copy(); score[~cand] = -1e9
        n_left = int(cand.sum())
        kk = min(k, n_left)
        sel = np.argsort(-score)[:kk]
        for pos in sel:
            cur[pos] = am[pos]
            revealed[pos] = True
    return cur[P:]  # 只返回生成部分


@torch.no_grad()
def judge_nll(jt, jm, dream_tok, gen_ids):
    text = dream_tok.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    if len(text) < 2:
        return None, text
    enc = jt(text, return_tensors="pt").to("cuda")
    if enc.input_ids.shape[1] < 2:
        return None, text
    out = jm(**enc, labels=enc.input_ids)
    return float(out.loss.item()), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="exp/out/layers.npz")
    ap.add_argument("--gen_len", type=int, default=12)
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("training certifier from", args.layers)
    cert = train_certifier(args.layers)
    print("certifier layers:", cert["use"])

    dream_tok, dream = load_dream()
    jt, jm = load_judge()

    ks = [1, 2, 3, 4, 6, args.gen_len]  # 每步敲定数：1≈AR ... gen_len=一步全并行
    res = {"confidence": {k: [] for k in ks}, "certifier": {k: [] for k in ks}}

    for pi, p in enumerate(PROMPTS):
        pid = dream_tok(p, return_tensors="pt").input_ids[0].to("cuda")
        for k in ks:
            for strat in ("confidence", "certifier"):
                gen = generate(dream, pid, args.gen_len, k, strat, cert)
                nll, _ = judge_nll(jt, jm, dream_tok, gen)
                if nll is not None:
                    res[strat][k].append(nll)
        print(f"[{pi+1}/{len(PROMPTS)}] {p!r} done")

    print("\n===== Exp10 质量(NLL,越低越好) vs 并行度(每步敲定k, 大=更并行) =====")
    print(f"{'k':>4} | {'steps':>5} | {'confidence':>10} | {'certifier':>9} | {'winner':>9}")
    conf_curve, cert_curve = [], []
    for k in ks:
        c = np.mean(res["confidence"][k]) if res["confidence"][k] else float("nan")
        g = np.mean(res["certifier"][k]) if res["certifier"][k] else float("nan")
        conf_curve.append(c); cert_curve.append(g)
        steps = int(np.ceil(args.gen_len / k))
        win = "certifier" if g < c else "confidence"
        print(f"{k:>4} | {steps:>5} | {c:>10.3f} | {g:>9.3f} | {win:>9}")

    print("\n解读：")
    print("  - k 越大=每步敲定越多=越并行=步数越少（k=gen_len 时一步生成完）")
    print("  - 若并行度提高时 certifier 的 NLL 比 confidence 涨得慢 -> certifier 在真实并行生成中更抗退化")
    print("  - 若两者接近 -> 我们相对 confidence planner 优势有限（印证 'vanilla/只是planner' 的担心）")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [int(np.ceil(args.gen_len / k)) for k in ks]
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(ks, conf_curve, "s-", color="#39c", label="confidence planner (base)")
        plt.plot(ks, cert_curve, "o-", color="#3b7", label="certifier planner (ours)")
        plt.xlabel("commits per step k  (larger = more parallel, fewer steps)")
        plt.ylabel("judge NLL of generation (lower=better)")
        plt.title("Exp10: quality vs parallelism (base vs ours)")
        plt.legend(); plt.tight_layout()
        pth = os.path.join(args.out, "gen_quality.png")
        plt.savefig(pth, dpi=130)
        print("saved", pth)
    except Exception as e:
        print("plot skip:", e)


if __name__ == "__main__":
    main()
