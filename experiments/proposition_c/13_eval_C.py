"""
命题 C 评估：对比三种生成策略的 质量 vs 并行度 曲线
  1) base confidence planner            (Exp10 蓝线)
  2) inference-only certifier planner    (Exp10 绿线, 不训练)
  3) 命题C: LoRA微调模型 + 训练的 readiness head 作 planner  (新)

关键判定：命题C 能否在【高并行区间(k>=4)】把生成质量压到 inference-only certifier 之下？
  若能 -> 训练诱导表征带来了 inference-time planner 给不了的收益(范式级证据)
  若不能 -> 训练没带来额外好处，certifier 已是天花板

用独立裁判 GPT-2 测 NLL。评估 prompt 与训练句子不同(避免过拟合污染)。

运行： python exp/13_eval_C.py --ckpt exp/out/ckpt_C --layers exp/out/layers.npz --out exp/out
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
JUDGE_ID = "openai-community/gpt2"
MASK_ID = 151666
L_READ = 4

PROMPTS = [
    "The weather today is", "My favorite kind of food is", "In the morning I usually",
    "The most important thing in life is", "Scientists recently discovered that",
    "When I travel I always", "The best way to learn a language is",
    "Yesterday the news reported that", "A good friend is someone who",
    "The future of technology will", "During the summer many people",
    "Reading a book can help you", "The city was crowded because",
    "She decided to leave early since", "The experiment failed because",
    "Music has the power to",
]


def load_base():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def load_judge():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    jt = GPT2TokenizerFast.from_pretrained(JUDGE_ID)
    jm = GPT2LMHeadModel.from_pretrained(JUDGE_ID).to("cuda").eval()
    return jt, jm


def train_infer_certifier(layers_npz):
    """inference-only certifier(线性), 复用 07/10 的做法。"""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    d = np.load(layers_npz)
    y = d["stability"]; n = len(y)
    avail = [int(x) for x in d["sel_layers"]]
    use = [li for li in [2, 4, 18, 20] if li in avail]
    mats, means = [], {}
    for li in use:
        Hh = d[f"H{li}"].astype(np.float32)
        m, s = Hh.mean(0), Hh.std(0) + 1e-6
        means[li] = (m, s); mats.append((Hh - m) / s)
    X = np.concatenate(mats, 1)
    pca = PCA(min(48, n // 6, X.shape[1]), random_state=0).fit(X)
    sc = StandardScaler().fit(pca.transform(X))
    reg = Ridge(alpha=10.0).fit(sc.transform(pca.transform(X)), y)
    return {"use": use, "pca": pca, "sc": sc, "reg": reg, "means": means}


def infer_cert_score(cert, hs):
    mats = []
    for li in cert["use"]:
        m, s = cert["means"][li]
        mats.append((hs[li] - m) / s)
    X = np.concatenate(mats, 1)
    return cert["reg"].predict(cert["sc"].transform(cert["pca"].transform(X)))


@torch.no_grad()
def gen(model, tok, pid, gen_len, k, strategy, cert=None, head=None,
        want_layers=(), is_peft=False):
    device = pid.device
    P = pid.shape[0]; L = P + gen_len
    cur = torch.full((L,), MASK_ID, device=device); cur[:P] = pid
    rev = torch.zeros(L, dtype=torch.bool, device=device); rev[:P] = True
    while not rev.all():
        need_hs = strategy != "confidence"
        out = model(input_ids=cur.unsqueeze(0), output_hidden_states=need_hs)
        logits = out.logits[0]
        probs = torch.softmax(logits.float(), -1)
        p1, am = probs.max(-1)
        cand = (~rev).cpu().numpy()
        if strategy == "confidence":
            score = p1.cpu().numpy()
        elif strategy == "infer_cert":
            hs = {li: out.hidden_states[li][0].float().cpu().numpy() for li in cert["use"]}
            score = infer_cert_score(cert, hs)
        else:  # trained head
            h = out.hidden_states[L_READ][0].float()
            score = head(h).squeeze(-1).detach().cpu().numpy()
        score = score.copy(); score[~cand] = -1e9
        n_left = int(cand.sum()); kk = min(k, n_left)
        for pos in np.argsort(-score)[:kk]:
            cur[pos] = am[pos]; rev[pos] = True
    return cur[P:]


@torch.no_grad()
def nll(jt, jm, dtok, ids):
    text = dtok.decode(ids.tolist(), skip_special_tokens=True).strip()
    if len(text) < 2:
        return None
    enc = jt(text, return_tensors="pt").to("cuda")
    if enc.input_ids.shape[1] < 2:
        return None
    return float(jm(**enc, labels=enc.input_ids).loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exp/out/ckpt_C")
    ap.add_argument("--layers", default="exp/out/layers.npz")
    ap.add_argument("--gen_len", type=int, default=12)
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()

    from peft import PeftModel
    dtok, base = load_base()
    jt, jm = load_judge()
    cert = train_infer_certifier(args.layers)

    # 命题C模型：加载 LoRA + head
    Hs = base.config.hidden_size
    head = nn.Sequential(nn.Linear(Hs, 256), nn.GELU(), nn.Linear(256, 1)).to("cuda").float()
    head.load_state_dict(torch.load(os.path.join(args.ckpt, "head.pt")))
    head.eval()
    cmodel = PeftModel.from_pretrained(base, os.path.join(args.ckpt, "lora"))
    cmodel.eval()

    ks = [1, 2, 3, 4, 6, args.gen_len]
    res = {"confidence": {k: [] for k in ks},
           "infer_cert": {k: [] for k in ks},
           "trainedC": {k: [] for k in ks},
           "lora_conf": {k: [] for k in ks}}  # 对照：LoRA模型+confidence planner

    for pi, p in enumerate(PROMPTS):
        pid = dtok(p, return_tensors="pt").input_ids[0].to("cuda")
        for k in ks:
            # base 模型跑 confidence / infer_cert（关闭 LoRA）
            with cmodel.disable_adapter():
                g1 = gen(cmodel, dtok, pid, args.gen_len, k, "confidence")
                v = nll(jt, jm, dtok, g1)
                if v is not None: res["confidence"][k].append(v)
                g2 = gen(cmodel, dtok, pid, args.gen_len, k, "infer_cert", cert=cert)
                v = nll(jt, jm, dtok, g2)
                if v is not None: res["infer_cert"][k].append(v)
            # 命题C：LoRA 开 + trained head 作 planner
            g3 = gen(cmodel, dtok, pid, args.gen_len, k, "trainedC", head=head)
            v = nll(jt, jm, dtok, g3)
            if v is not None: res["trainedC"][k].append(v)
            # 对照：LoRA 开 + confidence planner（分离"微调增益" vs "readiness planner增益"）
            g4 = gen(cmodel, dtok, pid, args.gen_len, k, "confidence")
            v = nll(jt, jm, dtok, g4)
            if v is not None: res["lora_conf"][k].append(v)
        print(f"[{pi+1}/{len(PROMPTS)}] done")

    print("\n===== 命题C 评估：质量(NLL,低=好) vs 并行度 =====")
    print(f"{'k':>4} | {'conf(base)':>10} | {'infer_cert':>10} | {'lora+conf':>9} | {'trainedC':>9}")
    cf, ic, tc, lc = [], [], [], []
    for k in ks:
        a = np.mean(res["confidence"][k]) if res["confidence"][k] else float("nan")
        b = np.mean(res["infer_cert"][k]) if res["infer_cert"][k] else float("nan")
        c = np.mean(res["trainedC"][k]) if res["trainedC"][k] else float("nan")
        e = np.mean(res["lora_conf"][k]) if res["lora_conf"][k] else float("nan")
        cf.append(a); ic.append(b); tc.append(c); lc.append(e)
        print(f"{k:>4} | {a:>10.3f} | {b:>10.3f} | {e:>9.3f} | {c:>9.3f}")

    # 高并行区间(k>=4)对比 + 关键的对照分解
    hi = [i for i, k in enumerate(ks) if k >= 4 and k < args.gen_len]
    if hi:
        ic_hi = np.nanmean([ic[i] for i in hi]); tc_hi = np.nanmean([tc[i] for i in hi])
        lc_hi = np.nanmean([lc[i] for i in hi])
        print(f"\n高并行区间(k>=4) 平均NLL:")
        print(f"  infer_cert(无训练)      = {ic_hi:.3f}")
        print(f"  lora+conf(微调,非readiness planner) = {lc_hi:.3f}  <- 对照:纯微调增益")
        print(f"  trainedC(微调+readiness planner)    = {tc_hi:.3f}")
        print(f"  Δ(trainedC - lora_conf) = {tc_hi - lc_hi:+.3f}  <- readiness planner 的净增益")
        print("  解读：")
        print("   - 若 trainedC << lora_conf -> readiness planner 本身有贡献(命题C真成立)")
        print("   - 若 trainedC ≈ lora_conf  -> 增益几乎全来自LoRA微调本身,与readiness表征无关(证据不成立)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.8))
        plt.plot(ks, cf, "s-", color="#39c", label="confidence (base)")
        plt.plot(ks, ic, "o-", color="#3b7", label="inference-only certifier")
        plt.plot(ks, lc, "d--", color="#999", label="LoRA + confidence (ablation)")
        plt.plot(ks, tc, "^-", color="#c63", label="trained (prop.C: LoRA+readiness)")
        plt.xlabel("commits per step k (larger=more parallel)")
        plt.ylabel("judge NLL (lower=better)")
        plt.title("Prop.C: does training the representation beat inference-only?")
        plt.legend(); plt.tight_layout()
        pth = os.path.join(args.out, "propC_quality.png")
        plt.savefig(pth, dpi=130); print("saved", pth)
    except Exception as e:
        print("plot skip:", e)


if __name__ == "__main__":
    main()
