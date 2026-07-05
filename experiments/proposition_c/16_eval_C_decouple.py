"""
命题C第三版评估：精准解耦训练后，
  (A) 训练集外的 pairwise dependency 是否下降 / 天然可并行子集是否变大（核心机制指标）
  (B) 高并行生成质量是否不掉（对照 base 与 lora+conf，分离微调副作用）

运行： python exp/16_eval_C_decouple.py --ckpt exp/out/ckpt_C_dec --out exp/out
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data_pool import EVAL_PROMPTS

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
JUDGE_ID = "openai-community/gpt2"
MASK_ID = 151666

# 训练集外的评估句（用于测 dependency，与 TRAIN_SENTS 不同）
EVAL_SENTS = [
    "The scientist carefully recorded the results of every experiment.",
    "A large crowd gathered outside the stadium before the game.",
    "The new policy will affect millions of people next year.",
    "Bright colorful lanterns lit up the entire street at night.",
    "The professor explained why the theory was widely accepted.",
    "Fresh snow covered the mountains after the long cold night.",
    "The committee reviewed the proposal and approved the budget.",
    "Many tourists visit the old town to see its architecture.",
]


def load_base():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    m = AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True,
                                  dtype=torch.bfloat16, device_map="cuda")
    m.eval()
    return tok, m


def load_judge():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained(JUDGE_ID), \
        GPT2LMHeadModel.from_pretrained(JUDGE_ID).to("cuda").eval()


@torch.no_grad()
def klrows(logp, logq):
    p = logp.exp()
    return (p * (logp - logq)).sum(-1)


@torch.no_grad()
def measure_dependency(model, tok, rng, sents, n_anchors=4, max_t=8):
    """测 pairwise dependency 分布 + 天然独立子集比例(τ=0.1,0.3)。"""
    deps = []
    frac = {0.1: [], 0.3: []}
    for sent in sents:
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 8:
            continue
        for _ in range(n_anchors):
            keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
            nk = max(1, int(round(rng.uniform(0.15, 0.5) * L)))
            keep[torch.as_tensor(rng.choice(L, nk, replace=False), device=gt.device)] = True
            s = gt.clone(); s[~keep] = MASK_ID
            masked = torch.where(~keep)[0].tolist()
            if len(masked) < 3:
                continue
            tgt = masked if len(masked) <= max_t else list(rng.choice(masked, max_t, replace=False))
            base = torch.log_softmax(model(input_ids=s.unsqueeze(0)).logits[0], -1)
            am = base.argmax(-1)
            n = len(tgt)
            D = np.zeros((n, n))
            for b, jj in enumerate(tgt):
                s2 = s.clone(); s2[jj] = am[jj]
                lg2 = torch.log_softmax(model(input_ids=s2.unsqueeze(0)).logits[0], -1)
                for a, ii in enumerate(tgt):
                    if a == b:
                        continue
                    D[a, b] = klrows(base[ii:ii+1], lg2[ii:ii+1]).item()
            D = np.maximum(D, D.T)
            iu = np.triu_indices(n, 1)
            deps.extend(D[iu].tolist())
            for t in frac:
                order = np.argsort(D.sum(1)); chosen = []
                for i in order:
                    if all(D[i, j] < t for j in chosen):
                        chosen.append(i)
                frac[t].append(len(chosen) / n)
    return np.array(deps), {t: np.mean(v) for t, v in frac.items()}


@torch.no_grad()
def gen(model, tok, pid, gen_len, k):
    device = pid.device; P = pid.shape[0]; Ln = P + gen_len
    cur = torch.full((Ln,), MASK_ID, device=device); cur[:P] = pid
    rev = torch.zeros(Ln, dtype=torch.bool, device=device); rev[:P] = True
    while not rev.all():
        lg = model(input_ids=cur.unsqueeze(0)).logits[0]
        p1, am = torch.softmax(lg.float(), -1).max(-1)
        sc = p1.cpu().numpy().copy(); sc[rev.cpu().numpy()] = -1e9
        kk = min(k, int((~rev).sum()))
        for pos in np.argsort(-sc)[:kk]:
            cur[pos] = am[pos]; rev[pos] = True
    return cur[P:]


@torch.no_grad()
def nll(jt, jm, dtok, ids):
    t = dtok.decode(ids.tolist(), skip_special_tokens=True).strip()
    if len(t) < 2:
        return None
    enc = jt(t, return_tensors="pt").to("cuda")
    if enc.input_ids.shape[1] < 2:
        return None
    return float(jm(**enc, labels=enc.input_ids).loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exp/out/ckpt_C_dec")
    ap.add_argument("--gen_len", type=int, default=12)
    ap.add_argument("--out", default="exp/out")
    args = ap.parse_args()
    from peft import PeftModel

    dtok, base = load_base()
    jt, jm = load_judge()
    cmodel = PeftModel.from_pretrained(base, os.path.join(args.ckpt, "lora"))
    cmodel.eval()

    # ---- (A) dependency: 训练前(disable adapter) vs 训练后 ----
    print("===== (A) pairwise dependency：训练前 vs 训练后（训练集外句子）=====")
    rng = np.random.default_rng(0)
    with cmodel.disable_adapter():
        dep_before, frac_before = measure_dependency(cmodel, dtok, np.random.default_rng(0), EVAL_SENTS)
    dep_after, frac_after = measure_dependency(cmodel, dtok, np.random.default_rng(0), EVAL_SENTS)
    print(f"  dep 中位数:  before={np.median(dep_before):.3f}  after={np.median(dep_after):.3f}")
    print(f"  dep>0.3占比: before={(dep_before>0.3).mean():.3f}  after={(dep_after>0.3).mean():.3f}")
    print(f"  独立子集比例 τ=0.1: before={frac_before[0.1]:.3f}  after={frac_after[0.1]:.3f}")
    print(f"  独立子集比例 τ=0.3: before={frac_before[0.3]:.3f}  after={frac_after[0.3]:.3f}")
    print("  解读：after 的 dep 下降 / 独立子集变大 -> 解耦训练确实增加了条件独立结构(机制成立)")

    # ---- (B) 生成质量：base(confidence) vs lora+confidence ----
    print("\n===== (B) 高并行生成质量 NLL（独立GPT-2裁判, 训练集外prompt）=====")
    ks = [1, 3, 6, args.gen_len]
    res = {"base": {k: [] for k in ks}, "lora": {k: [] for k in ks}}
    for p in EVAL_PROMPTS:
        pid = dtok(p, return_tensors="pt").input_ids[0].to("cuda")
        for k in ks:
            with cmodel.disable_adapter():
                v = nll(jt, jm, dtok, gen(cmodel, dtok, pid, args.gen_len, k))
                if v: res["base"][k].append(v)
            v = nll(jt, jm, dtok, gen(cmodel, dtok, pid, args.gen_len, k))
            if v: res["lora"][k].append(v)
    print(f"  {'k':>4} | {'base(conf)':>10} | {'decoupled(conf)':>15}")
    for k in ks:
        a = np.mean(res["base"][k]); b = np.mean(res["lora"][k])
        print(f"  {k:>4} | {a:>10.3f} | {b:>15.3f}")
    hi = [k for k in ks if k >= 3 and k < args.gen_len]
    if hi:
        ba = np.mean([np.mean(res["base"][k]) for k in hi])
        la = np.mean([np.mean(res["lora"][k]) for k in hi])
        print(f"\n  高并行(k>=3且<{args.gen_len}) 平均NLL: base={ba:.3f} decoupled={la:.3f} Δ={la-ba:+.3f}")
        print("  Δ<0 -> 解耦训练后并行生成质量提升(且是解耦贡献,因两者都用confidence planner)")
        print("  Δ>=0-> 解耦没帮助生成质量(即使dep降了)")


if __name__ == "__main__":
    main()
