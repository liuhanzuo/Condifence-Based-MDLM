"""
Exp19：快速诊断——命题C-v3的解耦损失是否误伤"推理必要耦合"(Flexibility Trap 警示)。

背景：The Flexibility Trap (arXiv:2601.15165) 指出 dLLM 的 easy-first unmask 策略会让模型
绕开高熵的逻辑分叉token(therefore/thus/since等)。我们的解耦损失是"无差别打当前最强依赖的pair"，
不区分这个依赖是"冗余"还是"推理必要的分叉耦合"。本实验直接统计：在GSM8K推理链上，
被我们算法选中要解耦的strong-dependency pair，其token具体是什么，是否显著落在推理连接词/
关系词上。

【结果预告，见 docs/findings.md Exp19 章节】：fork/connective 只占 3.6%，原始假说证据不强；
但数字+运算符合计~48%，说明真正的问题是"解耦在打数字间依赖"（对GSM8K算术正确性直接有害）。

方法：复用 15_train_C_decouple.py 里"找强依赖pair"的逻辑（无梯度，不训练），
在若干GSM8K训练样本的assistant回复片段上跑一遍，统计被选中pair里每个位置对应的token文本，
按类别归类：
  - fork/连接词类: therefore, thus, since, so, because, if, then, now, given, however
  - 数字/运算符类: 数字token、+-*/=、百分号等
  - 其他（限定词/介词/名词等冗余候选）

运行：
  python proposition_c/19_diagnose_fork_tokens.py --n 60 --out out
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666

FORK_WORDS = {
    "therefore", "thus", "since", "so", "because", "if", "then", "now",
    "given", "however", "hence", "so,", "thus,", "since,", "because,",
    "but", "and", "or", "since", "as", "when", "while", "although",
}


def load():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return tok, model


def classify_token(text):
    t = text.strip().lower()
    if not t:
        return "empty/space"
    if t in FORK_WORDS:
        return "fork/connective"
    if any(c.isdigit() for c in t):
        return "number"
    if t in {"+", "-", "*", "/", "=", "%", "$", "####"}:
        return "operator/symbol"
    if len(t) <= 3 and t.isalpha():
        return "short_word(article/prep/etc)"
    return "other_content_word"


@torch.no_grad()
def kl_rows(logp, logq):
    p = logp.exp()
    return (p * (logp - logq)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="用多少GSM8K训练样本统计")
    ap.add_argument("--dep_thresh", type=float, default=0.3)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tok, model = load()
    from gsm8k_corpus import load_gsm8k_chat_train
    data = load_gsm8k_chat_train(tok, n=args.n, seed=args.seed, max_len=320)
    print(f"loaded {len(data)} gsm8k chat samples")

    cat_counter = Counter()
    example_by_cat = {}
    n_pairs_total = 0

    for item in data:
        gt = item["input_ids"].to("cuda")
        L = gt.shape[0]
        resp_start = item["resp_start"]
        resp_len = L - resp_start
        if resp_len < 3:
            continue
        keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
        keep[:resp_start] = True
        kr = rng.uniform(0.15, 0.5)
        nk = max(1, int(round(kr * resp_len)))
        sel = resp_start + rng.choice(resp_len, size=nk, replace=False)
        keep[torch.as_tensor(sel, device=gt.device)] = True
        s = gt.clone(); s[~keep] = MASK_ID
        masked = torch.where(~keep)[0].tolist()
        if len(masked) < 3:
            continue

        base_logits = model(input_ids=s.unsqueeze(0)).logits[0]
        base_logp = torch.log_softmax(base_logits, -1)
        base_am = base_logits.argmax(-1)
        cand = masked if len(masked) <= 8 else list(rng.choice(masked, 8, replace=False))
        strong_pairs = []
        for jj in cand:
            s2 = s.clone(); s2[jj] = base_am[jj]
            lg2 = torch.log_softmax(model(input_ids=s2.unsqueeze(0)).logits[0], -1)
            for ii in cand:
                if ii == jj:
                    continue
                d = kl_rows(base_logp[ii:ii+1], lg2[ii:ii+1]).item()
                if d > args.dep_thresh:
                    strong_pairs.append((ii, jj, d))
        strong_pairs.sort(key=lambda x: -x[2])
        strong_pairs = strong_pairs[: args.n_pairs]

        for (ii, jj, d) in strong_pairs:
            n_pairs_total += 1
            for pos in (ii, jj):
                tid = int(gt[pos].item())
                text = tok.decode([tid])
                cat = classify_token(text)
                cat_counter[cat] += 1
                if cat not in example_by_cat:
                    example_by_cat[cat] = []
                if len(example_by_cat[cat]) < 8:
                    example_by_cat[cat].append(text)

    print(f"\n共统计 {n_pairs_total} 个被解耦算法选中的强依赖pair（{n_pairs_total*2} 个token端点）\n")
    print("===== 被解耦pair涉及的token类别分布 =====")
    total = sum(cat_counter.values())
    for cat, cnt in cat_counter.most_common():
        pct = cnt / total * 100 if total else 0
        examples = example_by_cat.get(cat, [])
        print(f"  {cat:>28}: {cnt:4d} ({pct:5.1f}%)  例: {examples}")

    fork_pct = cat_counter.get("fork/connective", 0) / total * 100 if total else 0
    num_op_pct = (cat_counter.get("number", 0) + cat_counter.get("operator/symbol", 0)) / total * 100 if total else 0
    print(f"\n判读：fork/connective 类占比 = {fork_pct:.1f}%；数字+运算符占比 = {num_op_pct:.1f}%")
    if fork_pct > 15:
        print("  -> fork占比不低，Flexibility Trap 警示可能成立：解耦算法在系统性地打掉推理分叉耦合。")
    else:
        print("  -> fork占比较低，Flexibility Trap 原始假说证据不强。")
    if num_op_pct > 30:
        print("  -> 数字+运算符占比高，真正问题可能是'解耦在打算术依赖'，对数学任务直接有害，"
              "建议在强依赖pair候选中排除数字-数字/数字-运算符pair。")

    np.savez(os.path.join(args.out, "exp19_fork_diagnosis.npz"),
             categories=list(cat_counter.keys()),
             counts=list(cat_counter.values()))


if __name__ == "__main__":
    main()
