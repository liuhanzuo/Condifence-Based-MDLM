"""
Exp18：真实下游任务验证——GSM8K 数学题正确率 vs 并行度。

补命题C第三版遗留的漏洞②（"降依赖能否换来更好/更快的并行生成"缺硬证据），
换掉此前的 GPT-2 NLL 代理指标，改用有明确对错的下游任务：GSM8K。

比较三个模型（均用 confidence planner，唯一变量是训练方式）：
  - base       : 未训练的 Dream-7B（对照）
  - lmonly     : 纯 LM 微调（排除微调副作用的对照）
  - decoupled  : 精准解耦训练（命题C第三版）
在不同并行度 k（每步敲定的 token 数）下生成答案，用规则抽取最终数字，
与 GSM8K ground truth 做 exact match，比较准确率曲线。

若 decoupled 在高并行度(k大)下准确率显著高于 base/lmonly，且优势随k增大而扩大，
则是"训练增加条件独立性 -> 真实下游任务在高并行下更鲁棒"的硬证据。

数据：本地下载的 out/gsm8k_test.parquet（HF datasets openai/gsm8k, test split, 1319题）。

【重要教训】用 data_pool/wiki 语料训练出的 checkpoint 在本评估上会严重 OOD 崩溃
（见 docs/h100_handoff.md），必须用 --corpus gsm8k 训练的 checkpoint 才有意义。

运行：
  python proposition_c/18_gsm8k_eval.py --n 24 --gen_len 200 \
      --ckpt_lmonly out/ckpt_C_lmonly_gsm8k --ckpt_dec out/ckpt_C_dec_gsm8k --out out
"""
import argparse
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666

SYS_PROMPT = (
    "You are a helpful assistant that solves math word problems. "
    "Think step by step, then give the final numeric answer on its own "
    "line in the exact format: #### <number>"
)


def load_models(ckpt_lmonly, ckpt_dec):
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    base = AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True,
                                     dtype=torch.bfloat16, device_map="cuda")
    base.eval()
    pm = PeftModel.from_pretrained(base, os.path.join(ckpt_lmonly, "lora"), adapter_name="lmonly")
    pm.load_adapter(os.path.join(ckpt_dec, "lora"), adapter_name="decoupled")
    pm.eval()
    return tok, pm


def extract_answer(text):
    """从文本抽取最终数字答案。优先匹配 '#### x'，否则取最后一个数字。"""
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if not m:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        if not nums:
            return None
        m_str = nums[-1]
    else:
        m_str = m.group(1)
    m_str = m_str.replace(",", "")
    try:
        v = float(m_str)
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def gt_answer(ans_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans_field)
    s = m.group(1).replace(",", "")
    v = float(s)
    return int(v) if v == int(v) else v


@torch.no_grad()
def gen_parallel(model, prompt_ids, gen_len, k):
    """从 prompt 之后铺 gen_len 个 mask，按 confidence 每步敲定 k 个，直到填满。"""
    device = prompt_ids.device
    P = prompt_ids.shape[0]
    L = P + gen_len
    cur = torch.full((L,), MASK_ID, device=device, dtype=torch.long)
    cur[:P] = prompt_ids
    revealed = torch.zeros(L, dtype=torch.bool, device=device)
    revealed[:P] = True
    while not revealed.all():
        logits = model(input_ids=cur.unsqueeze(0)).logits[0]
        probs = torch.softmax(logits.float(), -1)
        p1, am = probs.max(-1)
        cand = (~revealed).cpu().numpy()
        sc = p1.cpu().numpy().copy()
        sc[~cand] = -1e9
        n_left = int(cand.sum())
        kk = min(k, n_left)
        sel = np.argsort(-sc)[:kk]
        for pos in sel:
            cur[pos] = am[pos]
            revealed[pos] = True
    return cur[P:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="评测题目数")
    ap.add_argument("--gen_len", type=int, default=160)
    ap.add_argument("--ks", type=str, default="2,8,40")
    ap.add_argument("--data", default="out/gsm8k_test.parquet")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_lmonly", default="out/ckpt_C_lmonly_gsm8k")
    ap.add_argument("--ckpt_dec", default="out/ckpt_C_dec_gsm8k")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ks = [int(x) for x in args.ks.split(",")]

    df = pd.read_parquet(args.data)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)
    print(f"评测 {len(df)} 题, gen_len={args.gen_len}, ks={ks}")
    print(f"ckpt_lmonly={args.ckpt_lmonly}  ckpt_dec={args.ckpt_dec}")

    tok, pm = load_models(args.ckpt_lmonly, args.ckpt_dec)
    variants = ["base", "lmonly", "decoupled"]
    results = {v: {k: [] for k in ks} for v in variants}
    samples = {v: {k: [] for k in ks} for v in variants}

    t0 = time.time()
    for qi, row in df.iterrows():
        question = row["question"]
        gt = gt_answer(row["answer"])
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": question}]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pid = tok(prompt_text, return_tensors="pt").input_ids[0].to("cuda")
        for k in ks:
            for v in variants:
                if v == "base":
                    with pm.disable_adapter():
                        gen_ids = gen_parallel(pm, pid, args.gen_len, k)
                else:
                    pm.set_adapter(v)
                    gen_ids = gen_parallel(pm, pid, args.gen_len, k)
                text = tok.decode(gen_ids.tolist(), skip_special_tokens=True)
                pred = extract_answer(text)
                correct = (pred is not None and abs(pred - gt) < 1e-4)
                results[v][k].append(1.0 if correct else 0.0)
                if len(samples[v][k]) < 3:
                    samples[v][k].append((question[:60], gt, pred, text[:120].replace("\n", " ")))
        if (qi + 1) % 4 == 0:
            el = time.time() - t0
            print(f"[{qi+1}/{len(df)}] elapsed={el:.0f}s")

    print("\n===== Exp18 GSM8K 准确率 vs 并行度 =====")
    print(f"{'k':>4} | {'base':>8} | {'lmonly':>8} | {'decoupled':>9}")
    acc_table = {}
    for k in ks:
        row_acc = {}
        for v in variants:
            a = float(np.mean(results[v][k])) if results[v][k] else float("nan")
            row_acc[v] = a
        acc_table[k] = row_acc
        print(f"{k:>4} | {row_acc['base']:>8.3f} | {row_acc['lmonly']:>8.3f} | {row_acc['decoupled']:>9.3f}")

    print("\n解读：")
    print("  - k 越大 = 每步敲定越多 token = 越并行")
    print("  - 若 decoupled 相对 base/lmonly 的优势随 k 增大而扩大 -> 解耦训练在真实下游任务的")
    print("    高并行区间更鲁棒，是'训练增加独立性'的硬证据")
    print("  - 若三者接近或 decoupled 更差 -> 命题C第三版在下游任务上净贡献仍未坐实")

    np.savez(os.path.join(args.out, "gsm8k_eval.npz"),
             ks=np.array(ks),
             **{f"{v}_{k}": np.array(results[v][k]) for v in variants for k in ks})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        colors = {"base": "#888", "lmonly": "#39c", "decoupled": "#3b7"}
        for v in variants:
            ys = [acc_table[k][v] for k in ks]
            plt.plot(ks, ys, "o-", color=colors[v], label=v)
        plt.xlabel("commits per step k (larger = more parallel)")
        plt.ylabel("GSM8K accuracy")
        plt.title(f"Exp18: GSM8K accuracy vs parallelism (n={len(df)})")
        plt.legend()
        plt.tight_layout()
        pth = os.path.join(args.out, "gsm8k_eval.png")
        plt.savefig(pth, dpi=130)
        print("saved", pth)
    except Exception as e:
        print("plot skip:", e)

    for v in variants:
        for k in ks:
            print(f"\n-- samples: {v}, k={k} --")
            for q, gt, pred, txt in samples[v][k]:
                print(f"  Q:{q!r} GT={gt} PRED={pred} | {txt!r}")


if __name__ == "__main__":
    main()
