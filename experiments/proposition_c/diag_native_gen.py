"""
诊断3：用 Dream 原生扩散生成(diffusion_generate) 检验 gsm8k 格式训练是否保住 chat 能力。

naive greedy planner 对所有模型(含 base) 触发 eos 级联退化。改用 Dream 官方 diffusion_generate:
  - base 能否解 GSM8K(确认模型/环境/模板 OK + quality 基线)
  - gsm8k 格式训练的 lmonly/decoupled 是否仍连贯生成(格式对齐训练目标)
  - 三者准确率对比(原生生成)
"""
import os, re, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666
EOS_ID = 151643
SYS_PROMPT = ("You are a helpful assistant that solves math word problems. "
              "Think step by step, then give the final numeric answer on its own "
              "line in the exact format: #### <number>")


def extract_answer(text):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    s = m.group(1).replace(",", "") if m else None
    if s is None:
        nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
        s = nums[-1] if nums else None
    if s is None: return None
    try:
        v = float(s.replace(",", "")); return int(v) if v == int(v) else v
    except: return None


def gt_answer(a):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", a)
    v = float(m.group(1).replace(",", "")); return int(v) if v == int(v) else v


GEN_KW = dict(max_new_tokens=256, temperature=0.0, steps=256,
              mask_token_id=MASK_ID, pad_token_id=EOS_ID, eos_token_id=EOS_ID)


@torch.no_grad()
def native_gen(model, pid):
    x = model.diffusion_generate(pid.unsqueeze(0), **GEN_KW)
    if hasattr(x, "sequences"): x = x.sequences
    P = pid.shape[0]
    return x[0][P:]


def main():
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    n = 8
    df = pd.read_parquet("out/gsm8k_test.parquet")
    rng = np.random.default_rng(0)
    df = df.iloc[rng.choice(len(df), size=n, replace=False)].reset_index(drop=True)

    print("loading DreamModel (base) + 两个 LoRA adapter ...")
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    base = AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True,
                                     dtype=torch.bfloat16, device_map="cuda")
    base.eval()
    pm = PeftModel.from_pretrained(base, "out/ckpt_C_lmonly_gsm8k/lora", adapter_name="lmonly")
    pm.load_adapter("out/ckpt_C_dec_gsm8k/lora", adapter_name="decoupled")
    pm.eval()

    prompts = []
    for _, row in df.iterrows():
        msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": row["question"]}]
        ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append((tok(ptext, return_tensors="pt").input_ids[0].to("cuda"), gt_answer(row["answer"])))

    # 用 base 跑第一题，确认 diffusion_generate 可用 + 输出形态
    g0 = native_gen(pm.base_model.model if hasattr(pm, "base_model") else pm, prompts[0][0])
    print("base 首题原始生成 token 数(去 mask 后):",
          int((g0 != MASK_ID).sum()), "总:", g0.shape[0])
    print("base 首题 decode:", repr(tok.decode(g0.tolist(), skip_special_tokens=True)[:200]))

    variants = ["base", "lmonly", "decoupled"]
    correct = {v: 0 for v in variants}
    for v in variants:
        print("\n" + "#" * 70); print(f"# 变体: {v}"); print("#" * 70)
        if v == "base":
            cm = pm.disable_adapter(); cm.__enter__()
        else:
            pm.set_adapter(v)
        for i, (pid, gt) in enumerate(prompts):
            g = native_gen(pm, pid)
            txt = tok.decode(g.tolist(), skip_special_tokens=True)
            pred = extract_answer(txt)
            ok = (pred is not None and abs(pred - gt) < 1e-4)
            correct[v] += int(ok)
            print(f"[{i}] {'✓' if ok else '✗'} GT={gt} PRED={pred} | {txt[:110]!r}")
        if v == "base":
            cm.__exit__(None, None, None)

    print("\n" + "=" * 70)
    print(f"原生扩散生成准确率 (n={n}):")
    for v in variants:
        print(f"  {v:10s}: {correct[v]}/{n} = {correct[v]/n:.0%}")


if __name__ == "__main__":
    main()
