"""
诊断脚本：隔离 GSM8K 评估里 base 也全面崩溃(0准确率) 的根因。

回答三个问题：
  Q1: base 在 k=1(纯串行 confidence commit) 下能否产出连贯推理？
      - 若能 → 模型/环境/模板/planner 都 OK，k>=2 崩溃是"并行退化"现象本身
      - 若不能 → 更深的问题(模板/环境/自定义代码版本)
  Q2: 生成出来的原始 token id 是什么？(看是否 eos / mask / 重复 token 泛滥)
  Q3: base 用 Dream 原生 model.generate() (扩散采样) 能否解 GSM8K？
      - 隔离 confidence planner 与模型本身

运行：
  python experiments/proposition_c/diag_gen.py
"""
import os
import sys
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


@torch.no_grad()
def gen_parallel(model, prompt_ids, gen_len, k):
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
    from transformers import AutoModel, AutoTokenizer
    print("loading tokenizer + base model (bf16, cuda)...")
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print(f"eos_id={tok.eos_token_id} ({tok.eos_token!r})  mask_id={MASK_ID}")
    print(f"pad_id={tok.pad_token_id}  bos_id={getattr(tok,'bos_token_id',None)}")

    df = pd.read_parquet("out/gsm8k_test.parquet").head(3)
    for qi, row in df.iterrows():
        q = row["question"]; a = row["answer"]
        msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": q}]
        ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pid = tok(ptext, return_tensors="pt").input_ids[0].to("cuda")
        print("\n" + "=" * 70)
        print(f"Q{qi}: {q[:90]}")
        print(f"GT answer field: {a[-60:]!r}")
        print(f"prompt len P={pid.shape[0]}")

        # ---------- Q2: 第一步(全mask)时 base 对生成区每个位置最自信的 token ----------
        P = pid.shape[0]; GL = 64
        cur0 = torch.full((P + GL,), MASK_ID, device="cuda", dtype=torch.long)
        cur0[:P] = pid
        logits0 = model(input_ids=cur0.unsqueeze(0)).logits[0]
        probs0 = torch.softmax(logits0.float(), -1)
        p1_0, am_0 = probs0.max(-1)
        gen_region = slice(P, P + GL)
        top_ids = am_0[gen_region].tolist()
        top_conf = p1_0[gen_region].tolist()
        # 解码前几个 + 统计 eos/mask 占比
        n_eos = sum(1 for t in top_ids if t == tok.eos_token_id)
        n_mask = sum(1 for t in top_ids if t == MASK_ID)
        print(f"[Q2 all-mask step] gen区前12 argmax id: {top_ids[:12]}")
        print(f"   decoded: {tok.decode(top_ids[:24])!r}")
        print(f"   gen区64位中 eos={n_eos} mask={n_mask}  平均confidence={np.mean(top_conf):.3f}")

        # ---------- Q1: k=1 串行生成本题 ----------
        for k in [1, 2, 8]:
            g = gen_parallel(model, pid, gen_len=128, k=k)
            txt = tok.decode(g.tolist(), skip_special_tokens=True)
            print(f"[Q1 k={k}] gen(128) -> {txt[:160]!r}")

        # ---------- Q3: Dream 原生 generate ----------
        try:
            out = model.generate(
                pid.unsqueeze(0),
                max_new_tokens=160,
                mask_token_id=MASK_ID,
                do_sample=False,
                temperature=1.0,
            )
            native = tok.decode(out[0][P:], skip_special_tokens=True)
            print(f"[Q3 native generate] {native[:200]!r}")
        except Exception as e:
            print(f"[Q3 native generate] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
