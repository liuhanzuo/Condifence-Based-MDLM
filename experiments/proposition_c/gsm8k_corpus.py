"""
GSM8K chat 格式训练语料——修正 wiki_corpus 仍暴露的问题：
  Exp18 快速诊断发现，即便用 WikiText 真实语料训练，纯陈述句+随机mask的 infilling 训练
  仍会破坏 Dream-Instruct 的 chat/指令跟随能力（base 能勉强应答，lmonly 直接崩溃）。
  根因不是语料"窄"，而是训练格式与下游评估格式(chat模板+多轮结构)不匹配。

本模块直接用 GSM8K 训练集（与 out/gsm8k_test.parquet 测试集不重叠）构造完整的
chat 序列： system + user(question) + assistant(完整推理链+#### answer)，
只在 assistant 回复片段内做随机比例的 mask/去噪训练，保留 system+user+部分assistant
作为 context，这样训练分布与评估分布在格式上严格对齐。

用法：
    from gsm8k_corpus import load_gsm8k_chat_train
    samples = load_gsm8k_chat_train(tok, n=1500, seed=0)
    # 每个 sample: dict(input_ids=LongTensor[L], resp_start=int)
    #   resp_start = assistant 回复在 input_ids 中的起始位置(之前的位置永远 keep，不参与mask)

下载命令（若 out/gsm8k_train.parquet 不存在）：
  curl -L -o out/gsm8k_train.parquet \
    https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet
  curl -L -o out/gsm8k_test.parquet \
    https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet
"""
import os
import re

SYS_PROMPT = (
    "You are a helpful assistant that solves math word problems. "
    "Think step by step, then give the final numeric answer on its own "
    "line in the exact format: #### <number>"
)


def _clean_answer(ans_field):
    # 把 GSM8K 官方 <<...>> 计算注释去掉，保留自然推理链 + #### 数字
    a = re.sub(r"<<[^>]*>>", "", ans_field)
    return a.strip()


def load_gsm8k_chat_train(tok, n=1500, seed=0, max_len=512,
                           data_path=None):
    """返回 list[dict(input_ids: LongTensor[L], resp_start:int, L:int)]。"""
    import numpy as np
    import pandas as pd
    import torch

    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "out", "gsm8k_train.parquet")
    df = pd.read_parquet(data_path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)

    samples = []
    for _, row in df.iterrows():
        q = row["question"]
        a = _clean_answer(row["answer"])
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": q}]
        prefix_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prefix_ids = tok(prefix_text, return_tensors="pt").input_ids[0]
        # assistant 回复 + eos，作为要学习去噪生成的部分
        resp_ids = tok(a, return_tensors="pt", add_special_tokens=False).input_ids[0]
        eos_id = tok.eos_token_id
        full_ids = torch.cat([prefix_ids, resp_ids,
                               torch.tensor([eos_id], dtype=prefix_ids.dtype)])
        if full_ids.shape[0] > max_len or resp_ids.shape[0] < 3:
            continue
        samples.append({
            "input_ids": full_ids,
            "resp_start": prefix_ids.shape[0],
            "L": full_ids.shape[0],
        })
    return samples


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    ss = load_gsm8k_chat_train(tok, n=5)
    for s in ss:
        print(f"L={s['L']} resp_start={s['resp_start']} "
              f"resp_len={s['L']-s['resp_start']}")
        print("  resp:", repr(tok.decode(s["input_ids"][s["resp_start"]:].tolist())[:150]))
