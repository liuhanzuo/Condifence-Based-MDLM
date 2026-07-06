"""
真实预训练语料池（WikiText-103），替代 data_pool.TRAIN_SENTS 的人造短句，
用于命题C训练——目的：避免此前 50 句极简模板句造成的窄分布过拟合/OOD（见 Exp18 教训）。

来源：Salesforce/wikitext (wikitext-103-raw-v1, train split)，通过 hf-mirror 下载好的
本地 parquet：out/wikitext103_train.parquet（900675 行原始行，用 '\n\n' 级别的行聚合成段落）。

用法：
    from wiki_corpus import load_wiki_sents
    train_sents, eval_sents = load_wiki_sents(n_train=2000, n_eval=200, seed=0)

【重要】诊断结论：换成本语料仍不能解决 GSM8K 等下游任务的 OOD 崩溃问题，
根因是"纯陈述句+随机mask做infilling"训练格式与 chat/指令跟随行为冲突。
详见 gsm8k_corpus.py（v2 修正方案）与 docs/h100_handoff.md。

下载命令（若 out/wikitext103_train.parquet 不存在）：
  curl -L -o out/wikitext103_train.parquet \
    https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet
"""
import os
import re

_CACHE = {}


def _load_raw_lines():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "out", "wikitext103_train.parquet")
    import pandas as pd
    df = pd.read_parquet(path)
    return df["text"].tolist()


def _clean_sentences(lines, min_chars=40, max_chars=220):
    """把 wikitext 的行拆成句子级样本，过滤标题行(以 '=' 开头)/太短/太长。"""
    sents = []
    for line in lines:
        t = line.strip()
        if not t or t.startswith("="):
            continue
        t = t.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
        t = re.sub(r"\s+", " ", t)
        # 按句号/问号/感叹号粗切句子
        for piece in re.split(r"(?<=[.!?])\s+", t):
            piece = piece.strip()
            if min_chars <= len(piece) <= max_chars:
                sents.append(piece)
    return sents


def load_wiki_sents(n_train=2000, n_eval=200, seed=0, min_chars=40, max_chars=220):
    """返回 (train_sents, eval_sents)，严格不重叠（eval 取自不同的原始行区间）。"""
    key = (min_chars, max_chars)
    if key not in _CACHE:
        lines = _load_raw_lines()
        # 前 80% 行做训练来源，后 20% 做 eval 来源，避免同一文档片段互相污染
        cut = int(len(lines) * 0.8)
        train_sents = _clean_sentences(lines[:cut], min_chars, max_chars)
        eval_sents = _clean_sentences(lines[cut:], min_chars, max_chars)
        _CACHE[key] = (train_sents, eval_sents)
    train_pool, eval_pool = _CACHE[key]

    import numpy as np
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(train_pool), size=min(n_train, len(train_pool)), replace=False)
    ev_idx = rng.choice(len(eval_pool), size=min(n_eval, len(eval_pool)), replace=False)
    return [train_pool[i] for i in tr_idx], [eval_pool[i] for i in ev_idx]


if __name__ == "__main__":
    tr, ev = load_wiki_sents(n_train=20, n_eval=10)
    print(f"train pool sample ({len(tr)}):")
    for s in tr[:5]:
        print(" ", repr(s))
    print(f"\neval pool sample ({len(ev)}):")
    for s in ev[:5]:
        print(" ", repr(s))
