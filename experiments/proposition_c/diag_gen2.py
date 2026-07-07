"""
诊断2：定位 k=1 输出为空的根因(eos 级联假设) + 验证修复。
  (a) all-mask 时，最高 confidence 的若干位置预测什么 token？
  (b) k=1 原始 committed token id 全量 + special token 计数
  (c) 对照：planner 禁止提交 eos(及其它 special) 后，k=1 能否产出连贯文本？
"""
import os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(__file__))

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666
SYS_PROMPT = ("You are a helpful assistant that solves math word problems. "
              "Think step by step, then give the final numeric answer on its own "
              "line in the exact format: #### <number>")


@torch.no_grad()
def gen_parallel(model, prompt_ids, gen_len, k, forbid_special=False, tok=None):
    """ forbid_special=True 时不把 eos/mask/bos 提交进生成区(把它们 confidence 置 -1e9) """
    device = prompt_ids.device
    P = prompt_ids.shape[0]; L = P + gen_len
    cur = torch.full((L,), MASK_ID, device=device, dtype=torch.long)
    cur[:P] = prompt_ids
    revealed = torch.zeros(L, dtype=torch.bool, device=device); revealed[:P] = True
    forbid = set()
    if forbid_special and tok is not None:
        for t in (tok.eos_token_id, MASK_ID, getattr(tok, "bos_token_id", None), tok.pad_token_id):
            if t is not None: forbid.add(t)
    while not revealed.all():
        logits = model(input_ids=cur.unsqueeze(0)).logits[0]
        probs = torch.softmax(logits.float(), -1)
        p1, am = probs.max(-1)
        cand = (~revealed).cpu().numpy()
        sc = p1.cpu().numpy().copy()
        sc[~cand] = -1e9
        if forbid_special:
            am_np = am.cpu().numpy()
            for ft in forbid:
                sc[am_np == ft] = -1e9
        n_left = int(cand.sum()); kk = min(k, n_left)
        if np.max(sc) <= -1e9 + 1:  # 没有可提交的(全被禁止)
            break
        sel = np.argsort(-sc)[:kk]
        for pos in sel:
            cur[pos] = am[pos]; revealed[pos] = True
    return cur[P:]


def main():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    eos = tok.eos_token_id
    df = pd.read_parquet("out/gsm8k_test.parquet").head(2)

    for qi, row in df.iterrows():
        q = row["question"]
        msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": q}]
        ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pid = tok(ptext, return_tensors="pt").input_ids[0].to("cuda")
        P = pid.shape[0]; GL = 64
        print("\n" + "=" * 70); print(f"Q{qi}: {q[:80]}")

        # (a) all-mask: top-8 confidence 位置预测什么
        cur0 = torch.full((P + GL,), MASK_ID, device="cuda", dtype=torch.long); cur0[:P] = pid
        with torch.no_grad():
            pr = torch.softmax(model(input_ids=cur0.unsqueeze(0)).logits[0].float(), -1)
            p1, am = pr.max(-1)
        gen = slice(P, P + GL)
        order = np.argsort(-p1[gen].cpu().numpy())  # 按 confidence 降序的位置
        print(f"[a all-mask] 生成区按 confidence 降序前8位:")
        for rank, pos in enumerate(order[:8]):
            tid = int(am[gen][pos]); c = float(p1[gen][pos])
            tag = "EOS" if tid == eos else ("MASK" if tid == MASK_ID else "")
            print(f"   rank{rank} pos{pos} conf={c:.3f} id={tid} {tag} -> {tok.decode([tid])!r}")

        # (b) k=1 原始 token 统计
        g1 = gen_parallel(model, pid, gen_len=96, k=1, forbid_special=False)
        ids = g1.tolist()
        n_eos = sum(1 for t in ids if t == eos); n_mask = sum(1 for t in ids if t == MASK_ID)
        n_special = sum(1 for t in ids if t >= 151643)
        n_content = sum(1 for t in ids if t < 151643)
        print(f"[b k=1 raw] 96位中 content={n_content} eos={n_eos} mask={n_mask} special(>=151643)={n_special}")
        print(f"   前30 id: {ids[:30]}")
        print(f"   decode(special保留): {tok.decode(ids, skip_special_tokens=False)[:120]!r}")

        # (c) 对照：禁止提交 special
        g1b = gen_parallel(model, pid, gen_len=96, k=1, forbid_special=True, tok=tok)
        idsb = g1b.tolist()
        print(f"[c k=1 forbid_special] decode(skip_special): {tok.decode(idsb, skip_special_tokens=True)[:160]!r}")
        # 多几个 k
        for k in [2, 8]:
            gb = gen_parallel(model, pid, gen_len=128, k=k, forbid_special=True, tok=tok)
            print(f"[c k={k} forbid_special] decode: {tok.decode(gb.tolist(), skip_special_tokens=True)[:160]!r}")


if __name__ == "__main__":
    main()
