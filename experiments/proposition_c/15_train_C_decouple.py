"""
命题 C 第三版：精准解耦 (Targeted Decoupling)

思路（针对 Exp13 判决 + 前两版教训）：
  Exp13 显示 69% 的 masked pair 强依赖，天然可并行只 ~32%。
  目标：训练让"本可独立但当前强依赖"的 pair 变得条件独立，从而扩大可并行子集，
        但【只解耦、不损害预测正确性】以规避命题C前两版的语言退化。

解耦目标（直接用 factorization gap 的成对近似）：
  对状态 s、一对 masked 位置 (i,j)：
    dep(i->j) = KL( p(x_i|s) || p(x_i | s+reveal j的argmax) )
  我们希望降低 dep（揭示 j 后 i 的预测尽量不变 = 条件独立）。
  L_decouple = mean over selected strong-dep pairs of dep(i->j)

护栏：
  - 同时保 LM 去噪损失 L_lm（保预测正确性/语言能力）
  - 只对"强依赖(dep>阈值)"的 pair 施加解耦，弱依赖不动（精准）
  - 监控：训练中同时打印 dep 下降 与 lm 损失，若 lm 飙升=在破坏语言

LoRA(q/v) 微调，冻结主干其他。

【Exp18 教训修正 v1】：此前用 data_pool.TRAIN_SENTS（50句极简短陈述句）训练，
在 GSM8K 等真实下游任务上严重 OOD 崩溃（与解耦无关，是训练分布太窄的通用副作用）。
--corpus wiki 改用真实预训练语料（WikiText-103）训练，但诊断发现：**换语料不能解决
问题**——根因是"纯陈述句+随机mask做infilling"这种训练格式本身与 Dream-Instruct 的
chat/指令跟随行为冲突，短短几百步 LoRA 就会破坏 chat 模板下的生成能力（哪怕语料换了）。

【Exp18 教训修正 v2，--corpus gsm8k】：改用 GSM8K 训练集（与测试集不重叠）构造完整
chat 序列（system+user+assistant回复），只在 assistant 回复片段内做随机mask/解耦训练，
prompt 部分（system+user+response前缀）永远 keep 不动 —— 训练格式与下游评估格式严格对齐。
这是 H100 迁移后应采用的推荐配置，详见 docs/h100_handoff.md。

运行：
  python proposition_c/15_train_C_decouple.py --steps 300 --corpus data_pool --out out/ckpt_C_dec
  python proposition_c/15_train_C_decouple.py --steps 600 --corpus wiki --n_train_sents 2000 --out out/ckpt_C_dec_wiki
  python proposition_c/15_train_C_decouple.py --steps 400 --corpus gsm8k --n_train_sents 1500 --out out/ckpt_C_dec_gsm8k
"""
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666


def get_train_data(corpus, n_train_sents, seed, tok=None):
    """返回统一格式: list，每项是 str(纯句子) 或 dict(gsm8k chat样本)。"""
    if corpus == "data_pool":
        from data_pool import TRAIN_SENTS
        return TRAIN_SENTS
    elif corpus == "wiki":
        from wiki_corpus import load_wiki_sents
        train_sents, _ = load_wiki_sents(n_train=n_train_sents, n_eval=1, seed=seed)
        return train_sents
    elif corpus == "gsm8k":
        from gsm8k_corpus import load_gsm8k_chat_train
        return load_gsm8k_chat_train(tok, n=n_train_sents, seed=seed, max_len=320)
    else:
        raise ValueError(corpus)


def load():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam_dec", type=float, default=0.5, help="解耦损失权重")
    ap.add_argument("--dep_thresh", type=float, default=0.3, help="只解耦dep>此值的pair")
    ap.add_argument("--n_pairs", type=int, default=4, help="每步选几对强依赖pair解耦")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="out/ckpt_C_dec")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", choices=["data_pool", "wiki", "gsm8k"], default="data_pool",
                     help="data_pool=旧50句短句(OOD); wiki=WikiText-103(仍OOD,格式不匹配); "
                          "gsm8k=GSM8K训练集chat格式(格式对齐评估,推荐)")
    ap.add_argument("--n_train_sents", type=int, default=2000,
                     help="corpus=wiki/gsm8k 时从语料池采样的样本数")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, get_peft_model
    tok, model = load()

    TRAIN_DATA = get_train_data(args.corpus, args.n_train_sents, args.seed, tok=tok)
    print(f"corpus={args.corpus}  n_samples={len(TRAIN_DATA)}")

    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="FEATURE_EXTRACTION")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    emb = model.get_input_embeddings()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    ce = nn.CrossEntropyLoss()
    hist = []
    t0 = time.time()

    def kl_rows(logp, logq):
        p = logp.exp()
        return (p * (logp - logq)).sum(-1)

    def sample_gt_keep():
        """按 corpus 类型统一采样出 (gt_ids, keep_mask)。
        gsm8k: prompt部分(system+user+response前缀)永远keep，只在response片段内随机mask。
        其他: 全句范围内随机 keep_ratio。
        """
        item = TRAIN_DATA[rng.integers(len(TRAIN_DATA))]
        if args.corpus == "gsm8k":
            gt = item["input_ids"].to("cuda")
            L = gt.shape[0]
            resp_start = item["resp_start"]
            keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
            keep[:resp_start] = True  # prompt 永远保留
            resp_len = L - resp_start
            if resp_len < 3:
                return None
            kr = rng.uniform(0.15, 0.5)
            nk = max(1, int(round(kr * resp_len)))
            sel = resp_start + rng.choice(resp_len, size=nk, replace=False)
            keep[torch.as_tensor(sel, device=gt.device)] = True
            return gt, keep
        else:
            gt = tok(item, return_tensors="pt").input_ids[0].to("cuda")
            L = gt.shape[0]
            if L < 8:
                return None
            keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
            kr = rng.uniform(0.15, 0.5)
            nk = max(1, int(round(kr * L)))
            keep[torch.as_tensor(rng.choice(L, size=nk, replace=False), device=gt.device)] = True
            return gt, keep

    for step in range(args.steps):
        sample = sample_gt_keep()
        if sample is None:
            continue
        gt, keep = sample
        L = gt.shape[0]
        s = gt.clone(); s[~keep] = MASK_ID
        masked = torch.where(~keep)[0].tolist()
        if len(masked) < 3:
            continue

        # ---- 无梯度：找强依赖 pair ----
        with torch.no_grad():
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

        # ---- 有梯度：LM 损失 + 对强依赖 pair 的解耦损失 ----
        out = model(input_ids=s.unsqueeze(0))
        logits = out.logits[0]
        logp = torch.log_softmax(logits, -1)
        lm_loss = ce(logits[keep], gt[keep])

        dec_loss = torch.tensor(0.0, device="cuda")
        dep_before, dep_after = [], []
        if strong_pairs:
            for (ii, jj, d0) in strong_pairs:
                # 揭示 jj 后 ii 的预测（带梯度）
                s2 = s.clone(); s2[jj] = base_am[jj]
                lg2 = torch.log_softmax(model(input_ids=s2.unsqueeze(0)).logits[0], -1)
                dep = kl_rows(logp[ii:ii+1].detach(), lg2[ii:ii+1]).mean()
                # 目标：降低 dep（让揭示jj后ii预测不变）——只惩罚依赖
                dec_loss = dec_loss + dep
                dep_before.append(d0); dep_after.append(dep.item())
            dec_loss = dec_loss / len(strong_pairs)

        loss = lm_loss + args.lam_dec * dec_loss
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append((float(lm_loss.detach()),
                     float(dec_loss.detach()) if strong_pairs else 0.0,
                     np.mean(dep_after) if dep_after else 0.0))
        if (step + 1) % 20 == 0:
            a = np.mean([h[0] for h in hist[-20:]])
            b = np.mean([h[1] for h in hist[-20:]])
            el = time.time() - t0
            print(f"step {step+1:3d} | lm={a:.3f} decouple(dep)={b:.3f} elapsed={el:.0f}s")

    model.save_pretrained(os.path.join(args.out, "lora"))
    np.save(os.path.join(args.out, "loss_hist.npy"), np.array(hist))
    print(f"\nSaved to {args.out}, total_time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
