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

运行：
  python exp/15_train_C_decouple.py --steps 300 --out exp/out/ckpt_C_dec
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from data_pool import TRAIN_SENTS

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666


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
    ap.add_argument("--out", default="exp/out/ckpt_C_dec")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, get_peft_model
    tok, model = load()
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

    def kl_rows(logp, logq):
        p = logp.exp()
        return (p * (logp - logq)).sum(-1)

    for step in range(args.steps):
        sent = TRAIN_SENTS[rng.integers(len(TRAIN_SENTS))]
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 8:
            continue
        keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
        kr = rng.uniform(0.15, 0.5)
        nk = max(1, int(round(kr * L)))
        keep[torch.as_tensor(rng.choice(L, size=nk, replace=False), device=gt.device)] = True
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
            print(f"step {step+1:3d} | lm={a:.3f} decouple(dep)={b:.3f}")

    model.save_pretrained(os.path.join(args.out, "lora"))
    np.save(os.path.join(args.out, "loss_hist.npy"), np.array(hist))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
