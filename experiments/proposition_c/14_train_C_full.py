"""
命题 C 完整版：让 readiness 以【可微方式驱动并行 commit】，而非旁路预测 stability。

核心机制（针对最小版失败根因3）：
  给定部分揭示状态 s，模型对每个 masked 位置给 hidden -> readiness head -> 分数 r_i。
  模拟"一步并行 commit"：按 r_i 的 soft 权重(温度T的sigmoid门控)决定每个 masked 位置
  是否被"填入其预测的 argmax(用 straight-through 使之可微)"，得到新状态 s'。
  然后在 s' 上再前向，计算【剩余 masked 位置的 LM 去噪损失】。
  直觉：如果 readiness 选对了该 commit 的位置(填对且不干扰)，s' 下剩余位置更好预测 -> 损失低。
  => readiness 被训练成"选出 commit 后能让整体更好预测的位置"，直接对齐生成质量。

同时保留：
  - LM 去噪损失(保语言能力)
  - readiness 对齐蒙特卡洛 stability 的弱监督(warmup 作用)

训练：LoRA(q/v) + readiness head 联合训练。更大数据(50句)、更多步。

运行：
  python exp/14_train_C_full.py --steps 400 --out exp/out/ckpt_C_full
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
L_READ = 4


def load():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map="cuda")
    return tok, model


@torch.no_grad()
def mc_stability(model, gt, keep, tp, base_am, K, rng):
    device = gt.device
    other = [p for p in torch.where(~keep)[0].tolist() if p not in tp]
    kept = torch.where(keep)[0].tolist()
    hit = np.zeros(len(tp))
    for _ in range(K):
        keep2 = set(kept)
        if other:
            r = rng.uniform(0.0, 1.0); kk = int(round(r * len(other)))
            if kk > 0:
                keep2.update(int(x) for x in rng.choice(other, size=kk, replace=False))
        mf = torch.ones(gt.shape[0], dtype=torch.bool, device=device)
        if keep2:
            mf[torch.as_tensor(sorted(keep2), device=device)] = False
        for p in tp:
            mf[p] = True
        s = gt.clone(); s[mf] = MASK_ID
        logits = model(input_ids=s.unsqueeze(0)).logits[0]
        am = logits[tp].argmax(-1)
        hit += (am.cpu().numpy() == base_am).astype(float)
    return hit / K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lam_stab", type=float, default=0.3, help="stability弱监督权重")
    ap.add_argument("--lam_commit", type=float, default=1.0, help="可微commit损失权重")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--commit_frac", type=float, default=0.4, help="模拟一步commit的目标比例")
    ap.add_argument("--out", default="exp/out/ckpt_C_full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, get_peft_model
    tok, model = load()
    H = model.config.hidden_size
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="FEATURE_EXTRACTION")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    head = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Linear(256, 1)).to("cuda").float()
    emb = model.get_input_embeddings()  # 用于 straight-through commit

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    model.train()
    ce = nn.CrossEntropyLoss()
    hist = []

    for step in range(args.steps):
        sent = TRAIN_SENTS[rng.integers(len(TRAIN_SENTS))]
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 6:
            continue
        keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
        kr = rng.uniform(0.1, 0.5)
        nk = max(1, int(round(kr * L)))
        keep[torch.as_tensor(rng.choice(L, size=nk, replace=False), device=gt.device)] = True
        s = gt.clone(); s[~keep] = MASK_ID
        masked = torch.where(~keep)[0]
        if masked.numel() < 2:
            continue

        # ---- 前向1：拿 logits + readiness 分数 ----
        out = model(input_ids=s.unsqueeze(0), output_hidden_states=True)
        logits = out.logits[0]                       # [L,V]
        hs = out.hidden_states[L_READ][0].float()    # [L,H]
        r_all = head(hs).squeeze(-1)                 # [L] readiness

        # LM 去噪损失（已揭示位置）
        lm_loss = ce(logits[keep], gt[keep])

        # stability 弱监督（对部分 masked 位置）
        mp = masked.tolist()
        tp = mp if len(mp) <= 6 else list(rng.choice(mp, size=6, replace=False))
        with torch.no_grad():
            base_am = logits[tp].argmax(-1).cpu().numpy()
        stab = torch.tensor(mc_stability(model, gt, keep, tp, base_am, args.K, rng),
                            device="cuda").float()
        stab_loss = ((torch.sigmoid(r_all[tp]) - stab) ** 2).mean()

        # ---- 可微并行 commit：按 readiness 软门控 commit masked 位置 ----
        # 门控 g_i = sigmoid(r_i)，作为"该位置被 commit 的软权重"
        g = torch.sigmoid(r_all[masked])                          # [M]
        pred_ids = logits[masked].argmax(-1)                      # [M] 预测token(硬)
        # straight-through：用预测token的embedding替换原mask embedding，按 g 加权
        cur_emb = emb(s.unsqueeze(0))[0]                          # [L,H_emb]
        pred_emb = emb(pred_ids)                                  # [M,H_emb]
        mask_emb = emb(torch.full_like(pred_ids, MASK_ID))        # [M,H_emb]
        # 新的 masked 位置 embedding = g*pred + (1-g)*mask  (可微 in g)
        new_masked_emb = g.unsqueeze(-1).to(pred_emb.dtype) * pred_emb + \
                         (1 - g).unsqueeze(-1).to(mask_emb.dtype) * mask_emb
        cur_emb = cur_emb.clone()
        cur_emb[masked] = new_masked_emb
        # ---- 前向2：在软 commit 后的状态上，预测所有原 masked 位置的 GT ----
        out2 = model(inputs_embeds=cur_emb.unsqueeze(0))
        logits2 = out2.logits[0]
        # 目标：commit 后剩余位置更好预测 GT（用 g 加权：越被 commit 的位置越该已对）
        ce_tok = F.cross_entropy(logits2[masked], gt[masked], reduction="none")  # [M]
        # 被 commit(g大)的位置若预测错惩罚更大 -> 逼 readiness 只对"填了也对"的位置给高分
        commit_loss = (g.detach() * ce_tok).mean() + ce_tok.mean() * 0.1

        loss = lm_loss + args.lam_stab * stab_loss + args.lam_commit * commit_loss
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append((float(lm_loss.detach()), float(stab_loss.detach()),
                     float(commit_loss.detach())))
        if (step + 1) % 20 == 0:
            a = np.mean([h[0] for h in hist[-20:]])
            b = np.mean([h[1] for h in hist[-20:]])
            c = np.mean([h[2] for h in hist[-20:]])
            print(f"step {step+1:3d} | lm={a:.3f} stab={b:.4f} commit={c:.3f}")

    model.save_pretrained(os.path.join(args.out, "lora"))
    torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))
    np.save(os.path.join(args.out, "loss_hist.npy"), np.array(hist))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
