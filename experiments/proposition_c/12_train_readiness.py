"""
命题 C 训练：LoRA 微调 Dream + 辅助 readiness 头，诱导 hidden 显式编码 commit-readiness。

损失 = LM 去噪损失(保语言能力) + λ * readiness 头预测 stability 的 MSE(诱导表征)。
  - 冻结主干权重，只训 LoRA(q/v_proj) + readiness 头。
  - readiness 头：读第 L_READ 层 hidden -> MLP -> stability 预测。
  - stability 标签：在线蒙特卡洛(K条后续揭示轨迹, argmax 不变率)。

产出：保存 LoRA adapter + readiness 头到 exp/out/ckpt_C/，供 13 评估生成质量。

运行：
  python exp/12_train_readiness.py --steps 150 --out exp/out/ckpt_C
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
MASK_ID = 151666
L_READ = 4  # 读第几层 hidden 做 readiness (Exp5: 浅层最强)

SENTS = [
    "The capital of France is Paris and it is a beautiful city.",
    "Machine learning models require large amounts of training data.",
    "The sun rises in the east and sets in the west every day.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "She opened the door and walked quietly into the dark room.",
    "The stock market fell sharply after the announcement yesterday.",
    "A balanced diet includes fruits, vegetables, and whole grains.",
    "The ancient castle stood silently on top of the green hill.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The children played happily in the park all afternoon long.",
    "Quantum computers may one day solve very hard problems quickly.",
    "He carefully wrote down every word the teacher said today.",
    "The river flows gently through the valley toward the sea.",
    "Regular exercise improves both physical and mental health greatly.",
    "The museum displayed many rare paintings from the last century.",
    "Climate change is one of the biggest challenges of our time.",
    "The engineer fixed the broken bridge in just three days.",
    "Reading books every night helps children develop their imagination.",
    "The chef prepared a delicious meal for all the guests.",
    "Electric cars are becoming more popular around the world now.",
]


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
            r = rng.uniform(0.0, 1.0)
            kk = int(round(r * len(other)))
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
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lam", type=float, default=1.0, help="readiness 辅助损失权重")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="exp/out/ckpt_C")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, get_peft_model
    tok, model = load()
    H = model.config.hidden_size

    # LoRA
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="FEATURE_EXTRACTION")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    # readiness 头（fp32）
    head = nn.Sequential(nn.Linear(H, 256), nn.GELU(), nn.Linear(256, 1)).to("cuda").float()

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)

    model.train()
    ce = nn.CrossEntropyLoss()
    loss_hist = []

    for step in range(args.steps):
        sent = SENTS[rng.integers(len(SENTS))]
        gt = tok(sent, return_tensors="pt").input_ids[0].to("cuda")
        L = gt.shape[0]
        if L < 6:
            continue
        keep = torch.zeros(L, dtype=torch.bool, device=gt.device)
        kr = rng.uniform(0.1, 0.6)
        nk = max(1, int(round(kr * L)))
        keep[torch.as_tensor(rng.choice(L, size=nk, replace=False), device=gt.device)] = True
        s = gt.clone(); s[~keep] = MASK_ID
        masked = torch.where(~keep)[0].tolist()
        if not masked:
            continue
        tp = rng.choice(masked, size=min(6, len(masked)), replace=False).tolist()

        # 蒙特卡洛 stability 标签（用无梯度的当前模型）
        with torch.no_grad():
            base_logits = model(input_ids=s.unsqueeze(0)).logits[0]
            base_am = base_logits[tp].argmax(-1).cpu().numpy()
        stab = mc_stability(model, gt, keep, tp, base_am, args.K, rng)
        stab_t = torch.tensor(stab, device="cuda").float()

        # 前向(带梯度)：拿 logits(LM loss) + 第 L_READ 层 hidden(readiness)
        out = model(input_ids=s.unsqueeze(0), output_hidden_states=True)
        logits = out.logits[0]
        hs = out.hidden_states[L_READ][0].float()  # [L,H]

        # LM 去噪损失：对已揭示位置(它们是GT)做重建监督(保语言能力)
        lm_loss = ce(logits[keep], gt[keep])

        # readiness 损失：head(hidden_tp) 预测 stability
        pred = head(hs[tp]).squeeze(-1)
        read_loss = ((pred - stab_t) ** 2).mean()

        loss = lm_loss + args.lam * read_loss
        opt.zero_grad(); loss.backward(); opt.step()
        loss_hist.append((float(lm_loss.item()), float(read_loss.item())))
        if (step + 1) % 10 == 0:
            lm_m = np.mean([a for a, _ in loss_hist[-10:]])
            rd_m = np.mean([b for _, b in loss_hist[-10:]])
            print(f"step {step+1:3d} | lm={lm_m:.3f} read_mse={rd_m:.4f}")

    # 保存
    model.save_pretrained(os.path.join(args.out, "lora"))
    torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))
    np.save(os.path.join(args.out, "loss_hist.npy"), np.array(loss_hist))
    print(f"\nSaved LoRA + head to {args.out}")


if __name__ == "__main__":
    main()
