"""探测 Dream 模型结构：模块名、线性层、便于挂 LoRA 和辅助头。"""
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                  dtype=torch.bfloat16, device_map="cuda")
print("=== top-level named children ===")
for n, m in model.named_children():
    print(f"  {n}: {type(m).__name__}")

print("\n=== config ===")
for kk in ["hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads"]:
    print(f"  {kk}: {getattr(model.config, kk, '?')}")

print("\n=== 线性层名(取前若干,找 q_proj/v_proj 之类) ===")
seen = set()
for n, m in model.named_modules():
    if isinstance(m, torch.nn.Linear):
        # 归一化名字(去掉层号)
        key = ".".join([p for p in n.split(".") if not p.isdigit()])
        if key not in seen:
            seen.add(key)
            print(f"  {n}  -> shape {tuple(m.weight.shape)}")

print("\n=== 是否有 lm_head / 输出投影 ===")
for n, m in model.named_modules():
    if "head" in n.lower() or "lm" in n.lower():
        print(f"  {n}: {type(m).__name__}")
