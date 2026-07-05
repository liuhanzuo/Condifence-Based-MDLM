"""
环境 + MDLM 加载探针。
目标：在写正式实验前，确认在 py3.14 + transformers5.2 + torch2.10(cu128) + 5090 24GB 上
      能真正 load 一个 masked diffusion LM，并跑一次前向拿到 logits 和 hidden states。

策略：从小到大尝试若干候选 MDLM，任意一个成功即打印其配置并退出。
运行：  python exp/00_probe_model.py
"""

import sys
import traceback

import torch

# 候选模型：按 显存占用 从小到大 / 主流程度 排序
# 说明：LLaDA/Dream 使用自定义建模代码，需要 trust_remote_code=True
CANDIDATES = [
    # (repo_id, 说明)
    ("Dream-org/Dream-v0-Instruct-7B", "Dream 7B, diffusion LM"),
    ("GSAI-ML/LLaDA-8B-Base", "LLaDA 8B base, 最主流 MDLM"),
]


def report_env():
    print("=" * 60)
    print("ENV")
    print("  python     :", sys.version.split()[0])
    print("  torch      :", torch.__version__)
    print("  cuda avail :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device     :", torch.cuda.get_device_name(0))
        free, total = torch.cuda.mem_get_info(0)
        print(f"  vram       : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
    try:
        import transformers

        print("  transformers:", transformers.__version__)
    except Exception as e:
        print("  transformers: IMPORT FAILED ->", e)
    print("=" * 60)


def try_load(repo_id: str, note: str) -> bool:
    from transformers import AutoModel, AutoTokenizer

    print(f"\n[TRY] {repo_id}  ({note})")
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        model.eval()
        cfg = model.config
        print("  [OK] loaded.")
        print("    hidden_size :", getattr(cfg, "hidden_size", "?"))
        print("    vocab_size  :", getattr(cfg, "vocab_size", "?"))
        print("    mask_token  :", getattr(tok, "mask_token", None),
              getattr(tok, "mask_token_id", None))

        # 一次前向，确认能拿到 logits + hidden_states
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        logits = getattr(out, "logits", None)
        hs = getattr(out, "hidden_states", None)
        print("    logits shape:", None if logits is None else tuple(logits.shape))
        print("    n_hidden_layers returned:", None if hs is None else len(hs))
        if hs is not None:
            print("    last hidden shape:", tuple(hs[-1].shape))
        return True
    except Exception:
        print("  [FAIL]")
        traceback.print_exc()
        return False


def main():
    report_env()
    for repo_id, note in CANDIDATES:
        if try_load(repo_id, note):
            print(f"\n>>> USE THIS MODEL: {repo_id}")
            return
    print("\n>>> 所有候选均失败，请看上面的 traceback。")


if __name__ == "__main__":
    main()
