"""测某 checkpoint 的原生扩散生成是否恢复(对比 base)。用法: python diag_native_test.py <ckpt_lora_dir_or_base>"""
import os, re, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
DREAM_ID="Dream-org/Dream-v0-Instruct-7B"; MASK_ID=151666; EOS_ID=151643
SYS=("You are a helpful assistant that solves math word problems. "
     "Think step by step, then give the final numeric answer on its own "
     "line in the exact format: #### <number>")
def extract_answer(text):
    m=re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)",text)
    s=m.group(1).replace(",","") if m else (re.findall(r"-?[\d,]+(?:\.\d+)?",text)[-1] if re.findall(r"-?[\d,]+(?:\.\d+)?",text) else None)
    if s is None: return None
    try: v=float(s.replace(",","")); return int(v) if v==int(v) else v
    except: return None
def gt_answer(a):
    m=re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)",a); v=float(m.group(1).replace(",","")); return int(v) if v==int(v) else v
GEN_KW=dict(max_new_tokens=256, temperature=0.0, steps=256, mask_token_id=MASK_ID, pad_token_id=EOS_ID, eos_token_id=EOS_ID)
@torch.no_grad()
def native_gen(model,pid):
    x=model.diffusion_generate(pid.unsqueeze(0),**GEN_KW)
    if hasattr(x,"sequences"): x=x.sequences
    return x[0][pid.shape[0]:]

def main():
    ckpt = sys.argv[1] if len(sys.argv)>1 else "out/ckpt_C_lmonly_gsm8k_keep0/lora"
    n = int(sys.argv[2]) if len(sys.argv)>2 else 8
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    base=AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"); base.eval()
    pm=PeftModel.from_pretrained(base, ckpt, adapter_name="tuned"); pm.eval()
    df=pd.read_parquet("out/gsm8k_test.parquet")
    rng=np.random.default_rng(0); df=df.iloc[rng.choice(len(df),size=n,replace=False)].reset_index(drop=True)
    prompts=[]
    for _,row in df.iterrows():
        msgs=[{"role":"system","content":SYS},{"role":"user","content":row["question"]}]
        pt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        prompts.append((tok(pt,return_tensors="pt").input_ids[0].to("cuda"), gt_answer(row["answer"])))
    res={"base":0, "tuned":0}
    for v in ["base","tuned"]:
        print(f"\n### {v} ({ckpt})")
        if v=="base":
            cm=pm.disable_adapter(); cm.__enter__()
        else:
            pm.set_adapter("tuned")
        for i,(pid,gt) in enumerate(prompts):
            g=native_gen(pm,pid); txt=tok.decode(g.tolist(),skip_special_tokens=True)
            pred=extract_answer(txt); ok=pred is not None and abs(pred-gt)<1e-4; res[v]+=int(ok)
            print(f"[{i}] {'✓' if ok else '✗'} GT={gt} PRED={pred} | {txt[:100]!r}")
        if v=="base": cm.__exit__(None,None,None)
    print(f"\n准确率(n={n}): base={res['base']}/{n}  tuned={res['tuned']}/{n}")
main() if __name__=="__main__" else None
