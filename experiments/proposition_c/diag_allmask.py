"""快速确认: base vs lmonly 在"响应区全mask"生成起始态的预测差异。"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
DREAM_ID="Dream-org/Dream-v0-Instruct-7B"; MASK_ID=151666
SYS=("You are a helpful assistant that solves math word problems. "
     "Think step by step, then give the final numeric answer on its own "
     "line in the exact format: #### <number>")
@torch.no_grad()
def main():
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    base=AutoModel.from_pretrained(DREAM_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"); base.eval()
    pm=PeftModel.from_pretrained(base,"out/ckpt_C_lmonly_gsm8k/lora",adapter_name="lmonly"); pm.eval()
    df=pd.read_parquet("out/gsm8k_test.parquet").head(2)
    for qi,row in df.iterrows():
        msgs=[{"role":"system","content":SYS},{"role":"user","content":row["question"]}]
        pt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        pid=tok(pt,return_tensors="pt").input_ids[0].to("cuda"); P=pid.shape[0]
        cur=torch.full((P+40,),MASK_ID,device="cuda",dtype=torch.long); cur[:P]=pid
        print(f"\nQ{qi}: {row['question'][:70]}")
        for name,use in [("base",False),("lmonly",True)]:
            if use: pm.set_adapter("lmonly")
            else:
                cm=pm.disable_adapter(); cm.__enter__()
            lg=torch.log_softmax(pm(input_ids=cur.unsqueeze(0)).logits[0].float(),-1)
            p1,am=lg.max(-1)
            ids=am[P:P+24].tolist()
            conf=p1[P:P+24].mean().item()
            print(f"  [{name:7s}] conf={conf:.2f} 前24预测: {tok.decode(ids)!r}")
            if not use: cm.__exit__(None,None,None)
main() if __name__=="__main__" else None
