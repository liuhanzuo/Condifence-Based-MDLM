# H100 迁移交接文档（给远程 Claude Code）

> 你正在接手一个持续迭代的研究项目：**训练 masked diffusion LM 主动增加 token 间条件独立性，
> 以支持更可靠的并行生成**。本机此前在单卡 RTX 5090 Laptop(24GB) 上完成了大量实验，现在迁移到
> 一张 H100 上继续跑对下游任务（GSM8K）更有说服力的规模。请先完整读一遍本文档，再看
> `docs/findings.md`（全部实验记录）和 `docs/setup.md`（背景/概念）。

---

## 0. 一句话现状

命题C第三版"精准解耦"训练已经证明**机制成立**（用 LoRA 微调让强依赖 token 对占比从 69%→28%，
可并行独立子集翻倍，且有严格对照排除微调副作用）。但"降依赖是否真的换来更好的下游任务表现"
还没有干净的证据——之前用 GPT-2 NLL 代理指标净贡献很小；换成 GSM8K 真实任务后，发现**训练语料
格式不对会导致模型在 chat 场景下彻底 OOD 崩溃**（这才是 H100 上要优先解决和坐实的问题）。

---

## 1. 环境搭建

```bash
git clone <this-repo>
cd Condifence-Based-MDLM
python -m venv .venv && source .venv/bin/activate   # Linux
pip install torch --index-url https://download.pytorch.org/whl/cu124   # H100用cu124/cu121均可，不必cu128
pip install -r requirements.txt

# 如果 huggingface.co 直连慢，用镜像：
export HF_ENDPOINT=https://hf-mirror.com

# 验证模型能加载
python experiments/00_probe_model.py
```

模型：`Dream-org/Dream-v0-Instruct-7B`（28层，hidden=3584，vocab=152064，mask_id=**151666**）。
bfloat16 加载 7B 模型显存占用 ~15GB，H100(80GB) 完全够用，甚至可以去掉 LoRA 做全参数微调
（本机受限于24GB只能用 LoRA r=8, q/v_proj, 0.03%参数）。

---

## 2. 需要下载的数据（H100机器上执行）

```bash
mkdir -p out
# GSM8K 训练集(7473题) + 测试集(1319题)，通过 hf-mirror 下载 parquet
curl -L -o out/gsm8k_train.parquet \
  https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet
curl -L -o out/gsm8k_test.parquet \
  https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet

# (可选，wiki_corpus.py 用，但已证明该路线走不通，可以跳过)
curl -L -o out/wikitext103_train.parquet \
  https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet
```

---

## 3. 关键教训（务必先看，避免重复踩坑）

### 3.1 Exp18 教训：训练语料"窄"不是根因，训练"格式"不匹配才是根因

- 最初用 `experiments/data_pool.py` 里 50 句极简短陈述句训练 LoRA，在 GSM8K 上评估时，
  模型输出彻底退化成重复 token（`"the the the..."`）。
- 怀疑是语料太窄 → 换成 WikiText-103 真实语料（35万+句子，见 `wiki_corpus.py`）重训，
  **问题依旧存在**，甚至简单的"What is 2+2?"这种 chat 问答都会崩溃。
- **诊断结论**：Dream-7B-Instruct 是指令微调模型，靠 chat 模板（system/user/assistant）驱动。
  用"裸陈述句 + 随机位置mask做infilling"这种训练方式，即使只训几百步 LoRA，也会破坏它的
  chat/指令跟随能力——这是训练**格式**（无chat结构）与评估**格式**（chat结构+多轮）不匹配导致的
  灾难性遗忘，跟语料内容是否"真实/多样"无关。
- **正确做法**（已实现在 `gsm8k_corpus.py`）：直接用 GSM8K **训练集**（与测试集完全不重叠）构造
  完整 chat 序列 `system + user(question) + assistant(推理链+#### answer)`，训练时
  **system+user+response前缀永远 keep 不动**，只在 assistant 回复片段内做随机比例的 mask/解耦
  训练。这样训练分布与评估分布在格式上严格对齐，才能公平检验"解耦训练是否提升下游表现"。

### 3.2 本机（RTX 5090 Laptop）用 GSM8K chat 格式初步训练的结果

用 `--corpus gsm8k --steps 200 --n_train_sents 1000`：
- `lmonly`（lam_dec=0，纯LM微调对照）：200步顺利跑完，lm_loss 收敛到 ~0.3，无崩溃迹象。
- `decoupled`（lam_dec=0.5）：训练更慢（每步要多做几次前向算解耦损失），本机因为工具超时
  中途被打断在140步左右且未保存（**这个ckpt在本机没有跑完，需要在H100上重跑**）。
- 训练中期日志（decoupled, 到140步）显示 lm 收敛正常（11.7→0.7），dep 指标也在下降
  （2.26→0.38），方向正确。

### 3.3 单步耗时随语料变化（RTX 5090 Laptop 24GB 实测，供 H100 提速估算参考）

| corpus | 单步耗时 | 备注 |
|---|---|---|
| data_pool（50句短陈述句，8-50token） | ~1.4-1.6s/step | 已证明OOD，不要用 |
| wiki（WikiText真实句，40-220token） | ~1.4-1.6s/step | 已证明仍OOD（格式问题），不推荐 |
| gsm8k（chat格式，含推理链，截断到320token） | **~7-15s/step**（波动大，取决于回复长度） | 推荐用这个，但单步慢很多 |

gsm8k 语料慢的原因：每步要做 ~13 次完整 7B 模型前向（1次base + 最多8个候选位置的KL估计
+ 最多4对解耦loss的反向），且序列长度是 wiki/data_pool 的 2-6 倍，attention 开销随长度超线性增长。

---

## 4. H100 上的速度估算与建议规模

**RTX 5090 Laptop 实测**（bf16, batch=1, 无 flash-attn 优化确认，peft LoRA）：
gsm8k corpus 下约 7-15s/step（响应长度方差大，平均取 ~10s/step）。

**H100 相对 5090 Laptop 的理论算力比**：H100 FP16/BF16 Tensor Core 算力约 990 TFLOPS(含稀疏)/
495 TFLOPS(稠密)，5090 Laptop 约 100-120 TFLOPS(bf16稠密)量级，**理论加速比约 4-6x**，
但本任务是 batch=1、大量小前向、CPU-GPU同步频繁（`.item()`调用），并非充分利用张量核心的
大batch训练，实际加速比会明显低于理论值，保守估计 **2.5-4x**。

**据此估算 H100 上 gsm8k corpus 训练耗时**：
- 单步：约 2.5s-4s/step（保守取 3.5s/step）
- 400步：约 **23分钟/模型**
- 若把 batch size 从 1 提到 4-8（H100 80GB 显存充裕，可以做，见第5节改造建议），
  单步时间不会等比例增长（forward可以batch化处理多个样本的LM loss部分，
  但强依赖pair检测部分目前是逐样本串行的，需要改造才能真正吃满H100算力）。

**建议**：先不改并行度，直接跑单样本版本验证正确性和效果（预计每个模型 20-30分钟内），
若要认真扩大规模（如1500+步、更大n_train_sents），再考虑第5节的batch化改造把训练speed up到
分钟级/百步。

---

## 5. 接下来的 Plan（按优先级）

### P0：把命题C-v3的GSM8K验证跑完整（本机因超时/资源限制未完成的部分）

```bash
# 1. 纯LM微调对照（若本机ckpt_C_lmonly_gsm8k已完整，可以直接scp过来复用，跳过重训）
python experiments/proposition_c/15_train_C_decouple.py \
  --steps 400 --lam_dec 0.0 --corpus gsm8k --n_train_sents 1500 --seed 0 \
  --out out/ckpt_C_lmonly_gsm8k

# 2. 精准解耦（本机未跑完，必须重跑）
python experiments/proposition_c/15_train_C_decouple.py \
  --steps 400 --lam_dec 0.5 --corpus gsm8k --n_train_sents 1500 --seed 0 \
  --out out/ckpt_C_dec_gsm8k

# 3. GSM8K 下游评估：三模型(base/lmonly/decoupled) x 多个并行度k 的准确率对比
python experiments/proposition_c/18_gsm8k_eval.py \
  --n 100 --gen_len 220 --ks "1,2,4,8,16,32" \
  --ckpt_lmonly out/ckpt_C_lmonly_gsm8k --ckpt_dec out/ckpt_C_dec_gsm8k --out out
```

**判读标准**（写在 `18_gsm8k_eval.py` 里，也见 `docs/findings.md` Exp18 章节）：
- 若 `decoupled` 相对 `base`/`lmonly` 的准确率优势**随 k 增大而扩大**（即高并行度下更鲁棒），
  这是"训练增加条件独立性 → 真实下游任务在高并行下更好"的**硬证据**，是整条研究线最终要坐实的结论。
- 若三者接近或 decoupled 更差，命题C-v3 在下游任务上的净贡献仍未坐实，需要诚实记录，
  不要为了"有正向结果"而选择性报告。
- **务必先跑小规模（n=20-30, 少数k值）确认没有 OOD 崩溃**（即三个模型的输出都应该是连贯的
  数学推理文本，而不是重复token），再放大规模。参考本机 Exp18 日志里"崩溃样本"长什么样，
  见 `docs/findings.md` 的 "Exp18" 章节。

### P1：如果 P0 显示 decoupled 有效，扩大规模坐实结论

- 增大 `--n_train_sents`（GSM8K训练集共7473题，可以全用）、`--steps`（1000+）。
- 增大评估题目数 `--n`（100+，覆盖更多k值），做统计显著性检验（不只看均值，看置信区间/paired test）。
- 考虑去掉 LoRA 限制，H100显存充裕，可以尝试全参数微调或更大rank的LoRA(r=16/32)对比。

### P2：如果 P0 显示 decoupled 无效或退化，诊断原因

- 检查是不是解耦损失权重(`--lam_dec`)、强依赖阈值(`--dep_thresh`)、训练步数没调好，
  做超参扫描（这在H100上便宜很多，可以并行跑多组）。
- 检查是不是训练数据量(1500题)对7B模型LoRA微调还是太少，导致泛化不足。
- 若确认无效，如实记录在 `docs/findings.md`，这仍然是有价值的负向结果
  （命题C因果链后半段"降依赖→更好下游表现"不成立，但前半段"训练可主动增加独立性"依然成立）。

### P3（可选，工程改造，非必须）：把训练循环 batch 化以吃满 H100 算力

当前 `15_train_C_decouple.py` 是逐样本(batch=1)训练，每步内部还有多次串行小前向
（找强依赖pair时对每个候选位置单独跑一次前向）。这在H100上会造成算力浪费（GPU利用率低）。
如果需要显著提速（而不只是100+步的proof-of-concept），可以考虑：
- 把"找强依赖pair"的候选位置检测批量化（一次前向batch过8个候选位置的reveal变体，而非循环8次）。
- 把不同训练样本stack成batch（需要padding，注意mask/attention_mask正确处理）。
这部分改造工作量中等，建议先看P0/P1结果是否值得投入。

---

## 6. 文件地图

```
experiments/
  00_probe_model.py         模型加载探针，先跑这个确认环境OK
  data_pool.py              旧版50句短句(已证明OOD，勿用于新训练)
  analysis/                 现象分析实验(commit-readiness/逐层/AR退化/独立性诊断)，历史结果，不用重跑
  certifier/                certifier planner实验，历史结果，不用重跑
  proposition_c/            *** 本次要接着做的主线 ***
    12_train_readiness.py   命题C v1(失败版，仅供参考)
    14_train_C_full.py      命题C v2(失败版，仅供参考)
    15_train_C_decouple.py  命题C v3精准解耦训练脚本，本次工作的核心，支持 --corpus {data_pool,wiki,gsm8k}
    16_eval_C_decouple.py   命题C v3的机制评估(dependency分布)，训练完后可以跑一下确认机制依然成立
    18_gsm8k_eval.py        *** 本次要跑的下游任务评估 ***
    wiki_corpus.py          WikiText语料加载(已证明此路线走不通，仅留作记录)
    gsm8k_corpus.py         *** GSM8K chat格式语料构造，本次训练要用这个 ***
docs/
  setup.md                  背景概念、复现顺序
  findings.md               *** 完整实验记录，包括所有失败教训，务必读完Exp13-18部分 ***
  observations.md/.pdf      早期8个现象的图文汇总
out/                         checkpoint、npz结果、下载的parquet数据都放这里(gitignore)
```

---

## 7. 沟通与记录规范

- **诚实原则**：这个项目的历史充满"发现问题→自我纠错"的过程（命题C经历3版迭代，
  前两版都被证明是假阳性）。**不要为了呈现正向结果而美化数据或跳过负向发现**，
  用户明确要求过好几次"不要粉饰/过早乐观"。
- 每完成一个阶段，更新 `docs/findings.md`，用"## ExpN: 标题"的格式追加新章节，
  写清楚**设计、结果、能说明什么、不能说明什么**。
- 所有新代码放在 `experiments/proposition_c/` 下，复用 `gsm8k_corpus.py` 的语料接口。
- 如果时间/资源允许，训练完后跑一下 `16_eval_C_decouple.py` 确认机制指标(dependency下降)
  在GSM8K语料上依然成立（之前是在 data_pool 语料上验证的机制，换语料后需要重新确认）。
