# Confidence-Based MDLM — 掩码扩散语言模型的可靠并行生成

研究 **Masked Diffusion Language Model (MDLM)** 为什么退化成自回归(AR)、以及能否通过训练
真正支持并行 token 生成。实验模型 `Dream-org/Dream-v0-Instruct-7B`，单卡 RTX 5090。

## TL;DR

- **诊断**：MDLM 的 unmask 基本退化成 AR（顺序与 left-to-right 相关 0.62；81% 的揭示紧贴已填区）。
- **机制**：退化根因是 **confidence 门控**——高 confidence 只在上下文充分后出现，从而强制串行。
- **判据**：commit-readiness 独立于 confidence，且主要能从 hidden state 读出（但"用 hidden 选 commit"已被 TraceLock 等占据）。
- **主线结果（命题 C 第三版）**：用一个"只解耦冗余强依赖 token 对"的训练目标，
  能把**天然可并行的独立子集从 32% 翻倍到 64%**，且经纯微调对照证明是解耦的因果贡献——
  这落在所有现有 inference-only 方法（DEMASK/PUNT，只"检测"天然独立）的**盲区**。

## 这个工作能/不能说明什么

- ✅ **条件独立结构可被训练放大**（有对照，泛化到训练集外）——因果链前半段成立。
- ⚠️ **未证明"降依赖 → 并行生成质量更好"**（净贡献小且被微调噪声主导），未排除大规模下语言能力退化。
- 定位：**proof-of-concept + 揭示新方向**，非"解决了并行生成"。

## 目前状态（H100 实验已完成，见 Exp19）

命题C-v3"精准解耦"的**机制**已坐实，且在 H100 上用 GSM8K chat 语料训练后**跨分布依然成立**
（`16_eval`: dep 中位数 0.648→0.464、可并行独立子集 33%→38%，泛化到训练集外短句）。

但**因果链后半段（降依赖→下游并行生成更好）经 H100 实验被证明在当前训练范式下无法坐实**
（详见 `docs/findings.md` Exp19）：
- 任何能让模型真正学到任务的 LoRA 微调（lmonly / decoupled 均然；完整 LR×步数扫描无甜点），
  都会**破坏 Dream-7B 的原生扩散生成**（退化成 "the the the"）；base 未训练时反而能解 GSM8K（~25-37%）。
- 根因**不是**之前以为的"训练格式不匹配 chat 模板"（纯 LM 对照同样崩），而是**扩散 LM 对窄分布
  LoRA 微调的生成敏感性** + 解耦损失自身的 collapse 退化最小值。
- 此外 `18_gsm8k_eval` 的 naive confidence planner 对所有模型（含 base）触发 eos 级联（→全 0%）、
  GPT-2 NLL 指标奖励 mode-collapse，**这两个评估方法都不可信**。

可信的下一步是换训练目标/干预方式（见 Exp19"下一步建议"），而非在原范式上调参。
环境搭建、数据下载、关键教训仍见 `docs/h100_handoff.md`（顶部已加 Exp19 更新说明）。

## 目录结构

```
docs/
  setup.md            完整实验设定（先看这个）
  observations.md     8 个现象汇总（带图）
  observations.pdf    上文的 PDF 版
  findings.md         最详细的实验记录 + 相关工作核对 + 分寸
experiments/
  00_probe_model.py   模型加载探针
  11_probe_arch.py    模型结构探测
  data_pool.py        训练/评估句子池
  export_pdf.py       md -> pdf 导出工具
  analysis/           现象分析类实验（commit-readiness / 逐层 / AR退化 / 独立性诊断）
  certifier/          certifier planner（更好的 planner + 真实生成评估）
  proposition_c/      命题C：训练诱导 commit-ready / 精准解耦（主线）
figures/              所有结果图
```

## 相关工作定位（详见 docs/findings.md）

| 方向 | 代表工作 | 我们的关系 |
|---|---|---|
| 用 hidden 选 commit | TraceLock (The Path Matters) | 已被占 |
| 检测独立性并行 | DEMASK, PUNT | 已被占（只"检测"，天花板受限于天然独立~32%） |
| 并行失败理论 | Generation Order (info-theoretic) | 已被占 |
| **训练主动增加条件独立结构** | **本工作（命题C-v3）** | **真空区，有初步正向证据** |

## 快速开始

见 `docs/setup.md` 的"复现顺序"。核心：
```bash
pip install -r requirements.txt
python experiments/proposition_c/15_train_C_decouple.py --steps 300 --out out/ckpt_C_dec
python experiments/proposition_c/16_eval_C_decouple.py --ckpt out/ckpt_C_dec --out out
```
