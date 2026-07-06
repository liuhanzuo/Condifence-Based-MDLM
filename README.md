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

## 目前状态与下一步（H100迁移中）

命题C-v3"精准解耦"的**机制**已坐实（见上文TL;DR），但"降依赖是否真的提升下游任务表现"
这条因果链后半段还缺干净证据。用 GSM8K 真实任务验证时发现关键教训——训练语料若不是
chat 格式会导致模型在下游评估中严重 OOD 崩溃（与解耦机制无关）。已修正训练流程
（`experiments/proposition_c/gsm8k_corpus.py`），但受限于单卡资源，解耦版本训练未能完整跑完。

**正在迁移到 H100 服务器继续**：完整交接文档见 **`docs/h100_handoff.md`**
（含环境搭建、数据下载、关键教训、速度估算、分阶段plan）。

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
