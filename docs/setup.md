# 实验设定（Setup）

本项目研究 **Masked Diffusion Language Model (MDLM) 的可靠并行生成**：
为什么 MDLM 的 unmask 会退化成自回归(AR)、以及能否通过训练让它真正支持并行。

---

## 1. 底层设定（所有实验共用）

| 项 | 设定 |
|---|---|
| 模型 | `Dream-org/Dream-v0-Instruct-7B`（28 层，hidden=3584，vocab=152064，mask_id=**151666**） |
| 硬件 | 单张 RTX 5090 Laptop 24GB |
| 环境 | Python 3.11 + torch 2.11(cu128) + transformers 4.57 + peft；见 `requirements.txt` |
| 数据 | 数十条普通英文短句；训练集(`data_pool.TRAIN_SENTS`, 50句) 与 评估集 严格不重叠 |
| 质量裁判 | 独立的 **GPT-2**（`openai-community/gpt2`）打 NLL，避免 Dream 自评的同源偏差 |
| 模型下载 | `HF_ENDPOINT=https://hf-mirror.com`（镜像加速） |

---

## 2. 核心概念

- **commit / unmask**：MDLM 从全 `[M]` 起逐步把位置揭示成真实 token，通常不可逆。
- **并行生成**：一步同时 commit 多个位置。隐含假设：这些位置在当前状态下**条件独立**。
- **矛盾**：语言中 token 大多互相依赖 → 独立假设破 → 并行出错 → 退回串行(AR)。

### 两个关键测量量

**Stability（跨轨迹后验稳定性）** — 用于分析/certifier：
> Stability(i) = 从当前状态出发，随机继续揭示其他位置，位置 i 的 argmax 保持不变的概率。
> =1 → 现在 commit 绝对安全（commit-ready）。

**pairwise dependency（成对依赖，factorization gap 的成对近似）** — 用于命题 C：
> dep(i,j) = KL( p(x_i | s) ‖ p(x_i | s + 揭示 j) )
> 小 → i,j 条件独立 → 可安全并行 commit。

---

## 3. 当前主线实验：命题 C 第三版「精准解耦」

**目标**：用训练主动降低 token 间条件依赖，扩大一步可并行的范围（跳出 inference-only planner 的天花板）。

**训练**（`experiments/proposition_c/15_train_C_decouple.py`）：
- 冻结 Dream 主干，仅 LoRA(q/v_proj, r=8, 2.5M 参数=0.03%)。
- 每步：造部分揭示状态 → 找当前强依赖(dep>0.3)的 token 对 → 施加**解耦损失**（压低这些对的 dep）。
- 护栏：保 LM 去噪损失；**只打强依赖对**，不动天然独立的（区别于失败的前两版"无差别压"）。

**判决性对照**：同流程但 `lam_dec=0`（纯 LM 微调），排除"效果来自微调副作用"。

**评估**（`experiments/proposition_c/16_eval_C_decouple.py`，全部训练集外）：
- (A) 机制：训练前 / 纯微调 / 解耦 三者的 dependency 分布 + 可并行独立子集比例。
- (B) 质量：两模型各自真实生成整句，GPT-2 测 NLL，扫并行度 k。

**主要结果**：解耦后强依赖对 69%→28%、可并行子集 32%→64%（翻倍）；纯微调对照不降反升 →
证明是解耦的因果贡献。生成质量解耦净贡献约 -0.4（小，且被微调噪声主导，待坐实）。

---

## 4. 命题 C 三版对比（为什么第三版才成功）

| | v1 | v2 | v3（当前） |
|---|---|---|---|
| 训练目标 | 旁路预测 stability | 可微 soft-commit | 直接降 pairwise dependency |
| 施加范围 | 所有位置 | 所有位置 | 只打强依赖对 |
| 结果 | 增益=微调副作用 | 反而变差 | 机制干净成立（对照坐实） |

---

## 5. 复现顺序

```bash
# 0. 环境
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 可选，加速下载

# 1. 验证模型可加载
python experiments/00_probe_model.py

# 2. 分析类（现象 1-9，见 docs/observations.md）
python experiments/analysis/01_commit_ready.py --out out/data.npz
python experiments/analysis/02_analyze.py --data out/data.npz --out out
python experiments/analysis/04_collect_layers.py --out out/layers.npz
python experiments/analysis/05_layerwise.py --data out/layers.npz --out out
python experiments/analysis/06_ar_degeneracy.py --out out
python experiments/analysis/13_independence_capacity.py --out out   # 天然独立子集诊断

# 3. certifier（更好的 planner，现象 6-7）
python experiments/certifier/07_certifier_value.py --data out/layers.npz --out out
python experiments/certifier/10_generate_eval.py --out out

# 4. 命题 C 第三版（主线结果）
python experiments/proposition_c/15_train_C_decouple.py --steps 300 --out out/ckpt_C_dec
python experiments/proposition_c/15_train_C_decouple.py --steps 300 --lam_dec 0 --out out/ckpt_C_lmonly  # 对照
python experiments/proposition_c/16_eval_C_decouple.py --ckpt out/ckpt_C_dec --out out
```

详见 `docs/findings.md`（完整实验记录）与 `docs/observations.md`（现象汇总，带图）。
