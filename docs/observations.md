# MDLM Commit-Readiness：观察到的现象汇总

> 研究对象：Masked Diffusion Language Model（MDLM，实验模型 Dream-7B）
> 核心追问：MDLM 为什么退化成自回归（AR）？能否真正并行生成？
> 本文档只汇总**实验观察到的现象**，图片来自 `exp/out/`。

---

## 背景与核心概念

MDLM 从全 `[M]`（mask）开始，逐步把位置揭示（**commit**）成真实 token，且 commit 通常不可逆。
并行生成的关键问题是：**能否一次性、安全地敲定多个位置而不后悔？**

我们定义了一个可测量的量 **Stability（跨轨迹后验稳定性）**：

> Stability(位置 i) = 从当前部分揭示状态出发，随机继续揭示其他位置，i 的预测（argmax）保持不变的概率。

- Stability = 1 → 现在敲定绝对安全（**commit-ready**）
- Stability 低 → 敲定后很可能被后续揭示推翻

所有现有方法用 **confidence（predicted probability）** 决定敲不敲。我们的核心疑问：**confidence 是不是好的敲定判据？**

---

## 现象 1：commit-readiness 独立于 confidence

散点（左）：每个点是一个位置，横轴 predicted prob，纵轴真实 Stability。

![命题A/B](exp/out/result.png)

**观察**：
- `Spearman(prob, stability) = 0.32` —— 两者只是**中等相关，不是同一回事**。
- 存在大量"prob 低但 stability 高"（图左上）和"prob 高但 stability 低"的位置。
- 右图：用 hidden state 预测 stability（R²=0.34）明显强于只用 prob/margin（R²=0.11）。

**含义**：一个位置"是哪个词的把握（confidence）"和"敲定后会不会后悔（stability）"是两个不同的量。
commit reliability ≠ confidence。

---

## 现象 2：commit-readiness 主要能从单个位置的 hidden state 读出（state-local），但生成最早期例外

（同上图右侧柱状）在大样本(1570)下：
- hidden 单独预测 stability R²≈0.34；
- 再加入"全局揭示了哪些位置"的信息，仅额外提升约 +0.06。

**观察**：commit-readiness **主要是 state-local 的**——大体只看该位置自身内部状态就能判断，不太依赖全局。
**例外**：在生成最早期（只揭示了 1-2 个 token 时），"具体哪个邻居被揭示"仍有明显影响（Δ≈0.09），
即早期存在真实的**位置间耦合（path-dependence）**，之后逐渐消失。

---

## 现象 3：commit-readiness 信息在浅层最强（U 形）

逐层用 hidden 预测 stability：

![逐层](exp/out/layerwise.png)

**观察**：
- 第 0 层（embedding）几乎无信息；
- **第 2 层最强（R²≈0.38）**，随后中层下降，后层回升 —— 呈 U 形。

**含义**：commit-readiness 更像一种**浅层、局部的表面可预测性统计**，而非深层语义。
（注：浅层最强可能含位置/token 统计的 artifact，标为待复核。）

---

## 现象 4：模型自发生成是"边界锚点优先"，且锚点是位置性而非语义性

让模型按 confidence-first 自发揭示，记录顺序：

![AR退化](exp/out/ar_degeneracy.png)

**观察**：
- 揭示顺序与"从左到右"的相关性 = **0.62**（纯 AR 会是 1.0）→ **强偏 AR 但非纯 AR**。
- 相邻揭示间距 = +1 的比例约 0.53。
- 典型顺序如 `[0,1,2, 句尾, 3,4, 中间...]`：**先锚定句子开头连续几个 + 句尾，再往中间填**。
- 进一步检验（Exp8）：这个"先填哪里"由**位置**驱动（靠首尾更早，相关 -0.19），
  与"是不是内容词"几乎无关（控制位置后系数≈-0.02）。

**含义**：确实退化成近似 AR，保留"先定边界锚点"的痕迹；但锚点是**位置性（首尾边界）**，
**不是语义性**（不是先填信息量大的内容词）。这**反证**了"先建语义核心单元(information-unit)"的假说。

---

## 现象 5（机制）：AR 退化的根因是 "confidence 门控"

统计"confidence≥0.9 才敢敲定"的位置出现在什么时候：只有 3.6% 达标，且**几乎全在上下文充分揭示之后**。

**观察 → 机制**：
> confidence 要够高才敢敲定 → 但 confidence 只有在周围上下文揭示后才会变高
> → 模型只能"等邻居揭示完再敲定自己" → 从左到右的串行 = **AR 退化**。

**含义**：AR 退化**不是因为语言本质是 AR**，而是因为 **"用 confidence 当敲定判据" 这件事本身天然强制了串行**。

---

## 现象 6：用 certifier（读 hidden 预测 stability）挑 commit，比 confidence 更可靠

模拟"一步并行敲定 top-r 比例位置"，比较被敲定位置的真实 stability（越高越安全）：

![certifier价值](exp/out/certifier_value.png)

**观察**（8/8 随机种子稳健）：
- certifier（绿）在所有并行度下都优于 confidence（蓝），几乎贴合 oracle 上界（黑虚线）。
- 一步并行敲定 50% 时：confidence 出错率 5.2%，certifier 仅 1.0% —— **出错率降约 80%**。

**含义**：把敲定判据从 confidence 换成 certifier，能在激进并行下显著减少错误 commit。

---

## 现象 7：真实并行生成中，certifier 是更好的 planner——但优势主要在低并行、高并行时趋同

两种策略各自**真实生成整句**，独立 GPT-2 裁判测质量（NLL 越低越好），扫并行度 k：

![生成质量](exp/out/gen_quality.png)

**观察**：
- certifier（绿）在每个并行度都优于或等于 confidence（蓝）；k=1（串行）时差距最大（NLL 7.5→5.2）。
- **但优势随并行度增大而缩小**，到高并行（k≥4）两条线**迅速趋同**。

**含义**：certifier 的收益**主要来自"它是一个更好的揭示顺序 planner"**（尤其低并行时），
**并未解锁"高并行不退化"**。即当前形态本质是 **planner 谱系里"更好的评分函数"**，
高并行区间仍是未解决区。

---

## 现象 8：把 commit-readiness 塞进训练目标去重塑表征——无效，甚至有害

尝试用训练（LoRA + readiness 目标）诱导 commit-ready 表征，看能否压下高并行区间。做了两版并带对照：

![命题C](exp/out/propC_quality.png)

**观察**（高并行区间 k≥4 平均 NLL）：

| 策略 | 简单辅助损失版 | 可微commit版 |
|---|---|---|
| inference-only certifier（不训练） | 6.08 | **6.08（最好）** |
| LoRA微调 + confidence（纯微调对照） | 4.54 | 6.75 |
| LoRA微调 + readiness planner | 4.76 | 6.81 |

- 辅助损失版：增益**全来自 LoRA 微调副作用**（对照 lora+conf 已达 4.54），readiness planner 无净贡献。
- 可微 commit 版：训练目标反而**把模型整体搞差了**（对照从 4.54 退到 6.75），连不训练的 certifier 都反超。

**含义**：在单卡+小数据+LoRA 规模下，"训练诱导 commit-ready 表征" **无法带来超越 inference planner 的收益**，
甚至有害。readiness 信号作为**判据**有效，作为**训练目标**与语言建模冲突。

---

## 汇总：现象之间的逻辑闭环

1. MDLM 想并行却退化成近似 AR（现象4：L2R≈0.62）。
2. 根因是 confidence 门控——高 confidence 只在上下文充分后出现，逼着串行（现象5）。
3. 存在独立于 confidence 的量 Stability，能刻画"真正安不安全敲定"（现象1）。
4. Stability 主要能从单位置 hidden 读出（state-local，现象2），且藏在浅层（现象3）。
5. 用它做 certifier 挑 commit，比 confidence 可靠（现象6），真实生成中是更好的 planner（现象7）。
6. **但**：certifier 只在低并行占优、高并行趋同（现象7）；把 readiness 塞进训练重塑表征无效/有害（现象8）。

## 尚未解决的核心问题

- 现象 6-8 共同表明：**当前成果止步于"更好的 inference-time planner"**，
  没有真正解决"如何让 MDLM 具备可靠的高并行 commit 能力"这一原始问题。
- 现象 8 暗示一个更底层的张力：**"可靠并行 commit" 与 "以 token 为生成原子" 可能本质冲突**——
  只要生成原子是 token，confidence 就必然要等上下文充分、必然串行；
  在 token 粒度上加 planner 或辅助损失都绕不开这一点。
- 由此指向 GPT 讨论最后那个尚未回答的问题：**语言模型真正的 generation primitive 是不是 token？**
  若不是（而是某种 block / latent unit），MDLM 的可靠并行或许才有出口。

---

## 已知局限

- 单模型（Dream-7B）、英文短句、样本量有限；Stability 为蒙特卡洛近似（有噪）。
- probe/certifier 为线性；命题 C 受单卡算力限制（LoRA + 小数据）。
- 质量指标用 GPT-2 NLL（对流畅度敏感，微调过拟合会虚假拉低），需换真实下游任务指标复核。
- 结论为**强初步观察**，非定论级结果。
