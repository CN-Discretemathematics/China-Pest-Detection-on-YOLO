# 方法设计：ADEA —— 粘连与密度感知增强分配

> Adhesion- and Density-aware Enhanced Assignment
> 版本：v0.1（2026-10-23）｜ 对应代码：`ultralytics_ext/`
> 论文叙事：**D0 诊断（发现现象）→ FW-3/FW-4 失败案例（问题定位）→ I1/I2（监督层机制）
> → I3/I4（数据与采样层机制）→ 强基线（公平对照）→ 消融（证明净增益）→ I7（诚实的局限性）**

---

## 0. 贡献分类（**强制标注**，禁止把复现当创新）

| 编号 | 名称 | 类别 | 出处 |
|---|---|---|---|
| R1 | SPD-Conv（backbone 前 2 个 stride-2 卷积） | **复现** | EP-YOLO, Sensors 2026 |
| R2 | EMA 注意力（backbone ×2） | **复现** | EP-YOLO（实测最优注意力） |
| R3 | DySample（neck 上采样） | **复现** | CCDW-YOLO 2026 |
| R4 | P2 检测头（stride-4） | **复现** | arXiv:2512.16493 (YOLO11-4K) |
| R5 | Scale-Quality Aware EIoU | **复现** | AEP-PEST, JRTIP 2026 |
| R6 | 静态长尾重采样 | **复现（通用技巧）** | 实例数-显著性 |
| **I1** | **SLA：面积标定的公平正样本分配**（残差补偿 + 冲突消解递补） | **★创新（由本机诊断重构）** | 实测 `corr(log 相对尺度, n_pos)=+0.852`；面积标定后粘连残差 −0.761 |
| **I1a** | ALA：粘连感知配额补偿（初版方案，触发判据已被数据修正） | **降级为消融项** | 引擎保留，触发判据换为面积标定残差 |
| **I2** | **DLA：密度先验引导的分配与抑制** | **★创新** | FW-3 / FW-4（EP-YOLO 点名 DME） |
| **I3** | **DRA：难度感知的动态长尾重采样** | **★创新** | FW-1（EP-YOLO 点名 dynamic resampling） |
| **I4** | **ASM：β 分布匹配的粘连合成** | **★创新** | 查重后确认有增量 |
| I5 | AGS：粘连感知分级抑制 | **工程组件** | 与 SG-NMS 划界，见 §6.3 |
| I6 | 混淆对 / 分类群感知分类损失 | **可选应用型创新** | F5 + 论文分类学 |
| D0 | 粘连诊断分析 | **独立贡献（Analysis 章）** | 机制假说 M1 |
| **I7** | **天花板量化分析** | **独立贡献（Limitations 章）** | FW-2 |

**统一方法 ADEA 的关键性质**：I1–I4 全部作用在**监督分配层**与**数据/采样层**，
与已发表工作改造的**特征层 / 颈部 / 损失层正交** → 可叠加、互不冲突，
且**不改变推理时的模型结构**（I5 AGS 是唯一影响推理的，且是纯后处理）。

---

## 1. 机制现状：Ultralytics 的标签分配做了什么

`ultralytics/utils/tal.py::TaskAlignedAssigner.assign()` 的核心两步：

1. **每 GT 选正样本**：对每个 GT `g`，按对齐度量
   `m(a, g) = s(a, g)^α · u(a, g)^β`（`s` = 分类分，`u` = **CIoU**）取 **top-k（默认 10）** 个 anchor；
2. **独占归属**：每个 anchor 再用
   ```python
   target_gt_idx = mask_pos.argmax(-2)   # 在 GT 维度取最大
   ```
   决定它属于**哪一个** GT —— **每个 anchor 至多一个 GT**。

### 1.1 机制假说 M1（可证伪，本方法的存在前提）

> ⚠️ **实测结论：M1 的朴素形式已被证伪**（本机 68,009 个 GT 样本）。
> 粘连目标并**不是**正样本饥饿的主因；主因是**目标面积**。
> 详见 **§2.0b 诊断结论**（G0 裁决 = PASS，M1 以面积标定残差的形式修正后成立，
> M2 成立），方法据此重构为 **SLA**。
> 本节保留 M1 的原始表述，作为**方法演进的可追溯记录**。

> **M1**：当两个 GT 高度粘连（`β_gt` 大）且两者尺寸都很小时，
> 它们的候选 anchor 集合几乎完全重合（共享区域附近的 anchor 对二者的 CIoU 都高）。
> 由于第 2 步的 argmax **独占**分配，绝大多数正样本 anchor 被分配给其中一个 GT；
> 另一个 GT 只保留"严格更优"的少数 anchor；而每 GT 仅 10 个的 top-k 名额又被前者挤占。
> → 该实例的**正样本被系统性饿死**，表现为高粘连图上的漏检。

**为什么这个机制只伤害粘连小目标**：

| 情形 | 候选 anchor 集合 | argmax 结果 | 后果 |
|---|---|---|---|
| 两个大目标、彼此分开 | 几乎不重叠 | 各自独占自己的 anchor | 无影响 |
| 两个大目标、高度重叠 | 重叠但 anchor 数量充裕（每 GT 10 个名额） | 仍能各拿 10 个 | 影响有限 |
| **两个小目标、粘连** | 高度重合（anchor 总数 ≪ 20） | 一方几乎拿走全部 | **一方被饿死** ⬅ |

小目标的 anchor 候选本来就少（P3 上一层 10 px 目标只覆盖 1~2 个网格单元），
粘连进一步把候选集压到同一批 → argmax 独占 + 固定 top-k 的组合放大了这个效应。

### 1.2 验证设计（D0，零 GPU 成本 → **G0 kill gate**）

训练期用 `ultralytics_ext/diagnostics.py`（**只读**补丁，不改变分配行为）记录每个 GT 的：

- `n_pos`：分到的正样本 anchor 数；
- `β_max`：与最近邻 GT 的粘连度；
- `area`：目标面积。

离线统计（`scripts/diagnose_adhesion.py`）：

1. `corr(n_pos, β_max)`（Spearman + Pearson）—— 预期**显著负相关**；
2. `corr(n_pos, area)` —— 预期负相关（尺寸越小候选越少）；
3. 按 `β_max` 分桶的实例级召回率（Hungarian 匹配）；
4. 粘连子集 vs 非粘连子集 mAP 差值（复现论文 −4.96 pt）。

**G0 判据**：M1 成立 ⟺ (1) `corr(n_pos, β) ≤ −0.2` **或** (2) 粘连组平均 `n_pos` 显著低于孤立组
且被饿死样本中 ≥50% 属于粘连目标。

- **PASS** → 实现 I1 / I2；
- **FAIL** → **不实现 I1 / I2**，预算转 I3 / I4 / I7，论文转"D0 分析 + I7 上限论证 + 复现"路线
  （**分析型论文**，仍可发表，且不浪费算力）。

> **这是本方案最重要的防错设计**：先验证机制、再写代码。
> 避免造出一个"看起来合理、实则无机制基础"的模块。

---

## 2. I1 · ALA（Adhesion-aware Label Assignment）

### 2.0b 诊断结论（本机实测，G0 裁决 = PASS）

> 本节是全文的**事实基础**。数据来自本机在 Pest24 上采集的 **68,009 个 GT / 8,372 个 batch**
> （`scripts/diagnose_adhesion.py`，ALA 关闭时的原始分配）。
> **初版方案的核心假说 M1 被实测证伪**，方法据此重构 —— 这一纠正是本工作的第一项贡献。

#### 发现 1：正样本数由**目标面积**主导（M2 成立）

`corr(log 相对尺度, n_pos) = +0.852`（Spearman 0.884），且 n_pos 存在**饱和上限 11**：

| 相对尺度 | GT 样本 | n_pos 均值 | 中位数 | n_pos<3 占比 |
|---|---|---|---|---|
| <1e-4 | 1,086 | **2.27** | 2 | **59.7%** |
| 1e-4~3e-4 | 6,340 | 4.07 | 4 | 13.4% |
| 3e-4~1e-3 | 25,177 | 6.31 | 6 | 1.2% |
| 1e-3~3e-3 | 27,066 | 9.84 | 10 | 0.0% |
| ≥3e-3 | 8,340 | ~10.0 | 10 | 0.0% |

**机理**：Ultralytics 的 anchor 是**点**（`anc_points`），只有中心落在 GT 框内的 anchor
才是候选 → 候选数 ∝ 面积 / stride²。在 640 输入下 P3 层 stride=8，
一个边长 8px 的目标框内**只有约 1 个** anchor 中心 → 微目标天生拿不到足够正样本。
**这不是"特征不够"（EP-YOLO 的归因），而是"监督分配在几何上就覆盖不到"。**

#### 发现 2：初版 M1（粘连致饿死）被证伪 —— 面积效应掩盖了一切

朴素判据给出的结果与预期**方向相反**：

| 观测量 | 数值 | 初版预期 |
|---|---|---|
| `corr(n_pos, β)` | **+0.52**（正） | 负 |
| 粘连组 vs 孤立组平均 n_pos | **7.64 vs 1.24** | 粘连更少 |
| 被饿死样本中粘连目标占比 | **仅 0.23%** | 高 |
| `corr(n_pos, β)`（面积标定后） | **-0.247** | — |

**为什么会错**：粘连的虫通常**体型更大**（β 大的多是框大、彼此靠近的个体），
面积效应（+0.85）完全压过了粘连效应 → 直接比较绝对 n_pos 必然得出反向结论。

#### 发现 3：面积标定后，粘连剥夺效应**确实存在**（M1 以修正形式成立）

用"**在未被剥夺的样本（非粘连）上零残差**"这一原则标定期望模型：

```
E[n_pos | a_rel] = min(k0, c · a_rel)      k0 = 11（实测饱和上限），c = 9,476
```

| 分组 | 平均残差 n_pos − E[n_pos\|a] |
|---|---|
| 非粘连（标定锚点） | **0.000** |
| **粘连（β>0.05）** | **−0.761** |

`corr(residual, β) = −0.247`，且在 c ∈ [4,000, 20,000] 的**全区间稳健**
（−0.17 ~ −0.29，粘连残差恒低于孤立残差）→ **粘连目标确实被剥夺**，
只是幅度远小于面积效应。58.96% 的 GT 处于"被低估"状态。

**关键代码事实**（解释了粘连为何被剥夺）——**本项目已逐行核对 ultralytics 8.4.184 源码并实测验证**：

* `ultralytics/utils/tal.py::select_topk_candidates` 末尾确有
  `count_tensor.masked_fill_(count_tensor > 1, 0)`，
  但 `count_tensor` 是按 **`(gt, anchor)`** 索引的（`scatter_add_` 的 dim=-1），
  同一 GT 的 top-k 下标互不重复 → `count` 恒 ≤ 1，
  **该行其实是死代码**（实测 `count.max() == 1`，已写入 `scripts/selftest_sla.py` 的对照测试）；
* 真正的剥夺发生在 **`select_highest_overlaps`**：
  一个 anchor 被多个 GT 同时选中（top-k 列表重叠）时，
  **只有与预测框 IoU 最高的那个 GT 保留它**
  （源码注释：*"Anchors assigned to multiple gt_bboxes keep the highest overlap"*），
  其余 GT **直接失去**这些 anchor。

→ 粘连簇里，共享区域被"几何上最贴合"的那个目标独占；
**尺寸较小的成员被系统性地剥夺了监督预算**——这正是残差 −0.761 的来源。

#### 发现 4（**决定性**）：用**几何口径**复测后，瓶颈被重新定位

上表的"面积标定残差"依赖一个**参数化拟合**（`min(k0, c·a)`），
它把"面积效应"与"竞争剥夺"混在一起。为了得到**无拟合**的结论，
本项目在诊断采集器中补记了**两个几何字段**：

* `n_cand`：该 GT 的**可用候选数**（框内且 `align_metric>0` 的 anchor 数）→ 几何上限；
* `n_pos_usable`：它实际拿到的**可用**正样本数。

在 6,589 个 GT 样本（同 seed、同数据序的 baseline vs SLA 对照）上的结果：

| 观测量 | 数值 | 含义 |
|---|---|---|
| `n_pos == min(topk, n_cand)` 的比例 | **80.62%** | 分配器**基本已到几何上限** |
| 低于几何上限的 GT | **2.67%** | 可再分配的 headroom 很小 |
| 其中"池够却被抢"（竞争剥夺） | **0.24%** | **冲突剥夺极其稀疏** |
| `n_cand < 3`（几何不足） | **7.54%** | 微目标**天生**只有 1~2 个正样本 |
| `corr(log 相对面积, log n_cand)` | **0.965** | 候选池几乎**完全由面积决定** |
| 最小区间（相对面积 < 1e-4） | `n_cand` 均值 **0.41**、`n_pos` 均值 **1.85** | 微目标的监督预算**在几何上就不足** |

**结论（改写方法定位）**：

1. 正样本数是**目标尺度的函数**（`corr(log 相对尺度, n_pos) = 0.852`），
   根源是"候选池 ∝ 面积/stride²"这一**几何事实**，而不是"被邻居抢占"；
2. **竞争剥夺只占 0.24%**——初版"粘连感知配额补偿"（ALA）针对的机制**不是主要矛盾**；
3. 因此 SLA 的主机制必须转向**监督密度**：把"公平份额"从"几何上限"
   提升为"**配额下限**"，并用**面积标定的候选区扩展**去补足（见下）。

> 这也是本项目方法论的价值所在：**G0 kill gate 先证伪了朴素机制假说，
> 再用更细的几何口径重定位瓶颈**——避免了 35 GPU-h 的无效投入。

#### 发现 5：SLA 的本机端到端验证（baseline vs SLA，同 seed / 同数据序 / 同一 topk）

在 1,270 张训练图上各跑 1 epoch（1660S），用 `scripts/compare_sla.py` 做**同口径**对照
（`c` 由 baseline 标定后两次运行共用）：

| 指标 | baseline | SLA(M0+S1+S2) | Δ |
|---|---|---|---|
| 平均正样本数（fg 口径） | 8.406 | **9.507** | **+1.102** |
| 平均可用候选数 | 20.75 | 24.05 | +3.30 |
| **被饿死比例（n_pos<3）** | 8.04% | **6.68%** | **−1.36 pt** |
| **粘连目标残差** | −0.656 | **−0.127** | **+0.528** |
| 被低估比例 | 61.41% | 57.81% | −3.60 pt |
| `corr(残差, β)` | −0.177 | −0.150 | +0.027 |

**尺度公平（核心主张的直接度量）**：

| 相对面积档 | n | baseline n_pos | SLA n_pos | Δ |
|---|---|---|---|---|
| [0, 1e-4)（微目标） | 173 | 1.80 | **2.96** | +1.17 |
| [1e-4, 3e-4) | 486 | 3.75 | **7.50** | +3.75 |
| [3e-4, 1e-3) | 1,661 | 6.57 | **9.60** | +3.03 |
| [1e-3, 3e-3) | 2,393 | 9.87 | 9.94 | +0.07 |
| [3e-3, 1e-2) | 1,804 | 9.98 | 10.00 | +0.01 |
| [1e-2, 1) | 72 | 10.00 | 10.04 | +0.04 |

| `corr(n_pos, log 相对尺度)` | baseline **+0.8465** | SLA **+0.5609**（−34%） |
|---|---|---|

→ 微目标的监督预算被补齐，**大目标的预算不受影响**（topk 上限守住），
监督分配从"几何决定的尺度偏置"变成"接近尺度中立"。

**仍然存在的 headroom**（诚实记录）：
* S2 的公平再分配只触发了 **15 次 / 382 批次** —— 因为竞争剥夺本身只有 0.24%；
* 因此**吸附簇**方向的收益有限，粘连子集的提升主要要靠 AGS（后处理）与特征侧（SPD/DySample）。

#### 由此得到的最终方法：SLA（面积标定的公平正样本分配）

| 组件 | 作用 | 依据（均为本机实测） |
|---|---|---|
| **M0 监督密度补偿**（主） | 候选池不足配额（默认对齐 assigner 的 `topk`，无自由参数）的 GT，其候选框**每边扩张 `margin·stride_val`**（保持中心）；扩张区同样进入 `align_metric` 计算 → top-k 自动挑走对齐质量最好的 anchor | 微目标的几何候选池被 stride 网格卡死；`corr(log 面积, log 候选池)=0.965` |
| **公平份额（面积标定）** | 份额 `need_i = min(topk, 候选池_i)`——**纯几何量**，与分辨率/结构无关 | 候选数 ∝ 面积/stride²；加 P2 后自动变大 |
| **S1 冲突裁决的度量一致性** | 冲突 anchor 归给 `align_metric` 最高者（与 top-k 选择同一度量），且只在提名者中裁决 | vanilla 用纯 IoU 裁决，与 top-k 选择用了两套标准 |
| **S2 公平再分配** | 被挤占的 GT 从欠剥夺的 GT 手里取回**同时位于自己框内**的 anchor（匹配度护栏 ρ），零和转移 | 粘连残差 −0.761；max-min 公平（最大缺口下降） |
| **尺度分层** | 在更细 stride 的监督层（P2/stride-4）为微目标提供足够候选 | 候选数 ∝ 面积/stride²；与 M0 是**架构层 / 分配层**两条互补路径 |

> **为什么 M0 用"边距"而不是"倍数"**：ultralytics 的 `select_candidates_in_gts`
> 会把尺寸小于 `stride_val` 的边**抬到 `stride_val`** →
> 对极小框做乘性扩张会被这条规则吃掉（5×5 框 ×2 = 10×10 → 又被抬回 16×16 → 候选池毫无变化）。
> 用**以 stride 为单位**的边距则对任意尺寸都有确定效果，且**与分辨率无关**。

**为什么份额用几何量而不是"面积表"**：早期设计用经验表 `E[n_pos|a_rel]`（6 档）作期望，
实测有两个隐患：(1) `a_rel` 的口径依赖 letterbox 缩放，表与运行时会错档；
(2) 加入 P2 层后候选池结构改变，同一张表不再成立。
改用 `need_i = min(topk, 候选数)` 后两个隐患同时消失，
且**加 P2 层时公平份额会自动适配**——这正是"面积标定"的本意：
**份额由面积（→ 候选数）决定，而不是拍脑袋的常数**。

**与初版 ALA 的关系**：ALA 的引擎（配额补偿、供体保护、不变量）**完全保留**，
只更换了 (a) 触发判据：从"粘连簇 + 绝对 n_pos"改为"**面积标定的公平份额**"；
(b) 供体池：从"同粘连簇"放宽到"欠剥夺 GT"；(c) 新增 S1 度量一致性、M0 密度补偿与 S2 公平再分配。
初版 ALA 作为**消融项**保留在代码里（`ala.py`），用于论文里的对照实验。

#### 一处必须写进论文的批判性观察

EP-YOLO §4.3 把微目标困难归因为"接近成像极限、需更高分辨率或多模态"。
本项目的诊断给出了一个**前人未提的补充解释**：即使在固定分辨率下，
**监督分配本身对微目标就是不足的**（正样本密度差 5 倍）→
因此单纯做特征增强（SPD/EMA/DySample）不足以解决微目标，
必须同时在**监督层**（更细的 stride）与**分配层**（面积标定的公平配额）动手。
这正是 EP-YOLO 把它列为 future work、却无人实现的部分。

### 2.1 粘连度与簇结构

沿用 Pest24 原论文定义：

```
β_ij = |box_i ∩ box_j| / |box_i ∪ box_j|
```

> ⚠️ **事实更正（由单元自测发现）**：该式与 **IoU 在数值上完全等价**（分子分母相同）。
> 原论文的"粘连系数"与"交并比"是**同一公式**，差别只在**语义**：
> IoU 在检测中通常被读作"两个框是否指向同一个物体"（重复检测信号），
> 而论文把它读作"两只虫在捕捉板上物理黏连的程度"（粘连信号）。
> 本项目沿用论文的 β 记号与阈值（0.05 = 论文的"nontrivial adhesion"），
> 但**不得**声称 β 与 IoU 在数值上有区别 —— 论文中必须如实说明这一点。

粘连图 `E`：`E[i,j] = 1 ⟺ β_ij > β_thr`，`β_thr = 0.05`（论文"nontrivial adhesion"口径）。
连通分量 → **粘连簇** `C_1, C_2, ...`。

### 2.2 算法

```
输入: 原始分配 {target_gt_idx[a]}, anchor 框 A, GT 框 G,
      β_thr=0.05, q_min=3, max_reassign_ratio=0.5, min_iou=0.05
      [可选] per_gt_qmin[i]  ← I2 DLA 注入的密度调制配额

Step 1  粘连簇:  E[i,j] = (β_ij > β_thr);  连通分量 → C
Step 2  配额统计: n_i = |{a : target_gt_idx[a] == i}|
Step 3  回收 (ALA-2):
        for 每个簇 C, for 每个 starved GT i ∈ C (n_i < q_min[i]):
            donors = C \ {i}
            候选 = { a : target_gt_idx[a] ∈ donors ∧ IoU(a, G_i) ≥ min_iou }
            按 IoU 降序遍历:
                donor = target_gt_idx[a]
                若 donor 的可让出额度已耗尽 → 跳过
                否则: target_gt_idx[a] = i;  额度--
        供体额度: allow_d = min( max(0, n_d - q_min[d]), floor(n_d · max_reassign_ratio) )
Step 4  返回新的 target_gt_idx
```

**不变量**：每个 anchor 仍然**至多归属一个 GT** → **无需修改 loss**，
可直接与未打补丁的 `vflLoss` 配合（这正是 §7 降级路径的第一步）。

### 2.3 为什么这样设计

| 设计选择 | 理由 |
|---|---|
| 保持"一 anchor 一 GT"不变式 | 不侵入 Ultralytics 的 loss 计算 → 版本风险最低、可 A/B 对照 |
| 只在**粘连簇内**回收 | 非粘连场景**零行为改变** → baseline 与改进版的差异可归因于粘连处理 |
| 供体保留 `q_min` + 比例上限 | 防止把"支配者"掏空 —— 它本身也是真目标 |
| `min_iou` 下限 | 回收的 anchor 必须与目标有实质重叠，否则是噪声 |
| `β_thr = 0.05` | 与论文"nontrivial adhesion"口径一致，可与论文的 −4.96 pt 直接对照 |

### 2.4 与最接近工作的差异

| 工作 | 层面 | 做了什么 | 与 ALA 的差异 |
|---|---|---|---|
| TAL / TOOD / SimOTA / ATSS | 监督分配 | 优化"分类-定位对齐" | **均假设 GT 互不重叠**；ALA 首次把 **GT 间粘连度** 显式引入监督分配 |
| SG-Det / SG-NMS | 推理后处理 | 学习 SGE 嵌入判定"高 IoU 但不同个体" | ALA 在**训练监督**层；不需额外头与二阶段精修；二者可叠加 |
| EP-YOLO / AEP-PEST / YOLO-DCPG | 特征/颈部/损失 | 改模块 | **没有一篇动过标签分配** |

### 2.5 复杂度

- 粘连矩阵：O(n_gt²)（n_gt ≤ 214，均值 7.6）→ 可忽略；
- 回收：O(|C| · |A|) 的 IoU 计算，簇内 GT 与 anchor 数都小；
- **总开销 < 1% 训练时间**（相对前向/反向）。

### 2.6 实现位置与 ultralytics 8.4.x 的 API 事实（**实测，务必先读**）

> ultralytics 8.4.184 的 `TaskAlignedAssigner` 已**重构**，与 8.3.x 差异很大。
> 若照搬旧版钩子点，ALA 会**静默失效**（补丁挂在不存在的属性上，不报错、不生效）。

| 项 | 8.3.x | **8.4.184（实测）** |
|---|---|---|
| 分配入口 | `TaskAlignedAssigner.assign(...)` | `TaskAlignedAssigner.forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra)` |
| 独占分配发生处 | `target_gt_idx = mask_pos.argmax(-2)` | `select_highest_overlaps(...)`：*"Anchors assigned to multiple gt_bboxes keep the highest overlap"* ← **这正是 M1 的机制** |
| top-k | 10 | 类默认 13，但 **`v8DetectionLoss(tal_topk=10)` 实际传入 10**（实测每 GT 最多 10 个正样本，与诊断数据吻合）；`topk2=None` → `topk2==topk` → `select_highest_overlaps` 里的二次过滤**被跳过** |
| GT×anchor 重叠度 | 需自行计算 | 分配器**已算好** `overlaps`（IoU 矩阵，形状 (b, n_max_boxes, num_anchors)）→ ALA 直接复用 |
| 背景 anchor 的 `target_gt_idx` | `-1` | **`0`** ← ⚠️ 见下方陷阱 2 |
| 通道记账 | — | 未知模块走 `else` 分支（输出通道被强制等于输入、args 原样传入）→ 自定义模块**必须**注入 `base_modules` |

**两个必须知道的陷阱**：

1. **`base_modules` 是 `parse_model` 的函数内局部变量**（8.4.184），`tasks.base_modules = ...` 改不动。
   本项目用**源码级注入**：把 `base_modules = frozenset(` / `repeat_modules = frozenset(`
   替换为 `_adea_base_frozenset(` / `_adea_repeat_frozenset(`，在 tasks 命名空间内重新 exec。
   ⚠️ **base 与 repeat 必须分开注入**：`repeat_modules` 会执行 `args.insert(2, n)`，
   把第 3 个位置参数顶成 repeats 数 —— 若把 DySample 也加进 repeat，
   它的 `scale=2` 会变成 `scale=1` → 前向形状错乱。

2. **背景 anchor 的 `target_gt_idx` 是 0 而非 -1**。若按 8.3.x 的直觉直接计数，
   GT0 的"正样本数"会变成全部 anchor 数，配额补偿随之失效。
   → ALA 的计数与候选筛选**必须按 `fg_mask` 过滤**，且**只允许回收正样本 anchor**
   （把背景 anchor "回收"给某个 GT 会凭空制造正样本）。

**ALA 的挂接点（两个补丁）**：

| 补丁 | 作用 |
|---|---|
| `get_pos_mask` | 暂存 `gt_bboxes` / `mask_gt`（后续步骤需要 GT 间粘连度） |
| `select_highest_overlaps` | 在独占分配**之后**施加 ALA 配额补偿 |

由于 `_forward` 随后会用修改后的 `target_gt_idx` 去算 targets 与归一化，
下游全部保持自洽，**无需修改 loss**。

**SLA 的挂接点（一个补丁）**：`get_pos_mask`

| 补丁 | 作用 |
|---|---|
| `get_pos_mask` | 拿到 `mask_pos` + `align_metric` + `overlaps` 三个张量后，重做冲突裁决并施加 S2 公平再分配，返回新的 `mask_pos` |

选这个点的原因：**只有这里同时拿得到 `align_metric` 与 `overlaps`**——
`select_topk_candidates` 里没有 `overlaps`（无法复现 vanilla 的 IoU 裁决），
`select_highest_overlaps` 里拿不到 `align_metric` 原始值（已被原地乘算）。
返回的 `mask_pos` 保持 `(b, n_max_obj, h*w)` 的 0/1 语义，
下游 `select_highest_overlaps` 会照常做 `max(-2)` 取归属 → 完全自洽，
且**每个 anchor 至多归属一个 GT** → 独占分配不变式保持 → **无需改 loss**。

### 2.7 端到端验证（已通过）

`scripts/selftest_assigner_ala.py` 在**真实** `TaskAlignedAssigner` 上自动搜索能复现
M1 饥饿的配置（两个粘连小 GT，模型预测覆盖二者的框 → 两者 IoU 几乎相等 →
argmax 偏向先出现的 GT → 另一个实例正样本归零）：

```
基线（ADEA_ALA=0）:  GT0 = 10 正样本,  GT1 = 0 正样本   ← 饿死复现
开启 ALA:            GT0 = 5  正样本,  GT1 = 5 正样本   ← 被救回
不变量:              总正样本数 10 → 10 不变；供体仍保留 ≥5
```

该测试是 I1 成立的**直接证据**，也是 T0 诊断包（G0）的预演。

### 2.8 ultralytics 的 GT 填充与坐标约定（**本机训练实测踩到的三个坑**）

在真实训练里跑通 ALA 之前，连踩三个坑 —— 它们都源于 ultralytics 对 GT 的两个约定：

| 约定 | 说明 | 踩坑表现 | 处理 |
|---|---|---|---|
| **GT 按 batch 补齐** | `gt_bboxes` 形状 (b, n_max_boxes, 4)，每张图的 GT **补齐到同一上限**（本项目实测 n_max_boxes=25），空位是全零 | `per_gt_qmin length 25 != n_gt 2` → 补丁抛异常被吞 → **ALA 静默失效** | 配额按 `mask_gt` 的**有效 GT 数**切片；`ala_core` 对长度不一致改为截断/补齐而非抛异常 |
| **GT 框是 xyxy** | 而本项目 `per_gt_quota` 内部按 **xywh**（中心在 b[0],b[1]）解释 | 密度被采样在框的**左上角**而非中心 → 密度调制完全错位（不报错，只是无效） | 新增 `xyxy_to_xywh_list()`，调用前显式转换 |
| **零填充行污染密度场** | 补齐用的全零行若参与密度渲染，会在图上造出假的"密度峰" | 密度图出现伪峰 | 计算密度前先按 `mask_gt` 过滤有效 GT |

### 2.9 「补丁静默失效」是这类工作的头号风险

本项目连续两轮被"补丁没生效但训练照常跑"坑到（ALA 挂错 API 钩子 → `assign` 在 8.4.x 已不存在；
DLA 配额长度不匹配 → 异常被 try/except 吞掉）。因此加入两道防线：

1. **首次异常打印完整 traceback**（不是一行摘要），并累计错误次数；
2. **`assigner_health()`**：训练结束后报告 `errors / reassigned / starved`，
   `errors>0` 或 `reassigned=0` 时打印醒目告警 —— **该实验的 ALA 结果不可信**。

本机冒烟训练的最终健康报告（真实训练循环、10% 子集、1 epoch）::

    HEALTH: {'errors': 0, 'reassigned': 38, 'starved': 53, 'patched': True}

即：53 个"被饿死"的实例被检出，38 个 anchor 被成功回收，**零异常**。

> **给复现者的建议**：任何对训练栈打补丁的工作，都必须配套一个"补丁是否真的生效"的
> 运行时断言，而不能只看训练 loss 是否正常下降 —— loss 正常下降和补丁生效是两件事。

### 2.10 I1b · SFW（尺度公平的监督权重）—— 与 SLA 互补的**损失层**一手

SLA 解决的是"**正样本的个数**"（几何 → 预算），但即使个数被拉平，
**分类损失里每个目标的贡献仍然 ∝ 它的正样本数**：

```
loss_cls = Σ_anchors BCE(pred_scores, target_scores) / Σ target_scores
```

Ultralytics 对正样本**等权**，而一个 GT 有多少正样本是由面积决定的
（`corr(log 相对面积, log 候选池) = 0.965`）→ **大目标在分类损失里的权重
天然是微目标的数倍**。这是一条**隐式的尺度偏置**，
与长尾类别偏置同源，但发生在**尺度轴**上，此前未被处理。

SFW 把分类损失里正样本的权重从"等权"改成"**按 GT 归一**"：

```
w_a = (N_pos / N_gt) / count_{gt(a)}        （a 为正样本 anchor）
```

* 每个 GT 对分类损失的**总贡献恰好相等**；
* 背景 anchor 权重保持 1.0 → 正负样本平衡不变；
* `Σ_pos w = N_pos` → 分母 `target_scores_sum` 不需要动 → **损失量级不变**。

**与最接近工作的差异**：Balanced L1 Loss（CVPR 2019）处理的是**回归项**的尺度偏置
（大目标的 L1 天然更大）；SFW 处理的是**分类项**里"**正样本个数**"这条通道——
它在检测器里此前是空白。

**实现**：`ultralytics_ext/loss/scale_fair.py`
（补丁 1：`TaskAlignedAssigner.forward` 结束后算每 anchor 权重；
补丁 2：`v8DetectionLoss.bce` 乘上权重）。
开关：`ADEA_SFW=1`。

**与 SLA 的分工**：SLA 让"个数"更公平（微目标 1.80 → 2.96 个正样本），
SFW 让"权重"更公平（每个目标的分类贡献相等）—— 两者都指向同一个诊断结论
（监督预算被尺度主导），但作用在不同层，**可叠加**。

---

## 3. I2 · DLA（Density-aware Label Assignment）

### 3.1 动机（EP-YOLO 原文）

> §4.3："Future research could consider incorporating **Density Map Estimation** methods …
> to tackle this crowding dilemma."
> §4.2 Fig.12c："**46 false predictions** in a crowded region containing 172 ground truth targets …
> making the detection head **overly sensitive** in specific regions."

**查重结论**：密集计数的密度图估计已成熟（DM-Count 等），但**从未用于"检测的标签分配与抑制调制"**。

### 3.2 与 I1 的分工（正交）

| | 粒度 | 刻画什么 |
|---|---|---|
| **ALA（I1）** | **GT 级** | 两个目标之间**黏得多紧**（局部） |
| **DLA（I2）** | **图像级** | 某个区域**有多拥挤**（全局） |

### 3.3 算法

```
Step 1  密度场渲染 (训练期，GT 已知，零额外可学习参数):
        D(x,y) = Σ_i exp(-((x-cx_i)² + (y-cy_i)²) / (2σ_i²)),  σ_i = max(r_i·σ_scale, 0.5)
        r_i = 目标等效半径;  σ_scale = 0.5
        归一化: D̂ = clip(D / quantile(D, 0.99), 0, 1)   # 抑制离群峰值
Step 2  密度-配额调制 (与 ALA-2 互补):
        q_min[i] = ceil( q_min_base · (1 + μ · D̂(c_i)) )
        → 拥挤处的目标获得更多正样本配额，补偿密集处的 anchor 竞争
Step 3  密度-抑制调制 (推理期, 交给 I5 AGS):
        高密度区 → 放宽抑制阈值（保留近邻不同个体）; 低密度区 → 收紧
Step 4  密集区敏感度抑制 (可选, 风险较高, 单独消融):
        高密度区提高置信度门槛 → 抑制"把噪声纹理读成微目标"
```

### 3.4 首版范围与降级

- **首版只做 Step 1–2**：训练期用 GT 密度场，**零额外可学习参数、零推理开销**；
- Step 3 与 Step 4 作为独立子项消融；
- **降级**：只保留 Step 2（配额调制）→ 仍与 ALA 正交，仍可单独报告。

### 3.5 实现

- `ultralytics_ext/assign/dla.py` —— `render_density` / `normalize_density` / `per_gt_quota`
  （numpy 可选，无 numpy 走纯 Python）；
- `scripts/density_prior.py` —— 渲染密度图 PNG + 统计"密度 vs 实例数"相关性（论文插图 + P8 校验）。

---

## 4. I3 · DRA（Difficulty-aware Resampling）

### 4.1 动机（EP-YOLO 原文）

> §4.3："Future work could introduce **dynamic resampling strategies** or generative model-based
> data augmentation techniques for minority classes to mitigate this issue."

静态重采样（R6）在训练开始前一次性决定比例；DRA **在训练过程中按当前模型表现**动态调节。

### 4.2 算法

```
Step 1  类别难度估计: 维护每类滚动统计 (rolling_AP / rolling_recall / rolling_loss)
                      每个 epoch 结束更新
Step 2  动态采样权重:  p_c ∝ (1/n_c)^α · exp(τ · difficulty_c)
                      difficulty_c = 1 - rolling_mean(AP_c)        (越难 → 权重越高)
                      τ 随训练退火 τ → τ_final                     (早期激进, 后期均衡)
Step 3  难例定向增强:  被判定为"难"的类, 提高 ASM 合成样本比例与 β 采样权重
Step 4  早停保护:      某类滚动 AP 连续 N 轮无提升 → 降低其采样权重 (避免过拟合稀有类)
```

### 4.3 收益估计

只作用于 4 个稀有类（<500 实例），但 **mAP 是 24 类宏平均**：
若 4 类各提 5~10 pt → mAP50 **+0.8~1.7 pt**。这是**杠杆最高**的单点改动之一。

### 4.4 与最接近工作的差异

Class-balanced / repeat-factor sampling 是**静态**通用技术；
DRA 额外做到：① 按模型**当前表现**（而非仅按样本数）调节；② 与合成增强（I4）联动；
③ 带过拟合保护。EP-YOLO 明确点名 "dynamic" 却未实现。

### 4.5 实现

`ultralytics_ext/sampling/dra.py` —— `DRASampler`（纯 Python）+ `build_static_weights`（R6 对照）。

---

## 5. I4 · ASM（Adhesion-Statistics-Matched synthesis）

### 5.1 动机

现有增强都不产生**粘连**样本：
`mosaic` 拼接**不重叠**图块；Ultralytics `copy_paste` 放置的实例几乎**不重叠**且用 **alpha 混合**；
通用遮挡增强（Simple Copy-Paste, CVPR 2021）面向"被遮挡的完整物体"。

真实粘板上的粘连是**两虫物理黏连、边界融合、投下联合阴影** —— 与"半透明叠加的遮挡"在外观上**相反**。

### 5.2 算法

```
Step 1  统计: 从 train split 估计 (类别 c, 尺度 s, 粘连度 β) 的联合经验分布
              (β 用论文定义; 论文统计多数落在 0~0.2)
Step 2  采样: 按该分布采样 (c, s, β)
Step 3  合成: 贴入两个实例, 调整相对位置使交并比**恰为**采样到的 β
              ★ 重叠区「实心合并」渲染 (上层像素实心覆盖, 不 alpha 混合)
                并在并集外缘加一圈软阴影
              位置求解: 对固定尺寸的矩形对, β(dx) 随 dx 单调递减 → 二分求解
Step 4  密度对齐: 按论文统计 (8.89% 的图 >20 实例) 补齐高密度合成场景
Step 5  输出: 离线生成到 datasets/pest24_synth/, 训练时 train+synth 合并
              (离线而非在线 → 完全确定性、可审计、可复现)
```

### 5.3 与最接近工作的差异

| | Simple Copy-Paste (CVPR 2021) | **ASM** |
|---|---|---|
| 目标现象 | **遮挡**（被遮挡的完整物体） | **粘连**（两实体融合成一团） |
| 位置参数 | 随机 | **数据集自身的 β 经验分布** |
| 渲染方式 | alpha 混合（半透明） | **实心合并 + 联合软阴影** |
| 密度对齐 | 无 | 有（对齐论文的 8.89% 高密度统计） |

### 5.4 风险与降级

标签/图像错位是主要坑 → **降级**：先做"两图实例实心叠加 + β 采样"，放弃阴影与密度对齐。

### 5.5 实现

`scripts/synth_adhesion.py` —— 离线合成，输出 `synth_manifest.json`（每个样本的 target/achieved β）
与 `synth_stats.json`（可审计）。

---

## 6. I5 · AGS（工程组件）与 I6（可选）

### 6.1 I5 AGS

对候选框对先判断"实例同一性"：
- **高同一性**（同类别 + 形状/尺寸高度相似）→ **硬抑制**（标准 NMS 行为，Recall 不受损）；
- **低同一性但高重叠**（两个粘连个体）→ **分数按 β 平滑衰减而非删除**。

**实现**：替换 `ultralytics.utils.ops.batched_nms`（签名与 torchvision 一致），
AGS 在 `val` / `predict` 中**直接生效**，无需改训练/推理代码路径；
分数衰减**原地**写回 `scores` 张量，Ultralytics 随后的 `scores = scores[keep]` 会正确传播。

### 6.2 I6 混淆对感知分类损失（可选）

用 baseline 混淆矩阵构建 24 类混淆图，对高混淆类对加**成对 margin 惩罚**
（只惩罚"A 判成 B"这一对，**不做全局类别重平衡** —— 避免长尾类被拖垮，与 I3 互补）。
分类学层级头（5 目 13 科）亦为可选。

**诚实评价**：两者通用版均为经典技术，仅主张"用目标域真实分类学 + 数据驱动混淆图"的域特化应用。

### 6.3 I5 与 SG-NMS 的划界（论文中必须写明）

SG-Det 用学习式 SGE 嵌入判定同一性，KITTI 重遮挡 car AP 62.08（较 Greedy-NMS **+5 pt**），
CityPersons miss rate 51.1%；但需**额外 SGE 头 + 二阶段精修**，且面向行人/汽车**遮挡**。
AGS：单阶段 anchor-free、**无额外头**、用**数据集自身的 β 标定**、面向**粘连**。

---

## 7. D0 诊断与 I7 天花板分析（两项独立贡献）

### 7.1 D0（Analysis 章）
见 §1.2 的 4 组统计 + `scripts/diagnose_adhesion.py`。

### 7.2 I7（Limitations 章）—— 回应 FW-2

**动机**：EP-YOLO §4.3："the physical size of these micro-pests **approaches the imaging limit of
the camera sensor** … relying solely on algorithmic feature enhancement is insufficient."

**做法**：画 **极小类（Rice planthopper / Plutella xylostella）的 AP 上限曲线**：
横轴 = 有效输入分辨率（imgsz 640/800）× 模型容量（n/s），纵轴 = 该类 AP。

**判据**：
- AP 上限**仍随分辨率/容量上升** → 当前瓶颈是**分辨率与容量**而非算法
  → 支撑"换原图 + YOLO11s/m 可达 X"的**量化**结论（比 EP-YOLO 的定性说法更硬）；
- AP 上限**饱和** → 已触及**成像物理极限**，任何算法改进的边际收益趋零
  → 这是本工作的**负责任结论**，把"提升不上去"从失败重新定义为上限刻画。

---

## 8. 强基线（Strong Baseline）—— 方法论关键

```
yolo11n-strong = YOLO11n
                + (R1 SPD-Conv ×2)
                + (R2 EMA 注意力 ×2, backbone)
                + (R3 DySample, neck)
                + (R4 P2 检测头)
                + (R5 Scale-Quality Aware EIoU)
```

**我们的贡献 = 相对这个强基线的净增益**，而不是相对 vanilla YOLO11n 的增益。
这是让工作从"复现"变成"论文"的关键结构化调整：**在最强对手之上再赢**。

> ⚠️ 若强基线无法超过 vanilla baseline（闸门 G2 不通过），说明**复现本身有问题**，
> 必须先修复现实现，而不是继续叠加创新。

---

## 9. 实验设计与消融

| 档位 | 内容 | 闸门 |
|---|---|---|
| **T0** | D0 诊断（零 GPU 成本） | **G0**：M1 是否成立 |
| **T1** | baseline / ft_aug / ref_yolov8n / ft_aug_cls / AGS / conf-iou 扫描 | G1：`ft_aug` ≥ +0.5 pt |
| **T2** | 7 组复现筛查 + **强基线组装** | G2：强基线 ≥ +1.5 pt |
| **T3** | I1 ALA / I2 DLA / I3 DRA / I4 ASM 及组合 | G3：≥1 组件 ≥ +0.5 pt **且定向子集 ≥ +1 pt** |
| **T4** | 组合定稿 × 3 seed + YOLO11s 上限 + 天花板曲线 | — |

### 9.1 定向子集（固定口径，全程不变）

| 子集 | 定义 | 对应创新 |
|---|---|---|
| 粘连子集 | 存在 GT 对 β > 0.05 | I1 ALA / I4 ASM |
| 高密度子集 | 单图实例数 > 20 | I2 DLA |
| 稀有类子集 | 类别实例数 < 500（4 类） | I3 DRA |
| 极小类子集 | 类别平均相对尺度 < 0.05%（2 类） | I4 P2 头 / 天花板 |

### 9.2 验收标准（档二 · 创新增量）

- **B0** G0 诊断显示 M1 成立；
- **B1** 最优创新组合相对 **强基线**：mAP50 ≥ **+1.0 pt** 且 mAP50-95 ≥ **+1.0 pt**（3 seed 均值）；
- **B2 定向性**（每个组件在其目标子集上必须有提升）：

| 组件 | 目标子集 | 最低要求 |
|---|---|---|
| **I1 SLA**（M0 密度补偿 + S1 + S2） | **微目标子集**（相对面积 < 1e-3） | 微目标类平均 AP ≥ +2 pt；整体 mAP50 ≥ +1.0 pt |
| **I1b SFW**（尺度公平权重） | **微目标 + 稀有类子集** | 微目标类平均 AP ≥ +1 pt；稀有类平均 AP ≥ +1 pt |
| I2 DLA | 高密度子集 | mAP50 ≥ +1.0 pt **且密集区误检率下降 ≥ 15%** |
| I3 DRA | 稀有类子集 | 稀有类平均 AP ≥ +3 pt |
| I4 ASM / I5 AGS | 粘连子集 | mAP50 ≥ +1.0 pt |

- **B3 组件归因**：单测增益、组合增益、交互项有明确分解；
- **B4 显著性**：3 seed 配对检验（配对 t 检验 p<0.05，或 3/3 seed 方向一致 + 效应量）；
- **B5 若不达标**：诚实报告，交付 D0 + I7 + 复现作为论文主体。

### 9.3 风险与降级阶梯

| 风险 | 降级路径 |
|---|---|
| M1 证伪（G0 FAIL） | 不实现 I1/I2 → 转 I3/I4/I7，分析型论文路线 |
| ALA 改 assigner 后 loss 不收敛 | ① 仅 ALA-2 配额补偿（不放宽独占分配）→ ② 仅对粘连 GT 补采样 → ③ 放弃 |
| DLA 密度调制无增益 | 只保留 Step 2（配额调制） |
| DRA 采样不稳定 | τ 退火至 0，退化为静态 R6 |
| P2 yaml 构建失败 | 改走 SPD+EMA+DySample 组合 |
| SPD 后显存/FLOPs 暴涨 | SPD 后压回通道；或改单 SPD + 更强注意力 |
| SQA-EIoU 与版本不兼容 | 降级为原生 box 权重调整 + WIoU 轻量实现 |
| ASM 标签错位 | 放弃阴影与密度对齐，只做实心叠加 + β 采样 |

---

## 10. 复现与可审计性

- 全部实验**同实例、同卡型、同 ultralytics 版本**（`env_snapshot.txt` 存档）；
- **数据只转换一次**，三 split 固定，记录每类实例数；
- **seed 固定**（T1–T3 用 0；T4 用 0/1/2），同 seed 重跑差异 < 0.5 pt 视为噪声；
- **实验间对比**统一用默认 `conf=0.001 / iou=0.7`；conf/iou 扫描只定部署点；
- `results.md` 每行记录：实验名 / 改动 / imgsz / epochs / seed / eval split / P / R / mAP50 / mAP50-95 / 4 个定向子集 mAP / 耗时 / 参数量 / FLOPs / 结论。

---

## 11. 代码索引

| 文件 | 内容 |
|---|---|
| `ultralytics_ext/assign/sla.py` | **I1 SLA（核心创新）**：公平份额 `min(topk, 候选数)` + S1 冲突裁决（度量一致性）+ S2 面积标定公平再分配 |
| `ultralytics_ext/assign/ala.py` | I1 ALA（已降级为消融项）：纯算法层 `ala_core` + torch 适配 `apply_ala` |
| `ultralytics_ext/assign/dla.py` | I2 DLA：密度渲染 / 归一化 / 逐 GT 配额 |
| `ultralytics_ext/sampling/dra.py` | I3 DRA：动态采样器 + 静态对照 |
| `ultralytics_ext/post/ags.py` | I5 AGS：分级抑制 + `ags_batched_nms` |
| `ultralytics_ext/modules/spd_conv.py` | R1 SPD-Conv |
| `ultralytics_ext/modules/ema_attn.py` | R2 EMA（优先官方实现） |
| `ultralytics_ext/modules/dysample.py` | R3 DySample |
| `ultralytics_ext/loss/scale_fair.py` | **I1b SFW**：尺度公平的分类权重（每 GT 贡献归一） |
| `ultralytics_ext/loss/scale_eiou.py` | R5 SQA-EIoU（多挂载点探测） |
| `ultralytics_ext/loss/pair_aware_cls.py` | I6 混淆对感知分类损失（可选） |
| `ultralytics_ext/register.py` | 接线：模块注册 / **SLA+ALA assigner 补丁** / NMS 替换 / 损失安装 / `assigner_health()` |
| `ultralytics_ext/diagnostics.py` | D0 训练期只读采集（n_pos / **可用候选数 n_cand** / **可用正样本 n_pos_usable** / β / 面积） |
| `ultralytics_ext/train_adea.py` | 训练入口包装器（接线后交给 Ultralytics entrypoint） |
| `scripts/make_model_yamls.py` | 从官方 yolo11.yaml 程序化派生 6 套模型配置（含构建 + 前向 + 权重迁移率校验） |
| `scripts/diagnose_adhesion.py` | D0 分析 + **G0 裁决** |
| `scripts/compare_sla.py` | **SLA 前后对照分析**（几何缺口 + 面积标定残差，同 seed 同数据序） |
| `scripts/synth_adhesion.py` | I4 ASM 合成 |
| `scripts/density_prior.py` | I2 密度场渲染与统计 |
| `scripts/scan_thresholds.py` | conf/iou 扫描（定部署点） |
| `scripts/summarize.py` | runs/ → results.md 台账 |
| `scripts/selftest_adea.py` | **纯算法自测**（ALA/DLA/AGS/DRA，35 项，不依赖 torch） |
| `scripts/selftest_assigner_ala.py` | **ALA 端到端自测**（真实 assigner 上复现 M1 饥饿并验证救援） |
| `scripts/selftest_sla.py` | **SLA 自测**（27 项：与 vanilla 逐 anchor 等价 / S1 度量一致性 / S2 零和转移 / 护栏 / M0 扩张 / 不变式） |
