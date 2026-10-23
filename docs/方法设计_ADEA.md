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
| **I1** | **ALA：粘连感知标签分配** | **★创新** | FW-3 / FW-4 |
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
| top-k | 10 | **13**（`TaskAlignedAssigner(topk=13)`，loss 里另用 `tal_topk`） |
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
| I1 ALA | 粘连子集 | mAP50 ≥ +1.5 pt |
| I2 DLA | 高密度子集 | mAP50 ≥ +1.0 pt **且密集区误检率下降 ≥ 15%** |
| I3 DRA | 稀有类子集 | 稀有类平均 AP ≥ +3 pt |
| I4 ASM | 粘连子集 | mAP50 ≥ +1.0 pt |

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
| `ultralytics_ext/assign/ala.py` | I1 ALA：纯算法层 `ala_core` + torch 适配 `apply_ala` |
| `ultralytics_ext/assign/dla.py` | I2 DLA：密度渲染 / 归一化 / 逐 GT 配额 |
| `ultralytics_ext/sampling/dra.py` | I3 DRA：动态采样器 + 静态对照 |
| `ultralytics_ext/post/ags.py` | I5 AGS：分级抑制 + `ags_batched_nms` |
| `ultralytics_ext/modules/spd_conv.py` | R1 SPD-Conv |
| `ultralytics_ext/modules/ema_attn.py` | R2 EMA（优先官方实现） |
| `ultralytics_ext/modules/dysample.py` | R3 DySample |
| `ultralytics_ext/loss/scale_eiou.py` | R5 SQA-EIoU（多挂载点探测） |
| `ultralytics_ext/loss/pair_aware_cls.py` | I6 混淆对感知分类损失（可选） |
| `ultralytics_ext/register.py` | 接线：模块注册 / assigner 补丁 / NMS 替换 / 损失安装 |
| `ultralytics_ext/diagnostics.py` | D0 训练期只读采集 |
| `ultralytics_ext/train_adea.py` | 训练入口包装器（接线后交给 Ultralytics entrypoint） |
| `scripts/make_model_yamls.py` | 从官方 yolo11.yaml 程序化派生 6 套模型配置（含构建 + 前向 + 权重迁移率校验） |
| `scripts/diagnose_adhesion.py` | D0 分析 + **G0 裁决** |
| `scripts/synth_adhesion.py` | I4 ASM 合成 |
| `scripts/density_prior.py` | I2 密度场渲染与统计 |
| `scripts/scan_thresholds.py` | conf/iou 扫描（定部署点） |
| `scripts/summarize.py` | runs/ → results.md 台账 |
| `scripts/selftest_adea.py` | **纯算法自测**（ALA/DLA/AGS/DRA，35 项，不依赖 torch） |
| `scripts/selftest_assigner_ala.py` | **ALA 端到端自测**（真实 assigner 上复现 M1 饥饿并验证救援） |
