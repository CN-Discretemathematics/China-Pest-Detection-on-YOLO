# China-Pest-Detection-on-YOLO

**面向农业害虫密集粘连场景的 YOLO11n 改进方法 —— ADEA**
（*Adhesion- and Density-aware Enhanced Assignment*，粘连与密度感知增强分配）

> 目标数据集：**Pest24**（24 类田间农作物害虫，25,378 张 800×600 图像）
> 基线：**YOLO11n** ｜ 核心贡献：**在监督标签分配层与数据采样层**提出 4 项机制

---

## 1. 这个仓库在做什么

Pest24 的公开研究已把「目标粘连」确认为**第一影响因子**（论文：4657 张图存在非忽略粘连 β>0.05；粘连子集 mAP 比整体低 4.96 个点）。但**所有已发表工作**（EP-YOLO / AEP-PEST / YOLO-DCPG）都只在**特征层、颈部、损失层**做改进，后处理最多调一下 NMS 阈值 —— **没有人动过监督标签分配这一层**。

本仓库提出 **ADEA**，把创新点放在两个已发表论文明确点名、却至今无人实现的 future work 上：

| 编号 | 机制 | 层级 | 对应的 future work 出处 |
|---|---|---|---|
| **I1** | **ALA** 粘连感知标签分配 | 监督分配层 | EP-YOLO §4.3："NMS … could filter out valid targets as redundant predictions incorrectly" |
| **I2** | **DLA** 密度先验引导的分配与抑制 | 监督分配层 + 后处理 | EP-YOLO §4.3："Future research could consider incorporating **Density Map Estimation** methods" |
| **I3** | **DRA** 难度感知的动态长尾重采样 | 数据采样层 | EP-YOLO §4.3："Future work could introduce **dynamic resampling strategies** … for minority classes" |
| **I4** | **ASM** β 分布匹配的粘连合成 | 数据合成层 | 与通用 copy-paste（面向遮挡、半透明叠加）划界：ASM 面向**粘连**（实心融合渲染 + 数据集自身的 β 分布） |

同时提供 **D0 粘连诊断分析**（论文 Analysis 章）与 **I7 天花板量化分析**（论文 Limitations 章）两项独立贡献。

**方法学关键**：本项目不与「vanilla YOLO11n」比较，而是与**强基线**（把已发表最优模块组合移植到 YOLO11n 后的 `yolo11n-strong`）比较 —— **在最强对手之上再赢**。

---

## 2. 快速开始

```bash
# 0) 环境（GPU：CUDA 12.1，兼容 GTX 1660 SUPER / Turing sm_75）
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 1) 数据（Kaggle 镜像，800×600，2.38 GB）
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('boatshuai/pest24'))"

# 2) 转换 VOC XML → YOLO txt（只做一次）
python scripts/voc2yolo.py --src <下载目录>/VOCdevkit/voc2007 --dst datasets/pest24

# 3) 冒烟测试（10% 子集，3 epoch）
bash train/30_smoke.sh

# 3.5) 本地自测（不消耗 GPU）
python scripts/selftest_adea.py          # 纯算法自测：ALA/DLA/AGS/DRA（35 项）
python scripts/selftest_assigner_ala.py  # ALA 端到端：真实 assigner 上复现 M1 饥饿并验证救援

# 4) 正式实验矩阵
bash train/40_t1.sh    # T1 基线与工程线
bash train/50_t2.sh    # T2 复现筛查 + 强基线组装
bash train/60_t3.sh    # T3 创新模块包
bash train/70_t4.sh    # T4 定稿与统计
```

### 3.1 模型配置的生成方式（**不要手改 yaml**）

6 套模型配置由 `scripts/make_model_yamls.py` **从当前安装的官方 `yolo11.yaml` 程序化派生**，
以保证与所用 ultralytics 版本完全一致（通道缩放、层索引、Detect 输入列表都对得上）：

```bash
python scripts/make_model_yamls.py --validate   # 生成 + 构建 + 前向 + 权重迁移率校验
```

| 变体 | 内容 | 参数量 | 权重迁移率 | 检测尺度 |
|---|---|---|---|---|
| `p2/` | R4 P2 检测头 | 2.706M | 82.3% | 4 |
| `spd/` | R1 SPD-Conv×2 | 2.610M | 99.1% | 3 |
| `ema/` | R2 EMA×2（通道守恒） | 3.037M | 99.3% | 3 |
| `dysample/` | R3 DySample | 2.636M | 99.3% | 3 |
| `spd_ema/` | R1+R2 | 3.053M | 99.1% | 3 |
| **`strong/`** | **★强基线 R1+R2+R3+R4** | **3.205M** | 82.1% | 4 |

---

## 3. 数据集：Pest24

- **来源**：中科院合肥物质科学研究院，*Computers and Electronics in Agriculture* 175:105585 (2020)
  DOI: [10.1016/j.compag.2020.105585](https://doi.org/10.1016/j.compag.2020.105585)
- **获取**：Kaggle 镜像 [`boatshuai/pest24`](https://www.kaggle.com/datasets/boatshuai/pest24)
  （原图 2095×1944 版本的官方地址 `aisys.iim.ac.cn/zhibao.html` 目前返回 502）
- **许可**：**非商业免费使用**。本仓库**不包含**数据集本体（见 `.gitignore`），使用时请遵守原始许可。
- **规模**：25,378 张，24 类，平均 7.6 实例/图（最多 214），8.89% 的图 >20 实例。
- **难点**（论文统计）：
  1. 目标极小 —— 相对尺度跨度 1600×（0.00125% ~ 2.002%）；
  2. 密集 + 粘连 —— 4657 张图 β>0.05，粘连子集 mAP 低 4.96 pt；
  3. 类间相似度极高 —— 两两相似度均值 0.533，85.5% 的类对 > 0.5；
  4. 长尾极端 —— 53,347 实例 vs 108 实例（mAP 为 24 类宏平均 → 稀有类是杠杆点）。

### ⚠️ 评测口径（务必遵守）

- 官方划分 5:2:3；**官方仅公开 train/val 标注**。若镜像的 test 无标注，
  则统一用 `val` 作为 `EVAL_SPLIT` 并写进 `results.md` 头部，**全程不变**。
- **禁止自行从 train 切 test**（破坏与社区口径的可比性）。
- **实验间对比一律使用默认 `conf=0.001` / `iou=0.7`**；conf/iou 扫描只用于确定部署点。
- **跨论文的数值不可直接对表**（各论文的划分/分辨率/imgsz/轮数均不同）。

---

## 4. 仓库结构

```
.
├── configs/
│   ├── datasets/pest24.yaml        # 数据集配置（24 类 names）
│   └── models/                     # 模型配置（见 ultralytics_ext/yaml/）
├── scripts/                        # 数据准备 / 诊断 / 合成 / 扫描 / 汇总
│   ├── voc2yolo.py                 # VOC XML → YOLO txt
│   ├── diagnose_adhesion.py        # ★D0 粘连诊断 + G0 kill gate 裁决
│   ├── synth_adhesion.py           # ★I4 ASM 粘连合成
│   ├── density_prior.py            # I2 DLA 密度场渲染与统计
│   ├── scan_thresholds.py          # conf/iou 扫描（定部署点）
│   └── summarize.py                # runs/ → results.md 台账
├── ultralytics_ext/                # ★方法代码主体
│   ├── register.py                 # 接线入口（install_all）
│   ├── diagnostics.py              # 训练期 n_pos 采集（只读）
│   ├── modules/                    # R1 SPD-Conv / R2 EMA / R3 DySample
│   ├── assign/                     # ★I1 ALA / ★I2 DLA
│   ├── sampling/                   # ★I3 DRA
│   ├── post/                       # I5 AGS（工程组件）
│   ├── loss/                       # R5 SQA-EIoU / I6 混淆对感知分类损失
│   └── yaml/                       # p2 / spd / ema / dysample / strong
├── train/                          # 可直接执行的实验脚本（10_env → 70_t4）
├── docs/
│   ├── 调研报告_农业害虫检测前沿.md   # 前沿调研 + future work 溯源表
│   ├── 方法设计_ADEA.md               # I1–I4 推导、伪代码、与前作对比
│   ├── Pest24_YOLO11n_部署与微调指南.md
│   └── autodl_agent_action_plan.md
├── results.md                      # 实验台账
└── artifacts/                      # 结果表与图（不放权重）
```

---

## 5. 接线：ADEA 怎么接进 Ultralytics

接线全部通过**环境变量**完成（Ultralytics 的 CLI 会对未知键报错，故不用 CLI 参数）：

```bash
# 启用 I1 ALA（粘连感知标签分配）
ADEA_ALA=1 ADEA_ALA_BETA_THR=0.05 ADEA_ALA_Q_MIN=3 \
  yolo detect train model=ultralytics_ext/yaml/strong/yolo11n.yaml ...

# 启用 I2 DLA（密度先验调制）
ADEA_DLA=1 ADEA_DLA_MU=0.5 yolo detect train ...

# 启用 I5 AGS（推理期替换 batched_nms）
ADEA_AGS=1 ADEA_AGS_DECAY=0.5 yolo detect val ...

# 采集诊断数据（只读，不改行为）
ADEA_DIAG=1 ADEA_DIAG_OUT=runs/diag/assign_stats.jsonl yolo detect train ...
```

或用 Python API：

```python
from ultralytics_ext.register import install_all
install_all()                      # 打印接线报告
```

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `ADEA_ALA` | 0 | I1 开关 |
| `ADEA_ALA_BETA_THR` | 0.05 | 粘连阈值（论文口径） |
| `ADEA_ALA_Q_MIN` | 3 | 每 GT 最小正样本数 |
| `ADEA_DLA` | 0 | I2 开关 |
| `ADEA_DLA_MU` | 0.5 | 密度调制强度 |
| `ADEA_DRA` | 0 | I3 开关 |
| `ADEA_AGS` | 0 | I5 开关 |
| `ADEA_AGS_DECAY` | 0.5 | AGS 分数衰减强度 |
| `ADEA_SQA_EIOU` | 0 | R5 开关 |
| `ADEA_PAIR_CLS` | 0 | I6 开关 |
| `ADEA_DIAG` | 0 | 诊断采集开关 |
| `ADEA_CONFIG` | — | JSON 配置文件路径 |

> **所有开关默认关闭** —— 保证 baseline 训练不受任何影响（A/B 对照的前提）。

---

## 6. G0 kill gate：先验证机制，再写代码

创新是否成立，取决于机制假说 **M1** 是否为真：

> 两个高度粘连且都很小的目标，其候选 anchor 集合几乎完全重合；
> 由于 Ultralytics `TaskAlignedAssigner` 用 `mask_pos.argmax(-2)` 做**独占分配**，
> 加上每 GT 仅 `top-k=10` 的名额被"支配者"挤占，
> 其中一个实例的**正样本会被系统性饿死**。

**执行顺序（务必遵守）**：

```bash
# 1) 先跑一次 baseline（带诊断采集）
ADEA_DIAG=1 ADEA_DIAG_OUT=runs/diag/assign_stats.jsonl \
  yolo detect train model=yolo11n.pt data=datasets/pest24/pest24.yaml ...

# 2) 裁决
python scripts/diagnose_adhesion.py --data datasets/pest24 \
  --assign-stats runs/diag/assign_stats.jsonl --out artifacts/diag
```

裁决写在 `artifacts/diag/G0_verdict.md`：

- **PASS** → 继续实现 I1 ALA / I2 DLA；
- **FAIL** → **不实现 I1/I2**，预算转 I3 DRA / I4 ASM / I7 天花板分析，
  并把 D0 诊断 + I7 上限论证作为论文主体（分析型论文路线，仍可发表）。

---

## 7. 实验矩阵与闸门

| 档位 | 内容 | 闸门判据 |
|---|---|---|
| **T0** | 粘连诊断（零 GPU 成本） | G0：M1 是否成立 |
| **T1** | baseline / ft_aug / ref_yolov8n / ft_aug_cls / AGS / conf-iou 扫描 | G1：`ft_aug` 相对 baseline mAP50 ≥ 0.5 pt |
| **T2** | 7 组复现筛查 + **强基线 `yolo11n-strong`** 组装 | G2：强基线相对 vanilla ≥ 1.5 pt |
| **T3** | I1 ALA / I2 DLA / I3 DRA / I4 ASM 及其组合 | G3：≥1 个组件相对强基线 ≥ 0.5 pt **且在定向子集上 ≥ 1 pt** |
| **T4** | 组合定稿 × 3 seed 统计 + YOLO11s 上限 + 天花板曲线 | — |

**定向子集**（固定口径，全程不变）：① 粘连子集（β>0.05）；② 高密度子集（>20 实例）；
③ 稀有类子集（<500 实例的 4 类）；④ 极小类子集（相对尺度 <0.05%）。

---

## 8. 引用与致谢

- 数据集：Wang Q-J, Zhang S-Y, Dong S-F, et al. *Pest24: A large-scale very small object data set of agricultural pests for multi-target detection.* Computers and Electronics in Agriculture, 175:105585 (2020). DOI: 10.1016/j.compag.2020.105585
- 复现模块出处见各模块文件头部 docstring（EP-YOLO / AEP-PEST / CCDW-YOLO / YOLO11-4K）。
- 创新点的 future work 出处见 `docs/调研报告_农业害虫检测前沿.md` 的溯源表。

## 9. License

Apache License 2.0（见 `LICENSE`）。**数据集本身不随本仓库分发**，其使用受原始许可约束（非商业免费使用）。
