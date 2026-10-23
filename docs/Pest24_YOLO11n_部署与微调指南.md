# Pest24 + YOLO11n 本地部署与微调指南

> 目标读者：视觉处理经验不多、希望快速上手目标检测训练的同学。
> 目标：在本地以 **YOLO11n 为 baseline**，在 **Pest24** 数据集上训练并评测，再通过合理的微调手段使 **Precision / mAP 等指标可复现地提升**。
> 本文档所有命令均在 **Windows + PowerShell** 下给出（Linux/macOS 仅路径写法不同）。

---

## 目录

1. [数据集介绍：Pest24](#1-数据集介绍pest24)
2. [模型介绍：YOLO11n](#2-模型介绍yolo11n)
3. [为什么用这对组合（合理性分析）](#3-为什么用这对组合合理性分析)
4. [本地部署步骤（可复制执行）](#4-本地部署步骤可复制执行)
5. [Baseline 训练与评测](#5-baseline-训练与评测)
6. [微调方向（按预期收益排序）](#6-微调方向按预期收益排序)
7. [消融实验矩阵与验收标准](#7-消融实验矩阵与验收标准)
8. [常见问题（FAQ）](#8-常见问题faq)
9. [参考资料](#9-参考资料)

---

## 1. 数据集介绍：Pest24

- **来源**：中科院合肥物质科学研究院团队发表于 *Computers and Electronics in Agriculture* (2020) 的公开数据集，论文：[Pest24: A large-scale very small object data set of agricultural pests for multi-target detection](https://doi.org/10.1016/j.compag.2020.105585)；镜像下载：[Kaggle - boatshuai/pest24](https://www.kaggle.com/datasets/boatshuai/pest24)。
- **规模**：25,378 张田间害虫图像，由自动虫情测报灯/成像设备采集，共 **24 类**中国主要农作物害虫。⚠️ **分辨率注意**：论文原图为 2095×1944，**Kaggle 镜像已重采样为 800×600**（XML 标注与 800×600 图像一致），影响评估见下方专段。
- **划分与公开情况**：论文按 **train:val:test = 5:2:3** 划分（约 12689 / 5076 / 5076 张），但**官方只公开了 train 和 val**（test 标签未发布）。社区通行做法：**train 训练、官方 val 兼作统一评测集**，不要自行从 train 切 test（否则与社区口径不可比）。Kaggle 镜像以实际内容为准，§4.4 脚本两种情形都兼容。
- **标注格式**：Pascal VOC XML → 需转换为 YOLO txt（脚本见 §4.4）。
- **四大特点（也是难点）**：
  1. **目标极小**：原图 2095×1944 下害虫常只占十几到几十像素；Kaggle 800×600 版保留了相对尺寸，但纹理细节已丢失（细粒度分类更难）；
  2. **密集分布**：平均每图 2.3 类 / 7.6 实例，单图最多 **214** 个实例，8.89% 的图实例数 >20；
  3. **类间相似度高**：害虫两两相似度 0.45~0.7、均值 0.533（远高于 VOC/COCO），易误分（拉低 Precision）；
  4. **目标粘连**：**4657 张图**存在不可忽略粘连（β>0.05），NMS 容易误抑制。
- **拍摄噪声（论文统计）**：约 1100 张图含**外观相似的非目标害虫**（Precision 的主要杀手）、约 5000 张目标因拍摄角度过小、2200 张有反光斑、600 张有阴影/遮挡。
- **Kaggle 800×600 版的影响评估**：
  - **负面**：像素量约为原图 1/8.4，微小目标的触角/斑纹等细节**不可逆丢失** → 类间相似（均值 0.533）的细粒度判别更难，mAP50-95 受影响大于 mAP50；论文指标（2095×1944 口径）**不可直接对表**；
  - **正面**：`imgsz=640` 输入仅 0.8× 缩放、接近无损（原图口径下是 0.31×），baseline 更强；训练/推理成本大降、迭代更快；「very small object」在输入尺度上被缓解；
  - **结论**：对精度上限负面、对训练效率正面；对「baseline vs 微调」的内部对比有效性**无影响**（全部实验在同一数据上）。参数结论：imgsz 960/1280 移出计划，**800（32×25，原生档）为条件追加项**，主攻方向转向增强与后处理。
- **原论文 baseline 与结论（数值取自论文原文，2095×1944 口径）**：默认超参下 **YOLOv3 最优，mAP 61.42%**；调参后 **63.54%**，分别比 SSD、Faster R-CNN、Cascade R-CNN 高 13.03 / 11.82 / 3.57 个百分点（即约 **50.5 / 51.7 / 60.0**）；3-fold 交叉验证下 YOLOv3 57.28%、SSD(VGG-16) 44.12%。**粘连子集 mAP 仅 46.76%，比非粘连子集低 4.96 个点** → 因素排序确认为：**目标粘连 > 样本量（实例数）>> 颜色差异（几乎无影响）**——优化方向应锁定「密集样本增强 / NMS 与粘连处理 / 类间判别」，而不是颜色增强。
- **原始分辨率（2095×1944）从哪来**：论文原文声明数据集「已公开供非商业免费使用，地址 **http://aisys.iim.ac.cn/zhibao.html**」（中科院合肥研究院智能机械研究所 AI 系统页）。⚠️ **该地址当前返回 HTTP 502**（服务器端后端故障；页面为 2020 年的静态页，疑似长期无人维护）。原图获取建议走这三条路：
  1. **邮件向通讯作者申请（最可行）**：论文原文给出通讯作者邮箱 **hqwang@ustc.edu**（H.-Q. Wang, Hefei Institutes of Physical Science, CAS），数据集为非商业免费使用，可附用途说明索取原图；
  2. **Wayback 存档**：浏览器打开 https://web.archive.org/web/*/aisys.iim.ac.cn/zhibao.html 查快照——注意即使有页面快照，**原图下载链接多半仍指向那台已故障的服务器**，可能只能拿到页面说明而非数据；
  3. **关注迁移**：留意作者单位站点（iim.ac.cn）或中科院机构知识库是否重新托管。
  **主线不阻塞**：800×600 版足以完成 baseline 与全部改进对比；拿到原图后把 §6 的 imgsz 策略切回原图口径（640→960→1280）重跑即可。

  申请邮件模板（可直接改用）：

  > 主题：申请获取 Pest24 数据集（2095×1944 原始分辨率）用于科研/学习
  >
  > 王老师您好：
  > 我是在读研究生/工程师 XXX（单位：XXX），正在用 YOLO11n 在 Pest24 上做害虫检测的 baseline 与改进实验。论文中说明数据集发布于 http://aisys.iim.ac.cn/zhibao.html 供非商业免费使用，但该地址目前返回 502 无法访问。想申请获取 2095×1944 原始分辨率版本（25,378 张 train/val 划分 + VOC 标注），仅用于非商业研究，成果会引用论文（DOI: 10.1016/j.compag.2020.105585）。如可提供网盘链接或其他下载方式，万分感谢。祝好，XXX（联系方式）。

## 2. 模型介绍：YOLO11n

YOLO11 是 Ultralytics 于 2024 年发布的 YOLO 系列最新主线版本（[GitHub](https://github.com/ultralytics/ultralytics) / [官方文档](https://docs.ultralytics.com/)）。`n` 是最小的 nano 规格：

| 规格 | 输入尺寸 | COCO mAP50-95 | 参数量 | 计算量 (FLOPs) | 定位 |
|---|---|---|---|---|---|
| **YOLO11n** | 640 | ≈ 39.5 | ≈ 2.6M | ≈ 6.5G | 轻量、迭代快、适合 baseline |
| YOLO11s | 640 | ≈ 47.0 | ≈ 9.4M | ≈ 21.5G | 精度更高、训练更慢 |

> 数值来自 Ultralytics 官方模型表（COCO val，不同环境可能略有浮动）。

YOLO11n 的特点：

- **Anchor-free 检测头**：不需要手工设计 anchor，对新手友好；
- **C3k2 主干 + C2PSA 注意力模块**：同参数量下比 YOLOv8n 精度更高；
- **Ultralytics 统一 CLI/Python API**：训练、验证、预测、导出（ONNX/TensorRT）一条命令完成；
- **迁移学习开箱即用**：官方提供 COCO 预训练权重 `yolo11n.pt`，自定义数据集上微调收敛快。

## 3. 为什么用这对组合（合理性分析）

1. **难度匹配**：Pest24 是「极小目标 + 密集 + 高相似度」的专门 benchmark，比通用 COCO 更有针对性；YOLO11n 足够轻量，单轮训练快（消费级 GPU 几小时内），适合「改一个变量 → 看一次结果」的新手迭代节奏。
2. **瓶颈明确**：Pest24 的精度瓶颈不在模型容量而在**输入分辨率与小目标特征**，因此 YOLO11n（而非更大模型）+ 正确的微调策略，性价比最高。
3. **生态成熟**：Ultralytics 文档与社区项目丰富（见 §9），遇到问题容易找到答案；从 VOC→YOLO 转换到训练导出都有现成参考。
4. **可迁移**：在本流程中掌握的方法（小目标检测调参）可直接迁移到其他农业/工业检测任务。

## 4. 本地部署步骤（可复制执行）

### 4.0 前置条件

- Windows 10/11，建议有 NVIDIA GPU（≥6GB 显存体验较好；无 GPU 也能跑，见 §8-Q1）；
- 已安装 [Python 3.10+](https://www.python.org/) 与 [Git](https://git-scm.com/)；
- 能访问 Kaggle（下载数据集）与 PyPI/GitHub（装包、下预训练权重）。

### 4.0.1 GTX 1660 SUPER（6GB）适配确认与参数速查

**结论：可以本地部署。** 1660 SUPER 为 Turing 架构（sm_75），PyTorch 官方 CUDA 轮子直接支持，**无需安装 CUDA Toolkit**（轮子自带运行时），只需显卡驱动较新：

```powershell
nvidia-smi   # 看右上角 Driver / CUDA Version；CUDA 12.1 轮子要求驱动 ≥ 527.41
```

驱动过老：先升级 NVIDIA 驱动，或改装 cu118 轮子（`--index-url https://download.pytorch.org/whl/cu118`）。磁盘建议预留 10GB+（数据集约几 GB + runs 输出）。

**各阶段参数速查（6GB 显存）：**

| 阶段 | imgsz | batch | 显存预估 | 预计耗时（100 epochs，估算值） |
|---|---|---|---|---|
| baseline | 640 | 16 | ~4-5GB | 约 8~12 小时（一晚） |
| 主微调 | 960 | 6 | ~5-6GB | 约 20~28 小时 |
| 极限档 | 1280 | 2 | ~6GB | 约 2 天，不推荐首选 |

> Kaggle 镜像为 800×600：档位相应理解为 640（baseline）/ 800（原生上限档）；**1280 无意义，不用**。

长训练实用技巧：

- `patience=30`：涨点停滞自动提前停止，不浪费机时；
- 中断/断电后可续训：`yolo detect train model=runs/pest24/xxx/weights/last.pt resume=True`；
- `save_period=10` 每 10 轮存一次权重，防止意外丢进度；
- 训练时关掉占用显存的程序（游戏/视频剪辑），否则 6GB 很容易 OOM。

### 4.0.2 训练平台选择：本地太慢时的替代方案

> 📌 **决定全部放 AutoDL 执行？** 直接按 [autodl_agent_action_plan.md](autodl_agent_action_plan.md) 的 Agent 行动计划走（含误差控制约定、逐阶段成功判据与失败分支）。

**结论：本地 1660S 只用来「跑通流程 + 最终推理」，真实训练放云端。** 按省钱程度排序：

| 方案 | 费用 | GPU | 640/100ep 估算 | 960/100ep 估算 | 关键限制 |
|---|---|---|---|---|---|
| 本地 1660 SUPER | ¥0 | 6GB | 8~12h | 20~28h | 机器被长期占用 |
| **Kaggle Notebooks（免费首选）** | ¥0 | T4 16GB | ~4-5h | ~9-11h | 单会话 12h、GPU 约 30h/周；需 Google 账号 + 手机验证 |
| **AutoDL（付费推荐）** | 4090 约 ¥1.5~2/h | 24GB | ~1-1.5h | ~2.5-3.5h | 微信/支付宝充值，按量计费 |
| 百度 AI Studio | 免费算力点（签到积累） | V100 32G | ~3h | ~7h | 实名注册，点数有限 |
| 启智 OpenI | 免费 | V100/3080 | 类似 | 类似 | 任务制排队，流程较繁 |

> 时间为 YOLO11n 量级的估算、价格为撰写时的约数，均以平台页面为准。Colab 需境外支付且国内访问不稳，已排除。

**推荐组合（最省时间也最省钱）：**

1. **本地只做轻活（≤30 分钟）**：装环境、数据转换，再用 10% 子集 + `epochs=10` 把「转换→训练→评测」全流程跑通，确认无 bug 后本地不再训练；
2. **Kaggle 免费跑正式实验**：数据集本来就在 Kaggle 上，T4 比 1660S 快约 3 倍、显存 16GB 可开大 batch；免费额度足够跑完 baseline + ft_aug 全部实验；
3. **赶时间 → AutoDL**：4090 一晚（3~4h）跑完全部实验矩阵，总花费约 ¥15~30，微信/支付宝即可；
4. **本地 1660S 收尾**：只做 `best.pt` 的评测、推理部署、SAHI 验证。

**Kaggle Notebook 操作步骤（免费）：**

1. 登录 kaggle.com（Google 账号），账号设置完成手机验证才能开 GPU 与 Internet；
2. New Notebook → 右侧 Settings：`Accelerator = GPU T4 x2`、`Internet = On`（新手用单卡即可，双卡 DDP 不划算）；
3. `Add Input` 搜索 `pest24`（boatshuai/pest24）挂载数据集，路径 `/kaggle/input/pest24/`（只读）；
4. Cell 1 安装并转换（先把 §4.4 脚本粘贴进 notebook 新建文件或上传，注意输入/输出路径）：

   ```python
   !pip install -q ultralytics
   !python voc2yolo.py --src /kaggle/input/pest24 --dst /kaggle/working/pest24
   ```

5. Cell 2 训练（800×600 数据下 imgsz=640 即可，T4 16GB 可用 `batch=32`）：

   ```python
   !yolo detect train model=yolo11n.pt data=/kaggle/working/pest24/pest24.yaml \
     epochs=100 imgsz=640 batch=32 seed=0 patience=30 \
     project=/kaggle/working/runs name=baseline
   ```

6. **长训练务必用 Commit 模式**：`File → Save & Run All (Commit)` 无人值守跑完自动保存；交互模式关浏览器/闲置即断线；
7. 跑完在右侧 Output 面板下载 `best.pt` 与结果图（或 Save Version 永久保存）。

**AutoDL 操作步骤（约 ¥15~30 跑完全部实验）：**

1. autodl.com 手机号注册 → 充值（先充 ¥20 试水）；
2. 租一台 RTX 4090 **按量计费**实例 → 进入 JupyterLab；
3. 传数据：本地把 Kaggle 下载的 pest24 压缩包上传到实例网盘，随实例挂载；
4. 下载 GitHub/HuggingFace 资源慢，先执行学术加速：`source /etc/network_turbo`；
5. `pip install ultralytics` 后按 §5/§6 命令原样跑（4090 24GB：640 用 `batch=48`，800 用 `batch=32`）；
6. **用完立刻「关机」停止计费**（数据盘保留）；把 `best.pt`、`runs/` 打包传回本地。

**无论哪个平台都适用的省时技巧：**

- 先小后大：`epochs=10` + 10% 子集验证配置有效，再全量正式跑；
- `patience=30` 早停，省机时；
- 训练与评测放同一平台，避免数据来回拷贝；
- 多个实验写成 shell 脚本顺序执行（`yolo ... name=exp1 && yolo ... name=exp2`），人不用守着。

### 4.1 创建虚拟环境并安装依赖

```powershell
# 在工作目录（本指南所在仓库根目录，例如 F:\deepseek-harness\YOLO11）执行
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) 安装 PyTorch（按你的 CUDA 版本选择，下面以 CUDA 12.1 为例；
#    无 NVIDIA 显卡则改用: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) 安装 Ultralytics（自带 yolo CLI）
pip install ultralytics

# 3) 验证 GPU 是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

> PyTorch 安装命令以官方页面为准：https://pytorch.org/get-started/locally/

### 4.2 下载 Pest24 数据集

方式 A（推荐，命令行）：

```powershell
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('boatshuai/pest24'))"
# 打印出的路径就是数据集解压后的目录
```

方式 B（网页）：打开 https://www.kaggle.com/datasets/boatshuai/pest24 → Download，手动解压。

方式 C（原图，可选）：从论文声明地址 http://aisys.iim.ac.cn/zhibao.html 获取 2095×1944 原图（可达性需自行确认，见 §1；拿到后按 §6 方向 1 的原图口径调 imgsz）。

Kaggle 版实际结构（Version 1，2.38 GB，图片 800×600）：

```
Pest24\
  VOCdevkit\
    voc2007\
      Annotations\   *.xml（VOC 标注，转换的唯一依据）
      ImageSets\     train.txt  val.txt  test.txt
      images\        *.jpg（800×600）
      labels\        *.txt   ← 上传者预生成的 YOLO 标签（仅用作交叉校验）
      yolo_path\     *.txt   ← 上传者预生成的图片列表（本流程用不到）
```

> **`voc2007` 只是命名惯例，不是 PASCAL VOC 数据集**：这是 VOC 工具链脚本的默认输出目录名（工作区 `dataProcess.py` 的输出路径正是 `D:/projects/Pest24/VOCdevkit/voc2007/...`），里面的图仍是 Pest24 害虫图。论文提到 VOC/COCO 只是在统计量（每图类别数、实例数、相似度）上做对比，与目录名无关。
>
> **转换时 `--src` 指向 `...\Pest24\VOCdevkit\voc2007`**，并**以 `Annotations/` 重新转换为准**（保证 label id 与 §4.4 的 `CLASSES` 顺序一致）；镜像自带的 `labels/` 只作交叉校验：文件数应与我们输出的标签数一致。
>
> §4.4 脚本兼容该布局、`xml/ + ImageSets/` 布局、`train/val/test` 混放布局与 VOC 标准 `ImageSets/Main` 布局。

### 4.3 规划 YOLO 目录结构

转换脚本会生成：

```
datasets\pest24\
  images\train\  images\val\  images\test\
  labels\train\  labels\val\  labels\test\
  pest24.yaml
```

### 4.4 VOC XML → YOLO txt 转换脚本

将下面脚本保存为 `scripts\voc2yolo.py`。它做了几件新手容易忽略的事：

- 自动发现 XML（兼容 Kaggle 实际布局 `xml/ + ImageSets/`、VOC 标准布局与「图片+XML 混放」布局）；
- 跳过 `difficult=1` 标注（社区通行做法）；
- 坐标越界裁剪到 `[0,1]`，避免训练时报 label 警告；
- 未知名类别自动跳过并统计（防止类别名拼写不一致）；
- 输出**每类实例数量统计**，让你直观看到类别不均衡情况。

```python
# scripts/voc2yolo.py
# 用法: python scripts/voc2yolo.py --src <Pest24/VOCdevkit/voc2007> --dst datasets/pest24
import argparse, shutil, xml.etree.ElementTree as ET
from pathlib import Path

# Pest24 的 24 类——与 Kaggle 版 XML 中的真实类名一致（依据工作区 dataProcess.py；
# 顺序自定但固定即可，YOLO 训练用 0-23 编号，与论文原始编号无对应关系）
CLASSES = [
    "Bollworm", "Meadow borer", "Gryllotalpa orientalis", "Little Gecko",
    "Agriotes fuscicollis Miwa", "Nematode trench", "Athetis lepigone",
    "Scotogramma trifolii Rottemberg", "Armyworm", "Spodoptera cabbage",
    "Anomala corpulenta", "Spodoptera exigua", "Plutella xylostella",
    "holotrichia parallela", "Rice planthopper", "Yellow tiger",
    "Land tiger", "eight-character tiger", "holotrichia oblita", "Stem borer",
    "Striped rice bore", "Rice Leaf Roller", "Spodoptera litura", "Melahotus",
]
NAME2ID = {n.strip().lower(): i for i, n in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def convert_split(src: Path, dst: Path, split: str, stats: dict, unknown: set):
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    xmls = []
    # 布局 1: {split}/ 目录混放
    if (src / split).exists():
        xmls = sorted((src / split).rglob("*.xml"))
    # 布局 2: xml|Annotations + ImageSets(/Main)/{split}.txt（Kaggle 实际布局）
    if not xmls:
        ann = src / "xml" if (src / "xml").exists() else src / "Annotations"
        sets = next((p for p in (src / "ImageSets" / f"{split}.txt",
                                 src / "ImageSets" / "Main" / f"{split}.txt") if p.exists()), None)
        if ann.exists() and sets is not None:
            ids = sets.read_text(encoding="utf-8").split()
            xmls = [ann / f"{i}.xml" for i in ids if (ann / f"{i}.xml").exists()]
    assert xmls, f"未找到 {split} 划分的 XML，请检查 --src 目录结构"
    n_img = 0
    for xp in xmls:
        root = ET.parse(xp).getroot()
        w, h = int(root.findtext("size/width", "0")), int(root.findtext("size/height", "0"))
        assert w > 0 and h > 0, f"{xp} 缺少有效 size"
        lines = []
        for obj in root.findall("object"):
            diff = obj.findtext("difficult") or obj.findtext("Difficult") or "0"
            if diff.strip() and int(float(diff)) == 1:
                continue  # 跳过 difficult=1 标注（社区通行做法）
            name = obj.findtext("name", "").strip().lower()
            if name not in NAME2ID:
                unknown.add(name); continue
            b = obj.find("bndbox")
            xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
            xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
            cx = min(max((xmin + xmax) / 2 / w, 0.0), 1.0)
            cy = min(max((ymin + ymax) / 2 / h, 0.0), 1.0)
            bw = min(max((xmax - xmin) / w, 0.0), 1.0)
            bh = min(max((ymax - ymin) / h, 0.0), 1.0)
            if bw <= 0 or bh <= 0: continue
            lines.append(f"{NAME2ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats[split][NAME2ID[name]] = stats[split].get(NAME2ID[name], 0) + 1
        # 找同名图片：images/ 目录 → XML 同目录 → 全 src 搜索
        img = next((p for e in IMG_EXTS if (p := src / "images" / f"{xp.stem}{e}").exists()), None)
        if img is None:
            img = next((p for e in IMG_EXTS if (p := xp.with_suffix(e)).exists()), None)
        if img is None:
            img = next(iter(src.rglob(xp.stem + ".*")), None)
            if img is None or img.suffix.lower() not in IMG_EXTS: continue
        shutil.copy2(img, dst / "images" / split / img.name)
        (dst / "labels" / split / f"{xp.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
    print(f"[{split}] 图片 {n_img} 张，实例 {sum(stats[split].values())} 个")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    # 官方仅公开 train/val（test 标签未发布）；存在哪个 split 就转哪个
    splits = [s for s in ("train", "val", "test")
              if (src / s).exists()
              or (src / "ImageSets" / f"{s}.txt").exists()
              or (src / "ImageSets" / "Main" / f"{s}.txt").exists()]
    assert "train" in splits and "val" in splits, "至少需要 train 和 val 两个划分"
    if "test" not in splits:
        print("ℹ️ 未发现 test 划分（官方未公开 test 标签）：将以官方 val 作为统一评测集")
    stats, unknown = {s: {} for s in splits}, set()
    for s in splits: convert_split(src, dst, s, stats, unknown)
    print("\n各类实例数（train）:")
    for i, n in enumerate(CLASSES):
        print(f"  {i:2d} {n:30s} {stats['train'].get(i, 0)}")
    if unknown: print("\n⚠️ 未识别类别名（请核对 CLASSES 列表）:", sorted(unknown))
    yl = [f"path: {dst.resolve().as_posix()}", "train: images/train", "val: images/val"]
    if "test" in splits: yl.append("test: images/test")
    yl.append("names:")
    yl += [f"  {i}: {n!r}" for i, n in enumerate(CLASSES)]
    (dst / "pest24.yaml").write_text("\n".join(yl) + "\n", encoding="utf-8")
    print(f"\n已写出 {dst/'pest24.yaml'}")

if __name__ == "__main__":
    main()
```

运行：

```powershell
# Kaggle 版：--src 指向 VOCdevkit\voc2007 这一层
python scripts/voc2yolo.py --src datasets\pest24\VOCdevkit\voc2007 --dst datasets\pest24_yolo
```

**适配说明**：
- 评测口径：Kaggle 镜像的 ImageSets 含 test.txt，若 test 的 XML 有标注则统一用 `split=test`；若 test 无标注则退回 `split=val` 并全程统一（官方本只公开 train/val，**不要自行从 train 切 test**）；
- 数量 sanity check：train ≈ 12689、val/test ≈ 5076（5:2:3 口径，以 ImageSets 实际为准）；若差很多说明 `--src` 目录层级选错；
- 脚本末尾会打印「未识别类别名」。若出现，把打印出的真实名称**替换脚本里的 `CLASSES` 列表**（保持 24 个、顺序自定但需与 name 一一对应）后重跑；
- 转换完成后**务必**抽查 1~2 张图可视化验证（可用 `yolo detect predict` 或直接画框检查）；
- **交叉校验（Kaggle 版必做）**：把脚本输出到 `labels/{split}/` 的 txt 数量，与镜像自带 `...\VOCdevkit\voc2007\labels\` 的数量对比（应一致）；若不一致，说明上传者的类名顺序与我们的 `CLASSES` 不同，**一律以我们重新转换的结果为准**。

### 4.5 Smoke test：验证环境可用

```powershell
# 用 COCO 预训练权重在自带示例图上推理，自动下载权重和示例图
yolo detect predict model=yolo11n.pt source=https://ultralytics.com/images/bus.jpg
# 结果保存在 runs\detect\predict\
```

能出结果图即说明环境 OK。

## 5. Baseline 训练与评测

### 5.1 Baseline 参数设定（社区标准起点）

Baseline 采用 **Ultralytics 官方默认超参**——即开源社区训练自定义数据集的通行起点，只额外固定复现必需项（seed / patience / epochs / imgsz）。关键默认值供你理解（不用改）：

| 类别 | 参数 | 默认值 | 说明 |
|---|---|---|---|
| 优化 | optimizer / lr0 / lrf | auto(SGD) / 0.01 / 0.01 | 学习率线性衰减 |
| 动量 | momentum / weight_decay | 0.937 / 0.0005 | |
| 预热 | warmup_epochs | 3.0 | |
| 损失权重 | box / cls / dfl | 7.5 / 0.5 / 1.5 | |
| 增强 | mosaic / close_mosaic | 1.0 / 10 | 末 10 轮自动关 mosaic |
| 增强 | scale / fliplr / hsv_* | 0.5 / 0.5 / 小幅 | 颜色增强保留但收益小（论文：颜色差异几乎不影响 AP） |
| 精度 | amp | True | 混合精度 |
| 本实验固定 | seed / patience / epochs / imgsz | 0 / 30 / 100 / 640 | 复现 + 早停 |

训练命令（AutoDL 24G 用 `batch=48`；本地 1660S 按 §4.0.1 换算）：

```bash
yolo detect train model=yolo11n.pt data=datasets/pest24/pest24.yaml \
  epochs=100 imgsz=640 batch=48 seed=0 patience=30 \
  project=runs/pest24 name=baseline
```

- 训练结束后指标与曲线在 `runs/pest24/baseline/`（`results.csv`、`PR_curve.png`、`confusion_matrix.png` 等）。

### 5.2 在评测集上评测（口径：EVAL_SPLIT，Phase 3 确定）

Kaggle 镜像的 ImageSets 含 test.txt：若 test 的 XML 有标注则用 `split=test`；若 test 无标注（官方本只公开 train/val）则用 `split=val`。**确定后全程统一，不要中途更换**：

```bash
yolo detect val model=runs/pest24/baseline/weights/best.pt data=datasets/pest24/pest24.yaml split=test   # 或 val，以 EVAL_SPLIT 为准
```

> 实验间对比一律用默认 conf 的结果；conf/iou 扫描（§6）只用于定部署点。

### 5.3 指标含义速查

| 指标 | 含义 | 直觉 |
|---|---|---|
| **Precision (P)** | 检出的框里有多少是真的 | 误检少 → P 高 |
| **Recall (R)** | 真目标里有多少被检出 | 漏检少 → R 高 |
| **mAP50** | IoU=0.5 时各类 AP 的平均 | 常用主指标，**本实验以它为主要对比依据** |
| **mAP50-95** | IoU 0.5~0.95 平均 | 更严格，衡量框的贴合度 |
| **F1** | P 与 R 的调和平均 | 综合衡量 |

> 注意 P/R 存在权衡：提高置信度阈值（`conf`）→ P 升 R 降。比较两次实验时**必须在相同 `conf` 下比较**，或直接看 mAP（与阈值无关）。

### 5.4 记录 baseline

把评测集（官方 val）的 P / R / mAP50 / mAP50-95 填入 §7 的实验表，作为后续一切微调的对照基准。**没有记录就没有提升。**

## 6. 微调方向（按预期收益排序）

> 总原则：**一次只改一个变量**；每轮在同一评测集（EVAL_SPLIT）、同一 conf 口径下与 baseline 对比。
> 排序依据：论文因素分析（粘连 > 样本量 >> 颜色）+ Kaggle 镜像 800×600 的事实（分辨率杠杆大部分失效）。
> 以下命令按 AutoDL 24G 卡给 batch；本地 1660S 请按 §4.0.1 换算。

### 方向 1：输入分辨率（Kaggle 800×600 版下收益有限，降级）

原图 2095×1944 时这是第一杠杆；但 Kaggle 镜像已重采样为 **800×600**，`imgsz=640` 仅 0.8× 缩放、接近无损——分辨率收益大部分已「内置」在数据里：

- baseline 维持 `imgsz=640`；`imgsz=800`（32×25，原生档）为条件追加项；
- **imgsz 960/1280 超过原生分辨率，纯上采样无信息增益，移出计划**；
- 尺度鲁棒性改由 `multi_scale=True`（±50% 尺度抖动）低成本覆盖，已并入方向 2。

### 方向 2：密集与粘连处理（Kaggle 800×600 版下的第一主攻方向）

4657 张图有不可忽略粘连（β>0.05），NMS 误抑制是 Recall 的主要损失源：

- **NMS IoU 扫描（零训练成本）**：评测时 `iou ∈ {0.45, 0.5, 0.6, 0.7}`，粘连多时适当**调大** iou 保留更多近邻框；
- `copy_paste=0.1`：把虫体贴到新位置，定向增加小目标/密集样本，同时缓解类别不均衡（样本量是第二影响因素）；
- `multi_scale=True`：多尺度训练（±50% 随机缩放输入），补回分辨率方向的尺度鲁棒性；
- `close_mosaic=10`：末期关闭 mosaic，在真实分布上收敛。

```bash
yolo detect train model=yolo11n.pt data=datasets/pest24/pest24.yaml \
  epochs=100 imgsz=640 batch=48 seed=0 patience=30 \
  close_mosaic=10 copy_paste=0.1 multi_scale=True scale=0.9 \
  project=runs/pest24 name=ft_aug
```

### 方向 3：定向拉 Precision（类间相似 + 非目标背景）

类间相似度均值 0.533、约 1100 张图含外观相似的非目标害虫 → 误分是 Precision 的主要杀手：

- `cls=1.0`（默认 0.5）：加大分类损失权重，逼模型分清相似类；
- **conf 扫描（零训练成本）**：评测/部署时 `conf ∈ {0.10…0.50}`，按 PR 曲线选 F1 最大（或满足 P 目标的最小 conf）；
- 看逐类 AP（val 输出）：找出互相误分最严重的类对，写进 results.md 备注。

### 方向 4：训练策略（增益预期递减，预算允许再做）

- `epochs=150 cos_lr=True patience=30`：余弦退火 + 早停；
- `warmup_epochs=5`：稳定训练初期；
- 仅在前面方向仍在涨点时追加，避免白花机时。

### 方向 5：进阶（可选）

1. **SAHI 切片推理（可选）**：800×600 整图推理已可覆盖；仅当最终部署需处理 2095×1944 原图时启用（slice 512 / overlap 0.2）；
2. **加 P2 小目标检测头**：改 `yolo11n.yaml` 增加 stride=4 检测层，收益明显但要改模型结构；
3. **换 YOLO11s**：容量 ×3.6，作为上限参考；
4. **冻结主干 `freeze=10`**：仅用于快速验证流程。

### 建议执行顺序（AutoDL）

```
baseline(640) → ft_aug(640+增强) → [条件]ft800 → conf/iou 扫描（必做）→（可选）SAHI
```

> 条件分支：ft_aug 相对 baseline 提升 ≥ 0.5 pt 且预算允许 → 追加 ft800（原生档，预期收益小）；conf/iou 扫描零训练成本，必做。

每一步都与 baseline 比、与上一步比。

## 7. 消融实验矩阵与验收标准

### 实验记录表（模板，照着填；评测集 = 官方 val）

| 实验 | 改动（相对 baseline） | imgsz | epochs | eval P | eval R | eval mAP50 | eval mAP50-95 | 耗时 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | Ultralytics 默认超参 | 640 | 100 | | | | | | |
| ft_aug | +copy_paste/multi_scale/scale | 640 | 100 | | | | | | |
| ft800（条件） | imgsz→800（原生档） | 800 | 100 | | | | | | |
| ft_long（可选） | +150ep/cos_lr | 640 | 150 | | | | | | |
| 部署点 | +conf/iou 扫描选定 | — | — | | | | | | |

> Sanity 参照（弱）：论文在 **2095×1944 原图口径**下，YOLOv3 最佳 mAP **63.54%**（默认 61.42%），SSD ≈50.5%、Faster R-CNN ≈51.7%、Cascade R-CNN ≈60.0%。Kaggle 800×600 版**不可直接对表**——本实验一切结论以「同一数据上的 baseline vs 微调」内部对比为准。

### 验收标准

1. baseline 的评测集指标已记录且可复现（同 seed 重跑误差 < 0.5 pt）；
2. 至少一轮微调后，**eval mAP50 与 Precision 相对 baseline 有可复现提升**（建议 ≥ 1~2 pt 视为有效，且非阈值 trick 造成）；
3. 所有实验命令、超参、指标完整填入上表，结论可追溯。

## 8. 常见问题（FAQ）

**Q1：没有 NVIDIA GPU / 本地训练太慢能跑吗？**
能。本地可用 `imgsz=640 batch=4 workers=4 epochs=50` 先跑通流程；正式训练建议上云端免费/廉价 GPU（Kaggle Notebooks、AutoDL、AI Studio 等，见 §4.0.2），本地只做推理。

**Q2：CUDA out of memory？**
按顺序试：`batch` 减半 → `imgsz` 降一档 → 确认没开其他占显存的程序 → 加 `amp=True`（默认已开）。

**Q3：Windows 下训练卡住/报 DataLoader worker 错误？**
把 `workers=0`（CLI 或 Python 里 `if __name__ == "__main__":` 保护）。Windows 上多进程 DataLoader 偶尔不稳，`workers=0` 最稳。

**Q4：类别不均衡严重（某些类样本极少）怎么办？**
先看 §4.4 脚本打印的类别分布。少类可通过 `copy_paste`、针对性采集补充；评测时同时看**逐类 AP**（val 输出里有），不要只看均值。

**Q5：中文/空格路径问题？**
数据集与项目路径避免中文和空格；Python 读写统一 UTF-8（本文脚本已处理）。

**Q6：`yolo` 命令找不到？**
确认已激活虚拟环境（`.\.venv\Scripts\Activate.ps1`）且 `pip install ultralytics` 成功；也可用 `python -m ultralytics` 等价调用方式（或写 Python 脚本调用 `from ultralytics import YOLO`）。

**Q7：下载 yolo11n.pt 慢/失败？**
手动从 [Ultralytics  releases](https://github.com/ultralytics/assets/releases) 下载后放到当前目录，`model=./yolo11n.pt` 指向本地文件。

**Q8：Kaggle 镜像里只有 train 和 val，没有 test？**
Kaggle 镜像的 ImageSets 实际含 test.txt，test 是否有标注以转换脚本输出为准：有 → 用 `split=test`；无 → 退回 `split=val` 并全程统一，不要自己从 train 切 test。

**Q9：Kaggle 版分辨率 800×600（论文原图 2095×1944）有影响吗？**
两面性：负面是像素量约剩 1/8.4，细粒度纹理不可逆丢失、与论文指标不可直接对表；正面是 `imgsz=640` 接近无损、训练便宜得多。参数结论：960/1280 移出计划，800 为原生上限档（条件追加），主攻增强与 conf/iou 后处理（详见 §1 专段、§6）。

**Q10：目录里为什么叫 `VOCdevkit\voc2007`？这是 PASCAL VOC 数据集吗？**
不是。`VOCdevkit/voc2007` 是 VOC 工具链脚本的默认输出路径名（社区脚本 `dataProcess.py` 也这么写），里面装的是 Pest24 的害虫图（800×600）+ VOC 格式 XML。论文提到 VOC/COCO 只是做统计量对比。转换时 `--src` 指到 `voc2007` 即可（详见 §4.2）。

**Q11：哪里能拿到 2095×1944 原始分辨率数据？**
官方地址 http://aisys.iim.ac.cn/zhibao.html **当前返回 502**（服务器端故障）。三条路：① **邮件申请**——通讯作者邮箱 **hqwang@ustc.edu**（论文原文，非商业免费使用），模板见 §1；② Wayback 查快照（注意原图链接可能同样失效）；③ 关注作者单位后续迁移。**不必阻塞**：Kaggle 800×600 版可完整跑通 baseline 与改进对比；拿到原图后再按 §6 方向 1 的原图口径（960/1280）重跑。

## 9. 参考资料

- 数据集
  - Kaggle 下载页：https://www.kaggle.com/datasets/boatshuai/pest24
  - 原始论文（Computers and Electronics in Agriculture, 2020）：https://doi.org/10.1016/j.compag.2020.105585
  - 论文中文解读（含划分比例、噪声统计、AP 影响因素分析）：https://www.cnblogs.com/tccjx/p/16706119.html
- 模型与工具
  - Ultralytics 官方文档（训练/超参/导出权威参考）：https://docs.ultralytics.com/
  - Ultralytics GitHub：https://github.com/ultralytics/ultralytics
- 可借鉴的开源项目
  - YOLO11 自定义数据集训练→评估→ONNX 导出完整流程：https://github.com/ElmoData/YOLO11-Object-Detection-with-YOLO-and-Python-Training-Inference#1
  - VOC XML → YOLO 转换脚本参考：https://github.com/Arcaderix/XML-to-YOLO-#1
  - 农业害虫多模型对比项目（预处理流程参考）：https://github.com/davinagreen/AgroPest-12-Insect-Detection-Multi-Model-Comparison#1
  - GitHub pest-detection 主题页（更多同类项目）：https://github.com/topics/pest-detection#1
- 小目标检测进阶
  - SAHI 切片推理：https://github.com/obss/sahi
  - 轻量害虫检测改进思路（YOLO-LCE，YOLOv8 改进案例）：https://www.mdpi.com/2073-4395/15/9/2022#4

---

## 10. 与本仓库 ADEA 方案的关系（2026-10-23 追加）

> 本节说明本指南与仓库 `README.md` / `docs/方法设计_ADEA.md` 的分工，避免两份文档打架。

### 10.1 本指南的定位
本指南是**工程操作手册**：数据集获取、转换、部署、训练、评测口径、平台选型。
其中的**命令与口径约定全部有效**，可直接用于 ADEA 的 T1–T4 实验。

### 10.2 ADEA 方案在指南之上的增量
| 增量 | 位置 | 说明 |
|---|---|---|
| 强基线 `yolo11n-strong` | `docs/方法设计_ADEA.md` §8 | 指南 §6 的方向 5「加 P2 头」升级为核心候选；R1–R5 组合成对照基线 |
| 监督分配层创新（I1 ALA / I2 DLA） | `ultralytics_ext/assign/` | 指南 §6 全部是特征层/增强层改进，未触及标签分配 |
| 动态重采样（I3 DRA） | `ultralytics_ext/sampling/dra.py` | 指南 §6 方向 3 只调 `cls=1.0`（静态），DRA 是按模型当前表现动态调节 |
| 粘连合成（I4 ASM） | `scripts/synth_adhesion.py` | 指南用的是通用 `copy_paste=0.1`（面向遮挡、几乎不重叠），ASM 面向**粘连**并按数据集 β 分布采样 |
| G0 kill gate | `scripts/diagnose_adhesion.py` | 指南未涉及；用于**先验证机制、再决定是否实现 I1/I2** |
| 定向子集验收 | `results.md` §1 | 指南 §7 只看整体 mAP；ADEA 要求每个创新组件在**其目标子集**上有提升 |
| 未来工作溯源 | `docs/调研报告_农业害虫检测前沿.md` §5 | 创新点的出处逐条对应 EP-YOLO §4.2/§4.3 原文 |

### 10.3 指南中**不适用**的部分
- §6 方向 1（输入分辨率 960/1280）：本方案锁定 Kaggle 800×600 版，原生上限为 800，仍不适用；
- §6 方向 5 的「换 YOLO11s」：降级为 **T4-4 容量上限参考**，不是主线；
- §8 FAQ-Q11（原始分辨率获取）：本方案已决定**不追原图**（见问卷决策①）。

### 10.4 一个必须知道的事实更正
Pest24 原论文定义的**粘连系数 β = |A∩B| / |A∪B| 与 IoU 在数值上完全等价**（同一公式）。
差别只在**语义**：IoU 通常被读作"是否重复检出同一体"，β 在本数据集中被读作
"两只虫在捕捉板上物理黏连的程度"。本方案沿用论文的 β 记号与阈值（0.05），
但论文中**不得**声称 β 与 IoU 数值有别。

> 下一步：按 §4 部署环境并跑通 baseline，把指标填入 §7 表格后，再逐项执行 §6 的微调方向。祝训练顺利 🐛
>
> （ADEA 方案的执行顺序请以仓库根目录 `README.md` 的「快速开始」与 `train/*.sh` 为准。）
