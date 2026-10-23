# AutoDL 全流程训练行动计划（Agent 自动执行版）

> 执行者注意：本文档是给 agent 的执行剧本。每个步骤包含【操作】【成功判据】【失败处理】。
> 遵守脚本卫生纪律：**远程命令一律写成 `.sh` 文件 scp 上去执行**，禁止在 ssh 命令行里拼引号/`$()`/管道；
> 远程输出写文件再取回。数据划分、版本、seed 全程固定，保证同环境对比有效性。

---

## 0. 目标与验收

**目标**：在单台 AutoDL 实例上完成 Pest24 + YOLO11n 的数据转换、baseline、微调实验矩阵、评测、conf 扫描与推理验证，回收全部产物后关机停费。

**验收标准**：
1. `results.md` 中 baseline 与 ≥2 个微调实验的 val/test 指标（P / R / mAP50 / mAP50-95）齐全、同口径可比；
2. 最优实验 test mAP50 与 Precision 相对 baseline 提升 ≥ 1 pt；
3. 本地回收：`best.pt` ×3+、`results.md`、`PR_curve.png`、`confusion_matrix.png`、pip freeze 快照；
4. 实例已关机，费用 ≤ ¥20。

## 1. 环境误差控制约定（执行中不可违反）

- **单环境原则**：所有训练与评测在同一实例完成，禁止一半本地一半云端跑再比指标；
- **版本锁定**：ultralytics 固定一个版本（执行时 pin 当前稳定版，如 `ultralytics==8.3.x`），`pip freeze > env_snapshot.txt` 存档；
- **数据一次性**：VOC→YOLO 转换只做一次，三个 split 固定，后续所有实验共用，禁止重转换；
- **seed 固定**：所有训练 `seed=0`；同 seed 重跑允许 ±0.3 pt 噪声，超出才算真实差异；
- **同口径评测**：所有评测不加 `conf`（用默认）做实验间对比；conf 扫描只用于确定最终部署点，不参与实验对比；评测集 split（val/test）一经确定全程不变（官方未公开 test 标签时用 val，见 Phase 3）。

## 2. 需要用户提供的输入（人工闸门，agent 不可代做）

| # | 闸门 | 内容 | 提供形式 |
|---|---|---|---|
| H1 | AutoDL 实例 | 注册、充值 ≥¥30、租好按量计费实例（**4090 优先；4090D/3090 等 24G 级卡可接受**，选型理由见 §7） | 把 SSH 登录指令（形如 `ssh -p xxxxx root@region-x.autodl.com`）发给 agent |
| H2 | 数据集来源 | 三选一：(A) 本地已有 pest24 zip（Kaggle 800×600 版，告知路径）；(B) 提供 Kaggle `kaggle.json`（agent 在实例内下载）；(C) 官方原图——**地址 http://aisys.iim.ac.cn/zhibao.html 当前 502 不可用**，需先邮件申请（通讯作者 hqwang@ustc.edu，模板见主指南 §1）；拿到原图则 imgsz 策略切回 960/1280（主指南 §6 方向 1）。**默认按 A/B 的 800×600 版推进，不因 C 阻塞** | 路径字符串 / kaggle.json 文件 |
| H3 | 关机确认 | 全部产物校验回收后，agent 提醒用户在控制台关机 | 用户操作 |

## 3. 阶段计划

### Phase 1｜连接验证（0.1h，¥0.2）

- **操作**：本地生成 `10_ssh_check.sh`（内容：`hostname && nvidia-smi && df -h /root/autodl-tmp`），scp 到实例执行。
- **成功判据**：输出显示 24G 级 GPU（`RTX 4090` / `4090D` / `3090`，显存 ≥ 23000MiB）、数据盘可用 ≥ 40G。
- **失败处理**：连不上 → 让用户检查实例状态/防火墙；显存 < 24G → 建议换租 24G 级卡（换卡后 batch 适配见 §7）。

### Phase 2｜环境初始化（0.1h，¥0.2）

- **操作**：上传 `20_init.sh` 并执行：
  ```bash
  #!/bin/bash
  set -e
  source /etc/network_turbo 2>/dev/null || true   # AutoDL 学术加速
  pip install -q "ultralytics==8.3.*" kagglehub
  python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda ok')"
  yolo version
  pip freeze > /root/autodl-tmp/env_snapshot.txt
  ```
- **成功判据**：`yolo version` 正常输出；`env_snapshot.txt` 生成。
- **失败处理**：pip 超时 → 去掉 network_turbo 换清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### Phase 3｜数据就位与转换（0.3~1h，¥1~2）

- **操作**：
  1. 上传本地准备好的 `scripts/voc2yolo.py`（主指南 §4.4 的脚本）；
  2. 分支 A（本地 zip）：scp 上传 zip 到 `/root/autodl-tmp/`，`unzip`；
     分支 B（kaggle.json）：上传 kaggle.json 到 `/root/.kaggle/`，实例内 `kaggle datasets download boatshuai/pest24 -p /root/autodl-tmp/` 后解压；
  3. 探明解压后目录结构（`find /root/autodl-tmp/pest24* -maxdepth 3 -type d`）：Kaggle 版为 `Pest24/VOCdevkit/voc2007/{Annotations,ImageSets,images,labels,yolo_path}`（**`voc2007` 只是 VOC 工具链的命名惯例，不是 VOC 数据集**），**`--src` 指向 `.../Pest24/VOCdevkit/voc2007`**；
  4. 执行 `python voc2yolo.py --src <.../VOCdevkit/voc2007> --dst /root/autodl-tmp/datasets/pest24`（以 `Annotations/` 重新转换为准，保证 label id 与我们 `CLASSES` 顺序一致；镜像自带 `labels/` 仅作交叉校验：`ls <src>/labels/*.txt | wc -l` 应等于我们生成的标签总数，不一致则以我们的为准）；
  5. 确定评测口径：yaml 含 `test:` **且 labels/test/ 非空** → `EVAL_SPLIT=test`；否则（test 无标注或缺失）→ `EVAL_SPLIT=val`，写入 results.md 头部并全程统一；
  6. 分辨率核对：抽查几张图片尺寸应为 **800×600**；若实际是 2095×1944 原图，imgsz 策略改按主指南 §6 方向 1 的原图口径执行。
- **成功判据**：train/val 均有图片与标签（数量 sanity：train ≈ 12689、val/test ≈ 5076）；「未识别类别名」为空；`pest24.yaml` 生成；EVAL_SPLIT 已确定；抽查图片尺寸=800×600。
- **失败处理**：出现未识别类别名 → 把打印的真实类名回填 `CLASSES`（保持 24 个）重跑转换；图片数远小于预期 → 检查目录层级后重试；**禁止自行从 train 切 test**（与社区口径不可比）。
- **⚠️ 此步后记录各 split 图片数/实例数与 EVAL_SPLIT 到 results.md 头部，作为全实验固定基准。**

### Phase 4｜冒烟测试（0.15h，¥0.5）

- **操作**：`30_smoke.sh`：从 train 集拷 200 张图与标签到 `datasets/pest24_smoke/`（保持 yaml 结构，names 同 pest24），执行
  `yolo detect train model=yolo11n.pt data=.../pest24_smoke.yaml epochs=3 imgsz=640 batch=16 project=/root/autodl-tmp/runs name=smoke`
- **成功判据**：3 轮跑完无 OOM、无 label 警告刷屏、产出 `weights/last.pt`。
- **失败处理**：OOM → batch 减半；DataLoader 报错 → `workers=0`；label 警告多 → 回 Phase 3 检查越界框处理。

### Phase 5｜正式实验矩阵（4~6h，¥8~12，tmux 内执行）

- **操作**：上传 `40_experiments.sh`，在 tmux 中执行（防 ssh 断线）：
  ```bash
  #!/bin/bash
  set -e
  cd /root/autodl-tmp
  DATA=datasets/pest24/pest24.yaml
  EVAL_SPLIT=$(grep -q '^test:' $DATA && echo test || echo val)   # test 无标注则退回 val
  RUN="yolo detect train model=yolo11n.pt data=$DATA seed=0 patience=30 project=runs"
  VAL="yolo detect val data=$DATA split=$EVAL_SPLIT"

  # 5.1 baseline（Ultralytics 默认超参 + 固定 seed/patience，社区标准起点）
  $RUN epochs=100 imgsz=640 batch=48 name=baseline 2>&1 | tee exp_baseline.log
  $VAL model=runs/baseline/weights/best.pt 2>&1 | tee val_baseline.log

  # 5.2 主微调：密集/粘连增强（论文第一影响因素；Kaggle 800×600 下分辨率方向收益有限）
  $RUN epochs=100 imgsz=640 batch=48 close_mosaic=10 copy_paste=0.1 multi_scale=True scale=0.9 name=ft_aug 2>&1 | tee exp_aug.log
  $VAL model=runs/ft_aug/weights/best.pt 2>&1 | tee val_aug.log

  # 5.3 条件追加：ft_aug 相对 baseline 的 mAP50 提升 ≥ 0.5pt 且预算允许时，取消注释
  #     （imgsz=800 为原生上限档，预期收益小；960/1280 超原生分辨率无意义）
  # $RUN epochs=100 imgsz=800 batch=32 close_mosaic=10 copy_paste=0.1 multi_scale=True scale=0.9 name=ft800 2>&1 | tee exp_800.log
  # $VAL model=runs/ft800/weights/best.pt imgsz=800 2>&1 | tee val_800.log
  ```
  （tmux 用法：`tmux new -s train 'bash 40_experiments.sh'`；断线后 `tmux attach -t train`）
- **每个实验后**：`grep -E "^ +all" val_*.log | tail -1`（行内依次是 P / R / mAP50 / mAP50-95）追加写入 `results.md` 表格（注明 EVAL_SPLIT）。
- **分支判定**：ft_aug 相对 baseline 的 mAP50 提升 ≥ 0.5 pt → 取消注释跑 `ft800`（~2h，可选）；`ft_long`（150ep/cos_lr）默认不跑——增益预期最小，预算充裕再议。
- **成功判据**：`results.md` 含 baseline + ≥1 微调行；每个 `runs/<name>/weights/best.pt` 存在。
- **失败处理**：中途断电/断网 → 重连 tmux，从断掉那行改用 `model=runs/<name>/weights/last.pt resume=True` 续跑，已完成实验不重跑。

### Phase 6｜conf / iou 扫描与最终定位（0.3h，¥1）

- **操作**：对评测集 mAP50 最优的权重，做两轮零成本扫描（均在 EVAL_SPLIT 上）：
  1. **conf 扫描**：`conf ∈ {0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50}`，记录 P/R/F1，选 **F1 最大或满足 P 目标的最小 conf**；
  2. **IoU 扫描**（针对 4657 张粘连图）：`iou ∈ {0.45,0.5,0.6,0.7}`，与选定 conf 组合取 mAP50 最高者。
- **成功判据**：两张扫描表写入 results.md；明确给出「部署 conf / iou = x.xx / x.xx」结论。
- **注意**：扫描只用于定部署点，实验间对比仍用默认 conf/iou 的 mAP。

### Phase 7｜推理验证与 SAHI（0.3h，¥0.6）

- **操作**：
  1. `yolo detect predict model=<best.pt> source=<评测集前10张> conf=<Phase6选定> iou=<选定>`；
  2. **SAHI 切片验证（可选）**：800×600 整图推理已可覆盖；如需评估未来部署 2095×1944 原图的场景，`pip install sahi` 对 5 张原图做 slice 512 / overlap 0.2，与整图推理对比检出数与召回。
- **成功判据**：10 张整图结果图齐全（SAHI 若执行附对比记录）。
- **回收**：结果图 scp 回本地供用户人工抽查误检/漏检。

### Phase 8｜回收与停费（0.2h）

- **操作**：
  1. 实例内打包：`tar czf artifacts.tgz runs/*/weights/best.pt results.md exp_*.log val_*.log env_snapshot.txt datasets/pest24/pest24.yaml runs/*/PR_curve.png runs/*/confusion_matrix.png`；
  2. scp 回本地工作区 `artifacts/`；
  3. 校验：best.pt 数量 ≥3、results.md 齐全、文件大小 >0；
  4. **提醒用户执行 H3：AutoDL 控制台「关机」（停止计费；数据盘保留，续跑时开机即可）**。
- **成功判据**：本地 artifacts 完整；实例状态=已关机。

## 4. 预算与时间表（4090 按量计费，约 ¥1.7~2/h）

| 阶段 | GPU 时长 | 费用 |
|---|---|---|
| Phase 1-4 连接/环境/数据/冒烟 | ~1.5h | ~¥3 |
| Phase 5 baseline(640≈1-1.5h) + ft_aug(640+multi_scale≈2h) | ~3-3.5h | ~¥6-7 |
| Phase 5 条件追加 ft800 | +2h | +¥4 |
| Phase 6-7 conf/iou 扫描与推理验证 | ~0.5h | ~¥1 |
| **合计** | **~5-5.5h（约 ¥10-11）；含 ft800 约 7-7.5h（¥14-15）** | |

建议策略：一次跑完毫无压力；ft800 视 ft_aug 结果决定是否取消注释。

## 5. 可复现性保障清单

- [ ] ultralytics/torch 版本已 pin 并存 env_snapshot.txt
- [ ] 数据转换只做一次，三 split 统计数已记录在 results.md
- [ ] 所有实验 seed=0、同 conf 口径评测
- [ ] results.md 每行实验记录：名称、改动、imgsz、epochs、P/R/mAP50/mAP50-95、耗时
- [ ] 本地留存 best.pt + 配置 yaml + 日志，跨机器不再比指标

## 6. 风险与回滚

| 风险 | 信号 | 处置 |
|---|---|---|
| 显存 OOM | `CUDA out of memory` | batch 减半重跑该实验 |
| ssh 断线 | tmux 丢失 | 重连 `tmux attach`；训练中断则 last.pt resume=True |
| 网络加速失效 | pip/wget 超时 | 去掉 network_turbo / 换清华源 |
| 费用超支 | 实际时长超预估 50% | 停掉未开始的 ft800，先回收已有成果 |
| 类别名不匹配 | 转换脚本报未识别类别 | 回填 CLASSES 重跑 Phase 3（仅重转换，费用可忽略） |
| 镜像无 test split | yaml 无 `test:` 行 | 属正常（官方未公开），EVAL_SPLIT=val；禁止自切 test |

---

> 附：results.md 模板（Agent 按此填写；头部注明 EVAL_SPLIT=____ 与各 split 统计）
>
> | 实验 | 改动 | imgsz | epochs | eval P | eval R | eval mAP50 | eval mAP50-95 | 耗时 |
> |---|---|---|---|---|---|---|---|---|
> | baseline | Ultralytics 默认超参 | 640 | 100 | | | | | |
> | ft_aug | +copy_paste/multi_scale/scale | 640 | 100 | | | | | |
> | ft800（条件） | imgsz→800（原生档） | 800 | 100 | | | | | |
> | 部署点 | conf/iou=（Phase 6 选定） | — | — | | | | | |
> | SAHI（可选） | slice 512（对比整图推理） | — | — | | | | | |

## 7. GPU 选型与地区选择说明

**选卡逻辑**：YOLO11n 是「小模型 × 大吞吐」负载，瓶颈在 FP16 算力与数据加载，不在显存（640/batch48 约 10-14G）。因此按「FP16 算力 ÷ 时租」选，不堆显存。

| 卡型 | 本负载相对速度 | 显存 | 时租（约数，以下单页为准） | 结论 |
|---|---|---|---|---|
| **4090 24G** | 100%（基准） | 24G | ¥1.5~2 | **性价比最优，首选** |
| 4090D 24G | ~95% | 24G | 相近 | 4090 无货时第一替代 |
| 3090 24G | ~60-70% | 24G | 略低 | 第二替代，总时长 ×1.4、全矩阵 ≈¥30-40 |
| A6000 48G | 介于 3090-4090 | 48G | 略高 | 显存浪费，无必要 |
| V100 16/32G | ~40-60% | 16/32G | 相近或稍贵 | 老架构；16G 版需把 800 的 batch 降到 12-16 |
| A100 40/80G | ≤1.5~2x | 40/80G | ¥5~10 | 3~5 倍价买不满 2 倍速，本任务不划算 |

> 算总账：全矩阵 4090 约 5-7.5h / ¥10-15；A100 最多省 2-3h 却多花 ¥20-50。A100 的优势在大模型/多卡并行，nano 模型吃不到。**若 24G 级卡全无货，宁可等有货区，也不要为省一两个小时上 A100。**

**换卡后的参数适配**：3090/4090D 显存同为 24G，§3/§5 的 batch 参数**原样可用**；V100 16G 则 800 用 `batch=16`、640 用 `batch=32`。

**地区有没有要求：没有硬性要求。**
- 各区（西北/华北/华东/华南等）从国内任意城市均可 ssh，延迟差几十 ms，对训练无感；
- 4090 库存按区动态浮动 → **下单页「有货/余量足」优先**，同卡型各区价格基本一致；
- 唯一硬约束是复现性：**全实验矩阵必须同一实例跑完，中途不换卡不换区**——AutoDL 关机不释放实例、数据盘保留，再开机回到原实例原区，天然满足；
- 数据上传走 AutoDL 网盘中转，与所选区基本无关。
