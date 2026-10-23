# 实验台账（results.md）

> 本表仅汇总**本项目内部**同口径实验。跨论文数值不可直接对表
> （划分/分辨率/imgsz/轮数各异，见 `docs/调研报告_农业害虫检测前沿.md`）。

**评测集 EVAL_SPLIT = val**（实验间对比一律使用默认 conf=0.001 / iou=0.7）

| 实验 | model | imgsz | epochs | batch | seed | eval P | eval R | eval mAP50 | eval mAP50-95 | 耗时(h) | best.pt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke/smoke_strong_adea-2 | ultralytics_ext/yaml/strong/yolo11n.yaml | 640 | 3 | 4 | 0 | 0.45 | 38.86 | 0.38 | 0.16 | 437.305 | ✅ |
| runs/diag_baseline-2 | yolo11n.pt | 640 | 10 | 16 | 0 | 45.76 | 28.04 | 23.91 | 13.29 | 1934.54 | ✅ |
| runs/sla_base-2 | yolo11n.yaml | 640 | 1 | 4 | 0 | 16.93 | 32.58 | 2.28 | 1.35 | 56.961 | ✅ |
| runs/sla_sfw_smoke | yolo11n.yaml | 640 | 1 | 4 | 0 | 19.01 | 22.16 | 3.29 | 1.80 | 52.81 | ✅ |
| runs/sla_sla | yolo11n.yaml | 640 | 1 | 4 | 0 | 1.49 | 49.27 | 2.19 | 1.37 | 57.858 | ✅ |
| runs/sla_sla2 | yolo11n.yaml | 640 | 1 | 4 | 0 | 1.49 | 49.27 | 2.19 | 1.37 | 57.616 | ✅ |
| runs/sla_sla3 | yolo11n.yaml | 640 | 1 | 4 | 0 | 19.00 | 9.93 | 3.32 | 2.02 | 57.408 | ✅ |
| runs/smoke2_adea | ultralytics_ext/yaml/strong/yolo11n.yaml | 640 | 2 | 4 | 0 | 0.50 | 26.68 | 0.15 | 0.05 | 292.861 | ✅ |
| runs/smoke3 | yolo11n.pt | 640 | 1 | 4 | 0 | 23.60 | 9.28 | 2.24 | 1.30 | 120.691 | ✅ |
| runs/smoke4 | yolo11n.pt | 640 | 1 | 4 | 0 | 10.60 | 29.78 | 2.77 | 1.61 | 73.913 | ✅ |
| smoke_sla_sla3 | runs\sla_sla3\weights\best.pt | 640 | ? | ? | 0 | 18.98 | 17.07 | 3.35 | 2.03 | None | — |

## 定向子集 mAP（每个创新组件必须在其目标子集上有提升）

| 实验 | 粘连子集 | 高密度子集 | 稀有类子集 | 极小类子集 |
|---|---|---|---|---|
| smoke_sla_sla3 | 1.55 | 5.50 | 0.00 | - |
