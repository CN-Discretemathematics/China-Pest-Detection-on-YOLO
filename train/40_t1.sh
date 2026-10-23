#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# 40_t1.sh —— T1 基线与工程线（约 6 GPU-h）
#
#   T1-0 baseline      yolo11n.pt，默认超参，640，100ep，seed 0
#   T1-1 ft_aug        + copy_paste / multi_scale / scale / close_mosaic
#   T1-2 ref_yolov8n   YOLOv8n 同划分同口径（对标锚点）
#   T1-3 ft_aug_cls    T1-1 + cls=1.0（定向拉 Precision）
#   T1-4 ft_aug_ags    T1-1 的权重 + AGS 后处理（零训练成本验证 I5）
#   T1-5 scan          conf/iou 扫描（定部署点，零训练成本）
#
# 闸门 G1：baseline 指标齐全；ft_aug 相对 baseline mAP50 ≥ 0.5 pt
# 用法：tmux new -s t1 'bash train/40_t1.sh'
# --------------------------------------------------------------------------- #
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
# shellcheck disable=SC1091
source /etc/network_turbo 2>/dev/null || true

DATA="${ROOT}/datasets/pest24/pest24.yaml"
RUNS="${ROOT}/runs/pest24"
LOGS="${ROOT}/logs"
mkdir -p "${RUNS}" "${LOGS}" "${ROOT}/artifacts"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-48}"
SEED="${SEED:-0}"

# EVAL_SPLIT：yaml 含 test 且 test 有标签 → test，否则 val（全程固定）
EVAL_SPLIT="$(python - <<'PY'
import os
from pathlib import Path
y = Path("datasets/pest24/pest24.yaml")
split = "val"
if y.exists():
    txt = y.read_text(encoding="utf-8")
    if "test:" in txt and Path("datasets/pest24/labels/test").exists():
        if any(Path("datasets/pest24/labels/test").glob("*.txt")):
            split = "test"
print(split)
PY
)"
echo "=== [40_t1] EVAL_SPLIT = ${EVAL_SPLIT} ==="

run () {
  # run <name> <model> <extra args...>
  local name="$1"; shift
  local model="$1"; shift
  if [ -f "${RUNS}/${name}/weights/best.pt" ]; then
    echo "--- [40_t1] ${name} 已存在，跳过 ---"
    return 0
  fi
  echo "--- [40_t1] ${name} 开始 ---"
  ADEA_VERBOSE=1 python -m ultralytics_ext.train_adea detect train \
    model="${model}" data="${DATA}" \
    epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${BATCH}" seed="${SEED}" \
    patience=30 workers=8 \
    project="${RUNS}" name="${name}" "$@" \
    2>&1 | tee "${LOGS}/40_t1_${name}.log" | tail -12
  echo "--- [40_t1] ${name} 完成 ---"
}

eval_one () {
  local name="$1"; shift
  if [ ! -f "${RUNS}/${name}/weights/best.pt" ]; then
    echo "!!! ${name} 没有 best.pt，跳过评测"
    return 0
  fi
  echo "--- [40_t1] 评测 ${name} (split=${EVAL_SPLIT}) ---"
  ADEA_VERBOSE=0 python -m ultralytics_ext.train_adea detect val \
    model="${RUNS}/${name}/weights/best.pt" data="${DATA}" split="${EVAL_SPLIT}" \
    project="${RUNS}" name="${name}_val" "$@" \
    2>&1 | tee "${LOGS}/40_t1_eval_${name}.log" | tail -8
}

# ---- T1-0 baseline ----
run baseline yolo11n.pt
eval_one baseline

# ---- T1-1 ft_aug（调参线）----
run ft_aug yolo11n.pt copy_paste=0.1 multi_scale=True scale=0.9 close_mosaic=10
eval_one ft_aug

# ---- T1-2 ref_yolov8n（同划分同口径的 YOLOv8n 参考基线）----
run ref_yolov8n yolov8n.pt
eval_one ref_yolov8n

# ---- T1-3 ft_aug_cls（定向拉 Precision）----
run ft_aug_cls yolo11n.pt copy_paste=0.1 multi_scale=True scale=0.9 close_mosaic=10 cls=1.0
eval_one ft_aug_cls

# ---- T1-4 ft_aug_ags（AGS 后处理，零训练成本）----
echo "--- [40_t1] T1-4 AGS 评测（用 ft_aug 权重，AGS 开启）---"
ADEA_AGS=1 ADEA_AGS_DECAY=0.5 ADEA_VERBOSE=1 \
python -m ultralytics_ext.train_adea detect val \
  model="${RUNS}/ft_aug/weights/best.pt" data="${DATA}" split="${EVAL_SPLIT}" \
  project="${RUNS}" name="ft_aug_ags_val" \
  2>&1 | tee "${LOGS}/40_t1_eval_ft_aug_ags.log" | tail -8

# ---- T1-5 conf/iou 扫描（定部署点）----
echo "--- [40_t1] T1-5 conf/iou 扫描 ---"
python scripts/scan_thresholds.py \
  --model "${RUNS}/ft_aug/weights/best.pt" --data "${DATA}" --split "${EVAL_SPLIT}" \
  --out "${ROOT}/artifacts/scan" 2>&1 | tee "${LOGS}/40_t1_scan.log" | tail -30

# ---- 汇总 ----
python scripts/summarize.py --runs "${ROOT}/runs" --out "${ROOT}/results.md" --eval-split "${EVAL_SPLIT}"

echo "=== [40_t1] 完成 ==="
