#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# 70_t4.sh —— T4 定稿与统计（约 11 GPU-h）
#
#   T4-1 final_best × seed 0/1/2   200 epochs（对齐 EP-YOLO 230ep / Attention-PestNet 200ep）
#   T4-2 baseline    × seed 1/2     200 epochs，配对检验用
#   T4-3 strong      × seed 1/2     200 epochs，创新增量的配对检验用
#   T4-4 ref_yolo11s                容量上限参考（区分"方法问题" vs "nano 容量上限"）
#   T4-5 天花板曲线组               I7：imgsz{640,800} × model{n,s} → 逐类 AP 曲线
#
# 训练轮数说明：T1–T3 用 100 epochs 筛查；T4 定稿组统一升到 200 epochs，
#             并**重跑** T1-0 baseline 于 200 epochs 作为最终对照（否则不可比）。
# --------------------------------------------------------------------------- #
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
source /etc/network_turbo 2>/dev/null || true

DATA="${ROOT}/datasets/pest24/pest24.yaml"
RUNS="${ROOT}/runs/pest24"
LOGS="${ROOT}/logs"
mkdir -p "${RUNS}" "${LOGS}"

EPOCHS="${EPOCHS:-200}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-48}"
STRONG="ultralytics_ext/yaml/yolo11n-strong.yaml"

# 定稿组：默认用 T3 的全量组合；可用 FINAL_ENV 覆盖
FINAL_NAME="${FINAL_NAME:-strong_ADEA}"
FINAL_ENV="${FINAL_ENV:-ADEA_ALA=1 ADEA_DLA=1 ADEA_DRA=1 ADEA_AGS=1}"

EVAL_SPLIT="$(python - <<'PY'
from pathlib import Path
split = "val"
y = Path("datasets/pest24/pest24.yaml")
if y.exists() and "test:" in y.read_text(encoding="utf-8"):
    d = Path("datasets/pest24/labels/test")
    if d.exists() and any(d.glob("*.txt")):
        split = "test"
print(split)
PY
)"
echo "=== [70_t4] EVAL_SPLIT = ${EVAL_SPLIT} ; FINAL = ${FINAL_NAME} (${FINAL_ENV}) ==="

run () {
  local name="$1"; shift
  local envs="$1"; shift
  local model="$1"; shift
  if [ -f "${RUNS}/${name}/weights/best.pt" ]; then
    echo "--- [70_t4] ${name} 已存在，跳过 ---"; return 0
  fi
  echo "--- [70_t4] ${name} 开始 (${envs}) ---"
  # shellcheck disable=SC2086
  env ${envs} ADEA_VERBOSE=1 python -m ultralytics_ext.train_adea detect train \
    model="${model}" data="${DATA}" \
    epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${BATCH}" \
    patience=50 workers=8 project="${RUNS}" name="${name}" "$@" \
    2>&1 | tee "${LOGS}/70_t4_${name}.log" | tail -8
  # shellcheck disable=SC2086
  env ${envs} ADEA_VERBOSE=0 python -m ultralytics_ext.train_adea detect val \
    model="${RUNS}/${name}/weights/best.pt" data="${DATA}" split="${EVAL_SPLIT}" \
    project="${RUNS}" name="${name}_val" \
    2>&1 | tee "${LOGS}/70_t4_eval_${name}.log" | tail -6
}

# ---- T4-1 定稿组 × 3 seed ----
for s in 0 1 2; do
  run "${FINAL_NAME}_s${s}" "${FINAL_ENV}" "${STRONG}" seed="${s}"
done

# ---- T4-2 baseline × 3 seed（200 epochs，最终对照）----
for s in 0 1 2; do
  run "baseline200_s${s}" "ADEA_VERBOSE=0" yolo11n.pt seed="${s}"
done

# ---- T4-3 strong × 3 seed（创新增量的配对检验）----
for s in 0 1 2; do
  run "strong200_s${s}" "ADEA_VERBOSE=0" "${STRONG}" seed="${s}"
done

# ---- T4-4 容量上限参考 ----
run ref_yolo11s "ADEA_VERBOSE=0" yolo11s.pt seed=0

# ---- T4-5 天花板曲线组（I7）----
echo "--- [70_t4] T4-5 天花板曲线组 ---"
for sz in 640 800; do
  for m in yolo11n yolo11s; do
    nm="ceiling_${m}_${sz}"
    if [ -f "${RUNS}/${nm}/weights/best.pt" ]; then
      echo "--- ${nm} 已存在，跳过 ---"; continue
    fi
    python -m ultralytics_ext.train_adea detect train \
      model="${m}.pt" data="${DATA}" epochs=100 imgsz="${sz}" batch="${BATCH}" \
      seed=0 patience=30 workers=8 project="${RUNS}" name="${nm}" \
      2>&1 | tee "${LOGS}/70_t4_${nm}.log" | tail -6
    python -m ultralytics_ext.train_adea detect val \
      model="${RUNS}/${nm}/weights/best.pt" data="${DATA}" split="${EVAL_SPLIT}" \
      project="${RUNS}" name="${nm}_val" \
      2>&1 | tee "${LOGS}/70_t4_eval_${nm}.log" | tail -6
  done
done

python scripts/summarize.py --runs "${ROOT}/runs" --out "${ROOT}/results.md" --eval-split "${EVAL_SPLIT}"
echo "=== [70_t4] 完成 ==="
