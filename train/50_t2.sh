#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# 50_t2.sh —— T2 复现筛查 + 强基线组装（约 10 GPU-h）
#
#   T2-1 spd              复现 R1（EP-YOLO）—— 单用预期掉点
#   T2-2 ema              复现 R2（EP-YOLO）—— 实测最优注意力
#   T2-3 p2               复现 R4（arXiv 2512.16493）
#   T2-4 dysample         复现 R3（CCDW-YOLO）
#   T2-5 sqe_eiou         复现 R5（AEP-PEST）
#   T2-6 balanced_static  复现 R6（静态长尾重采样）
#   T2-7 spd+ema          复现组合（EP-YOLO 已证二者强耦合）
#   T2-8 yolo11n-strong   ★强基线组装（R1+R2+R3+R4+R5）
#
# 筛查协议：全量数据、100 epochs、seed 0，与 T1-0 同档直接比。
# 闸门 G2：yolo11n-strong 相对 vanilla baseline mAP50 ≥ 1.5 pt
# --------------------------------------------------------------------------- #
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
source /etc/network_turbo 2>/dev/null || true

DATA="${ROOT}/datasets/pest24/pest24.yaml"
RUNS="${ROOT}/runs/pest24"
LOGS="${ROOT}/logs"
mkdir -p "${RUNS}" "${LOGS}"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-48}"
SEED="${SEED:-0}"

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
echo "=== [50_t2] EVAL_SPLIT = ${EVAL_SPLIT} ==="

run () {
  local name="$1"; shift
  local model="$1"; shift
  if [ -f "${RUNS}/${name}/weights/best.pt" ]; then
    echo "--- [50_t2] ${name} 已存在，跳过 ---"; return 0
  fi
  echo "--- [50_t2] ${name} 开始 ---"
  ADEA_VERBOSE=1 python -m ultralytics_ext.train_adea detect train \
    model="${model}" data="${DATA}" \
    epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${BATCH}" seed="${SEED}" \
    patience=30 workers=8 project="${RUNS}" name="${name}" "$@" \
    2>&1 | tee "${LOGS}/50_t2_${name}.log" | tail -10
  ADEA_VERBOSE=0 python -m ultralytics_ext.train_adea detect val \
    model="${RUNS}/${name}/weights/best.pt" data="${DATA}" split="${EVAL_SPLIT}" \
    project="${RUNS}" name="${name}_val" \
    2>&1 | tee "${LOGS}/50_t2_eval_${name}.log" | tail -6
}

# ---- 复现筛查（单模块）----
run spd              ultralytics_ext/yaml/yolo11n-spd.yaml
run ema              ultralytics_ext/yaml/yolo11n-ema.yaml
run p2               ultralytics_ext/yaml/yolo11n-p2.yaml
run dysample         ultralytics_ext/yaml/yolo11n-dysample.yaml
run sqe_eiou         ultralytics_ext/yaml/yolo11n-strong.yaml ADEA_SQA_EIOU=1
run balanced_static  ultralytics_ext/yaml/yolo11n-strong.yaml

# ---- T2-7 SPD+EMA 组合（EP-YOLO 完整配方）----
run spd_ema          ultralytics_ext/yaml/yolo11n-spd-ema.yaml

# ---- T2-8 强基线（对照组：本文创新增量的参照系）----
run yolo11n-strong   ultralytics_ext/yaml/yolo11n-strong.yaml

python scripts/summarize.py --runs "${ROOT}/runs" --out "${ROOT}/results.md" --eval-split "${EVAL_SPLIT}"
echo "=== [50_t2] 完成 ==="
