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
  local envs="$1"; shift
  local model="$1"; shift
  if [ -f "${RUNS}/${name}/weights/best.pt" ]; then
    echo "--- [50_t2] ${name} 已存在，跳过 ---"; return 0
  fi
  echo "--- [50_t2] ${name} 开始 (${envs}) ---"
  # shellcheck disable=SC2086
  env ${envs} ADEA_VERBOSE=1 python -m ultralytics_ext.train_adea detect train \
    model="${model}" data="${DATA}" \
    epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${BATCH}" seed="${SEED}" \
    patience=30 workers=8 project="${RUNS}" name="${name}" "$@" \
    2>&1 | tee "${LOGS}/50_t2_${name}.log" | tail -10
  # shellcheck disable=SC2086
  env ${envs} ADEA_VERBOSE=0 python -m ultralytics_ext.train_adea detect val \
    model="${RUNS}/${name}/weights/best.pt" data="${DATA}" split="${EVAL_SPLIT}" \
    project="${RUNS}" name="${name}_val" \
    2>&1 | tee "${LOGS}/50_t2_eval_${name}.log" | tail -6
}

# 模型配置由 scripts/make_model_yamls.py 从官方 yolo11.yaml 派生（不要手改 yaml）
Y="ultralytics_ext/yaml"

# ---- 复现筛查（单模块，每组只相对 baseline 变一处）----
run spd              "ADEA_VERBOSE=0" "${Y}/spd/yolo11n.yaml"
run ema              "ADEA_VERBOSE=0" "${Y}/ema/yolo11n.yaml"
run p2               "ADEA_VERBOSE=0" "${Y}/p2/yolo11n.yaml"
run dysample         "ADEA_VERBOSE=0" "${Y}/dysample/yolo11n.yaml"
run sqe_eiou         "ADEA_SQA_EIOU=1"  yolo11n.pt
run balanced_static  "ADEA_VERBOSE=0"  yolo11n.pt
# ---- T2-7 SPD+EMA 组合（EP-YOLO 完整配方）----
run spd_ema          "ADEA_VERBOSE=0" "${Y}/spd_ema/yolo11n.yaml"
# ---- T2-8 强基线（对照组：本文创新增量的参照系）----
run yolo11n-strong   "ADEA_VERBOSE=0" "${Y}/strong/yolo11n.yaml"

python scripts/summarize.py --runs "${ROOT}/runs" --out "${ROOT}/results.md" --eval-split "${EVAL_SPLIT}"
echo "=== [50_t2] 完成 ==="
