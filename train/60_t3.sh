#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# 60_t3.sh —— T3 创新模块包（约 8 GPU-h）★本方案的价值所在
#
#   T3-1 strong+ALA         ★I1  粘连子集
#   T3-2 strong+DLA         ★I2  高密度子集 + 密集区误检率
#   T3-3 strong+DRA         ★I3  稀有类子集
#   T3-4 strong+ASM         ★I4  粘连子集
#   T3-5 strong+ALA+DLA     I1+I2 互补性
#   T3-6 strong+全量        I1+I2+I3+I4
#   T3-7 strong+I6          可选：混淆对感知分类损失
#
# 前置条件：G0 通过（M1 成立）。若 G0 判定 FAIL，则**只跑 T3-3 / T3-4**
#           （I3 DRA / I4 ASM），并把 D0 + I7 作为论文主体。
# 闸门 G3：≥1 个组件相对强基线 mAP50 ≥ 0.5 pt **且在其定向子集上 ≥ 1 pt**
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
STRONG="ultralytics_ext/yaml/strong/yolo11n.yaml"

# 读 G0 裁决（若存在）
G0_VERDICT="UNKNOWN"
if [ -f "${ROOT}/artifacts/diag/G0_verdict.md" ]; then
  G0_VERDICT="$(grep -oE '\*\*(PASS|FAIL|PENDING)\*\*' "${ROOT}/artifacts/diag/G0_verdict.md" | head -1 | tr -d '*')"
fi
echo "=== [60_t3] G0 裁决 = ${G0_VERDICT} ==="

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

run () {
  local name="$1"; shift
  local envs="$1"; shift
  local model="$1"; shift
  if [ -f "${RUNS}/${name}/weights/best.pt" ]; then
    echo "--- [60_t3] ${name} 已存在，跳过 ---"; return 0
  fi
  echo "--- [60_t3] ${name} 开始 (${envs}) ---"
  # shellcheck disable=SC2086
  env ${envs} ADEA_VERBOSE=1 python -m ultralytics_ext.train_adea detect train \
    model="${model}" data="${DATA}" \
    epochs="${EPOCHS}" imgsz="${IMGSZ}" batch="${BATCH}" seed="${SEED}" \
    patience=30 workers=8 project="${RUNS}" name="${name}" "$@" \
    2>&1 | tee "${LOGS}/60_t3_${name}.log" | tail -10
  # shellcheck disable=SC2086
  env ${envs} ADEA_VERBOSE=0 python -m ultralytics_ext.train_adea detect val \
    model="${RUNS}/${name}/weights/best.pt" data="${DATA}" split="${EVAL_SPLIT}" \
    project="${RUNS}" name="${name}_val" \
    2>&1 | tee "${LOGS}/60_t3_eval_${name}.log" | tail -6
}

if [ "${G0_VERDICT}" = "FAIL" ] || [ "${G0_VERDICT}" = "PENDING" ]; then
  echo "!!! G0 = ${G0_VERDICT}：按降级路径，只跑 I3 DRA / I4 ASM"
  run strong_DRA   "ADEA_DRA=1"   "${STRONG}"
  run strong_ASM   "ADEA_AGS=1"   "${STRONG}"
else
  # ---- I1 ~ I4 单组件 ----
  run strong_ALA   "ADEA_ALA=1"   "${STRONG}"
  run strong_DLA   "ADEA_DLA=1"   "${STRONG}"
  run strong_DRA   "ADEA_DRA=1"   "${STRONG}"
  run strong_ASM   "ADEA_AGS=1"   "${STRONG}"
  # ---- 互补性验证 ----
  run strong_ALA_DLA "ADEA_ALA=1 ADEA_DLA=1" "${STRONG}"
  # ---- 全量组合 ----
  run strong_ADEA  "ADEA_ALA=1 ADEA_DLA=1 ADEA_DRA=1 ADEA_AGS=1" "${STRONG}"
  # ---- 可选 I6 ----
  if [ "${RUN_I6:-0}" = "1" ]; then
    run strong_I6   "ADEA_PAIR_CLS=1" "${STRONG}"
  fi
fi

python scripts/summarize.py --runs "${ROOT}/runs" --out "${ROOT}/results.md" --eval-split "${EVAL_SPLIT}"
echo "=== [60_t3] 完成 ==="
