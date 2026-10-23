#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# 10_env.sh —— 环境初始化（AutoDL / Linux）
#
# 幂等：可重复执行。执行后生成 env_snapshot.txt 供复现。
# --------------------------------------------------------------------------- #
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "=== [10_env] AutoDL 学术加速 ==="
# shellcheck disable=SC1091
source /etc/network_turbo 2>/dev/null || true

echo "=== [10_env] 目录准备 ==="
mkdir -p "${ROOT}/datasets" "${ROOT}/runs" "${ROOT}/artifacts" "${ROOT}/logs"

echo "=== [10_env] Python 依赖 ==="
# ultralytics 精确锁定：全实验矩阵必须同版本，否则指标不可比
if ! python -c "import ultralytics, torch" 2>/dev/null; then
  pip install -q -r requirements.txt
else
  echo "  已安装，跳过（当前版本：$(python -c 'import ultralytics;print(ultralytics.__version__)')）"
fi

python - <<'PY'
import torch, ultralytics, sys
print(f"  python      : {sys.version.split()[0]}")
print(f"  ultralytics : {ultralytics.__version__}")
print(f"  torch       : {torch.__version__}")
print(f"  cuda        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  gpu         : {torch.cuda.get_device_name(0)}")
    print(f"  vram        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PY

pip freeze > "${ROOT}/env_snapshot.txt"
echo "=== [10_env] 已写出 env_snapshot.txt ==="
echo "=== [10_env] 完成 ==="
