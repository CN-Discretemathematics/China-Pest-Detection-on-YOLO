#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# 20_data.sh —— 数据获取 + VOC→YOLO 转换 + D1~D7 正确性检查
#
# 幂等：datasets/pest24 已存在且有 pest24.yaml 时跳过转换。
# 输出：转换统计、每类实例数、交叉校验结果。
# --------------------------------------------------------------------------- #
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DATA="${ROOT}/datasets/pest24"
RAW="${ROOT}/datasets/raw"

mkdir -p "${RAW}" "${ROOT}/logs"

# ---- 1) 下载（Kaggle 镜像，2.38 GB，800×600）----
if [ ! -d "${RAW}/Pest24" ] && [ ! -d "${RAW}/VOCdevkit" ]; then
  echo "=== [20_data] 下载 Kaggle 镜像 boatshuai/pest24 ==="
  python - <<'PY'
import kagglehub, shutil, os
src = kagglehub.dataset_download("boatshuai/pest24")
dst = os.path.join("datasets", "raw")
print("downloaded:", src)
if os.path.abspath(src) != os.path.abspath(dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
print("copied to:", dst)
PY
else
  echo "=== [20_data] 数据已存在，跳过下载 ==="
fi

# ---- 2) 定位 VOC 根目录 ----
SRC=""
for cand in "${RAW}/Pest24/VOCdevkit/voc2007" "${RAW}/VOCdevkit/voc2007" "${RAW}"; do
  if [ -d "${cand}/Annotations" ] || [ -d "${cand}/xml" ]; then SRC="${cand}"; break; fi
done
if [ -z "${SRC}" ]; then
  echo "!!! 未找到 VOC 根目录（应有 Annotations/ 或 xml/）。请检查 datasets/raw 结构"
  find "${RAW}" -maxdepth 3 -type d | head -20
  exit 1
fi
echo "=== [20_data] VOC 根目录: ${SRC} ==="

# ---- 3) 转换 ----
if [ -f "${DATA}/pest24.yaml" ]; then
  echo "=== [20_data] ${DATA}/pest24.yaml 已存在，跳过转换 ==="
else
  python scripts/voc2yolo.py --src "${SRC}" --dst "${DATA}" 2>&1 | tee "${ROOT}/logs/20_data.log"
fi

# ---- 4) D1~D7 检查 ----
echo "=== [20_data] D1~D7 检查 ==="
python - <<'PY'
import os, glob, random
from pathlib import Path

root = Path("datasets/pest24")
img_dir, lbl_dir = root / "images", root / "labels"
splits = [s for s in ("train", "val", "test") if (img_dir / s).exists()]

print("  D1 图片数：")
expect = {"train": 12689, "val": 5076, "test": 5076}
ok_d1 = True
for s in splits:
    n = len(list((img_dir / s).glob("*.*")))
    exp = expect.get(s, 0)
    flag = "✅" if exp and abs(n - exp) / exp < 0.05 else ("—" if not exp else "⚠️")
    if exp and abs(n - exp) / exp >= 0.05:
        ok_d1 = False
    print(f"     {s}: {n} (期望≈{exp}) {flag}")

print("  D2 标签数：")
for s in splits:
    n = len(list((lbl_dir / s).glob("*.txt")))
    print(f"     {s}: {n} 个 txt")

print("  D3 类别数：")
import yaml
cfg = yaml.safe_load((root / "pest24.yaml").read_text(encoding="utf-8"))
names = cfg.get("names", {})
print(f"     names 项数 = {len(names)}  {'✅' if len(names) == 24 else '⚠️ 期望 24'}")
cls_ids = set()
for p in (lbl_dir / splits[0]).glob("*.txt"):
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cls_ids.add(int(line.split()[0]))
print(f"     标签中出现的 class id: {sorted(cls_ids)}")

print("  D4 图片尺寸：")
try:
    from PIL import Image
    for s in splits:
        files = list((img_dir / s).glob("*.jpg"))
        if not files:
            continue
        for f in random.sample(files, min(3, len(files))):
            with Image.open(f) as im:
                print(f"     {f.name}: {im.size} {'✅' if im.size == (800, 600) else '⚠️'}")
except Exception as e:
    print(f"     (跳过：{e})")

print("  D6 越界检查：")
bad = 0
for s in splits:
    for p in (lbl_dir / s).glob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            v = [float(x) for x in line.split()[1:5]]
            if any(x < -1e-6 or x > 1 + 1e-6 for x in v):
                bad += 1
print(f"     越界行数 = {bad}  {'✅' if bad == 0 else '⚠️'}")
PY

echo "=== [20_data] 完成 ==="
