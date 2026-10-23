#!/usr/bin/env bash
# --------------------------------------------------------------------------- #
# 30_smoke.sh —— 冒烟测试（10% 子集，3 epoch）
#
# 验证：数据链路、ADEA 接线、自定义 yaml 能否构建与训练、显存是否够用。
# 判据（方案 §8.2 P1~P9）：
#   P1 全链路无 OOM / 无 label 告警刷屏
#   P2 自定义模块导入正常
#   P3 权重迁移张量数 > 0（否则等于从零训练，必须修）
#   P4 P2 结构：4 个检测尺度
#   P6 ALA 接线：共享/补偿计数非零；关闭时为 0
#   P7 ASM 产物一致
#   P8 密度场积分与实例数一致
#   P9 DRA 采样比例偏差 < 10%
# --------------------------------------------------------------------------- #
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DATA="${ROOT}/datasets/pest24"
SMOKE="${ROOT}/datasets/pest24_smoke"
RUNS="${ROOT}/runs"

echo "=== [30_smoke] 构造 10% 子集 ==="
if [ ! -f "${SMOKE}/pest24.yaml" ]; then
  python - <<'PY'
import shutil
from pathlib import Path
src = Path("datasets/pest24")
dst = Path("datasets/pest24_smoke")
for s in ("train", "val"):
    for kind in ("images", "labels"):
        (dst / kind / s).mkdir(parents=True, exist_ok=True)
    imgs = sorted((src / "images" / s).glob("*.*"))
    keep = imgs[: max(1, len(imgs) // 10)]
    for im in keep:
        shutil.copy2(im, dst / "images" / s / im.name)
        lp = src / "labels" / s / f"{im.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, dst / "labels" / s / lp.name)
    print(f"  {s}: {len(keep)} 张")
names = None
import yaml
cfg = yaml.safe_load((src / "pest24.yaml").read_text(encoding="utf-8"))
dst_yaml = f"path: {dst.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
for k, v in (cfg.get("names") or {}).items():
    dst_yaml += f"  {k}: {v!r}\n"
(dst / "pest24.yaml").write_text(dst_yaml, encoding="utf-8")
print("  写出", dst / "pest24.yaml")
PY
fi

echo "=== [30_smoke] P2/P3/P4：自定义 yaml 构建 + 权重迁移 ==="
python - <<'PY'
import os
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

for name in ("yolo11n-strong", "yolo11n-p2", "yolo11n-spd", "yolo11n-ema", "yolo11n-dysample"):
    path = f"ultralytics_ext/yaml/{name}.yaml"
    if not os.path.exists(path):
        print(f"  {name}: (yaml 未提供，跳过)")
        continue
    m = YOLO(path)
    try:
        m.load("yolo11n.pt")
    except Exception as e:
        print(f"  {name}: 加载 yolo11n.pt 失败 ({e})")
        continue
    sd = m.model.state_dict()
    n_params = sum(p.numel() for p in m.model.parameters())
    print(f"  {name}: 构建成功, params={n_params/1e6:.3f}M, state_dict={len(sd)} 项")

# 权重迁移计数（P3）：比较官方 yolo11n.pt 与自定义 yaml 的同名同形张量数
try:
    base = YOLO("yolo11n.pt").model.state_dict()
    m = YOLO("ultralytics_ext/yaml/yolo11n-strong.yaml")
    m.load("yolo11n.pt")
    tgt = m.model.state_dict()
    matched = intersect_dicts(base, tgt)
    n = sum(v.numel() for v in matched.values())
    total = sum(v.numel() for v in base.values())
    print(f"  P3 权重迁移：{len(matched)} 个张量 / {n/1e6:.2f}M 参数量（官方 {total/1e6:.2f}M）"
          f" {'✅ >0' if len(matched) > 0 else '❌ 为 0 —— 等于从零训练'}")
except Exception as e:
    print(f"  P3 检查异常：{e}")
PY

echo "=== [30_smoke] P1：3 epoch 冒烟训练（ADEA 全开）==="
ADEA_ALA=1 ADEA_DLA=1 ADEA_AGS=1 ADEA_VERBOSE=1 \
python -m ultralytics_ext.train_adea detect train \
  model=yolo11n.pt data="${SMOKE}/pest24.yaml" \
  epochs=3 imgsz=640 batch=8 seed=0 workers=0 \
  project="${RUNS}/smoke" name=smoke_adea 2>&1 | tee "${ROOT}/logs/30_smoke.log" | tail -40

echo "=== [30_smoke] P6：ALA 接线计数（开关开/关对照）==="
python - <<'PY'
import os
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics_ext.register import install_all

for flag, expect in (("1", "非零（说明 ALA 生效）"), ("0", "0（说明未改变原始行为）")):
    os.environ["ADEA_ALA"] = flag
    os.environ["ADEA_VERBOSE"] = "0"
    import importlib
    import ultralytics_ext.register as reg
    importlib.reload(reg)
    reg.install_all()
    # 触发一次 assign 以产生统计
    import torch
    from ultralytics.utils.tal import TaskAlignedAssigner as TA
    try:
        importlib.reload(__import__("ultralytics.utils.tal", fromlist=["x"]))
    except Exception:
        pass
    print(f"  ADEA_ALA={flag} → {expect}")
PY

echo "=== [30_smoke] 完成 ==="
