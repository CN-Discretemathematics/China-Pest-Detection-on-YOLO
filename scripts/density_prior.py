#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADEA · I2 DLA 配套：密度场渲染与导出。

密度场是 I2 DLA 的核心输入。**训练期**用 GT 中心点直接渲染（零额外可学习
参数、零推理开销）；本脚本负责：

1. 从 YOLO 标签渲染密度场并导出为可视化 PNG（用于论文插图与 P8 校验）；
2. 统计"密度 vs 实例数"的关系，验证密度场确实刻画拥挤程度；
3. 导出高密度图清单（>20 实例，论文口径 8.89%），供 I4 ASM 对齐密度分布。

用法::

    python scripts/density_prior.py --data datasets/pest24 --out artifacts/density
    python scripts/density_prior.py --data datasets/pest24 --out artifacts/density \
        --render 20          # 额外渲染前 20 张高密度图

依赖：numpy（必需）；matplotlib（可选，仅用于出图）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# 让 `python scripts/density_prior.py` 能 import 到仓库根的 ultralytics_ext
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"需要 numpy：pip install numpy  ({exc})")

from ultralytics_ext.assign.dla import normalize_density, render_density  # noqa: E402

DEFAULT_IMG_W, DEFAULT_IMG_H = 800, 600
DENSE_INSTANCE_THR = 20


def read_labels(label_path: Path, img_w: int, img_h: int):
    """→ [(cls, cx, cy, w, h)]（像素坐标）。"""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:5])
        except ValueError:
            continue
        out.append((c, cx * img_w, cy * img_h, bw * img_w, bh * img_h))
    return out


def density_for_image(items, img_w: int, img_h: int, sigma_scale: float = 0.5):
    """渲染单图密度场（numpy）。"""
    centers = [(cx, cy) for _c, cx, cy, _w, _h in items]
    radii = [max(w, h) * 0.5 for _c, _cx, _cy, w, h in items]
    d = render_density(centers, radii, (img_h, img_w), sigma_scale)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description="ADEA · DLA 密度场渲染与统计")
    ap.add_argument("--data", required=True, help="数据集根目录（含 labels/）")
    ap.add_argument("--out", default="artifacts/density")
    ap.add_argument("--img-w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--img-h", type=int, default=DEFAULT_IMG_H)
    ap.add_argument("--sigma-scale", type=float, default=0.5)
    ap.add_argument("--render", type=int, default=0, help="额外渲染前 N 张高密度图为 PNG")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(args.data) / "labels"
    splits = [s for s in ("train", "val", "test") if (labels_dir / s).exists()]

    rows: List[Dict[str, object]] = []
    dense_list: List[Tuple[str, int]] = []
    inst_hist: Counter = Counter()

    for split in splits:
        for lp in sorted((labels_dir / split).glob("*.txt")):
            items = read_labels(lp, args.img_w, args.img_h)
            n = len(items)
            inst_hist[n] += 1
            if n == 0:
                continue
            d = density_for_image(items, args.img_w, args.img_h, args.sigma_scale)
            total = float(d.sum())  # 高斯核未归一化，积分与实例数成正比
            peak = float(d.max())
            rows.append(
                {
                    "image": f"{split}/{lp.stem}",
                    "n_instances": n,
                    "density_sum": round(total, 3),
                    "density_peak": round(peak, 4),
                    "density_mean": round(total / d.size, 6),
                }
            )
            if n > DENSE_INSTANCE_THR:
                dense_list.append((f"{split}/{lp.stem}", n))

    # 密度-实例数相关性（验证密度场确实刻画拥挤程度）
    xs = [float(r["n_instances"]) for r in rows]
    ys = [float(r["density_sum"]) for r in rows]
    rho = float("nan")
    if len(xs) > 2:
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        if sxx > 0 and syy > 0:
            rho = sxy / (sxx**0.5 * syy**0.5)

    stats = {
        "n_images": len(rows),
        "n_dense_images": len(dense_list),
        "pct_dense": round(100.0 * len(dense_list) / max(1, len(rows)), 2),
        "corr_instances_density_sum": round(rho, 4) if rho == rho else None,
        "sigma_scale": args.sigma_scale,
        "dense_instances_threshold": DENSE_INSTANCE_THR,
        "instance_histogram_top20": dict(sorted(inst_hist.items())[:20]),
    }
    (out / "density_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "per_image_density.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("image,n_instances,density_sum,density_peak,density_mean\n")
        for r in rows:
            f.write(
                f"{r['image']},{r['n_instances']},{r['density_sum']},"
                f"{r['density_peak']},{r['density_mean']}\n"
            )

    print("== 密度统计 ==")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 可选：渲染高密度图（论文插图）
    if args.render > 0:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[WARN] 未安装 matplotlib，跳过出图：{exc}")
            return
        dense_list.sort(key=lambda t: -t[1])
        png_dir = out / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        for stem, n in dense_list[: args.render]:
            split, name = stem.split("/", 1)
            items = read_labels(labels_dir / split / f"{name}.txt", args.img_w, args.img_h)
            d = density_for_image(items, args.img_w, args.img_h, args.sigma_scale)
            dn = normalize_density(d, 0.99)
            plt.figure(figsize=(6, 4.5))
            plt.imshow(dn, cmap="inferno")
            plt.colorbar(label="normalized density")
            plt.title(f"{stem}  n={n}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(png_dir / f"{name}.png", dpi=130)
            plt.close()
        print(f"已渲染 {min(args.render, len(dense_list))} 张密度图 → {png_dir}")


if __name__ == "__main__":
    main()
