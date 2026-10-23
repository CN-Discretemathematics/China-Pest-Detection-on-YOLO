#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADEA · I4 ASM —— β 分布匹配的粘连合成增强 (Adhesion-Statistics-Matched synthesis)

出处 / 出处依据
--------------
Pest24 的第一影响因子是**目标粘连**（论文：4657 张图 β>0.05；粘连子集 mAP
比整体低 4.96 pt）。但现有增强都不产生"粘连"样本：

* ``mosaic`` 拼接的是**不重叠**的图块；
* Ultralytics 的 ``copy_paste`` 放置的实例几乎**不重叠**，且用 **alpha 混合**；
* 通用遮挡增强（Simple Copy-Paste, CVPR 2021）面向的是"被遮挡的完整物体"。

真实粘板上的粘连是**两虫物理黏连、边界融合、投下联合阴影** —— 与上述
"半透明叠加的遮挡"在外观上相反。

ASM 的做法
----------
1. **统计**：从 train split 估计 (类别 c, 尺度 s, 粘连度 β) 的联合经验分布
   （β 用论文定义 |A∩B|/|A∪B|；论文统计多数落在 0~0.2）。
2. **采样**：按该分布采样 (c, s, β)。
3. **合成**：贴入两个实例并调整相对位置使交并比**恰为**采样到的 β；
   重叠区采用**实心合并**渲染（上层像素实心覆盖，不 alpha 混合），
   并在并集外缘加一圈软阴影。
4. **密度对齐**：按论文统计（8.89% 的图 >20 实例）补齐高密度合成场景。
5. **输出**：**离线**生成到 ``datasets/pest24_synth/``，训练时 train+synth 合并
   → 完全确定性、可审计、可复现（不走在线增强）。

用法::

    python scripts/synth_adhesion.py --data datasets/pest24 --out datasets/pest24_synth \
        --n-samples 2000 --seed 0

依赖：numpy + Pillow。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"需要 numpy：pip install numpy ({exc})")

try:
    from PIL import Image, ImageFilter
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"需要 Pillow：pip install pillow ({exc})")

DEFAULT_IMG_W, DEFAULT_IMG_H = 800, 600
CLASSES = [
    "Bollworm", "Meadow borer", "Gryllotalpa orientalis", "Little Gecko",
    "Agriotes fuscicollis Miwa", "Nematode trench", "Athetis lepigone",
    "Scotogramma trifolii Rottemberg", "Armyworm", "Spodoptera cabbage",
    "Anomala corpulenta", "Spodoptera exigua", "Plutella xylostella",
    "holotrichia parallela", "Rice planthopper", "Yellow tiger",
    "Land tiger", "eight-character tiger", "holotrichia oblita", "Stem borer",
    "Striped rice bore", "Rice Leaf Roller", "Spodoptera litura", "Melahotus",
]
BETA_THR = 0.05
DENSE_INSTANCE_THR = 20
SOURCE_MARGIN = 0.06  # 实例裁剪时向外扩的比例，避免把背景一起贴过去


# --------------------------------------------------------------------------- #
# 标签读取
# --------------------------------------------------------------------------- #
def read_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """→ [(cls, xyxy 像素框)]。"""
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
        out.append(
            (
                c,
                (
                    (cx - bw / 2) * img_w,
                    (cy - bh / 2) * img_h,
                    (cx + bw / 2) * img_w,
                    (cy + bh / 2) * img_h,
                ),
            )
        )
    return out


def _inter(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def adhesion(a, b) -> float:
    inter = _inter(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# 统计：(类别, 尺度, β) 联合分布
# --------------------------------------------------------------------------- #
def collect_stats(
    labels_dir: Path, img_w: int, img_h: int
) -> Tuple[Dict[int, List[Tuple[float, float]]], List[float], List[str]]:
    """返回 (每类的 (尺度, β) 列表, 全局 β 列表, 可用图片列表)。"""
    per_class: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    betas: List[float] = []
    images: List[str] = []
    area_img = float(img_w * img_h)

    for lp in sorted((labels_dir / "train").glob("*.txt")):
        items = read_labels(lp, img_w, img_h)
        if not items:
            continue
        images.append(lp.stem)
        for i, (ci, bi) in enumerate(items):
            rel = _area(bi) / area_img
            best = 0.0
            for j, (_cj, bj) in enumerate(items):
                if i == j:
                    continue
                b = adhesion(bi, bj)
                if b > best:
                    best = b
            per_class[ci].append((rel, best))
            betas.append(best)
    return per_class, betas, images


# --------------------------------------------------------------------------- #
# 粘连位置求解
# --------------------------------------------------------------------------- #
def solve_offset_for_beta(
    w1: float, h1: float, w2: float, h2: float, target_beta: float
) -> Optional[Tuple[float, float]]:
    """求两个矩形（尺寸固定）的相对平移，使 IoU(=β) 命中目标值。

    在 x 方向做一维搜索（y 方向保持 0），因为沿对角线平移会更快降低 IoU，
    单轴搜索即可覆盖 [0, β_max] 的全部取值。
    """
    if target_beta <= 0:
        return (max(w1, w2) + 2.0, 0.0)
    # β 的上限：完全重合时 IoU = min(a1,a2)/max(a1,a2)
    a1, a2 = w1 * h1, w2 * h2
    beta_max = min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0.0
    if target_beta > beta_max:
        return None  # 该尺寸组合无法达到目标 β

    lo, hi = 0.0, max(w1, w2) + 2.0

    def beta_at(dx: float) -> float:
        a = (0.0, 0.0, w1, h1)
        b = (dx, 0.0, dx + w2, h2)
        return adhesion(a, b)

    # beta_at(dx) 随 dx 单调递减 → 二分
    if beta_at(lo) < target_beta:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if beta_at(mid) > target_beta:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2.0, 0.0)


# --------------------------------------------------------------------------- #
# 实例裁剪与贴图
# --------------------------------------------------------------------------- #
def crop_instance(img: Image.Image, box, margin: float = SOURCE_MARGIN):
    """按框裁剪实例（外扩 margin），返回 (patch, 实际框)。"""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin, h * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(img.width, int(x2 + mx))
    y2 = min(img.height, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def paste_solid(canvas: Image.Image, patch: Image.Image, xy: Tuple[int, int]) -> None:
    """**实心**贴图（不使用 alpha 混合）—— 模拟两虫黏连后的边界融合。"""
    canvas.paste(patch, xy)


def add_union_shadow(canvas: Image.Image, boxes: Sequence[Tuple[int, int, int, int]]) -> None:
    """在并集外缘加一圈软阴影（模拟联合投影）。"""
    if not boxes:
        return
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad = max(2, int(0.12 * max(x2 - x1, y2 - y1)))
    sw, sh = canvas.size
    mask = Image.new("L", (sw, sh), 0)
    from PIL import ImageDraw

    d = ImageDraw.Draw(mask)
    d.ellipse([x1 - pad, y1 - pad, x2 + pad, y2 + pad], fill=110)
    d.rectangle([x1, y1, x2, y2], fill=0)  # 去掉内部，只留外缘
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, pad // 2)))
    dark = Image.new("RGB", (sw, sh), (0, 0, 0))
    canvas.paste(dark, (0, 0), mask)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="ADEA · I4 ASM 粘连合成")
    ap.add_argument("--data", required=True, help="Pest24 根目录（含 labels/ 与 images/）")
    ap.add_argument("--out", default="datasets/pest24_synth")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--img-h", type=int, default=DEFAULT_IMG_H)
    ap.add_argument("--dense-frac", type=float, default=0.089,
                    help="高密度合成图占比（论文：8.89%% 的图 >20 实例）")
    ap.add_argument("--no-shadow", action="store_true", help="不渲染联合阴影（降级路径）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    data = Path(args.data)
    labels_dir = data / "labels"
    images_dir = data / "images"
    if not (labels_dir / "train").exists():
        raise SystemExit(f"未找到 {labels_dir / 'train'}，请先运行 scripts/voc2yolo.py")

    out = Path(args.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)

    per_class, betas, image_ids = collect_stats(labels_dir, args.img_w, args.img_h)
    usable = [c for c in per_class if per_class[c]]
    if not usable:
        raise SystemExit("train split 无有效实例，无法合成")
    print(f"[ASM] 类别覆盖 {len(usable)}/{len(CLASSES)}；train 图片 {len(image_ids)} 张；"
          f"GT 对粘连 β 分布均值 {np.mean(betas):.4f}")

    # ---- 背景图池：优先选实例较少的图，避免背景本身过于拥挤 ----
    bg_pool: List[str] = []
    for stem in image_ids:
        n = len(read_labels(labels_dir / "train" / f"{stem}.txt", args.img_w, args.img_h))
        bg_pool.append((stem, n))
    bg_pool.sort(key=lambda t: t[1])

    # ---- 实例来源池：每类可用的 (图名, 框) ----
    inst_pool: Dict[int, List[Tuple[str, tuple]]] = defaultdict(list)
    for stem in image_ids:
        for c, b in read_labels(labels_dir / "train" / f"{stem}.txt", args.img_w, args.img_h):
            inst_pool[c].append((stem, b))

    manifest: List[Dict[str, object]] = []
    n_dense_target = int(round(args.n_samples * args.dense_frac))
    produced = 0

    for si in range(args.n_samples):
        want_dense = si < n_dense_target
        cls = rng.choice(usable)
        pool = inst_pool.get(cls, [])
        if len(pool) < 2:
            continue

        # 采样 (尺度 s, 粘连度 β)
        s_ref, b_ref = rng.choice(per_class[cls])
        if b_ref <= BETA_THR:
            # 该样本无粘连 → 从全局 β 分布采样一个低值（多数样本本就不粘连）
            b_ref = float(np_rng.choice(betas)) if betas else 0.0
        target_beta = float(np_rng.choice(betas)) if betas else 0.0
        # 合成图**刻意**偏向粘连（这是 ASM 的目的），但不超过分布上界
        target_beta = max(target_beta, min(b_ref + 0.05, 0.30))

        # 背景
        bg_idx = rng.randrange(min(len(bg_pool), max(1, len(bg_pool) // 4)))
        bg_stem = bg_pool[bg_idx][0]
        bg_path = images_dir / "train" / f"{bg_stem}.jpg"
        if not bg_path.exists():
            cand = list((images_dir / "train").glob(f"{bg_stem}.*"))
            if not cand:
                continue
            bg_path = cand[0]
        try:
            canvas = Image.open(bg_path).convert("RGB")
        except Exception:
            continue
        if canvas.size != (args.img_w, args.img_h):
            canvas = canvas.resize((args.img_w, args.img_h))

        placed: List[Tuple[int, Tuple[float, float, float, float]]] = []
        boxes_px: List[Tuple[int, int, int, int]] = []
        ok = True

        # ---- 第一个实例 ----
        src1, b1 = pool[rng.randrange(len(pool))]
        p1_path = images_dir / "train" / f"{src1}.jpg"
        if not p1_path.exists():
            continue
        try:
            img1 = Image.open(p1_path).convert("RGB")
        except Exception:
            continue
        patch1, box1 = crop_instance(img1, b1)
        if patch1 is None:
            continue
        w1, h1 = patch1.size

        # ---- 第二个实例（与第一个形成目标 β 的重叠）----
        src2, b2 = pool[rng.randrange(len(pool))]
        p2_path = images_dir / "train" / f"{src2}.jpg"
        if not p2_path.exists():
            continue
        try:
            img2 = Image.open(p2_path).convert("RGB")
        except Exception:
            continue
        patch2, box2 = crop_instance(img2, b2)
        if patch2 is None:
            continue
        w2, h2 = patch2.size

        off = solve_offset_for_beta(w1, h1, w2, h2, target_beta)
        if off is None:
            off = (0.0, 0.0)  # 尺寸不兼容时退化为不重叠
            target_beta = 0.0
        dx, dy = off

        # 放置位置（保证不出界）
        px1 = rng.randint(0, max(0, args.img_w - w1))
        py1 = rng.randint(0, max(0, args.img_h - h1))
        px2 = int(px1 + dx)
        py2 = int(py1 + dy)
        if px2 < 0 or py2 < 0 or px2 + w2 > args.img_w or py2 + h2 > args.img_h:
            px1 = rng.randint(0, max(0, args.img_w - w1 - int(abs(dx))))
            py1 = rng.randint(0, max(0, args.img_h - h1 - int(abs(dy))))
            px2 = int(px1 + dx)
            py2 = int(py1 + dy)
        if px2 < 0 or py2 < 0 or px2 + w2 > args.img_w or py2 + h2 > args.img_h:
            continue

        if not args.no_shadow:
            add_union_shadow(
                canvas,
                [
                    (px1, py1, px1 + w1, py1 + h1),
                    (px2, py2, px2 + w2, py2 + h2),
                ],
            )
        paste_solid(canvas, patch1, (px1, py1))
        paste_solid(canvas, patch2, (px2, py2))
        placed.append((cls, (px1, py1, px1 + w1, py1 + h1)))
        boxes_px.append((px1, py1, px1 + w1, py1 + h1))
        # ⚠️ 第二个实例的框也必须记录 —— 否则 achieved_beta 永远算不出来
        #    （曾经只记了第一个，导致统计显示"合成样本零粘连"，而实际是重叠的）
        boxes_px.append((px2, py2, px2 + w2, py2 + h2))

        # ---- 高密度场景：补齐到 >20 实例（论文口径）----
        if want_dense:
            extra_needed = DENSE_INSTANCE_THR + rng.randint(1, 10) - 2
            tries = 0
            while len(placed) < extra_needed and tries < 200:
                tries += 1
                c2 = rng.choice(usable)
                p2 = inst_pool.get(c2, [])
                if not p2:
                    continue
                s2, bx2 = p2[rng.randrange(len(p2))]
                ip = images_dir / "train" / f"{s2}.jpg"
                if not ip.exists():
                    continue
                try:
                    im = Image.open(ip).convert("RGB")
                except Exception:
                    continue
                pt, bxx = crop_instance(im, bx2)
                if pt is None:
                    continue
                ex = rng.randint(0, max(0, args.img_w - pt.size[0]))
                ey = rng.randint(0, max(0, args.img_h - pt.size[1]))
                paste_solid(canvas, pt, (ex, ey))
                placed.append((c2, (ex, ey, ex + pt.size[0], ey + pt.size[1])))

        # ---- 落盘 ----
        name = f"synth_{si:06d}_{bg_stem}"
        canvas.save(out / "images" / "train" / f"{name}.jpg", quality=95)
        lines = []
        for c, b in placed:
            cx = (b[0] + b[2]) / 2 / args.img_w
            cy = (b[1] + b[3]) / 2 / args.img_h
            bw = (b[2] - b[0]) / args.img_w
            bh = (b[3] - b[1]) / args.img_h
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        manifest.append(
            {
                "name": name,
                "background": bg_stem,
                "n_instances": len(placed),
                "target_beta": round(target_beta, 4),
                "achieved_beta": round(adhesion(boxes_px[0], boxes_px[1]), 4) if len(boxes_px) > 1 else 0.0,
                "classes": [c for c, _ in placed],
                "dense": len(placed) > DENSE_INSTANCE_THR,
            }
        )
        produced += 1

    # ---- 清单与统计 ----
    (out / "synth_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    achieved = [m["achieved_beta"] for m in manifest if m["achieved_beta"] > 0]
    stats = {
        "n_produced": produced,
        "n_dense": sum(1 for m in manifest if m["dense"]),
        "pct_dense": round(100.0 * sum(1 for m in manifest if m["dense"]) / max(1, produced), 2),
        "mean_achieved_beta": round(float(np.mean(achieved)), 4) if achieved else None,
        "pct_with_adhesion": round(100.0 * len(achieved) / max(1, produced), 2),
        "seed": args.seed,
    }
    (out / "synth_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ASM] 合成完成：", json.dumps(stats, ensure_ascii=False))
    print(f"[ASM] 输出：{out}")


if __name__ == "__main__":
    main()
