#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按**目标尺寸**分档的 AP（COCO 口径：AP_S / AP_M / AP_L）—— 尺度公平主张的标准指标。

为什么必须加这个
----------------
本项目的核心主张是"**监督预算被目标尺度决定**"，
而 COCO 体系里与"尺度"直接对应的官方指标就是
``AP_S / AP_M / AP_L``（小 / 中 / 大目标的 AP，面积阈值 32² / 96² 像素）。
只报整体 mAP，读者无法判断"提升是不是真的来自微目标"；
只报我们的机制指标（n_pos、缺口），又属于自定义量，审稿人不会买账。

本脚本用 **pycocotools（COCOeval）** 走官方口径：
1. 把该 split 的 YOLO 标签转成 COCO GT json；
2. 跑一次 ``model.val(save_json=True)`` 让 ultralytics 输出 COCO 格式预测；
3. COCOeval 求 AP / AP50 / AP75 / **AP_S / AP_M / AP_L**；
4. 写回 ``artifacts/eval/<run>.json``（与 ``eval_subsets.py`` 同一份报告）。

用法
----
::

    python scripts/eval_ap_by_size.py --model runs/pest24/strong_SLA/weights/best.pt \
        --data datasets/pest24/pest24.yaml --split test --run-name strong_SLA

依赖：``pycocotools``（``pip install pycocotools``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_subsets import read_yaml_simple, resolve_split_dir  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_label_dir(img_dir: Path) -> Path:
    """由图片目录推出标签目录（datasets/<x>/images/<split> → datasets/<x>/labels/<split>）。"""
    cands = [
        img_dir.parent.parent / "labels" / img_dir.name,
        img_dir.parent / "labels",
        img_dir.parent.parent / "labels",
    ]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit(f"[!] 找不到标签目录（试过：{[str(c) for c in cands]}）")


def yolo_to_coco_gt(
    img_dir: Path, label_dir: Path, names: List[str], out_json: Path
) -> Dict[str, int]:
    """把 YOLO txt 标签转成 COCO GT json。image_id 用**文件名词干**（与 ultralytics 一致）。"""
    from PIL import Image

    images: List[dict] = []
    annotations: List[dict] = []
    ann_id = 1
    img_files = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    for p in img_files:
        stem = p.stem
        # ⚠️ 必须与 ultralytics 的 image_id 约定完全一致：
        #    `int(stem) if stem.isnumeric() else stem`（见 ultralytics/models/yolo/detect/val.py）
        img_id = int(stem) if stem.isnumeric() else stem
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        images.append({"id": img_id, "file_name": str(p), "width": w, "height": h})
        lp = label_dir / f"{stem}.txt"
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            bw_px, bh_px = bw * w, bh * h
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": c + 1,
                    "bbox": [x, y, bw_px, bh_px],
                    "area": bw_px * bh_px,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
    gt = {"images": images, "annotations": annotations, "categories": categories}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    print(f"[ap] GT: {len(images)} 张图 / {len(annotations)} 个标注 → {out_json}")
    return {"n_images": len(images), "n_annotations": len(annotations)}


def main() -> int:
    ap = argparse.ArgumentParser(description="按目标尺寸分档的 AP（COCO AP_S/AP_M/AP_L）")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:
        print(f"[!] 需要 pycocotools：pip install pycocotools（{exc}）")
        return 2

    data_yaml = Path(args.data)
    weights = Path(args.model)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = list(model.names.values())
    img_dir = resolve_split_dir(data_yaml, args.split)
    label_dir = find_label_dir(img_dir)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_apsize"
    work.mkdir(parents=True, exist_ok=True)

    gt_json = work / "gt.json"
    yolo_to_coco_gt(img_dir, label_dir, names, gt_json)

    # ---- 让 ultralytics 输出 COCO 格式预测 ----
    # ⚠️ project/name 必须是**绝对路径**，否则 ultralytics 会解析到 runs/detect/ 下
    save_dir = (work / "pred").resolve()
    res = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        save_json=True,
        project=str(save_dir),
        name="val",
        exist_ok=True,
        verbose=False,
    )
    pred_json = save_dir / "val" / "predictions.json"
    if not pred_json.exists():
        cand = list(save_dir.rglob("predictions.json"))
        if not cand:
            print("[!] 没找到 predictions.json")
            return 2
        pred_json = cand[0]
    print(f"[ap] 预测: {pred_json}")

    # ---- COCOeval ----
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    ap_coco, ap50, ap75, ap_s, ap_m, ap_l = [float(stats[i]) for i in range(6)]

    report_path = out_dir / f"{args.run_name or weights.parent.parent.name}.json"
    report: Dict[str, object] = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {}
    report.setdefault("run", args.run_name or weights.parent.parent.name)
    report.update(
        {
            "ap_coco": ap_coco,
            "ap50_coco": ap50,
            "ap75_coco": ap75,
            "ap_small": ap_s,
            "ap_medium": ap_m,
            "ap_large": ap_l,
            "ap_size_source": "pycocotools COCOeval (areaRng: small<32^2, medium 32^2~96^2, large>96^2 px)",
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[ap] COCO 口径：AP={ap_coco:.4f} AP50={ap50:.4f} AP75={ap75:.4f} | "
        f"AP_S={ap_s:.4f} AP_M={ap_m:.4f} AP_L={ap_l:.4f}"
    )
    print(f"[ap] 已写入: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
