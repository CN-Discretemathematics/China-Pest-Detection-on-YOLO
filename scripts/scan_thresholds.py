#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""conf / iou 阈值扫描 —— **零训练成本**，确定最终部署点。

重要约定（见方案 §4.2）
------------------------
* **实验间对比一律使用默认 conf(0.001) / iou(0.7)**；
* 本脚本的扫描结果**只用于确定部署点**，**不参与**实验间对比。

用法::

    # conf 扫描（一次 val 即可从 PR 曲线得到全部阈值点）+ iou 扫描（需多次 val）
    python scripts/scan_thresholds.py --model runs/pest24/ft_aug/weights/best.pt \
        --data datasets/pest24/pest24.yaml --split val --out artifacts/scan

输出
----
* ``artifacts/scan/threshold_scan.md``  扫描表 + 部署点结论
* ``artifacts/scan/conf_curve.csv``      PR/F1 曲线数据
* ``artifacts/scan/iou_scan.csv``        iou 扫描数据
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple

CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
IOU_GRID = [0.45, 0.50, 0.60, 0.70]
TARGET_PRECISION = 0.80


def run_val(model_path: str, data: str, split: str, conf: float, iou: float, device: str = ""):
    """跑一次 val，返回 (metrics dict, curves)。"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    kwargs = dict(data=data, split=split, conf=conf, iou=iou, verbose=False)
    if device:
        kwargs["device"] = device
    r = model.val(**kwargs)
    box = r.box
    out = {
        "conf": conf,
        "iou": iou,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "fitness": float(box.fitness) if hasattr(box, "fitness") else None,
    }
    curves = {}
    for attr in ("p_curve", "r_curve", "f1_curve"):
        c = getattr(box, attr, None)
        if c is not None:
            curves[attr] = [list(map(float, arr)) for arr in c]
    return out, curves


def main() -> None:
    ap = argparse.ArgumentParser(description="conf/iou 阈值扫描（定部署点）")
    ap.add_argument("--model", required=True, help="待扫描的 best.pt")
    ap.add_argument("--data", required=True, help="数据集 yaml")
    ap.add_argument("--split", default="val", help="评测 split（val 或 test）")
    ap.add_argument("--out", default="artifacts/scan")
    ap.add_argument("--device", default="")
    ap.add_argument("--skip-iou", action="store_true", help="只做 conf 扫描（更快）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) 一次默认 val：拿到 PR/F1 曲线（可离线读出任意 conf 点的 P/R）----
    base, curves = run_val(args.model, args.data, args.split, 0.001, 0.7, args.device)
    print(f"[base] P={base['precision']:.4f} R={base['recall']:.4f} "
          f"mAP50={base['mAP50']:.4f} mAP50-95={base['mAP50-95']:.4f}")

    conf_rows: List[dict] = []
    if "f1_curve" in curves:
        x, y, p, r = curves["f1_curve"]
        for xi, yi, pi, ri in zip(x, y, p, r):
            conf_rows.append({"conf": xi, "f1": yi, "precision": pi, "recall": ri})
        with (out / "conf_curve.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["conf", "f1", "precision", "recall"])
            w.writeheader()
            w.writerows(conf_rows)

        best_f1_idx = max(range(len(y)), key=lambda i: y[i])
        conf_f1 = float(x[best_f1_idx])
        p_at_f1 = float(p[best_f1_idx])
        r_at_f1 = float(r[best_f1_idx])

        # 满足目标 Precision 的最小 conf
        conf_target = None
        for xi, pi, ri in sorted(conf_rows, key=lambda t: t["conf"]):
            if pi >= TARGET_PRECISION:
                conf_target = float(xi)
                break
    else:
        conf_f1, p_at_f1, r_at_f1, conf_target = None, None, None, None

    # ---- 2) iou 扫描（每个 iou 一次 val）----
    iou_rows: List[dict] = []
    if not args.skip_iou:
        for iou in IOU_GRID:
            m, _ = run_val(args.model, args.data, args.split, 0.001, iou, args.device)
            iou_rows.append(m)
            print(f"[iou={iou}] P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50-95']:.4f}")
        with (out / "iou_scan.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["conf", "iou", "precision", "recall", "mAP50", "mAP50-95", "fitness"])
            w.writeheader()
            w.writerows(iou_rows)

    best_iou_row = max(iou_rows, key=lambda r: r["mAP50"]) if iou_rows else None

    # ---- 3) 结论 ----
    lines = [
        "# conf / iou 扫描结论（部署点）",
        "",
        "> ⚠️ 本表**只用于确定部署点**，不参与实验间对比（实验间对比一律用默认 conf=0.001 / iou=0.7）。",
        "",
        "## 基线（默认阈值）",
        "",
        "| conf | iou | P | R | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|---|",
        f"| {base['conf']} | {base['iou']} | {base['precision']:.4f} | {base['recall']:.4f} "
        f"| {base['mAP50']:.4f} | {base['mAP50-95']:.4f} |",
        "",
    ]
    if conf_f1 is not None:
        lines += [
            "## conf 扫描（从 PR 曲线离线读取）",
            "",
            f"- **F1 最优 conf = {conf_f1:.4f}**（P={p_at_f1:.4f}, R={r_at_f1:.4f}）",
            f"- 满足 P ≥ {TARGET_PRECISION:.2f} 的最小 conf = "
            f"{('%.4f' % conf_target) if conf_target is not None else '未达到'}",
            "",
            "| conf | P | R | F1 |",
            "|---|---|---|---|",
        ]
        for row in sorted(conf_rows, key=lambda t: -t["conf"])[:40]:
            lines.append(f"| {row['conf']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")
        lines.append("")
    if iou_rows:
        lines += [
            "## iou 扫描（每个 iou 一次 val）",
            "",
            "| iou | P | R | mAP50 | mAP50-95 |",
            "|---|---|---|---|---|",
        ]
        for r in iou_rows:
            lines.append(
                f"| {r['iou']} | {r['precision']:.4f} | {r['recall']:.4f} "
                f"| {r['mAP50']:.4f} | {r['mAP50-95']:.4f} |"
            )
        lines.append("")
    if conf_f1 is not None and best_iou_row is not None:
        lines += [
            "## 部署点结论",
            "",
            f"**推荐部署 conf = {conf_f1:.3f}，iou = {best_iou_row['iou']}**",
            "",
            f"- 该点 mAP50 = {best_iou_row['mAP50']:.4f}，mAP50-95 = {best_iou_row['mAP50-95']:.4f}",
            f"- 若业务更看重 Precision（如现场预警误报成本高），改用 conf = "
            f"{('%.3f' % conf_target) if conf_target is not None else 'N/A'}",
            "",
        ]
    (out / "threshold_scan.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "scan_summary.json").write_text(
        json.dumps(
            {
                "base": base,
                "conf_f1_optimal": conf_f1,
                "conf_for_target_precision": conf_target,
                "target_precision": TARGET_PRECISION,
                "iou_rows": iou_rows,
                "best_iou": best_iou_row["iou"] if best_iou_row else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\n已写出：{out / 'threshold_scan.md'}")


if __name__ == "__main__":
    main()
