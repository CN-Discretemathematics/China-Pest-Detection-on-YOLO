#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实验台账汇总：解析 ``runs/*/results.csv`` 与 ``args.yaml``，生成 results.md 表格。

用法::

    python scripts/summarize.py --runs runs/pest24 --out results.md
    python scripts/summarize.py --runs runs/pest24 --out results.md --eval-json artifacts/eval/*.json

产物
----
* ``results.md`` —— 统一口径的实验台账表（方案 §8.4 交付物 1）
* 表格列：实验 / 改动 / imgsz / epochs / seed / eval split / P / R / mAP50 / mAP50-95 / 耗时 / 参数量 / FLOPs / 结论

⚠️ 口径提醒
-----------
跨论文的数值**不可直接对表**（划分/分辨率/imgsz/轮数各异），
本表只汇总**本项目内部**同口径实验。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _last_row(rows: List[Dict[str, str]]) -> Dict[str, str]:
    return rows[-1] if rows else {}


def _f(row: Dict[str, str], *names: str) -> Optional[float]:
    for n in names:
        if n in row and row[n] not in ("", None):
            try:
                return float(row[n])
            except ValueError:
                continue
    return None


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.2f}"


def collect(runs_root: Path) -> List[Dict[str, object]]:
    """遍历 runs 目录，收集每个实验的 summary。"""
    out: List[Dict[str, object]] = []
    if not runs_root.exists():
        return out

    for results_csv in sorted(runs_root.rglob("results.csv")):
        run_dir = results_csv.parent
        rows = _read_csv(results_csv)
        if not rows:
            continue
        last = _last_row(rows)

        args_yaml = run_dir / "args.yaml"
        args: Dict[str, object] = {}
        if args_yaml.exists():
            try:
                import yaml  # type: ignore

                args = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
            except Exception:
                args = {}

        name = run_dir.name
        parent = run_dir.parent.name
        out.append(
            {
                "run": f"{parent}/{name}" if parent not in ("", ".") else name,
                "model": args.get("model", "?"),
                "imgsz": args.get("imgsz", "?"),
                "epochs": args.get("epochs", "?"),
                "batch": args.get("batch", "?"),
                "seed": args.get("seed", "?"),
                "precision": _f(last, "metrics/precision(B)", "metrics/precision"),
                "recall": _f(last, "metrics/recall(B)", "metrics/recall"),
                "map50": _f(last, "metrics/mAP50(B)", "metrics/mAP50"),
                "map50_95": _f(last, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
                "time_h": round(_f(last, "time") or 0.0, 3),
                "has_best": (run_dir / "weights" / "best.pt").exists(),
            }
        )
    return out


def merge_eval_json(items: List[Dict[str, object]], eval_dir: Optional[Path]) -> None:
    """把 ``yolo val`` 导出的 JSON 合并进表格（可选）。"""
    if not eval_dir or not eval_dir.exists():
        return
    by_run = {str(i["run"]): i for i in items}
    for jp in sorted(eval_dir.glob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        run = str(data.get("run") or jp.stem)
        target = by_run.get(run)
        if target is None:
            target = {"run": run, "model": data.get("model", "?"), "imgsz": data.get("imgsz", "?"),
                      "epochs": data.get("epochs", "?"), "batch": data.get("batch", "?"),
                      "seed": data.get("seed", "?"), "time_h": data.get("time_h"),
                      "has_best": data.get("has_best")}
            items.append(target)
            by_run[run] = target
        for key, jkey in (
            ("precision", "precision"),
            ("recall", "recall"),
            ("map50", "mAP50"),
            ("map50_95", "mAP50-95"),
        ):
            if data.get(jkey) is not None:
                target[key] = float(data[jkey])


def render(items: List[Dict[str, object]], out_path: Path, eval_split: str = "val") -> str:
    lines = [
        "# 实验台账（results.md）",
        "",
        "> 本表仅汇总**本项目内部**同口径实验。跨论文数值不可直接对表",
        "> （划分/分辨率/imgsz/轮数各异，见 `docs/调研报告_农业害虫检测前沿.md`）。",
        "",
        f"**评测集 EVAL_SPLIT = {eval_split}**（实验间对比一律使用默认 conf=0.001 / iou=0.7）",
        "",
        "| 实验 | model | imgsz | epochs | batch | seed | eval P | eval R | eval mAP50 | eval mAP50-95 | 耗时(h) | best.pt |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for it in items:
        lines.append(
            f"| {it.get('run')} | {it.get('model')} | {it.get('imgsz')} | {it.get('epochs')} "
            f"| {it.get('batch')} | {it.get('seed')} | {_pct(it.get('precision'))} "
            f"| {_pct(it.get('recall'))} | {_pct(it.get('map50'))} | {_pct(it.get('map50_95'))} "
            f"| {it.get('time_h')} | {'✅' if it.get('has_best') else '—'} |"
        )
    lines.append("")

    # 定向子集列（若有）
    sub = [i for i in items if any(k.startswith("subset_") for k in i.keys())]
    if sub:
        lines += [
            "## 定向子集 mAP（每个创新组件必须在其目标子集上有提升）",
            "",
            "| 实验 | 粘连子集 | 高密度子集 | 稀有类子集 | 极小类子集 |",
            "|---|---|---|---|---|",
        ]
        for it in sub:
            lines.append(
                f"| {it.get('run')} | {_pct(it.get('subset_adhesion'))} | {_pct(it.get('subset_dense'))} "
                f"| {_pct(it.get('subset_rare'))} | {_pct(it.get('subset_tiny'))} |"
            )
        lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总 runs/ 为 results.md")
    ap.add_argument("--runs", default="runs", help="训练输出根目录")
    ap.add_argument("--out", default="results.md")
    ap.add_argument("--eval-json", default=None, help="可选：yolo val 导出的 JSON 目录")
    ap.add_argument("--eval-split", default="val")
    args = ap.parse_args()

    items = collect(Path(args.runs))
    if args.eval_json:
        merge_eval_json(items, Path(args.eval_json))
    text = render(items, Path(args.out), args.eval_split)
    print(text)
    print(f"已写出：{args.out}（{len(items)} 条实验记录）")


if __name__ == "__main__":
    main()
