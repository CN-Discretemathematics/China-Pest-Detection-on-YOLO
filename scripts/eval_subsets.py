#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定向子集评测 —— 产出 ``artifacts/eval/<run>.json``（供 ``scripts/summarize.py`` 汇总）。

为什么需要它
------------
验收标准 **B2「定向性」** 要求：每个创新组件在**它自己的目标子集**上必须有提升。
但 ``summarize.py`` 只**读取** ``subset_*`` 字段，没有任何脚本产出它们 —— 本脚本补上这一环。

四个子集（口径固定，全程不变；与 `docs/方法设计_ADEA.md` §9.1 一致）
----------------------------------------------------------------
| 子集 | 定义 | 怎么算 mAP | 对应创新 |
|---|---|---|---|
| 粘连子集 | 存在 GT 对 β > 0.05（论文定义）的**图** | 在这些图上重跑一次 val | I1 SLA / I4 ASM |
| 高密度子集 | 单图实例数 > 20 的**图** | 同上 | I2 DLA |
| 稀有类子集 | 全库实例数 < 500 的**类别** | 该子集各类别 AP 的平均 | I3 DRA |
| 极小类子集 | 全库平均相对尺度 < 0.05% 的**类别** | 同上 | I4 P2 头 / I7 天花板 |

用法
----
::

    python scripts/eval_subsets.py --model runs/pest24/strong_SLA/weights/best.pt \
        --data datasets/pest24/pest24.yaml --split test \
        --out artifacts/eval --run-name strong_SLA

产物：``artifacts/eval/<run-name>.json``，字段与 `scripts/summarize.py` 的期望完全一致：
``run / model / imgsz / epochs / batch / seed / precision / recall / map50 / map50_95 /
subset_adhesion / subset_dense / subset_rare / subset_tiny``。

⚠️ **口径提醒**
------------
* 评测一律使用 **EVAL_SPLIT = test**（7,600 张，与论文 5:2:3 一致）；
* 实验间对比一律用默认 ``conf=0.001 / iou=0.7``（不要用扫出来的部署点做对比）；
* 图像子集是**整图参与评测**（子集图里的所有目标都算），这与 Pest24 原论文
  "粘连子集 mAP 比非粘连子集低 4.96 个点" 的口径一致（按图划分子集）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.diagnose_adhesion import (  # noqa: E402
    BETA_THR,
    CLASSES,
    DEFAULT_IMG_H,
    DEFAULT_IMG_W,
    DENSE_INSTANCE_THR,
    RARE_INSTANCE_THR,
    TINY_RELATIVE_SCALE,
    adhesion,
    read_labels,
)

IMG_AREA = float(DEFAULT_IMG_W * DEFAULT_IMG_H)


# --------------------------------------------------------------------------- #
# 数据集解析
# --------------------------------------------------------------------------- #
def read_yaml_simple(path: Path) -> Dict[str, object]:
    """极简 yaml 读取（只处理 path/train/val/test/names 顶层键，避免引入 pyyaml 依赖差异）。"""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    out: Dict[str, object] = {}
    key = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if raw.startswith((" ", "\t")) and key:
            out[key] = raw.strip()
            continue
        if ":" in raw:
            key, _, val = raw.partition(":")
            key = key.strip()
            out[key] = val.strip()
    return out


def resolve_split_dir(data_yaml: Path, split: str) -> Path:
    """把 yaml 里的 split 条目解析成实际的图片目录。"""
    cfg = read_yaml_simple(data_yaml)
    raw = str(cfg.get(split, "") or "").strip()
    if not raw:
        raise SystemExit(f"[!] {data_yaml} 里没有 {split}: 条目")
    p = Path(raw)
    if not p.is_absolute():
        p = (data_yaml.parent / p).resolve()
    return p


def collect_subset_stats(labels_root: Path) -> Dict[str, object]:
    """扫描全部 split 的标签，算出 4 个子集的成员。"""
    adhesion_imgs: List[str] = []
    dense_imgs: List[str] = []
    per_class_inst: Dict[int, int] = {}
    per_class_area: Dict[int, List[float]] = {}
    img_of_ann: Dict[str, str] = {}

    for split_dir in sorted(labels_root.glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for lp in sorted(split_dir.glob("*.txt")):
            items = read_labels(lp, DEFAULT_IMG_W, DEFAULT_IMG_H)
            key = f"{split}/{lp.stem}"
            img_of_ann[key] = key
            if len(items) > DENSE_INSTANCE_THR:
                dense_imgs.append(key)
            has_adh = False
            for i in range(len(items)):
                ci, bi = items[i]
                per_class_inst[ci] = per_class_inst.get(ci, 0) + 1
                per_class_area.setdefault(ci, []).append(
                    max(0.0, bi[2] - bi[0]) * max(0.0, bi[3] - bi[1]) / IMG_AREA
                )
                for j in range(i + 1, len(items)):
                    if adhesion(bi, items[j][1]) > BETA_THR:
                        has_adh = True
                        break
                if has_adh:
                    break
            if has_adh:
                adhesion_imgs.append(key)

    rare_classes = sorted(c for c, n in per_class_inst.items() if n < RARE_INSTANCE_THR)
    tiny_classes = sorted(
        c
        for c, areas in per_class_area.items()
        if areas and sum(areas) / len(areas) < TINY_RELATIVE_SCALE
    )
    return {
        "adhesion_images": adhesion_imgs,
        "dense_images": dense_imgs,
        "rare_classes": rare_classes,
        "tiny_classes": tiny_classes,
        "per_class_instances": per_class_inst,
    }


# --------------------------------------------------------------------------- #
# 子集评测
# --------------------------------------------------------------------------- #
def write_subset_yaml(
    img_dir: Path, keys: Sequence[str], names: Sequence[str], out_dir: Path, tag: str
) -> Optional[Tuple[Path, int]]:
    """为一批图生成"val: 文本清单"的临时 yaml；没有图则返回 None。

    Returns
    -------
    (yaml_path, n_picked) —— ``n_picked`` 是**本次实际评测**的图数
    （可能小于子集在全库中的规模，因为评测集本身可能只覆盖其中一部分）。
    """
    if not keys:
        return None
    stem2path = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            stem2path.setdefault(p.stem, p)
    picked: List[str] = []
    for k in keys:
        stem = k.split("/")[-1]
        p = stem2path.get(stem)
        if p is not None:
            picked.append(str(p))
    if not picked:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    list_txt = (out_dir / f"{tag}.txt").resolve()
    list_txt.write_text("\n".join(picked) + "\n", encoding="utf-8")
    yaml_path = out_dir / f"{tag}.yaml"
    # ⚠️ ultralytics 会把 `val:` 相对 `path:` 拼接 → 这里必须写**绝对路径**
    lines = [f"path: {img_dir}", f"train: {list_txt}", f"val: {list_txt}", "names:"]
    for i, n in enumerate(names):
        lines.append(f"  {i}: '{n}'")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path, len(picked)


def val_metrics(model, data_yaml: Path, imgsz: int, batch: int, device: str, workers: int):
    """跑一次 val，返回 (整体指标 dict, 逐类 AP 数组, 出现的类别下标)。"""
    res = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=False,
        save_json=False,
        verbose=False,
    )
    box = res.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }
    all_ap = getattr(box, "all_ap", None)
    cls_raw = getattr(box, "ap_class_index", None)
    cls_idx = [] if cls_raw is None else [int(c) for c in list(cls_raw)]
    return overall, all_ap, cls_idx


def subset_class_ap(all_ap, cls_idx: Sequence[int], classes: Sequence[int]) -> Dict[str, float]:
    """子集内各类别 AP 的平均（mAP50 / mAP50-95）。"""
    if all_ap is None or len(cls_idx) == 0:
        return {"map50": float("nan"), "map50_95": float("nan")}
    import numpy as np

    arr = np.asarray(all_ap, dtype=float)  # (nc_present, 10)
    ap50 = {int(c): float(arr[i, 0]) for i, c in enumerate(cls_idx)}
    ap_all = {int(c): float(arr[i].mean()) for i, c in enumerate(cls_idx)}
    vals50 = [ap50[c] for c in classes if c in ap50]
    valsall = [ap_all[c] for c in classes if c in ap_all]
    return {
        "map50": (sum(vals50) / len(vals50)) if vals50 else float("nan"),
        "map50_95": (sum(valsall) / len(valsall)) if valsall else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="定向子集评测（产出 artifacts/eval/<run>.json）")
    ap.add_argument("--model", required=True, help="权重路径（best.pt）")
    ap.add_argument("--data", required=True, help="数据集 yaml")
    ap.add_argument("--split", default="test", help="评测 split（默认 test，与 EVAL_SPLIT 一致）")
    ap.add_argument("--out", default="artifacts/eval", help="JSON 输出目录")
    ap.add_argument("--run-name", default=None, help="台账里的实验名（默认用权重父目录名）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--stats-root",
        default=None,
        help="类别子集统计所用的 labels 根目录（默认 = 数据集 yaml 同级的 labels/）。"
        "口径固定：应指向**全库**标签，保证稀有类/极小类的成员不随评测集变化",
    )
    args = ap.parse_args()

    data_yaml = Path(args.data)
    weights = Path(args.model)
    if not weights.exists():
        print(f"[!] 权重不存在: {weights}")
        return 2

    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = list(model.names.values()) if hasattr(model, "names") else CLASSES
    img_dir = resolve_split_dir(data_yaml, args.split)
    if args.stats_root:
        labels_root = Path(args.stats_root)
    else:
        labels_root = data_yaml.parent / "labels"
        if not labels_root.exists():
            labels_root = img_dir.parent / "labels"

    stats = collect_subset_stats(labels_root)
    print(
        f"[eval] 子集规模：粘连图 {len(stats['adhesion_images'])}、密集图 {len(stats['dense_images'])}、"
        f"稀有类 {[CLASSES[c] for c in stats['rare_classes']]}、"
        f"极小类 {[CLASSES[c] for c in stats['tiny_classes']]}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = out_dir / "_subsets"

    report: Dict[str, object] = {
        "run": args.run_name or weights.parent.parent.name,
        "model": str(weights),
        "imgsz": args.imgsz,
        "seed": args.seed,
        "split": args.split,
    }

    # ---- 全 split：整体指标 + 逐类 AP（供稀有类/极小类子集）----
    overall, all_ap, cls_idx = val_metrics(
        model, data_yaml, args.imgsz, args.batch, args.device, args.workers
    )
    report.update(overall)
    print(f"[eval] {args.split} 整体：P={overall['precision']:.4f} R={overall['recall']:.4f} "
          f"mAP50={overall['map50']:.4f} mAP50-95={overall['map50_95']:.4f}")

    rare = subset_class_ap(all_ap, cls_idx, stats["rare_classes"])
    tiny = subset_class_ap(all_ap, cls_idx, stats["tiny_classes"])
    report["subset_rare"] = rare["map50"]
    report["subset_tiny"] = tiny["map50"]
    report["subset_rare_map50_95"] = rare["map50_95"]
    report["subset_tiny_map50_95"] = tiny["map50_95"]

    # ---- 图像子集：重跑 val ----
    for tag, key, label in (
        ("adhesion", "adhesion_images", "粘连子集"),
        ("dense", "dense_images", "高密度子集"),
    ):
        sub = write_subset_yaml(img_dir, stats[key], names, sub_dir, tag)
        if sub is None:
            print(f"[eval] {label} 在本次评测集里没有图，跳过")
            report[f"subset_{tag}"] = float("nan")
            continue
        sub_yaml, n_picked = sub
        m, _, _ = val_metrics(model, sub_yaml, args.imgsz, args.batch, args.device, args.workers)
        report[f"subset_{tag}"] = m["map50"]
        report[f"subset_{tag}_map50_95"] = m["map50_95"]
        report[f"subset_{tag}_precision"] = m["precision"]
        report[f"subset_{tag}_recall"] = m["recall"]
        report[f"subset_{tag}_n_images"] = n_picked
        report[f"subset_{tag}_n_images_full_dataset"] = len(stats[key])
        print(
            f"[eval] {label}（本次评测 {n_picked} 张 / 全库 {len(stats[key])} 张）："
            f"mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f}"
        )

    out_json = out_dir / f"{report['run']}.json"
    # NaN → None（严格 JSON 不允许 NaN）；并补上 summarize.py 期望的键名别名
    clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in report.items()}
    if clean.get("map50") is not None:
        clean["mAP50"] = clean["map50"]
    if clean.get("map50_95") is not None:
        clean["mAP50-95"] = clean["map50_95"]
    out_json.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 报告已写入: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
