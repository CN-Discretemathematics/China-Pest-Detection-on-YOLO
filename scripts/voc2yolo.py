#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VOC XML → YOLO txt 转换（Pest24）。

来源：本项目 ``docs/Pest24_YOLO11n_部署与微调指南.md`` §4.4。

用法::

    python scripts/voc2yolo.py --src <Pest24/VOCdevkit/voc2007> --dst datasets/pest24

要点
----
* 自动发现 XML，兼容三种常见布局：
  - ``{split}/`` 目录混放；
  - ``xml|Annotations/ + ImageSets(/Main)/{split}.txt``（Kaggle 镜像实际布局）；
  - VOC 标准 ``ImageSets/Main`` 布局。
* 跳过 ``difficult=1`` 标注（社区通行做法）。
* 坐标越界裁剪到 ``[0, 1]``，避免训练时报 label 警告。
* 未知名类别自动跳过并统计（防止类别名拼写不一致静默丢数据）。
* 输出每类实例数统计（观察长尾分布）。
* 官方仅公开 train/val 标注；**不会**自行从 train 切 test。
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Pest24 的 24 类 —— 与 Kaggle 版 XML 中的真实类名一致
# （顺序自定但固定，训练用 0-23 编号，与论文原始编号无对应关系）
CLASSES = [
    "Bollworm", "Meadow borer", "Gryllotalpa orientalis", "Little Gecko",
    "Agriotes fuscicollis Miwa", "Nematode trench", "Athetis lepigone",
    "Scotogramma trifolii Rottemberg", "Armyworm", "Spodoptera cabbage",
    "Anomala corpulenta", "Spodoptera exigua", "Plutella xylostella",
    "holotrichia parallela", "Rice planthopper", "Yellow tiger",
    "Land tiger", "eight-character tiger", "holotrichia oblita", "Stem borer",
    "Striped rice bore", "Rice Leaf Roller", "Spodoptera litura", "Melahotus",
]
NAME2ID = {n.strip().lower(): i for i, n in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert_split(src: Path, dst: Path, split: str, stats: dict, unknown: set) -> int:
    """转换单个 split，返回图片数。"""
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    xmls: list[Path] = []
    # 布局 1：{split}/ 目录混放
    if (src / split).exists():
        xmls = sorted((src / split).rglob("*.xml"))
    # 布局 2：xml|Annotations + ImageSets(/Main)/{split}.txt（Kaggle 实际布局）
    if not xmls:
        ann = src / "xml" if (src / "xml").exists() else src / "Annotations"
        sets = next(
            (
                p
                for p in (
                    src / "ImageSets" / f"{split}.txt",
                    src / "ImageSets" / "Main" / f"{split}.txt",
                )
                if p.exists()
            ),
            None,
        )
        if ann.exists() and sets is not None:
            ids = sets.read_text(encoding="utf-8").split()
            xmls = [ann / f"{i}.xml" for i in ids if (ann / f"{i}.xml").exists()]
    assert xmls, f"未找到 {split} 划分的 XML，请检查 --src 目录结构"

    n_img = 0
    for xp in xmls:
        root = ET.parse(xp).getroot()
        w = int(root.findtext("size/width", "0"))
        h = int(root.findtext("size/height", "0"))
        assert w > 0 and h > 0, f"{xp} 缺少有效 size"

        lines: list[str] = []
        for obj in root.findall("object"):
            diff = obj.findtext("difficult") or obj.findtext("Difficult") or "0"
            if diff.strip() and int(float(diff)) == 1:
                continue  # 跳过 difficult=1
            name = obj.findtext("name", "").strip().lower()
            if name not in NAME2ID:
                unknown.add(name)
                continue
            b = obj.find("bndbox")
            xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
            xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
            cx = min(max((xmin + xmax) / 2 / w, 0.0), 1.0)
            cy = min(max((ymin + ymax) / 2 / h, 0.0), 1.0)
            bw = min(max((xmax - xmin) / w, 0.0), 1.0)
            bh = min(max((ymax - ymin) / h, 0.0), 1.0)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{NAME2ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats[split][NAME2ID[name]] = stats[split].get(NAME2ID[name], 0) + 1

        # 找同名图片：images/ 目录 → XML 同目录 → 全 src 搜索
        img = next((p for e in IMG_EXTS if (p := src / "images" / f"{xp.stem}{e}").exists()), None)
        if img is None:
            img = next((p for e in IMG_EXTS if (p := xp.with_suffix(e)).exists()), None)
        if img is None:
            img = next(iter(src.rglob(xp.stem + ".*")), None)
            if img is None or img.suffix.lower() not in IMG_EXTS:
                continue
        shutil.copy2(img, dst / "images" / split / img.name)
        (dst / "labels" / split / f"{xp.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1

    print(f"[{split}] 图片 {n_img} 张，实例 {sum(stats[split].values())} 个")
    return n_img


def main() -> None:
    ap = argparse.ArgumentParser(description="Pest24: VOC XML -> YOLO txt")
    ap.add_argument("--src", required=True, help="VOC 根目录（Kaggle 版指向 .../VOCdevkit/voc2007）")
    ap.add_argument("--dst", required=True, help="输出目录（将生成 images/ labels/ 与 pest24.yaml）")
    ap.add_argument("--no-yaml", action="store_true", help="不生成 pest24.yaml")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    splits = [
        s
        for s in ("train", "val", "test")
        if (src / s).exists()
        or (src / "ImageSets" / f"{s}.txt").exists()
        or (src / "ImageSets" / "Main" / f"{s}.txt").exists()
    ]
    assert "train" in splits and "val" in splits, "至少需要 train 和 val 两个划分"
    if "test" not in splits:
        print("ℹ️ 未发现 test 划分（官方未公开 test 标注）：将以官方 val 作为统一评测集")

    stats: dict = {s: {} for s in splits}
    unknown: set = set()
    for s in splits:
        convert_split(src, dst, s, stats, unknown)

    print("\n各类实例数（train）:")
    for i, n in enumerate(CLASSES):
        print(f"  {i:2d} {n:30s} {stats['train'].get(i, 0)}")
    if unknown:
        print("\n⚠️ 未识别类别名（请核对 CLASSES 列表）:", sorted(unknown))

    if not args.no_yaml:
        yl = [f"path: {dst.resolve().as_posix()}", "train: images/train", "val: images/val"]
        if "test" in splits:
            yl.append("test: images/test")
        yl.append("names:")
        yl += [f"  {i}: {n!r}" for i, n in enumerate(CLASSES)]
        (dst / "pest24.yaml").write_text("\n".join(yl) + "\n", encoding="utf-8")
        print(f"\n已写出 {dst / 'pest24.yaml'}")

    # 交叉校验：与镜像自带 labels/ 的文件数对比
    mirror_labels = src / "labels"
    if mirror_labels.exists():
        n_mirror = len(list(mirror_labels.glob("*.txt")))
        n_ours = len(list((dst / "labels" / "train").glob("*.txt"))) + len(
            list((dst / "labels" / "val").glob("*.txt"))
        )
        if "test" in splits:
            n_ours += len(list((dst / "labels" / "test").glob("*.txt")))
        print(f"\n交叉校验：镜像自带 labels {n_mirror} 个 / 我们生成 {n_ours} 个"
              f" {'✅ 一致' if n_mirror == n_ours else '⚠️ 不一致，以我们的为准并记录'}")


if __name__ == "__main__":
    main()
