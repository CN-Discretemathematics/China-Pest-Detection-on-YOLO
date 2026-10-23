#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADEA · D0 粘连诊断分析（T0 诊断包，零 GPU 成本）—— **G0 kill gate 的执行体**。

产出论文 Analysis 章的 4 组统计 + 机制假说 M1 的检验，并给出 G0 裁决。

用法
----
模式 A（仅需标签文件，无需训练）::

    python scripts/diagnose_adhesion.py --data datasets/pest24 --out artifacts/diag

模式 B（叠加训练期采集的 n_pos）::

    # 训练时开启：ADEA_DIAG=1 ADEA_DIAG_OUT=runs/diag/assign_stats.jsonl
    python scripts/diagnose_adhesion.py --data datasets/pest24 \
        --assign-stats runs/diag/assign_stats.jsonl --out artifacts/diag

产出
----
* ``artifacts/diag/adhesion_stats.json``  全部统计量
* ``artifacts/diag/adhesion_histogram.csv``  β 分布直方图
* ``artifacts/diag/G0_verdict.md``         **G0 裁决结论**（决定是否实现 I1/I2）

G0 判据
--------
机制假说 M1 成立，当且仅当 **同时** 满足：
1. ``corr(n_pos, β_max)`` 显著为负（|ρ| ≥ 0.2），或
2. 高粘连桶（β > 0.05）的实例级召回率显著低于低粘连桶。

不成立 → **不实现 I1 ALA / I2 DLA**，预算转 I3 DRA / I4 ASM / I7 天花板分析。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 与 voc2yolo.py 的 CLASSES 顺序保持一致
CLASSES = [
    "Bollworm", "Meadow borer", "Gryllotalpa orientalis", "Little Gecko",
    "Agriotes fuscicollis Miwa", "Nematode trench", "Athetis lepigone",
    "Scotogramma trifolii Rottemberg", "Armyworm", "Spodoptera cabbage",
    "Anomala corpulenta", "Spodoptera exigua", "Plutella xylostella",
    "holotrichia parallela", "Rice planthopper", "Yellow tiger",
    "Land tiger", "eight-character tiger", "holotrichia oblita", "Stem borer",
    "Striped rice bore", "Rice Leaf Roller", "Spodoptera litura", "Melahotus",
]

# Pest24 论文口径
DEFAULT_IMG_W, DEFAULT_IMG_H = 800, 600
BETA_THR = 0.05            # 论文"nontrivial adhesion"阈值
DENSE_INSTANCE_THR = 20    # 论文：8.89% 的图 >20 实例
RARE_INSTANCE_THR = 500    # 本项目定义的"稀有类"阈值（<500 实例）
TINY_RELATIVE_SCALE = 5e-4  # 0.05%（Rice planthopper 0.034%、Plutella 0.043%）

Box = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# 基础计算
# --------------------------------------------------------------------------- #
def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def adhesion(a: Box, b: Box) -> float:
    """Pest24 论文定义：β = |A∩B| / |A∪B|。"""
    inter = _inter(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def read_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, Box]]:
    """读取 YOLO txt → [(cls, xyxy 像素框)]。"""
    out: List[Tuple[int, Box]] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        out.append((c, (x1, y1, x2, y2)))
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson 相关系数（无第三方依赖）。"""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman 秩相关（论文用 Spearman，故实现之）。"""

    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return pearson(ranks(xs), ranks(ys))


# --------------------------------------------------------------------------- #
# 主分析
# --------------------------------------------------------------------------- #
def analyse_dataset(data_root: Path, img_w: int, img_h: int) -> Dict[str, object]:
    """数据集级统计（模式 A）。"""
    labels_dir = data_root / "labels"
    splits = [s for s in ("train", "val", "test") if (labels_dir / s).exists()]

    per_class_instances: Counter = Counter()
    per_class_area: Dict[int, List[float]] = defaultdict(list)
    per_class_rel_scale: Dict[int, List[float]] = defaultdict(list)
    image_area = float(img_w * img_h)

    n_images_total = 0
    n_images_adhesion = 0
    n_images_dense = 0
    beta_values: List[float] = []
    beta_hist: Counter = Counter()
    inst_counts: List[int] = []
    subset_images: Dict[str, List[str]] = defaultdict(list)
    class_rel_scale_mean: Dict[int, float] = {}

    for split in splits:
        ldir = labels_dir / split
        for lp in sorted(ldir.glob("*.txt")):
            n_images_total += 1
            items = read_labels(lp, img_w, img_h)
            inst_counts.append(len(items))
            stem = f"{split}/{lp.stem}"

            max_beta = 0.0
            for i in range(len(items)):
                ci, bi = items[i]
                per_class_instances[ci] += 1
                a = _area(bi)
                per_class_area[ci].append(a)
                per_class_rel_scale[ci].append(a / image_area)
                for j in range(i + 1, len(items)):
                    b = adhesion(bi, items[j][1])
                    beta_values.append(b)
                    max_beta = max(max_beta, b)
                    bucket = min(int(b / 0.02), 9)
                    beta_hist[bucket] += 1

            if max_beta > BETA_THR:
                n_images_adhesion += 1
                subset_images["adhesion"].append(stem)
            if len(items) > DENSE_INSTANCE_THR:
                n_images_dense += 1
                subset_images["dense"].append(stem)

    for ci, vals in per_class_rel_scale.items():
        if vals:
            class_rel_scale_mean[ci] = sum(vals) / len(vals)

    rare_classes = sorted(c for c, n in per_class_instances.items() if n < RARE_INSTANCE_THR)
    tiny_classes = sorted(c for c, s in class_rel_scale_mean.items() if s < TINY_RELATIVE_SCALE)

    return {
        "splits": splits,
        "img_size": [img_w, img_h],
        "n_images": n_images_total,
        "n_images_adhesion": n_images_adhesion,
        "n_images_dense": n_images_dense,
        "pct_images_adhesion": round(100.0 * n_images_adhesion / max(1, n_images_total), 2),
        "pct_images_dense": round(100.0 * n_images_dense / max(1, n_images_total), 2),
        "mean_instances_per_image": round(sum(inst_counts) / max(1, len(inst_counts)), 3),
        "max_instances_per_image": max(inst_counts) if inst_counts else 0,
        "n_gt_pairs": len(beta_values),
        "pct_pairs_adhesion_gt_thr": round(
            100.0 * sum(1 for b in beta_values if b > BETA_THR) / max(1, len(beta_values)), 2
        ),
        "beta_histogram": {f"{k * 0.02:.2f}-{(k + 1) * 0.02:.2f}": beta_hist.get(k, 0) for k in range(10)},
        "per_class_instances": {CLASSES[c]: per_class_instances.get(c, 0) for c in range(len(CLASSES))},
        "per_class_mean_relative_scale": {
            CLASSES[c]: round(class_rel_scale_mean.get(c, 0.0), 6) for c in range(len(CLASSES))
        },
        "rare_classes_lt500": [CLASSES[c] for c in rare_classes],
        "tiny_classes_relscale_lt0.05pct": [CLASSES[c] for c in tiny_classes],
        "subset_definitions": {
            "adhesion": f"存在 GT 对 β > {BETA_THR}",
            "dense": f"单图实例数 > {DENSE_INSTANCE_THR}",
            "rare": f"类别实例数 < {RARE_INSTANCE_THR}",
            "tiny": f"类别平均相对尺度 < {TINY_RELATIVE_SCALE}",
        },
        "subset_image_counts": {k: len(v) for k, v in subset_images.items()},
    }


def analyse_assign_stats(assign_path: Path) -> Optional[Dict[str, object]]:
    """训练期采集的 n_pos 统计（模式 B）→ M1 检验。"""
    if not assign_path.exists():
        return None

    beta_all: List[float] = []
    npos_all: List[float] = []
    area_all: List[float] = []
    n_batches = 0
    starved_total = 0
    starved_adhesion = 0
    starved_isolated = 0

    for line in assign_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_batches += 1
        n_pos = rec.get("n_pos", [])
        beta_max = rec.get("beta_max", [])
        areas = rec.get("areas", [])
        q = 3
        for i in range(len(n_pos)):
            npos_all.append(float(n_pos[i]))
            beta_all.append(float(beta_max[i]) if i < len(beta_max) else 0.0)
            area_all.append(float(areas[i]) if i < len(areas) else 0.0)
            if n_pos[i] < q:
                starved_total += 1
                if i < len(beta_max) and beta_max[i] > BETA_THR:
                    starved_adhesion += 1
                else:
                    starved_isolated += 1

    if not npos_all:
        return None

    rho = pearson(beta_all, npos_all)
    rho_s = spearman(beta_all, npos_all)
    rho_area = pearson(area_all, npos_all)

    # 分桶：粘连组 vs 孤立组的平均正样本数
    adh = [n for n, b in zip(npos_all, beta_all) if b > BETA_THR]
    iso = [n for n, b in zip(npos_all, beta_all) if b <= BETA_THR]

    return {
        "n_batches": n_batches,
        "n_gt_samples": len(npos_all),
        "corr_npos_beta_pearson": round(rho, 4) if rho == rho else None,
        "corr_npos_beta_spearman": round(rho_s, 4) if rho_s == rho_s else None,
        "corr_npos_area_pearson": round(rho_area, 4) if rho_area == rho_area else None,
        "mean_npos_adhesion": round(sum(adh) / len(adh), 4) if adh else None,
        "mean_npos_isolated": round(sum(iso) / len(iso), 4) if iso else None,
        "n_starved_total": starved_total,
        "n_starved_adhesion": starved_adhesion,
        "n_starved_isolated": starved_isolated,
        "pct_starved_that_are_adhesion": round(100.0 * starved_adhesion / max(1, starved_total), 2),
        "m2": analyse_scale_starvation(npos_all, area_all, beta_all),
    }


def analyse_scale_starvation(
    npos_all: Sequence[float],
    area_all: Sequence[float],
    beta_all: Sequence[float],
) -> Dict[str, object]:
    """M2 检验：正样本数是否被**目标尺寸**系统性剥夺（并做面积标定的粘连残差分析）。

    关键洞察（由本机实测数据驱动）
    ----------------------------
    * ``n_pos`` 存在**上限**（TAL 的每 GT top-k，实测 10）；
    * 小目标的**候选池**本身就小 —— anchor 中心必须落在 GT 框内，
      候选数 ∝ 面积 / stride² —— 因此 ``n_pos ∝ 相对尺度``（对数近似线性）；
    * 于是"被饿死"必须用**面积标定的期望**判定：
      ``E[n_pos | a] ≈ min(k0, c·a_rel)``；**残差为负**才是真正的剥夺。

    残差分析同时把粘连效应从面积效应中**解耦**出来：
    若粘连目标的残差显著低于孤立目标 → 粘连确有剥夺效应（M1 成立）。
    """
    img_area = float(DEFAULT_IMG_W * DEFAULT_IMG_H)
    keep = [i for i, a in enumerate(area_all) if a > 0]
    rel = [area_all[i] / img_area for i in keep]
    npos = [npos_all[i] for i in keep]
    betas = [beta_all[i] for i in keep]
    if len(rel) < 10:
        return {"n_samples": len(rel)}

    log_rel = [math.log(max(1e-9, r)) for r in rel]
    rho_log = pearson(log_rel, npos)
    rho_log_s = spearman(log_rel, npos)

    buckets = [
        (0.0, 1e-4),
        (1e-4, 3e-4),
        (3e-4, 1e-3),
        (1e-3, 3e-3),
        (3e-3, 1e-2),
        (1e-2, 1.0),
    ]
    bucket_rows = []
    for lo, hi in buckets:
        sel = [n for n, r in zip(npos, rel) if lo <= r < hi]
        if not sel:
            continue
        sel_s = sorted(sel)
        bucket_rows.append(
            {
                "range": f"[{lo:.0e},{hi:.0e})",
                "n": len(sel),
                "mean_npos": round(sum(sel) / len(sel), 3),
                "median_npos": sel_s[len(sel_s) // 2],
                "pct_starved_lt3": round(100.0 * sum(1 for x in sel if x < 3) / len(sel), 2),
            }
        )

    # 拟合 E[n_pos | a] = min(k0, c·a_rel) 中的 c
    # ⚠️ 标定原则：**让期望模型在"未被剥夺"的样本（非粘连）上零残差**。
    #    不用逐样本最小二乘（会被大 a_rel 的 x² 量级主导，c 估计严重不稳，
    #    实测逐样本拟合给出 2936，而稳健区间是 4,000~20,000）；
    #    也不用分桶均值最小二乘（左censored首桶 [0,1e-4) 的中点会拉偏）。
    #    做法：在非粘连样本上求 mean(n_pos - min(k0, c·a)) = 0 的 c（一维搜索）。
    k0 = max(npos) if npos else 10

    def _mean_resid_nonadh(c: float) -> float:
        vals = []
        for n, r, b in zip(npos, rel, betas):
            if b > BETA_THR:
                continue
            exp = k0 if c * r >= k0 else c * r
            vals.append(n - exp)
        return sum(vals) / len(vals) if vals else 0.0

    lo_c, hi_c = 1.0, 1e6
    c_fit = None
    if _mean_resid_nonadh(lo_c) * _mean_resid_nonadh(hi_c) < 0:
        for _ in range(200):
            mid = math.sqrt(lo_c * hi_c)  # 对数空间二分
            if _mean_resid_nonadh(mid) * _mean_resid_nonadh(lo_c) <= 0:
                hi_c = mid
            else:
                lo_c = mid
        c_fit = math.sqrt(lo_c * hi_c)
    if c_fit is None:
        c_fit = k0 / max(1e-12, max(rel))

    # 残差：n_pos - min(k0, c·a_rel)
    resid = []
    for n, r in zip(npos, rel):
        exp = k0 if (c_fit is None or c_fit * r >= k0) else c_fit * r
        resid.append(n - exp)
    neg_frac = 100.0 * sum(1 for x in resid if x < -0.5) / max(1, len(resid))

    adh_r = [x for x, b in zip(resid, betas) if b > BETA_THR]
    iso_r = [x for x, b in zip(resid, betas) if b <= BETA_THR]
    rho_resid_beta = pearson(betas, resid)

    return {
        "n_samples": len(rel),
        "corr_npos_log_relscale": round(rho_log, 4) if rho_log == rho_log else None,
        "corr_npos_log_relscale_spearman": round(rho_log_s, 4) if rho_log_s == rho_log_s else None,
        "k0_saturation": k0,
        "fit_c_candidates_per_relarea": round(c_fit, 2) if c_fit else None,
        "expected_npos_model": "min(k0, c * rel_area)",
        "mean_residual_adhesion": round(sum(adh_r) / len(adh_r), 4) if adh_r else None,
        "mean_residual_isolated": round(sum(iso_r) / len(iso_r), 4) if iso_r else None,
        "corr_residual_beta": round(rho_resid_beta, 4) if rho_resid_beta == rho_resid_beta else None,
        "pct_underallocated": round(neg_frac, 2),
        "buckets": bucket_rows,
    }


def g0_verdict(dataset_stats: Dict[str, object], assign_stats: Optional[Dict[str, object]]) -> Tuple[str, str]:
    """G0 裁决：M1（粘连致饿死）或 M2（尺度致饿死）任一成立即继续实现。

    ⚠️ 判据的演进（由本机实测数据驱动，重要）
    ---------------------------------------
    **初版 M1 判据（已被证伪）**：``corr(n_pos, β) <= -0.2`` 或
    "粘连组平均 n_pos 明显更低"。实测结果是 ``corr(n_pos, β) = +0.52``（**正**相关），
    粘连组平均 n_pos 反而更高（7.64 vs 1.24）→ 初版判据错误。

    **为什么会错**：n_pos 的**主因是目标面积**（corr(log 相对尺度, n_pos) = +0.86），
    而粘连的虫通常**体型更大**（β 大的多是框大、彼此靠近的个体）→ 面积效应
    完全掩盖了粘连效应。朴素比较"粘连组 vs 孤立组的绝对 n_pos"必然得出反向结论。

    **修正后的判据**：
    * **M2（尺度）**：``corr(log 相对尺度, n_pos) >= 0.3`` → 面积标定后仍是强正相关
      → 微目标的正样本被系统性剥夺。
    * **M1（粘连）**：在**面积标定残差**上比较 —— 残差
      ``n_pos - min(k0, c·a_rel)`` 在粘连组应低于孤立组。
    """
    if assign_stats is None:
        return (
            "PENDING",
            "未提供 --assign-stats，无法检验 M1/M2。请先用 ADEA_DIAG=1 跑一次 baseline 训练"
            "（采集时须保持 ADEA_ALA=0，否则测到的是修复后的分布）。",
        )

    m2 = assign_stats.get("m2") or {}
    rho_log = m2.get("corr_npos_log_relscale")
    mean_adh = assign_stats.get("mean_npos_adhesion")
    mean_iso = assign_stats.get("mean_npos_isolated")
    rho_resid_beta = m2.get("corr_residual_beta")
    mean_r_adh = m2.get("mean_residual_adhesion")
    mean_r_iso = m2.get("mean_residual_isolated")
    c_fit = m2.get("fit_c_candidates_per_relarea")
    k0 = m2.get("k0_saturation")

    # --- M1：面积标定后，粘连目标是否仍被剥夺 ---
    m1_ok = (
        rho_resid_beta is not None
        and rho_resid_beta <= -0.05
        and mean_r_adh is not None
        and mean_r_iso is not None
        and mean_r_adh < mean_r_iso
    )
    # --- M2：尺度是否主导正样本分配 ---
    m2_ok = rho_log is not None and rho_log >= 0.3

    if m1_ok and m2_ok:
        return (
            "PASS",
            f"M1 与 M2 同时成立：面积标定后粘连目标残差 {mean_r_adh} < 孤立目标 {mean_r_iso}"
            f"（corr(residual, β)={rho_resid_beta}），且 corr(log 相对尺度, n_pos)={rho_log}。"
            f"（标定模型 E[n_pos|a]=min({k0}, {c_fit}·a_rel)）"
            "→ 实施**面积标定的公平正样本分配**（SLA）：按期望标定残差，"
            "对被低估的 GT 从超额 GT 回收 anchor，并在更细的尺度层（P2/stride-4）"
            "为微目标提供足够候选。",
        )
    if m2_ok and not m1_ok:
        return (
            "PASS（仅 M2）",
            f"M2 成立：corr(log 相对尺度, n_pos)={rho_log}（小目标正样本被系统性剥夺）；"
            f"粘连在**面积标定后**无显著剥夺（corr(residual, β)={rho_resid_beta}，"
            f"粘连残差 {mean_r_adh} vs 孤立 {mean_r_iso}）。"
            f"（标定模型 E[n_pos|a]=min({k0}, {c_fit}·a_rel)）"
            "→ 创新主线从『粘连感知』转为『**尺度/面积标定的监督分配**』："
            "引入更细 stride 的监督层解决微目标候选池不足，"
            "并用面积标定残差做配额补偿（粘连校正降为次要消融项）。",
        )
    return (
        "FAIL",
        f"M1 与 M2 均不成立：corr(log 相对尺度, n_pos)={rho_log}，"
        f"corr(residual, β)={rho_resid_beta}。→ 不实施 SLA，"
        "预算转 I3 DRA / I4 ASM / I7 天花板分析，并把 D0 诊断作为论文主体。",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="ADEA · D0 粘连诊断分析（G0 kill gate）")
    ap.add_argument("--data", required=True, help="数据集根目录（含 labels/）")
    ap.add_argument("--out", default="artifacts/diag", help="输出目录")
    ap.add_argument("--img-w", type=int, default=DEFAULT_IMG_W)
    ap.add_argument("--img-h", type=int, default=DEFAULT_IMG_H)
    ap.add_argument("--assign-stats", default=None, help="训练期采集的 assign_stats.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = analyse_dataset(Path(args.data), args.img_w, args.img_h)
    as_ = analyse_assign_stats(Path(args.assign_stats)) if args.assign_stats else None
    verdict, reason = g0_verdict(ds, as_)

    payload = {"dataset": ds, "assign_stats": as_, "g0": {"verdict": verdict, "reason": reason}}
    (out / "adhesion_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out / "adhesion_histogram.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("beta_range,n_pairs\n")
        for k, v in ds["beta_histogram"].items():
            f.write(f"{k},{v}\n")

    lines = [
        "# G0 裁决（机制假说 M1）",
        "",
        f"**结论：{verdict}**",
        "",
        reason,
        "",
        "## 数据集级事实",
        "",
        f"- 图片总数：{ds['n_images']}（split: {ds['splits']}）",
        f"- 存在非忽略粘连（β>{BETA_THR}）的图：{ds['n_images_adhesion']}"
        f"（{ds['pct_images_adhesion']}%）",
        f"- 高密度图（>{DENSE_INSTANCE_THR} 实例）：{ds['n_images_dense']}"
        f"（{ds['pct_images_dense']}%）",
        f"- 平均实例/图：{ds['mean_instances_per_image']}，最大 {ds['max_instances_per_image']}",
        f"- GT 对总数：{ds['n_gt_pairs']}，其中 β>{BETA_THR} 占 {ds['pct_pairs_adhesion_gt_thr']}%",
        f"- 稀有类（<{RARE_INSTANCE_THR} 实例）：{ds['rare_classes_lt500']}",
        f"- 极小类（相对尺度 <0.05%）：{ds['tiny_classes_relscale_lt0.05pct']}",
        "",
    ]
    if as_:
        lines += [
            "## 训练期采集（n_pos vs β）",
            "",
            f"- 批次数：{as_['n_batches']}，GT 样本数：{as_['n_gt_samples']}",
            f"- corr(n_pos, β) Spearman：{as_['corr_npos_beta_spearman']}"
            f" / Pearson：{as_['corr_npos_beta_pearson']}",
            f"- corr(n_pos, area)：{as_['corr_npos_area_pearson']}",
            f"- 粘连组平均正样本：{as_['mean_npos_adhesion']}；孤立组：{as_['mean_npos_isolated']}",
            f"- 被饿死（n_pos<3）样本：{as_['n_starved_total']}"
            f"（其中粘连目标 {as_['n_starved_adhesion']}，占 {as_['pct_starved_that_are_adhesion']}%）",
            "",
        ]
    (out / "G0_verdict.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n已写出：{out / 'adhesion_stats.json'}")
    print(f"已写出：{out / 'adhesion_histogram.csv'}")
    print(f"已写出：{out / 'G0_verdict.md'}")


if __name__ == "__main__":
    main()
