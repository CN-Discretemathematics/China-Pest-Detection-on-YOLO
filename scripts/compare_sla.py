#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SLA 前后对照分析：baseline 分配 vs SLA 分配（**同口径**）。

用途
----
验证 SLA 的核心主张——**面积标定的公平再分配**确实把被剥夺的目标拉回来了：

1. **几何缺口**（SLA 的原生指标，同口径）：
   ``gap_i = min(topk, n_cand_i) - n_pos_usable_i``
   其中 ``n_cand_i`` 是该 GT 的可用候选数（框内且 align>0），
   ``n_pos_usable_i`` 是它实际拿到的可用正样本数。
   → 零和转移下 **总缺口不变**，变的是**分布**（最大缺口下降）。

2. **面积标定残差**（G0 诊断口径）：
   ``resid_i = n_pos_i - min(k0, c·a_rel_i)``，
   ``c`` 由 **baseline** 的非粘连样本标定后**固定**给两次运行使用，
   保证两次运行的残差可直接相减。

用法::

    # 1) 先跑两次训练（相同 seed / 数据序），各开 ADEA_DIAG=1
    #    ADEA_DIAG_OUT=.../sla_base.jsonl   （ADEA_SLA=0）
    #    ADEA_DIAG_OUT=.../sla_sla.jsonl    （ADEA_SLA=1）
    # 2) 对照分析
    python scripts/compare_sla.py --base runs/diag/sla_base.jsonl --sla runs/diag/sla_sla.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_adhesion import pearson  # noqa: E402

IMG_W, IMG_H = 800, 600
IMG_AREA = float(IMG_W * IMG_H)
BETA_THR = 0.05


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
class Samples:
    """一次运行的逐 GT 样本集合。"""

    def __init__(self) -> None:
        self.n_pos: List[float] = []
        self.n_pos_usable: List[float] = []
        self.n_cand: List[float] = []
        self.beta: List[float] = []
        self.area: List[float] = []
        self.has_usable = False
        self.skipped_records = 0

    def __len__(self) -> int:
        return len(self.n_pos)


def load_samples(path: Path, max_records: Optional[int] = None) -> Samples:
    """读取 assign_stats.jsonl → 逐 GT 样本。

    只保留**含几何字段**（``n_cand`` / ``n_pos_usable``）的记录——
    旧版诊断采集器写出的记录没有这些字段，混入会破坏同口径比较。
    """
    s = Samples()
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for n_line, line in enumerate(f):
            if max_records and n_line >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_use = rec.get("n_pos_usable")
            n_cand = rec.get("n_cand")
            if n_use is None or n_cand is None:
                skipped += 1
                continue
            s.has_usable = True
            n_pos = rec.get("n_pos", [])
            beta = rec.get("beta_max", [])
            areas = rec.get("areas", [])
            for i in range(len(n_pos)):
                area = float(areas[i]) if i < len(areas) else 0.0
                if area <= 0:
                    continue
                s.n_pos.append(float(n_pos[i]))
                s.n_pos_usable.append(float(n_use[i]))
                s.n_cand.append(float(n_cand[i]))
                s.beta.append(float(beta[i]) if i < len(beta) else 0.0)
                s.area.append(area)
    s.skipped_records = skipped
    return s


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def fit_c(s: Samples, k0: float = 11.0) -> float:
    """标定 ``E[n_pos|a] = min(k0, c·a_rel)`` 的 c（非粘连样本零残差）。"""

    def mean_resid(c: float) -> float:
        vals = []
        for n, a, b in zip(s.n_pos, s.area, s.beta):
            if b > BETA_THR:
                continue
            rel = a / IMG_AREA
            vals.append(n - (k0 if c * rel >= k0 else c * rel))
        return sum(vals) / len(vals) if vals else 0.0

    lo, hi = 1.0, 1e6
    if mean_resid(lo) * mean_resid(hi) >= 0:
        return 9476.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if mean_resid(mid) * mean_resid(lo) <= 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def geometric_gap(s: Samples, topk: int = 10) -> List[float]:
    """几何缺口 ``min(topk, n_cand) - n_pos_usable``（NaN 候选数视为不可用）。"""
    gaps: List[float] = []
    for n_use, n_cand in zip(s.n_pos_usable, s.n_cand):
        if n_cand != n_cand:  # NaN
            continue
        gaps.append(max(0.0, min(float(topk), n_cand) - n_use))
    return gaps


def summarize(s: Samples, c: float, k0: float = 11.0, topk: int = 10) -> Dict[str, object]:
    """一次运行的汇总指标。"""
    n = len(s.n_pos)
    if n == 0:
        return {"n_samples": 0}

    rel = [a / IMG_AREA for a in s.area]
    resid = [
        ni - (k0 if c * r >= k0 else c * r) for ni, r in zip(s.n_pos, rel)
    ]
    gaps = geometric_gap(s, topk=topk)
    # 只在有几何字段时统计粘连/孤立分组
    adh_idx = [i for i, b in enumerate(s.beta) if b > BETA_THR]
    iso_idx = [i for i, b in enumerate(s.beta) if b <= BETA_THR]

    def mean(xs: Sequence[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    rho = pearson(s.beta, resid)
    out: Dict[str, object] = {
        "n_samples": n,
        "mean_n_pos": mean(s.n_pos),
        "mean_n_pos_usable": mean(s.n_pos_usable),
        "mean_n_cand": mean([x for x in s.n_cand if x == x]) if s.has_usable else None,
        "pct_starved_lt3": round(100.0 * sum(1 for x in s.n_pos_usable if x < 3) / n, 2),
        # 面积标定残差（G0 口径，c 固定）
        "mean_residual": mean(resid),
        "mean_residual_adhesion": mean([resid[i] for i in adh_idx]),
        "mean_residual_isolated": mean([resid[i] for i in iso_idx]),
        "corr_residual_beta": round(rho, 4) if rho == rho else None,
        "pct_underallocated": round(100.0 * sum(1 for x in resid if x < -0.5) / n, 2),
    }
    if gaps:
        out.update(
            {
                "mean_geometric_gap": mean(gaps),
                "max_geometric_gap": round(max(gaps), 4),
                "pct_gap_gt05": round(100.0 * sum(1 for x in gaps if x > 0.5) / len(gaps), 2),
                "pct_gap_gt2": round(100.0 * sum(1 for x in gaps if x > 2.0) / len(gaps), 2),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def scale_table(s: Samples) -> List[Dict[str, object]]:
    """按相对面积分档的 n_pos 表（"尺度公平"的直接度量）。"""
    buckets = [
        (0.0, 1e-4),
        (1e-4, 3e-4),
        (3e-4, 1e-3),
        (1e-3, 3e-3),
        (3e-3, 1e-2),
        (1e-2, 1.0),
    ]
    rows: List[Dict[str, object]] = []
    for lo, hi in buckets:
        sel = [
            (n, u)
            for n, u, a in zip(s.n_pos, s.n_pos_usable, s.area)
            if lo <= a / IMG_AREA < hi
        ]
        if not sel:
            continue
        rows.append(
            {
                "range": f"[{lo:.0e},{hi:.0e})",
                "n": len(sel),
                "mean_n_pos": round(sum(x[0] for x in sel) / len(sel), 3),
                "mean_n_pos_usable": round(sum(x[1] for x in sel) / len(sel), 3),
            }
        )
    return rows


def scale_correlation(s: Samples) -> Optional[float]:
    """``corr(n_pos, log 相对面积)`` —— 越接近 0 越"尺度公平"。"""
    xs = [math.log(max(1e-9, a / IMG_AREA)) for a in s.area]
    rho = pearson(xs, s.n_pos)
    return round(rho, 4) if rho == rho else None


def main() -> int:
    ap = argparse.ArgumentParser(description="SLA 前后对照分析")
    ap.add_argument("--base", required=True, help="baseline 的 assign_stats.jsonl")
    ap.add_argument("--sla", required=True, help="SLA 运行的 assign_stats.jsonl")
    ap.add_argument("--topk", type=int, default=10, help="TAL 的每 GT topk（默认 10）")
    ap.add_argument("--max-records", type=int, default=None, help="每个文件最多读多少条记录")
    ap.add_argument("--out", default=None, help="报告 JSON 输出路径")
    args = ap.parse_args()

    base_path, sla_path = Path(args.base), Path(args.sla)
    for p in (base_path, sla_path):
        if not p.exists():
            print(f"[!] 文件不存在: {p}")
            return 2

    base = load_samples(base_path, args.max_records)
    sla = load_samples(sla_path, args.max_records)
    if not len(base) or not len(sla):
        print("[!] 样本为空")
        return 2

    c = fit_c(base)
    print("=" * 78)
    print("SLA 前后对照（同 seed / 同数据序 / 同一 topk）")
    print("=" * 78)
    print(f"标定常数 c（由 baseline 非粘连样本标定，两次运行共用）= {c:.2f}")
    print(f"baseline 样本 {len(base)}  SLA 样本 {len(sla)}")

    sb = summarize(base, c, topk=args.topk)
    ss = summarize(sla, c, topk=args.topk)

    rows = [
        ("GT 样本数", "n_samples", "{:.0f}"),
        ("平均正样本数 (fg 口径)", "mean_n_pos", "{:.3f}"),
        ("平均可用正样本数", "mean_n_pos_usable", "{:.3f}"),
        ("平均可用候选数", "mean_n_cand", "{:.3f}"),
        ("被饿死比例 n_pos<3 (%)", "pct_starved_lt3", "{:.2f}"),
        ("面积标定残差(平均)", "mean_residual", "{:.3f}"),
        ("  ├ 粘连目标残差", "mean_residual_adhesion", "{:.3f}"),
        ("  └ 孤立目标残差", "mean_residual_isolated", "{:.3f}"),
        ("corr(残差, β)", "corr_residual_beta", "{:.3f}"),
        ("被低估比例 (%)", "pct_underallocated", "{:.2f}"),
        ("几何缺口(平均)", "mean_geometric_gap", "{:.3f}"),
        ("几何缺口(最大)", "max_geometric_gap", "{:.3f}"),
        ("缺口>0.5 比例 (%)", "pct_gap_gt05", "{:.2f}"),
        ("缺口>2 比例 (%)", "pct_gap_gt2", "{:.2f}"),
    ]
    print(f"\n{'指标':<28}{'baseline':>14}{'SLA':>14}{'Δ':>16}")
    print("-" * 78)
    for label, key, fmt in rows:
        b, s = sb.get(key), ss.get(key)
        if b is None or s is None:
            continue
        delta = s - b
        print(f"{label:<28}{fmt.format(b):>14}{fmt.format(s):>14}{delta:>+16.3f}")

    report = {"c_fit": c, "topk": args.topk, "baseline": sb, "sla": ss}

    # ---- 尺度公平表（论文主表候选）----
    tb, ts = scale_table(base), scale_table(sla)
    print(f"\n尺度公平：按相对面积分档的平均正样本数")
    print(f"{'尺度档':<18}{'n':>7}{'baseline':>12}{'SLA':>12}{'Δ':>10}")
    print("-" * 60)
    for rb, rs in zip(tb, ts):
        print(
            f"{rb['range']:<18}{rb['n']:>7}{rb['mean_n_pos']:>12.2f}"
            f"{rs['mean_n_pos']:>12.2f}{rs['mean_n_pos'] - rb['mean_n_pos']:>+10.2f}"
        )
    cb, cs = scale_correlation(base), scale_correlation(sla)
    print(f"\ncorr(n_pos, log 相对尺度): baseline {cb:+.4f} → SLA {cs:+.4f}"
          f"（越接近 0 越尺度公平）")
    report["scale_table"] = {"baseline": tb, "sla": ts}
    report["scale_corr_npos_log_area"] = {"baseline": cb, "sla": cs}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
