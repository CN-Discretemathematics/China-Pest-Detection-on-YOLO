#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADEA 核心算法的单元自测（**不依赖 torch**，纯标准库即可运行）。

覆盖
----
* **ALA（I1）**：粘连系数 β 的定义正确性、粘连簇构建、被"饿死"实例的配额回收、
  供体保护、关闭开关时零行为改变。
* **DLA（I2）**：密度场渲染、归一化、逐 GT 配额随密度单调递增。
* **AGS（I5）**：同一实例 → 硬抑制；不同粘连个体 → 软衰减且保留。
* **DRA（I3）**：难度越高权重越高；静态对照与早停保护。

用法::

    python scripts/selftest_adea.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Windows 控制台默认 GBK，无法编码 ✅/❌ —— 强制 UTF-8 并改用 ASCII 标记
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics_ext.assign.ala import (  # noqa: E402
    ALAConfig,
    ala_core,
    adhesion_coefficient,
    build_clusters,
)
from ultralytics_ext.assign.dla import (  # noqa: E402
    DLAConfig,
    per_gt_quota,
    render_density,
    normalize_density,
)
from ultralytics_ext.post.ags import AGSConfig, Detection, ags_core  # noqa: E402
from ultralytics_ext.sampling.dra import DRASampler, DRAConfig, build_static_weights  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


# --------------------------------------------------------------------------- #
def test_adhesion_definition() -> None:
    print("\n=== ALA · 粘连系数与簇 ===")
    # 完全重合 → β = 1
    check("完全重合 β=1", abs(adhesion_coefficient((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9)
    # 部分重合：A=[0,10]×[0,10], B=[5,15]×[0,10] → 交集 5×10=50, 并集 15×10=150 → 1/3
    b = adhesion_coefficient((0, 0, 10, 10), (5, 0, 15, 10))
    check("部分重叠 β=1/3", abs(b - 1.0 / 3.0) < 1e-9, f"β={b:.4f}")
    # 不相交 → 0
    check("不相交 β=0", adhesion_coefficient((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0)
    # ⚠️ 事实更正：论文的 β = |A∩B|/|A∪B| 与 IoU **数值等价**（同一公式），
    #    二者只在语义上不同（"物理粘连" vs "重复检出同一体"）。
    big, small = (0, 0, 100, 100), (40, 40, 60, 60)
    beta = adhesion_coefficient(big, small)
    inter = 20 * 20
    iou = inter / (100 * 100)
    check("β 与 IoU 数值等价（论文定义即 IoU）", abs(beta - iou) < 1e-9, f"β={beta:.4f} IoU={iou:.4f}")

    # 簇构建：1-2 粘连，3 孤立，4-5 粘连
    boxes = [(0, 0, 10, 10), (2, 0, 12, 10), (100, 100, 110, 110), (200, 0, 210, 10), (202, 0, 212, 10)]
    beta = [[0.0] * 5 for _ in range(5)]
    for i in range(5):
        for j in range(i + 1, 5):
            beta[i][j] = beta[j][i] = adhesion_coefficient(boxes[i], boxes[j])
    clusters = sorted(build_clusters(beta, 0.05))
    check("粘连簇划分正确", clusters == [[0, 1], [3, 4]], f"clusters={clusters}")


def test_ala_quota_rescue() -> None:
    print("\n=== ALA · 配额回收（核心机制）===")
    # 两个粘连的小 GT；候选 anchor 全部挤在重叠区 → argmax 独占导致 GT1 被饿死
    gt_boxes = [(10.0, 10.0, 30.0, 30.0), (12.0, 10.0, 32.0, 30.0)]
    # 6 个 anchor 全部更靠近 GT0 → GT0 独占，GT1 得到 0 个
    anchors = []
    target = []
    for i in range(6):
        cx = 13.0 + i * 2.0
        anchors.append((cx - 4.0, 14.0, cx + 4.0, 26.0))
        target.append(0)
    # 再加 2 个只贴近 GT1 的 anchor（GT1 因此有 2 个正样本，低于 q_min=3）
    for i in range(2):
        cx = 26.0 + i * 2.0
        anchors.append((cx - 4.0, 14.0, cx + 4.0, 26.0))
        target.append(1)

    counts_before = [target.count(0), target.count(1)]
    check("构造：GT1 初始正样本 < q_min", counts_before[1] < 3, f"n_pos={counts_before}")

    cfg = ALAConfig(enabled=True, beta_thr=0.05, q_min=3)
    new_target, st = ala_core(target, anchors, gt_boxes, cfg)
    counts_after = [new_target.count(0), new_target.count(1)]
    check("ALA 抬高了被饿死 GT 的正样本数", counts_after[1] > counts_before[1],
          f"{counts_before} → {counts_after}")
    check("被救 GT 达到 q_min", counts_after[1] >= 3, f"n_pos={counts_after[1]}")
    check("供体仍保留 q_min", counts_after[0] >= 3, f"n_pos={counts_after[0]}")
    check("总正样本数不变（不变量）", sum(counts_after) == sum(counts_before),
          f"{sum(counts_before)} → {sum(counts_after)}")
    check("每个 anchor 至多一个 GT", all(t in (0, 1) for t in new_target))
    check("统计量报告了回收数", st.n_reassigned > 0, f"reassigned={st.n_reassigned}")

    # 关闭开关 → 零行为改变
    cfg_off = ALAConfig(enabled=False, q_min=3)
    off_target, st_off = ala_core(target, anchors, gt_boxes, cfg_off)
    check("关闭 ALA 时分配完全不变", off_target == list(target))
    check("关闭 ALA 时回收数为 0", st_off.n_reassigned == 0)

    # 无粘连场景 → 不应触发回收
    gt_far = [(10.0, 10.0, 30.0, 30.0), (200.0, 200.0, 220.0, 220.0)]
    far_target, st_far = ala_core(list(target), anchors, gt_far, cfg)
    check("无粘连时不触发回收", st_far.n_reassigned == 0, f"reassigned={st_far.n_reassigned}")


def test_dla() -> None:
    print("\n=== DLA · 密度场与配额调制 ===")
    centers = [(10.0, 10.0), (12.0, 11.0), (14.0, 9.0)]
    radii = [2.0, 2.0, 2.0]
    d = render_density(centers, radii, (40, 40), sigma_scale=0.5)
    check("密度图形状正确", len(d) == 40 and len(d[0]) == 40)
    peak = max(max(r) for r in d)
    check("峰值出现在实例附近", d[10][10] >= 0.9 * peak, f"d[10][10]={d[10][10]:.4f} peak={peak:.4f}")

    dn = normalize_density(d, 0.99)
    flat = [v for row in dn for v in row]
    check("归一化后落在 [0,1]", min(flat) >= 0.0 and max(flat) <= 1.0)

    # 密集区 vs 稀疏区的配额
    gt_dense = [(14.0, 10.0, 18.0, 14.0)]
    gt_sparse = [(2.0, 38.0, 6.0, 38.0)]
    cfg = DLAConfig(enabled=True, mu=1.0, sigma_scale=0.5)
    q_dense = per_gt_quota(gt_dense, 3, cfg, density=dn)
    q_sparse = per_gt_quota(gt_sparse, 3, cfg, density=dn)
    check("密集区配额 ≥ 稀疏区配额", q_dense[0] >= q_sparse[0], f"dense={q_dense} sparse={q_sparse}")
    check("配额为正整数", all(isinstance(q, int) and q >= 3 for q in q_dense + q_sparse))

    cfg_off = DLAConfig(enabled=False, mu=1.0)
    q_off = per_gt_quota(gt_dense, 3, cfg_off)
    check("关闭 DLA 时配额为基线值", q_off == [3], f"{q_off}")


def test_ags() -> None:
    print("\n=== AGS · 分级抑制 ===")
    # 场景 1：同一目标的重复框（高度重合，IoU≈0.90）→ 应被硬抑制
    dets = [
        Detection((0.0, 0.0, 20.0, 20.0), 0.9, 0),
        Detection((1.0, 1.0, 21.0, 21.0), 0.8, 0),
    ]
    iou_dup = 19 * 19 / (20 * 20 + 20 * 20 - 19 * 19)
    kept, st = ags_core(dets, AGSConfig(enabled=True, iou_thr=0.5, same_instance_iou=0.80, decay=0.5))
    check("重复框（高度重合）被硬抑制", st["n_hard_suppressed"] == 1 and len(kept) == 1,
          f"kept={len(kept)} hard={st['n_hard_suppressed']} IoU={iou_dup:.3f}")

    # 场景 2：两个粘连的不同个体（中等重合，IoU≈0.50，形状完全相同）
    #          → 关键回归用例：初版用"形状相似"判同一性会把它误杀
    dets2 = [
        Detection((0.0, 0.0, 20.0, 20.0), 0.9, 0),
        Detection((10.0, 0.0, 30.0, 20.0), 0.8, 0),
    ]
    iou_adh = 10 * 20 / (20 * 20 + 20 * 20 - 10 * 20)
    kept2, st2 = ags_core(dets2, AGSConfig(enabled=True, iou_thr=0.3, same_instance_iou=0.80, decay=0.5))
    check("粘连个体（中等重合、同形状）被保留", len(kept2) == 2,
          f"kept={len(kept2)} IoU={iou_adh:.3f}")
    check("其分数被衰减", st2["n_soft_decayed"] >= 1, f"soft={st2['n_soft_decayed']}")
    if len(kept2) == 2:
        low = min(d.score for d in kept2)
        check("被衰减者分数低于原值", low < 0.8, f"min_score={low:.4f}")

    # 场景 3：不同类别 → 不做任何处理（与 per-class NMS 一致）
    dets3 = [
        Detection((0.0, 0.0, 20.0, 20.0), 0.9, 0),
        Detection((1.0, 1.0, 21.0, 21.0), 0.8, 1),
    ]
    kept3, st3 = ags_core(dets3, AGSConfig(enabled=True, iou_thr=0.5, same_instance_iou=0.80))
    check("跨类别不抑制", len(kept3) == 2 and st3["n_hard_suppressed"] == 0)

    # 关闭开关 → 不做任何处理
    dets4 = [Detection((0.0, 0.0, 20.0, 20.0), 0.9, 0), Detection((0.5, 0.5, 20.5, 20.0), 0.8, 0)]
    kept4, st4 = ags_core(dets4, AGSConfig(enabled=False))
    check("关闭 AGS 时不抑制", len(kept4) == 2 and st4["n_hard_suppressed"] == 0)


def test_dra() -> None:
    print("\n=== DRA · 动态重采样 ===")
    names = ["common", "rare"]
    cfg = DRAConfig(enabled=True, alpha=1.0, tau=1.0)
    s = DRASampler(names, cfg, n_instances={"common": 10000, "rare": 108})
    w0 = s.weights(0)
    check("初始权重：稀有类权重更高", w0[1] > w0[0], f"common={w0[0]:.5f} rare={w0[1]:.5f}")
    check("权重归一化到 1", abs(sum(w0) - 1.0) < 1e-9, f"sum={sum(w0):.9f}")

    # 回灌 AP：common 表现好（AP 高），rare 表现差（AP 低）→ 难度项应进一步抬高 rare
    s.update(1, {"common": 0.9, "rare": 0.1})
    w1 = s.weights(1)
    check("难度项进一步提高困难类权重", w1[1] > w0[1], f"{w0[1]:.5f} → {w1[1]:.5f}")

    # 早停保护：rare 长期无提升（AP 持平）→ 权重被降
    # ⚠️ 关键回归用例：初版用 "ap < best_ap" 判无提升，AP **持平**时不触发 → 保护失效
    before = s.stats["rare"].penalty
    for e in range(2, 8):
        s.update(e, {"common": 0.95, "rare": 0.1})
    check("连续无提升触发降权保护（严格下降）", s.stats["rare"].penalty < before,
          f"penalty {before} → {s.stats['rare'].penalty}")
    check("降权后仍保留下限", s.stats["rare"].penalty >= s.stats["rare"].min_penalty,
          f"penalty={s.stats['rare'].penalty} min={s.stats['rare'].min_penalty}")

    # 平台期（AP 完全持平）也必须被判定为"无提升"
    s2 = DRASampler(["a"], DRAConfig(enabled=True, alpha=0.0, tau=0.0), n_instances={"a": 10})
    s2.update(0, {"a": 0.5})
    s2.update(1, {"a": 0.5})
    check("AP 持平被判定为无提升", s2.stats["a"].no_improve >= 1,
          f"no_improve={s2.stats['a'].no_improve}")

    # 静态对照
    ws = build_static_weights(names, {"common": 10000, "rare": 108}, alpha=1.0)
    check("静态权重归一化", abs(sum(ws) - 1.0) < 1e-9)
    check("静态对照：稀有类权重更高", ws[1] > ws[0])


def main() -> int:
    print("=" * 68)
    print("ADEA 核心算法自测（纯标准库，不依赖 torch）")
    print("=" * 68)
    test_adhesion_definition()
    test_ala_quota_rescue()
    test_dla()
    test_ags()
    test_dra()
    print("\n" + "=" * 68)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 68)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
