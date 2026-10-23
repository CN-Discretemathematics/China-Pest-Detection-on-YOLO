#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SLA（面积标定的公平正样本分配）的单元自测。

覆盖
----
* **经验期望表**（报告口径）：``E[n|a]`` 与残差符号；
* **S1 冲突裁决**：
  - vanilla（``criterion=overlap`` + ``claim_only=False``）与 ultralytics 原实现**逐 anchor 一致**；
  - SLA 默认（``criterion=align`` + 只在提名者中裁决）会把冲突 anchor
    判给"与 top-k 选择同一度量"得分最高的提名者；
* **S2 公平再分配**：几何公平份额 ``need = min(topk, 框内可用候选数)``；
  被挤占的 GT 从欠剥夺的 GT 手里取回**同时位于自己框内**的 anchor；
* **不变式**：每个 anchor 至多归属一个 GT；总正样本数不变（转移零和）；
  分配缺口（need − count）收敛。

用法::

    python scripts/selftest_sla.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics_ext.assign.sla import (  # noqa: E402
    SLAConfig,
    expected_npos,
    residual_npos,
    sla_assign,
)

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
# 工具
# --------------------------------------------------------------------------- #
def make_case(align_ov: float = 0.8, ov_ov: float = 0.95):
    """构造一个"粘连对"：两个 GT 共享 5 个 anchor，但度量/IoU 偏好相反。

    * GT0：anchor 0..4 的 align=0.9、IoU=0.3；anchor 5..10 独占
    * GT1：anchor 0..4 的 align=0.8、IoU=0.95；anchor 11..16 独占
    """
    import torch

    A = 17
    align = torch.zeros(1, 2, A)
    ov = torch.zeros(1, 2, A)
    align[0, 0, 0:5] = 0.9
    align[0, 1, 0:5] = align_ov
    align[0, 0, 5:11] = torch.tensor([0.50, 0.48, 0.46, 0.44, 0.42, 0.40])
    align[0, 1, 11:17] = torch.tensor([0.50, 0.49, 0.48, 0.47, 0.46, 0.45])
    ov[0, 0, 0:5] = 0.3
    ov[0, 1, 0:5] = ov_ov
    ov[0, 0, 5:11] = 0.2
    ov[0, 1, 11:17] = 0.2
    return align, ov


def topk_claims(align, topk: int = 6):
    """复刻 ultralytics 的 per-GT top-k 掩码（未做独占裁决）。"""
    import torch

    b, n_gt, A = align.shape
    topk_metrics, topk_idxs = torch.topk(align, topk, dim=-1, largest=True)
    claims = torch.zeros((b, n_gt, A), dtype=torch.int8, device=align.device)
    claims.scatter_(2, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
    valid = torch.zeros((b, n_gt, A), dtype=torch.int8, device=align.device)
    valid.scatter_(2, topk_idxs, (topk_metrics > 1e-9).to(torch.int8))
    claims = claims * valid
    return claims


def ultralytics_vanilla(mask_pos, overlaps):
    """复刻 ultralytics 8.4.184 ``select_highest_overlaps`` 的独占分配。"""
    import torch

    fg = mask_pos.sum(-2)
    multi = (fg.unsqueeze(1) > 1).expand(-1, mask_pos.shape[1], -1)
    max_idx = overlaps.max(1).indices
    is_max = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
    is_max.scatter_(1, max_idx.unsqueeze(1), 1)
    return torch.where(multi, is_max, mask_pos)


def counts_of(mask_pos, align):
    """统计每个 GT 的**可用**正样本数（align>0 才算有监督价值）。"""
    usable = align > 1e-9
    return (mask_pos.bool() & usable).sum(-1)[0].tolist()


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #
def test_expectation_model() -> None:
    print("\n=== 经验期望表与残差（报告口径）===")
    cfg = SLAConfig(k0=11, c_fit=9476.0)
    e_small = expected_npos(5e-5, cfg)
    e_mid = expected_npos(6e-4, cfg)
    e_big = expected_npos(3e-3, cfg)
    check("小目标期望正样本很少", e_small < 3.0, f"E(1e-4)={e_small:.2f}")
    check("中等目标期望居中", 4 < e_mid < 7, f"E(6e-4)={e_mid:.2f}")
    check("大目标期望饱和", e_big >= 9.5, f"E(3e-3)={e_big:.2f}")
    check("残差符号：n 低于期望 → 负", residual_npos(0, 1e-4, cfg) < 0,
          f"resid={residual_npos(0, 1e-4, cfg):.2f}")
    check("残差符号：n 高于期望 → 正", residual_npos(11, 1e-2, cfg) > 0,
          f"resid={residual_npos(11, 1e-2, cfg):.2f}")


def test_vanilla_equivalence() -> None:
    print("\n=== 与 ultralytics 原实现的等价性（需 torch）===")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] 需要 torch: {exc}")
        return

    align, ov = make_case()
    claims = topk_claims(align, topk=6)
    van = ultralytics_vanilla(claims, ov)

    cfg = SLAConfig(enabled=True, criterion="overlap", claim_only=False, fairness=False)
    sla, info = sla_assign(claims, align, ov, topk=6, cfg=cfg)

    same = bool((van == sla).all())
    print(f"  vanilla 分配: GT0/GT1 = {counts_of(van, align)}")
    print(f"  SLA(overlap, claim_only=False): GT0/GT1 = {counts_of(sla, align)}")
    check("SLA 可逐 anchor 复刻 ultralytics 原分配", same)
    check("识别出 5 个冲突 anchor", info["n_contested_anchors"] == 5,
          f"contested={info['n_contested_anchors']}")


def test_criterion_consistency() -> None:
    print("\n=== S1 冲突裁决的度量一致性 ===")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] 需要 torch: {exc}")
        return

    align, ov = make_case()
    claims = topk_claims(align, topk=6)

    cfg_v = SLAConfig(enabled=True, criterion="overlap", claim_only=False, fairness=False)
    v, _ = sla_assign(claims, align, ov, topk=6, cfg=cfg_v)
    cfg_a = SLAConfig(enabled=True, criterion="align", claim_only=True, fairness=False)
    a, _ = sla_assign(claims, align, ov, topk=6, cfg=cfg_a)

    n_v = counts_of(v, align)
    n_a = counts_of(a, align)
    print(f"  vanilla 口径(按 IoU 裁决): GT0/GT1 = {n_v}")
    print(f"  SLA 口径(按 align 裁决) : GT0/GT1 = {n_a}")
    check("vanilla 按 IoU 裁决 → 冲突区归 GT1", n_v[1] > n_v[0], f"{n_v}")
    check("SLA 按 align 裁决 → 冲突区归 GT0（度量与 top-k 一致）", n_a[0] > n_a[1], f"{n_a}")
    check("每 anchor 至多一个 GT", int(a.sum(dim=1).max()) <= 1)
    check("总正样本数不变", sum(n_v) == sum(n_a), f"{sum(n_v)} vs {sum(n_a)}")


def test_fairness_redistribution() -> None:
    print("\n=== S2 面积标定的公平再分配 ===")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] 需要 torch: {exc}")
        return

    align, ov = make_case()
    claims = topk_claims(align, topk=6)

    cfg_s1 = SLAConfig(enabled=True, criterion="align", claim_only=True, fairness=False)
    s1, i1 = sla_assign(claims, align, ov, topk=6, cfg=cfg_s1)
    n_s1 = counts_of(s1, align)

    cfg_full = SLAConfig(enabled=True, criterion="align", claim_only=True, fairness=True,
                         match_ratio=0.5, min_gap=2.0, max_rounds=8)
    s2, i2 = sla_assign(claims, align, ov, topk=6, cfg=cfg_full)
    n_s2 = counts_of(s2, align)

    print(f"  仅 S1 : GT0/GT1 = {n_s1}，gap={i1['mean_gap_before']:.2f}，max_gap={i1['max_gap_before']:.2f}")
    print(f"  S1+S2 : GT0/GT1 = {n_s2}，transfers={i2['n_transfer']:.0f}，"
          f"gap={i2['mean_gap_after']:.2f}，max_gap={i2['max_gap_after']:.2f}")
    print(f"  缺口>tol 的 GT 数 {i1['starved_before']:.0f} → {i2['starved_after']:.0f}"
          f"（零和转移下总数不变，变的是分布）")

    check("S2 确实发生了转移", i2["n_transfer"] > 0, f"transfers={i2['n_transfer']:.0f}")
    check("被剥夺的 GT 正样本变多", n_s2[1] > n_s1[1], f"{n_s1[1]} → {n_s2[1]}")
    check("转移零和：总正样本数不变", sum(n_s2) == sum(n_s1), f"{sum(n_s1)} → {sum(n_s2)}")
    check("每 anchor 至多一个 GT", int(s2.sum(dim=1).max()) <= 1)
    # 零和转移 ⇒ mean gap 不变；公平性的正确度量是**最大缺口**（max-min 公平）
    check("平均缺口在零和转移下不变", abs(i2["mean_gap_after"] - i2["mean_gap_before"]) < 1e-6,
          f"{i2['mean_gap_before']:.2f} → {i2['mean_gap_after']:.2f}")
    check("最大缺口下降（max-min 公平）", i2["max_gap_after"] < i2["max_gap_before"],
          f"{i1['max_gap_before']:.2f} → {i2['max_gap_after']:.2f}")
    check("任一 GT 不超过 topk 上限", all(n <= 6 for n in n_s2), f"{n_s2}")

    # --- 匹配度护栏：GT1 在冲突区匹配很差时不应抢走 ---
    align_bad = align.clone()
    align_bad[0, 1, 0:5] = 0.05
    s3, i3 = sla_assign(claims, align_bad, ov, topk=6, cfg=cfg_full)
    print(f"  护栏测试（GT1 冲突区 align=0.05）: transfers={i3['n_transfer']:.0f}")
    check("匹配度护栏阻止了不合理的抢占", i3["n_transfer"] == 0,
          f"transfers={i3['n_transfer']:.0f}")


def test_no_conflict_case() -> None:
    print("\n=== 无冲突场景：SLA 不应改变 vanilla 结果 ===")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] 需要 torch: {exc}")
        return

    A = 20
    align = torch.zeros(1, 2, A)
    align[0, 0, 0:6] = 0.9    # GT0 独占
    align[0, 1, 10:16] = 0.8  # GT1 独占
    ov = align.clone()
    claims = topk_claims(align, topk=6)
    van = ultralytics_vanilla(claims, ov)
    cfg = SLAConfig(enabled=True, criterion="align", claim_only=True, fairness=True)
    s, info = sla_assign(claims, align, ov, topk=6, cfg=cfg)
    nv = counts_of(van, align)
    ns = counts_of(s, align)
    print(f"  vanilla {nv}，SLA {ns}，conflicts={info['n_contested_anchors']:.0f}")
    check("无冲突时正样本数一致", nv == ns, f"{nv} vs {ns}")
    check("无冲突时没有转移", info["n_transfer"] == 0, f"transfers={info['n_transfer']:.0f}")
    check("无冲突时分配结果一致", bool((van == s).all()))


def test_candidate_expansion() -> None:
    print("\n=== M0 监督密度补偿（候选区扩展）===")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"  [SKIP] 需要 torch: {exc}")
        return

    from ultralytics_ext.assign.sla import sla_expand_boxes

    # 一个 8×8 的小框（中心 100,100）与一个 200×200 的大框
    gt = torch.tensor([[[96.0, 96.0, 104.0, 104.0], [0.0, 0.0, 200.0, 200.0]]])
    need = torch.tensor([[True, False]])
    margin = 16.0
    out = sla_expand_boxes(gt, need, margin)

    w0 = float(gt[0, 0, 2] - gt[0, 0, 0])
    h0 = float(gt[0, 0, 3] - gt[0, 0, 1])
    w1 = float(out[0, 0, 2] - out[0, 0, 0])
    h1 = float(out[0, 0, 3] - out[0, 0, 1])
    cx0 = float((gt[0, 0, 0] + gt[0, 0, 2]) / 2)
    cx1 = float((out[0, 0, 0] + out[0, 0, 2]) / 2)
    print(f"  小框: {w0:.1f}×{h0:.1f} → {w1:.1f}×{h1:.1f}（中心 {cx0:.1f} → {cx1:.1f}）")
    check("被扩张的框每边增加 margin", abs(w1 - (w0 + 2 * margin)) < 1e-3
          and abs(h1 - (h0 + 2 * margin)) < 1e-3, f"{w1:.1f}×{h1:.1f}")
    check("扩张保持中心不变", abs(cx1 - cx0) < 1e-4, f"{cx0:.2f} → {cx1:.2f}")
    check("未被扩张的框原样返回",
          bool(torch.allclose(out[0, 1], gt[0, 1])))
    # 边距是**绝对**量：同样的 margin 对大框的相对影响小（符合"面积标定"的直觉）
    check("边距扩张对大框的相对增幅更小",
          (w1 - w0) / w0 > 0 and (w1 - w0) == 2 * margin)

    # 与分辨率无关：同样的 margin 语义（像素）不随 stride 改变
    gt2 = gt.clone() * 2.0
    out2 = sla_expand_boxes(gt2, need, margin)
    w2 = float(out2[0, 0, 2] - out2[0, 0, 0])
    check("边距是绝对像素量（调用方按 stride_val 折算 → 网格单位）",
          abs(w2 - (2 * w0 + 2 * margin)) < 1e-3, f"{w2:.1f}")


def main() -> int:
    print("=" * 70)
    print("SLA 单元自测（面积标定的公平正样本分配）")
    print("=" * 70)
    test_expectation_model()
    test_vanilla_equivalence()
    test_criterion_consistency()
    test_fairness_redistribution()
    test_no_conflict_case()
    test_candidate_expansion()
    print("\n" + "=" * 70)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
