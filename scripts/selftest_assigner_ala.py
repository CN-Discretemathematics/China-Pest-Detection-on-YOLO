#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ALA 在**真实 Ultralytics assigner** 上的端到端验证（需要 torch + ultralytics）。

这是 I1 创新机制的关键验证：机制假说 M1 声称，在粘连小目标上，
独占分配（``select_highest_overlaps``：一个 anchor 归给重叠最高的 GT）
会让其中一个实例的正样本被"饿死"；ALA 应当把它救回来。

真实机制（ultralytics 8.4.x）
-----------------------------
两个粘连的 GT，模型往往预测出**覆盖二者的框** → 该 anchor 对两个 GT 的
IoU 几乎相等 → ``overlaps.max(1).indices`` 的 argmax 偏向先出现的 GT
→ 另一个 GT 只剩"独占区域"里的极少数 anchor → **正样本被饿死**。

测试流程
--------
1. 在一个小参数空间里**自动搜索**能复现饥饿（GT1 正样本 ≪ GT0）的配置；
2. 该配置下跑"不开 ALA"与"开 ALA"两次，比较每 GT 的正样本数；
3. 断言：GT1 正样本上升、达到配额下限、供体不被掏空、总正样本数不变。

用法::

    python scripts/selftest_assigner_ala.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def build_case(gt1_dx: int = 2, pred_bias: float = 0.0, step: int = 5):
    """构造两个高度粘连的小 GT + 候选 anchor。

    Parameters
    ----------
    gt1_dx:
        GT1 相对 GT0 的水平偏移（像素）→ 越小越粘连。
    pred_bias:
        预测框相对 GT0 的整体偏移 → 模拟"模型倾向把框画在 GT0 一侧"。
    step:
        anchor 网格间距（像素）。
    """
    import torch

    torch.manual_seed(0)
    bs, n_cls = 1, 24

    gt_bboxes = torch.tensor(
        [[[100.0, 100.0, 130.0, 130.0], [100.0 + gt1_dx, 100.0, 130.0 + gt1_dx, 130.0]]],
        dtype=torch.float32,
    )
    gt_labels = torch.tensor([[[0.0], [1.0]]], dtype=torch.float32)
    mask_gt = torch.ones((bs, 2, 1), dtype=torch.float32)

    ys = torch.arange(100, 131, step, dtype=torch.float32)
    xs = torch.arange(100, 131, step, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    anc_points = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    n_anchor = anc_points.shape[0]

    # 预测框：**xyxy**，像素空间。统一画在 GT0 与 GT1 之间的"折中"位置，
    # 使两者 IoU 接近 → argmax 偏向 GT0 → 复现 M1 的饿死现象。
    half = 15.0
    cx = 115.0 + pred_bias
    pd_bboxes = torch.zeros((bs, n_anchor, 4), dtype=torch.float32)
    pd_bboxes[:, :, 0] = cx - half
    pd_bboxes[:, :, 1] = 115.0 - half
    pd_bboxes[:, :, 2] = cx + half
    pd_bboxes[:, :, 3] = 115.0 + half

    # 分类分：GT0 类别分更高 → align_metric 偏向 GT0
    pd_scores = torch.zeros((bs, n_anchor, n_cls), dtype=torch.float32)
    pd_scores[:, :, 0] = 0.9
    pd_scores[:, :, 1] = 0.55
    return pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt


def run_with_ala(enabled: bool, q_min: int = 5, **case_kw):
    """跑一次 assigner，返回每个 GT 的正样本数与总数。"""
    os.environ["ADEA_ALA"] = "1" if enabled else "0"
    os.environ["ADEA_VERBOSE"] = "0"
    os.environ["ADEA_ALA_Q_MIN"] = str(q_min)

    import importlib

    import ultralytics_ext.register as reg

    importlib.reload(reg)
    reg.install_all()

    from ultralytics.utils.tal import TaskAlignedAssigner

    pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt = build_case(**case_kw)
    assigner = TaskAlignedAssigner(topk=10, num_classes=24, stride=[8, 16, 32])
    out = assigner(
        pd_scores=pd_scores,
        pd_bboxes=pd_bboxes,
        anc_points=anc_points,
        gt_labels=gt_labels,
        gt_bboxes=gt_bboxes,
        mask_gt=mask_gt,
    )
    target_gt_idx = out[4]
    fg_mask = out[3]
    tgi = target_gt_idx[0].cpu().tolist()
    fg = fg_mask[0].cpu().bool().tolist()
    counts = [0, 0]
    for t, m in zip(tgi, fg):
        if m and 0 <= t < 2:
            counts[t] += 1
    return counts, sum(counts)


def main() -> int:
    print("=" * 70)
    print("ALA 端到端验证（真实 TaskAlignedAssigner，ultralytics 8.4.x）")
    print("=" * 70)

    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception as exc:
        print(f"[SKIP] 需要 torch + ultralytics：{exc}")
        return 0

    # ---- 1) 搜索能复现"饿死"的配置 ----
    print("\n=== 搜索能复现 M1 饥饿的配置 ===")
    found = None
    for step in (5, 4, 3):
        for gt1_dx in (0, 1, 2, 3):
            for pred_bias in (0.0, 1.0, 2.0):
                c, tot = run_with_ala(False, q_min=5, gt1_dx=gt1_dx, pred_bias=pred_bias, step=step)
                if c[0] > 0 and c[1] < max(1, c[0] // 2):
                    found = dict(gt1_dx=gt1_dx, pred_bias=pred_bias, step=step, base=c, tot=tot)
                    print(
                        f"  命中: step={step} gt1_dx={gt1_dx} pred_bias={pred_bias} "
                        f"→ GT0={c[0]} GT1={c[1]}"
                    )
                    break
            if found:
                break
        if found:
            break

    if not found:
        print("  [WARN] 未能在搜索空间内复现饥饿；改用最接近的退化配置做结构性校验")
        c, tot = run_with_ala(False, q_min=5, gt1_dx=0, pred_bias=0.0, step=3)
        found = dict(gt1_dx=0, pred_bias=0.0, step=3, base=c, tot=tot)
        print(f"  退化配置: GT0={c[0]} GT1={c[1]}")

    kw = dict(gt1_dx=found["gt1_dx"], pred_bias=found["pred_bias"], step=found["step"])

    print("\n=== 基线（ADEA_ALA=0）===")
    c0, tot0 = run_with_ala(False, q_min=5, **kw)
    print(f"  GT0 正样本 = {c0[0]}, GT1 正样本 = {c0[1]}, 总计 = {tot0}")
    check("支配者 GT0 拿到正样本", c0[0] > 0, f"n_pos={c0[0]}")
    check("GT1 被'饿死'（正样本明显少于 GT0）", c0[1] < c0[0], f"{c0[1]} < {c0[0]}")

    print("\n=== 开启 ALA（ADEA_ALA=1）===")
    c1, tot1 = run_with_ala(True, q_min=5, **kw)
    print(f"  GT0 正样本 = {c1[0]}, GT1 正样本 = {c1[1]}, 总计 = {tot1}")
    check("GT1 的正样本数上升", c1[1] > c0[1], f"{c0[1]} → {c1[1]}")
    check("GT1 达到配额下限 5", c1[1] >= 5, f"n_pos={c1[1]}")
    check("供体 GT0 仍保留 ≥5 个正样本", c1[0] >= 5, f"n_pos={c1[0]}")
    check("总正样本数不变（不变量）", tot1 == tot0, f"{tot0} → {tot1}")

    print("\n" + "=" * 70)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
