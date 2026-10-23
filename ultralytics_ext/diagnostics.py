"""ADEA · D0 诊断采集器 —— 训练期记录"每 GT 正样本数 vs 粘连度"。

用途
----
为 G0 kill gate 提供数据：机制假说 M1 声称

    高粘连 (β 大) 且小尺寸的目标，其被分配到的正样本 anchor 数 ``n_pos``
    会系统性地偏少（被同簇的"支配者"目标挤占）。

本模块在 ``TaskAlignedAssigner`` 上打**只读**记录补丁，把每个 batch 的
(GT 粘连度 β_max, GT 正样本数 n_pos, GT 面积) 落盘为 JSONL，
供 ``scripts/diagnose_adhesion.py`` 离线统计。

⚠️ ultralytics 8.4.x 的 API（实测）
-----------------------------------
* 8.3.x：分配入口是 ``TaskAlignedAssigner.assign(...)``；
* **8.4.184**：重构为 ``nn.Module``，分配入口是 ``forward(...)``，
  独占分配发生在 ``select_highest_overlaps(...)``。
  → 采集补丁必须挂在 ``select_highest_overlaps`` 上，否则**静默失效**。

设计约束
--------
* **只记录、不改行为**：记录层不改变分配结果；
* **必须与 ALA 互斥**：做 G0 检验时要测的是**未修复**的分布，
  因此采集时须保持 ``ADEA_ALA=0``（ALA 开启时测到的是修复后的分布）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["is_enabled", "diag_path", "record_assignment", "install_diagnostics"]


def is_enabled() -> bool:
    return os.environ.get("ADEA_DIAG", "0").strip().lower() in ("1", "true", "yes", "on")


def diag_path() -> Optional[Path]:
    raw = os.environ.get("ADEA_DIAG_OUT")
    if not raw:
        raw = "runs/diag/assign_stats.jsonl"
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _adhesion(a, b) -> float:
    """xyxy 框的粘连系数 β = |A∩B| / |A∪B|（Pest24 论文定义；数值上等同 IoU）。"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def record_assignment(
    target_gt_idx,
    fg_mask,
    gt_bboxes,
    beta_thr: float = 0.05,
    batch_idx: int = 0,
) -> Dict[str, Any]:
    """记录一次 assign 的统计量（逐 batch 元素）。

    Parameters
    ----------
    target_gt_idx:
        (b, num_anchors) 每个 anchor 归属的 GT 下标。
    fg_mask:
        (b, num_anchors) 正样本掩码。
    gt_bboxes:
        (b, n_max_boxes, 4) GT 框（xyxy，像素）。
    beta_thr:
        判定"非忽略粘连"的阈值。

    Returns
    -------
    dict
        本次调用的统计（同时追加写入 JSONL）。
    """
    n_b = int(target_gt_idx.shape[0])
    per_batch: List[Dict[str, Any]] = []

    for b in range(n_b):
        tgi = target_gt_idx[b].detach().cpu().tolist()
        fg = fg_mask[b].detach().cpu().bool().tolist()
        gtb = gt_bboxes[b].detach().cpu().tolist()
        n_gt = len(gtb)

        counts = [0] * n_gt
        for t, m in zip(tgi, fg):
            if m and 0 <= t < n_gt:
                counts[t] += 1

        boxes = [tuple(float(v) for v in g) for g in gtb]
        beta_max = [0.0] * n_gt
        for i in range(n_gt):
            best = 0.0
            for j in range(n_gt):
                if i == j:
                    continue
                v = _adhesion(boxes[i], boxes[j])
                if v > best:
                    best = v
            beta_max[i] = best

        areas = [
            max(0.0, g[2] - g[0]) * max(0.0, g[3] - g[1]) for g in gtb
        ]

        per_batch.append(
            {
                "batch": int(batch_idx),
                "b": b,
                "n_gt": n_gt,
                "n_pos": counts,
                "beta_max": beta_max,
                "beta_nn": [1.0 if v > beta_thr else 0.0 for v in beta_max],
                "areas": areas,
            }
        )

    p = diag_path()
    if p is not None:
        try:
            with p.open("a", encoding="utf-8") as f:
                for rec in per_batch:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return {"n_batches": 1, "records": per_batch}


def install_diagnostics() -> Dict[str, str]:
    """安装只读记录补丁。返回接线报告。

    补丁点（ultralytics 8.4.x）：
    * ``get_pos_mask`` —— 暂存 ``gt_bboxes``（后续算 GT 间粘连度需要）；
    * ``select_highest_overlaps`` —— 记录分配结果（**只读**，不改行为）。
    """
    if not is_enabled():
        return {"diagnostics": "skipped (ADEA_DIAG=0)"}

    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as exc:  # pragma: no cover
        return {"diagnostics": f"FAILED to import TaskAlignedAssigner: {exc}"}

    if getattr(TaskAlignedAssigner, "_adea_diag_patched", False):
        return {"diagnostics": "already patched"}

    counter = {"n": 0}

    orig_get_pos_mask = TaskAlignedAssigner.get_pos_mask

    def patched_get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra):
        self._adea_gt_bboxes = gt_bboxes
        self._adea_mask_gt = mask_gt
        return orig_get_pos_mask(
            self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra
        )

    orig_select = TaskAlignedAssigner.select_highest_overlaps

    def patched_select(self, mask_pos, overlaps, n_max_boxes, align_metric):
        target_gt_idx, fg_mask, mask_pos_out = orig_select(
            self, mask_pos, overlaps, n_max_boxes, align_metric
        )
        try:
            gt_bboxes = getattr(self, "_adea_gt_bboxes", None)
            if gt_bboxes is not None and gt_bboxes.shape[1] > 0:
                record_assignment(target_gt_idx, fg_mask, gt_bboxes, batch_idx=counter["n"])
                counter["n"] += 1
        except Exception:
            pass
        return target_gt_idx, fg_mask, mask_pos_out

    TaskAlignedAssigner.get_pos_mask = patched_get_pos_mask
    TaskAlignedAssigner.select_highest_overlaps = patched_select
    TaskAlignedAssigner._adea_diag_patched = True
    return {"diagnostics": f"patched (read-only) -> {diag_path()}"}
