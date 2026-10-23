"""ADEA · D0 诊断采集器 —— 训练期记录"每 GT 正样本数 vs 粘连度"。

用途
----
为 G0 kill gate 提供数据：机制假说 M1 声称

    高粘连 (β 大) 且小尺寸的目标，其被分配到的正样本 anchor 数 ``n_pos``
    会系统性地偏少（被同簇的"支配者"目标挤占）。

本模块在 ``TaskAlignedAssigner.assign`` 上再打一层**只读记录**的补丁，
把每个 batch 的 (GT 粘连度, GT 正样本数, GT 面积) 落盘为 JSONL，
供 ``scripts/diagnose_adhesion.py`` 离线统计。

启用方式（环境变量）::

    ADEA_DIAG=1 ADEA_DIAG_OUT=runs/diag/assign_stats.jsonl yolo detect train ...

设计约束
--------
* **只记录、不改行为**：记录层不改变分配结果，可与 ALA 同时开启
  （但做 G0 检验时应关闭 ALA，否则测的是修复后的分布）。
* 写盘采用**追加 + 行缓冲**，训练中断也不丢已完成的部分。
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
    """xyxy 框的粘连系数 β = |A∩B| / |A∪B|。"""
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
    """记录一次 assign 调用的统计量。

    Parameters
    ----------
    target_gt_idx:
        (num_anchors,) 每个 anchor 归属的 GT 下标（-1 为背景）。
    fg_mask:
        (num_anchors,) 正样本掩码。
    gt_bboxes:
        (num_gt, 4) GT 框（xyxy，像素）。
    beta_thr:
        判定"非忽略粘连"的阈值。

    Returns
    -------
    dict
        本次调用的统计（同时会被追加写入 JSONL）。
    """
    n_gt = int(gt_bboxes.shape[0]) if gt_bboxes is not None else 0
    n_pos = [0] * n_gt
    if n_gt:
        idx = target_gt_idx.detach().cpu().tolist()
        mask = fg_mask.detach().cpu().tolist() if fg_mask is not None else [1] * len(idx)
        for a, m in zip(idx, mask):
            if m and a is not None and 0 <= a < n_gt:
                n_pos[a] += 1

    beta_max = [0.0] * n_gt
    beta_nn = [0.0] * n_gt
    if n_gt > 1:
        boxes = [tuple(float(v) for v in b) for b in gt_bboxes.detach().cpu().tolist()]
        for i in range(n_gt):
            best = 0.0
            for j in range(n_gt):
                if i == j:
                    continue
                b = _adhesion(boxes[i], boxes[j])
                if b > best:
                    best = b
            beta_max[i] = best
            beta_nn[i] = 1.0 if best > beta_thr else 0.0

    areas = []
    if n_gt:
        for b in gt_bboxes.detach().cpu().tolist():
            areas.append(max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1])))

    rec = {
        "batch": int(batch_idx),
        "n_gt": n_gt,
        "n_pos": n_pos,
        "beta_max": beta_max,
        "beta_nn": beta_nn,
        "areas": areas,
    }

    p = diag_path()
    if p is not None:
        try:
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return rec


def install_diagnostics() -> Dict[str, str]:
    """安装只读记录补丁。返回接线报告。"""
    if not is_enabled():
        return {"diagnostics": "skipped (ADEA_DIAG=0)"}

    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as exc:  # pragma: no cover
        return {"diagnostics": f"FAILED to import TaskAlignedAssigner: {exc}"}

    if getattr(TaskAlignedAssigner, "_adea_diag_patched", False):
        return {"diagnostics": "already patched"}

    orig_assign = TaskAlignedAssigner.assign
    counter = {"n": 0}

    def patched_assign(self, *args, **kwargs):
        result = orig_assign(self, *args, **kwargs)
        try:
            gt_idx, fg_mask, target_gt_idx = result[0], result[1], result[2]
            gt_bboxes = None
            for a in args:
                if hasattr(a, "shape") and len(a.shape) == 2 and a.shape[-1] == 4:
                    if gt_bboxes is None:
                        gt_bboxes = a
            if gt_bboxes is None and len(args) >= 3:
                gt_bboxes = args[2]
            if gt_bboxes is not None and len(gt_bboxes) > 0:
                record_assignment(target_gt_idx, fg_mask, gt_bboxes, batch_idx=counter["n"])
                counter["n"] += 1
        except Exception:
            pass
        return result

    TaskAlignedAssigner.assign = patched_assign
    TaskAlignedAssigner._adea_diag_patched = True
    return {"diagnostics": f"patched (read-only) -> {diag_path()}"}
