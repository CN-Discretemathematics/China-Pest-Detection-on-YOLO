"""ADEA · I1b · SFW —— 尺度公平的监督权重 (Scale-Fair supervision Weighting)

动机（由本项目 G0 诊断实测导出，不是拍脑袋）
------------------------------------------------
Ultralytics 的分类损失是

    loss_cls = Σ_anchors BCE(pred_scores, target_scores) / Σ target_scores

即**每个正样本等权**。而一个 GT 能拿到多少正样本，是由**几何**决定的
（候选池 ∝ 面积/stride²，实测 ``corr(log 相对面积, log 候选池) = 0.965``），
于是：

    一个 GT 对分类损失的贡献 ∝ 它的正样本数 ∝ 它的面积

→ 大目标在分类损失里的权重天然是微目标的 **数倍**，
这就是一条**隐式的尺度偏置**（与长尾类别偏置同源，但发生在**尺度轴**上）。
Pest24 数据集恰好同时具备"尺度跨度 1600×"与"长尾 53,347 vs 108"两个特征，
微目标与稀有类的召回被这条偏置系统性地压低。

SFW 的做法
----------
把分类损失里**正样本**的权重从"等权"改成"**按 GT 归一**"：

    w_a = (N_pos / N_gt) / count_{gt(a)}      （a 为正样本 anchor）

于是每个 GT 对分类损失的**总贡献恰好相等**（都等于 ``N_pos/N_gt``），
而背景 anchor 的权重保持 1.0 → 正负样本的平衡与损失量级都**不变**
（Σ_pos w = N_pos，分母 ``target_scores_sum`` 不需要动）。

实现
----
* 补丁 1（``TaskAlignedAssigner.forward``）：分配完成后统计每个 GT 的正样本数，
  算出每个 anchor 的权重并存到模块级 ``_STATE``；
* 补丁 2（``v8DetectionLoss.__init__``）：把 ``self.bce`` 包一层，
  乘上刚才算好的权重（形状 (b, A, 1)，广播到类别维）。

为什么放在分类损失而不是回归损失
------------------------------
回归损失（DFL/L1）已由 ultralytics 用 ``target_scores_sum`` 归一，
其尺度效应与 Balanced L1 Loss（CVPR 2019）处理的是同一件事；
而**分类项**的尺度偏置此前**没有被处理过**——正样本数这个通道是新的。

⚠️ 失效保护：任何异常一律回退原始 BCE（权重视为 1），并计数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

__all__ = ["ScaleFairConfig", "install_scale_fair_weighting", "scale_fair_health"]

# 模块级状态：assigner 写入 → loss 的 BCE 读取（同一进程内串行调用，顺序由框架保证）
_STATE: Dict[str, Any] = {"weight": None}


@dataclass
class ScaleFairConfig:
    """SFW 配置。

    Attributes
    ----------
    enabled:
        总开关。
    clip:
        单个正样本权重的上下界（相对 ``N_pos/N_gt`` 的倍数），
        防止极端情况下（某 GT 只有 1 个正样本）权重过大。
    verbose:
        是否打印接线/统计信息。
    """

    enabled: bool = False
    clip: float = 10.0
    verbose: bool = True


def install_scale_fair_weighting(cfg: ScaleFairConfig) -> Dict[str, str]:
    """安装尺度公平监督权重补丁。返回接线报告。"""
    if not cfg.enabled:
        return {"scale_fair": "skipped (ADEA_SFW=0)"}

    try:
        from ultralytics.utils.loss import v8DetectionLoss
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as exc:  # pragma: no cover
        return {"scale_fair": f"FAILED to import ultralytics loss/assigner: {exc}"}

    if getattr(TaskAlignedAssigner, "_adea_sfw_patched", False):
        return {"scale_fair": "already patched"}

    state = {"errors": 0, "calls": 0, "n_gt": 0, "n_pos": 0, "first_tb": None}

    # ---- 补丁 1：assigner.forward 结束后算每 anchor 的权重 ----
    orig_forward = TaskAlignedAssigner.forward

    def patched_forward(self, *args, **kwargs):
        result = orig_forward(self, *args, **kwargs)
        try:
            target_gt_idx, fg_mask = result[4], result[3]
            b, n_anchor = fg_mask.shape
            fg = fg_mask.bool()
            idx = target_gt_idx.long().clamp(min=0)
            counts = torch.zeros((b, self.n_max_boxes), device=fg.device, dtype=torch.float32)
            counts.scatter_add_(
                1, idx, (fg & (target_gt_idx >= 0)).to(torch.float32)
            )
            n_pos = float(counts.sum())
            n_gt = float((counts > 0).sum())
            if n_pos <= 0 or n_gt <= 0:
                _STATE["weight"] = None
                return result
            base = n_pos / n_gt                      # 保持总权重 = N_pos
            w = base / counts.gather(1, idx).clamp(min=1.0)
            w = w * fg.to(w.dtype)
            if cfg.clip > 0:
                w = w.clamp(max=base * float(cfg.clip))
            w = torch.where(fg, w, torch.ones_like(w))
            _STATE["weight"] = w.detach()
            state["calls"] += 1
            state["n_gt"] += n_gt
            state["n_pos"] += n_pos
        except Exception:  # pragma: no cover
            state["errors"] += 1
            _STATE["weight"] = None
        return result

    # ---- 补丁 2：loss 的 BCE 乘上权重 ----
    orig_init = v8DetectionLoss.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        orig_bce = self.bce

        def weighted_bce(pred, target):
            out = orig_bce(pred, target)
            w = _STATE.get("weight")
            if w is not None and w.shape == out.shape[:2]:
                out = out * w.unsqueeze(-1).to(out.dtype)
            return out

        self.bce = weighted_bce

    TaskAlignedAssigner.forward = patched_forward
    v8DetectionLoss.__init__ = patched_init
    TaskAlignedAssigner._adea_sfw_patched = True
    TaskAlignedAssigner._adea_sfw_state = state
    return {
        "scale_fair": (
            f"patched TaskAlignedAssigner.forward + v8DetectionLoss.bce "
            f"(clip={cfg.clip}) —— 每个 GT 的分类损失贡献归一为相等"
        )
    }


def scale_fair_health() -> Dict[str, object]:
    """读取 SFW 的运行统计（用于训练后确认它真的生效了）。"""
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception:  # pragma: no cover
        return {"patched": False, "errors": -1, "calls": -1}
    st = getattr(TaskAlignedAssigner, "_adea_sfw_state", None)
    if st is None:
        return {"patched": False, "errors": 0, "calls": 0}
    return {
        "patched": True,
        "errors": st["errors"],
        "calls": st["calls"],
        "mean_pos_per_gt": round(st["n_pos"] / st["n_gt"], 4) if st["n_gt"] else None,
    }
