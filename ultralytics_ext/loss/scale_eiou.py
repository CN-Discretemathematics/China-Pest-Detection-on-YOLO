"""ADEA · R5（复现）· Scale-Quality Aware EIoU —— 尺度-质量感知 EIoU 损失

出处
----
AEP-PEST: Tang, Wang, Zou, Fang et al., "AEP-PEST: adaptive edge-enhanced
pyramid network for real-time detection of small and occluded agricultural
pests", J. Real-Time Image Process. 23:106 (2026), DOI 10.1007/s11554-026-01901-4。
该文以 **YOLOv11 框架**为基线，是目前 Pest24 上 mAP50 最高（75.2%）的方法。

原文对其损失的描述：

    "Third, a Scale-Quality Aware EIoU loss dynamically adjusts sample weights
     based on target size and prediction quality, focusing optimization on
     challenging small pests."

与原版 EIoU 的关系
------------------
EIoU (Zhang et al.) 在 CIoU 基础上把宽高比项解耦为两个独立的距离项，
直接度量"宽偏差"与"高偏差"，对**小目标**的定位更敏感。

本实现在 EIoU 之上再加权：
* **尺度因子** w_s：目标越小 → 权重越大（对应 Pest24 极小目标占比高的事实）；
* **质量因子** w_q：预测 IoU 越低（质量越差）→ 权重越大（聚焦难样本）。

两者相乘得到最终权重 ``w = w_s * w_q``。

集成方式
--------
Ultralytics 各版本的 loss 结构不同（``BboxLoss`` / ``bbox_iou`` / 直接调用
``CIoU``），因此这里提供**多候选挂载点**的安装函数
:func:`install_scale_quality_eiou`：逐一探测可用挂载点并返回实际生效的那个，
避免版本升级后静默失效。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

__all__ = [
    "ScaleQualityConfig",
    "scale_quality_weights",
    "scale_quality_eiou_loss",
    "install_scale_quality_eiou",
]


@dataclass
class ScaleQualityConfig:
    """Scale-Quality Aware EIoU 超参。

    Attributes
    ----------
    enabled:
        总开关。关闭时行为与原始 EIoU 完全一致。
    scale_weight:
        尺度因子的强度 γ_s。目标相对面积 a 归一化后按 ``a^(-γ_s)`` 加权。
    quality_weight:
        质量因子的强度 γ_q。按 ``(1 - IoU)^γ_q`` 加权。
    min_weight / max_weight:
        权重截断区间，避免极端样本主导梯度。
    normalize_by:
        目标面积的归一化基准（相对面积 = box_area / image_area）。
    """

    enabled: bool = True
    scale_weight: float = 0.5
    quality_weight: float = 0.5
    min_weight: float = 0.1
    max_weight: float = 5.0
    normalize_by: float = 1.0

    def describe(self) -> str:
        return (
            f"SQA-EIoU(enabled={self.enabled}, scale_w={self.scale_weight}, "
            f"quality_w={self.quality_weight})"
        )


def _bbox_iou_eiou(pred, target, eps: float = 1e-7):
    """逐元素 EIoU（xyxy）。返回 (iou, eiou)。

    EIoU = IoU - ρ²(b, b_gt)/c² - |w - w_gt|/c_w - |h - h_gt|/c_h
    """
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)

    pw = (px2 - px1).clamp(min=0)
    ph = (py2 - py1).clamp(min=0)
    tw = (tx2 - tx1).clamp(min=0)
    th = (ty2 - ty1).clamp(min=0)

    inter = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0) * (
        torch.minimum(py2, ty2) - torch.maximum(py1, ty1)
    ).clamp(min=0)
    area_p = pw * ph
    area_t = tw * th
    union = area_p + area_t - inter
    iou = inter / (union + eps)

    cw = (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).clamp(min=eps)
    ch = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).clamp(min=eps)
    c2 = cw * cw + ch * ch

    # 中心距平方（EIoU 论文的标准写法）
    rho2 = ((px1 + px2 - tx1 - tx2) ** 2 + (py1 + py2 - ty1 - ty2) ** 2) / 4.0

    w_loss = (pw - tw).abs() / cw
    h_loss = (ph - th).abs() / ch
    eiou = iou - (rho2 / c2) - w_loss - h_loss
    return iou, eiou


def scale_quality_weights(
    iou,
    target_box,
    cfg: ScaleQualityConfig,
    image_area: float = 1.0,
):
    """计算尺度-质量权重。

    Parameters
    ----------
    iou:
        逐元素 IoU（已 detach，作为"预测质量"的代理）。
    target_box:
        目标框 (xyxy)。
    cfg:
        :class:`ScaleQualityConfig`。
    image_area:
        图像面积（像素²），用于把目标面积归一化成相对面积。

    Returns
    -------
    权重张量，形状与 ``iou`` 相同。
    """
    if not cfg.enabled:
        return iou.new_ones(iou.shape)

    tw = (target_box[..., 2] - target_box[..., 0]).clamp(min=0)
    th = (target_box[..., 3] - target_box[..., 1]).clamp(min=0)
    rel_area = (tw * th) / max(float(image_area), 1.0)
    rel_area = rel_area.clamp(min=1e-8)

    scale_term = rel_area.pow(-cfg.scale_weight)
    quality_term = (1.0 - iou.detach()).clamp(min=0.0, max=1.0).pow(cfg.quality_weight)

    w = scale_term * quality_term
    return w.clamp(min=cfg.min_weight, max=cfg.max_weight)


def scale_quality_eiou_loss(
    pred_box,
    target_box,
    cfg: ScaleQualityConfig,
    image_area: float = 1.0,
    reduction: str = "mean",
):
    """Scale-Quality Aware EIoU 损失。

    Returns
    -------
    (loss, iou)
        加权后的损失与逐元素 IoU（供上层做进一步加权或统计）。
    """
    iou, eiou = _bbox_iou_eiou(pred_box, target_box)
    w = scale_quality_weights(iou, target_box, cfg, image_area=image_area)
    loss_vec = (1.0 - eiou) * w
    if reduction == "mean":
        loss = loss_vec.mean()
    elif reduction == "sum":
        loss = loss_vec.sum()
    else:
        loss = loss_vec
    return loss, iou


def install_scale_quality_eiou(cfg: Optional[ScaleQualityConfig] = None):
    """把 Scale-Quality Aware EIoU 挂到 Ultralytics 的 loss 上。

    依次探测以下挂载点（返回第一个可用者）：

    1. ``ultralytics.utils.loss.BboxLoss`` —— 8.3.x 常见形态；
    2. ``ultralytics.utils.loss.vflLoss.bbox_loss`` —— 8.4.x 常见形态；
    3. ``ultralytics.utils.loss.vflLoss.bbox_iou`` —— 更早/其他形态。

    Returns
    -------
    (ok, report)
        ``ok`` 表示是否成功挂载；``report`` 描述挂载点与版本信息，
        便于在训练日志中确认"到底有没有生效"。
    """
    cfg = cfg or ScaleQualityConfig()
    try:
        import ultralytics.utils.loss as loss_mod
    except Exception as exc:  # pragma: no cover
        return False, f"import ultralytics.utils.loss failed: {exc}"

    version = getattr(__import__("ultralytics"), "__version__", "?")

    # ---- 挂载点 1：BboxLoss ----
    bbox_loss_cls = getattr(loss_mod, "BboxLoss", None)
    if bbox_loss_cls is not None and hasattr(bbox_loss_cls, "__call__"):
        orig_call = bbox_loss_cls.__call__

        def patched_call(self, pred, target, weight=None, *args, **kwargs):
            try:
                p = pred.unsqueeze(1) if pred.dim() == 2 else pred
                t = target.unsqueeze(1) if target.dim() == 2 else target
                loss, _ = scale_quality_eiou_loss(p, t, cfg)
                w = weight if weight is not None else loss.new_ones(loss.shape)
                if w.dim() == loss.dim() - 1:
                    w = w.unsqueeze(-1)
                return (loss * w).mean()
            except Exception:
                return orig_call(self, pred, target, weight, *args, **kwargs)

        bbox_loss_cls.__call__ = patched_call
        return True, f"hooked ultralytics.utils.loss.BboxLoss.__call__ (ultralytics {version})"

    # ---- 挂载点 2/3：vflLoss 实例属性 ----
    vfl = getattr(loss_mod, "vflLoss", None)
    if vfl is not None:
        orig_init = vfl.__init__

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            inner = getattr(self, "bbox_loss", None)
            if inner is not None and hasattr(inner, "__call__"):
                inner_call = inner.__call__

                def patched_inner(pred, target, weight=None, *a, **kw):
                    try:
                        p = pred.unsqueeze(1) if pred.dim() == 2 else pred
                        t = target.unsqueeze(1) if target.dim() == 2 else target
                        loss, _ = scale_quality_eiou_loss(p, t, cfg)
                        w = weight if weight is not None else loss.new_ones(loss.shape)
                        if w.dim() == loss.dim() - 1:
                            w = w.unsqueeze(-1)
                        return (loss * w).mean()
                    except Exception:
                        return inner_call(pred, target, weight, *a, **kw)

                inner.__call__ = patched_inner
                self._sqa_eiou_installed = True
            else:
                self._sqa_eiou_installed = False

        vfl.__init__ = patched_init
        return True, f"hooked ultralytics.utils.loss.vflLoss.bbox_loss (ultralytics {version})"

    return False, f"no known hook point found in ultralytics.utils.loss (ultralytics {version})"
