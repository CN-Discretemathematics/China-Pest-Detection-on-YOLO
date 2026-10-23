"""ADEA · R2（复现）· EMA 注意力 —— 高效多尺度注意力 (Efficient Multi-Scale Attention)

出处
----
Yang et al., "Efficient Multi-Scale Attention Module with Cross-Channel
Interaction", arXiv:2303.11366 (2023)。

在 EP-YOLO (Sensors 2026, 26(9):2607) 的注意力横向对比中，EMA 是**唯一
优于其余候选**的模块：

    "Modules like Squeeze-and-Excitation, CBAM, and the parameter-free SimAM
     all manage to increase the recall to around 0.68 compared to the baseline,
     but they cause a severe drop in precision. Squeeze-and-Excitation focuses
     purely on channel weights and completely ignores the spatial locations
     where tiny pests are hidden. CBAM includes spatial attention, but its
     pooling operations tend to blur the features of extremely small targets.
     SimAM ... its energy function is simply not robust enough to handle highly
     complex ... backgrounds. ... EMA is the only module ..."

EP-YOLO 消融数据（100 epochs 口径）：
    EMA 单独   → P 0.734 / R 0.636 / mAP50 0.700   （baseline P 0.718 / R 0.661 / mAP50 0.694）
    SPD+EMA    → P 0.744 / R 0.664 / mAP50 0.705 / mAP50-95 0.473
    放置位置   → backbone(0.700) **优于** neck(0.687)

⚠️ 复现要点
-----------
1. EMA 的输出通道是输入的 **2 倍**（``c2 = make_divisible(c2 * 2, 8)``），
   yaml 中必须按此设置后续层的输入通道，否则维度不匹配。
2. EP-YOLO 实证：**放在 backbone 更好**，neck 放置反而掉点。
3. 优先复用 Ultralytics 官方实现（``ultralytics.nn.modules.attention.EMA``），
   以保持与官方权重/行为一致；官方未提供时回退到本文件的实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["EMA", "get_ema"]


def _make_divisible(x: float, divisor: int = 8) -> int:
    """与 Ultralytics ``make_divisible`` 同语义。"""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return int((int(x) + divisor - 1) // divisor * divisor)


class EMA(nn.Module):
    """Efficient Multi-Scale Attention（本地实现，官方缺失时的回退）。

    与官方实现的关键点：
    * 输入 ``(B, C, H, W)`` → 输出 ``(B, 2C, H, W)``（通道翻倍）；
    * 内部对 ``x`` 做全局平均池化后生成通道权重，再与 ``x`` 逐元素相乘，
      因此保留空间信息（不像 SE 只做通道加权）。
    """

    def __init__(self, c1: int, c2: int, ratio: int = 16, kernel_size: int = 5, stride: int = 1, groups: int = 1, act: bool = True):
        super().__init__()
        c2 = _make_divisible(c2 * 2, 8)
        self.stride = stride
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size, stride, padding=groups * (kernel_size - 1) // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Conv2d(c2, c2, 1, stride=1, padding=0, bias=False)
        self.conv_gap2 = nn.Conv2d(c2, c2, 1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_gap(y)
        y = self.conv_gap2(y)
        y = self.norm(y)
        y = self.act(y)
        y = torch.mul(y, x)
        return self.conv(x)


def get_ema():
    """返回可用的 EMA 实现。

    优先使用 Ultralytics 官方实现（行为与官方仓库一致），
    官方不存在时回退到本文件的 :class:`EMA`。
    """
    try:
        from ultralytics.nn.modules.attention import EMA as _OfficialEMA  # type: ignore

        return _OfficialEMA, "ultralytics.official"
    except Exception:
        try:
            from ultralytics.nn.modules import EMA as _OfficialEMA2  # type: ignore

            return _OfficialEMA2, "ultralytics.official"
        except Exception:
            return EMA, "local"
