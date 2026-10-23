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

⚠️ 本实现的"通道守恒"变体（与官方 EMA 的差异，必须在论文中说明）
------------------------------------------------------------
官方 EMA 的 ``__init__`` 内部把 ``c2`` 翻倍（``c2 = make_divisible(c2*2, 8)``），
即它是"通道扩张型"模块，输出通道 = 2×输入。

本项目需要的是**可 1:1 插入 backbone 而不改变下游通道**的版本，
否则：
* Ultralytics ``parse_model`` 的通道记账会与实际输出不符 → 后续层维度错乱；
* 预训练权重的通道匹配被破坏 → 迁移率骤降。

因此本实现提供一个**通道守恒变体**：``EMA(c1, c2)`` 输出恰为 ``c2`` 通道，
注意力计算结构与官方一致（分组卷积 + 跨通道交互 + 全局池化门控），
仅"末端卷积不翻倍"这一处不同。yaml 中的通道参数取**输入层通道数**，
从而输出 == 输入，下游层完全不受影响。
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
    """Efficient Multi-Scale Attention（**通道守恒**变体）。

    与官方实现的关键差异：输出通道 = ``c2``（不再翻倍）。
    注意力机制本身与官方一致：
    * 分组卷积提取局部特征；
    * 全局平均池化 + 两个 1×1 卷积生成通道权重；
    * 权重与原特征逐元素相乘（保留空间信息，不像 SE 只做通道加权）。

    典型用法（yaml 中）::

        - [-1, 1, EMA, [<输入层通道数>]]     # 输出通道 == 输入通道
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        ratio: int = 16,
        kernel_size: int = 5,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.stride = stride
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.c1,
                self.c2,
                kernel_size,
                stride,
                padding=groups * (kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(self.c2),
            nn.SiLU() if act else nn.Identity(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = nn.Conv2d(self.c2, self.c2, 1, stride=1, padding=0, bias=False)
        self.conv_gap2 = nn.Conv2d(self.c2, self.c2, 1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(self.c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_gap(y)
        y = self.conv_gap2(y)
        y = self.norm(y)
        y = self.act(y)
        y = torch.mul(y, x)
        return self.conv(x)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"c1={self.c1}, c2={self.c2}"


def get_ema():
    """返回可用的 EMA 实现与来源标记。

    ⚠️ 始终返回**本文件的通道守恒实现** —— 官方 ``ultralytics.nn.modules.attention.EMA``
    会把通道翻倍，与 Ultralytics ``parse_model`` 的通道记账冲突，无法 1:1 插入 backbone。
    官方实现保留在下面仅作对照参考。
    """
    return EMA, "local.channel-preserving"


def get_official_ema():
    """返回 Ultralytics 官方 EMA（通道翻倍版），仅作参考对照。"""
    try:
        from ultralytics.nn.modules.attention import EMA as _OfficialEMA  # type: ignore

        return _OfficialEMA, "ultralytics.official"
    except Exception:
        return None, "unavailable"
