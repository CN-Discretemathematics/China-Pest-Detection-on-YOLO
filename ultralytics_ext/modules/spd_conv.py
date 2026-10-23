"""ADEA · R1（复现）· SPD-Conv —— 保真下采样 (Space-to-Depth Convolution)

出处
----
Sunkara & Luo, "No More Strided Convolutions or Pooling: A New CNN Building
Block for Low-Resolution Images and Small Objects", arXiv:2202.01248 (2022)。

在 EP-YOLO (Sensors 2026, 26(9):2607) 中用于 Pest24：

    "the network is prone to lose fine-grained features during the downsampling
     process. The backbone of YOLOv8n heavily relies on 3×3 convolutions with a
     stride of 2 ... For the numerous micro-scale pests in Pest24, continuous
     strided convolutions inevitably destroy their already [degraded] features."

⚠️ EP-YOLO 消融给出的关键教训（务必遵守）
----------------------------------------
* **单用 SPD 会掉点**：mAP50 0.694 → 0.685，Precision 0.718 → 0.702。
  原因是 SPD 保留了微目标像素的同时**引入了大量背景纹理噪声**。
* **必须与注意力配对**（双 SPD + EMA → mAP50 0.705 / P 0.744）。
* **只替换前两个 stride-2 卷积最优**；替换更多反而掉（0.685）。
* 因此本实现的验收点之一是 **Δparams ≤ 0.3M**（EP-YOLO 原版仅 +0.025M）。

与标准卷积的接口对齐
--------------------
``SPDConv(c1, c2, k, s=2)`` 与 ``Conv(c1, c2, k, s=2)`` 的
**输出通道数与空间分辨率完全一致**，因此可在 yaml 中 1:1 替换，
后续层的输入通道无需改动（利于预训练权重迁移）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SpaceToDepth", "SPDConv"]


class SpaceToDepth(nn.Module):
    """Space-to-Depth：把 (B, C, H, W) 重排为 (B, 4C, H/2, W/2)。

    无损地把局部 2×2 邻域的空间信息折入通道维 —— 这是"不丢信息"的关键，
    与 stride-2 卷积（丢掉 3/4 采样点）本质不同。

    若 H 或 W 为奇数，则在右侧/底部做 replicate padding 到偶数，
    以保证任何输入尺寸都能工作（Ultralytics 会把输入 pad 到 32 的倍数，
    正常路径下不会触发）。
    """

    def __init__(self, block_size: int = 2):
        super().__init__()
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = self.block_size
        b, c, h, w = x.shape
        pad_h = (bs - h % bs) % bs
        pad_w = (bs - w % bs) % bs
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            h, w = x.shape[-2:]
        x = x.view(b, c, h // bs, bs, w // bs, bs)
        # (b, c, h//bs, bs, w//bs, bs) -> (b, c*bs*bs, h//bs, w//bs)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * bs * bs, h // bs, w // bs)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"block_size={self.block_size}"


class SPDConv(nn.Module):
    """Space-to-Depth + 卷积。

    用法（与 Ultralytics ``Conv`` 等价替换）::

        # 标准:  [-1, 1, Conv, [32, 3, 2]]     # 3->32, stride 2
        # SPD  :  [-1, 1, SPDConv, [32, 3, 2]] # 3->32, stride 2 (空间分辨率相同)

    ⚠️ **内部卷积必须是 stride=1**
    Space-to-Depth 本身已经把 H×W 变成 (H/2)×(W/2)；
    若后续卷积再用 stride=2，空间会被**二次下采样**（H/4），
    导致与 neck 的其它分支尺寸不匹配（Concat 时报
    "Sizes of tensors must match except in dimension 1"）。
    因此 yaml 里沿用原 Conv 的 ``s=2``，但内部卷积固定 stride=1。
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None, g: int = 1, act: bool = True):
        super().__init__()
        self.c1, self.c2 = int(c1), int(c2)
        self.s = s  # 记录 yaml 中的 stride（语义上已由 SpaceTo-Depth 实现）
        self.spd = SpaceToDepth(2)
        # SpaceToDepth 后通道变为 4*c1；卷积 stride 固定为 1（避免二次下采样）
        conv_pad = (k - 1) // 2 if p is None else p
        self.conv = nn.Sequential(
            nn.Conv2d(4 * int(c1), int(c2), k, 1, conv_pad, groups=g, bias=False),
            nn.BatchNorm2d(int(c2)),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.spd(x))

    def extra_repr(self) -> str:  # pragma: no cover
        return f"c1={self.c1}, c2={self.c2}, stride={self.s}(by SpaceToDepth)"
