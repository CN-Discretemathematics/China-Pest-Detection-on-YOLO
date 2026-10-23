"""ADEA · R3（复现）· DySample —— 动态上采样 (Dynamic Upsampling)

出处
----
Liao et al., "Learning to Upsample by Learning to Sample", ICLR 2023
(arXiv:2308.07096)。

在 YOLO11 体系上的应用证据：CCDW-YOLO (2026) 以 **YOLO11n 为基线**
引入 DySample 替换 neck 上采样，相对基线
**mAP@50 +1.39 pt / mAP@50:95 +1.89 pt**。

机制
----
标准 ``nn.Upsample``（nearest / bilinear）是**静态**的：采样位置由几何固定。
DySample 额外预测一张**偏移图 (offset)** 与一张**调制掩码 (mask)**：

* offset：每个输出子像素位置相对标准网格的采样偏移 → 自适应地对齐真实物体；
* mask：对每个输出子像素给出 [0,1] 的置信权重 → 抑制无效/噪声区域的贡献。

两者都由输入特征回归得到，因此上采样位置是**内容自适应**的。

与 Ultralytics ``nn.Upsample`` 的接口对齐
----------------------------------------
``DySample(c1, c2, scale=2)`` 在 yaml 中可替换 ``nn.Upsample``：

    # 标准:  [-1, 1, nn.Upsample, [None, 2, 'nearest']]
    # DySample: [-1, 1, DySample, [c_in, c_out, 2]]     # c_out = c_in

若 ultralytics 的 ``parse_model`` 对 ``nn.Upsample`` 有特殊处理，
本模块同时提供 ``fuse()`` 兼容入口（见 ``ultralytics_ext/register.py``）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DySample"]


class DySample(nn.Module):
    """动态（内容自适应）上采样。

    Parameters
    ----------
    in_channels:
        输入通道。
    out_channels:
        输出通道；通常等于 ``in_channels``（用于替换 nn.Upsample）。
    kernel_size:
        预测 offset/mask 的卷积核大小。
    stride / padding:
        预测分支的卷积参数。
    scale:
        上采样倍数。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        scale: int = 2,
        sampling_type: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels) if out_channels else int(in_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2 if padding is None else padding
        self.groups = groups
        self.scale = int(scale)
        self.sampling_type = sampling_type
        self.align_corners = align_corners

        self.offset = nn.Conv2d(
            self.in_channels,
            2 * self.scale * self.scale,
            kernel_size,
            stride,
            self.padding,
            groups=groups,
            bias=True,
        )
        self.mask = nn.Conv2d(
            self.in_channels,
            self.scale * self.scale,
            kernel_size,
            stride,
            self.padding,
            groups=groups,
            bias=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        s = self.scale
        out_h, out_w = h * s, w * s

        offset = self.offset(x)  # (b, 2*s*s, h, w)
        mask = torch.sigmoid(self.mask(x))  # (b, s*s, h, w)

        # ---- 基础归一化网格（对应 F.interpolate 的 align_corners=False 语义）----
        dev, dt = x.device, x.dtype
        oy = torch.arange(out_h, device=dev)
        ox = torch.arange(out_w, device=dev)
        base_y = (oy.to(dt) + 0.5) / out_h * 2.0 - 1.0
        base_x = (ox.to(dt) + 0.5) / out_w * 2.0 - 1.0
        base_y = base_y.view(1, out_h, 1).expand(1, out_h, out_w)
        base_x = base_x.view(1, 1, out_w).expand(1, out_h, out_w)

        # ---- 展平索引：每个输出像素对应 (cell_y, cell_x, sub_y, sub_x) ----
        n = out_h * out_w
        flat = torch.arange(n, device=dev)
        fy = flat // out_w
        fx = flat % out_w
        cy = fy // s
        cx = fx // s
        sy = fy % s
        sx = fx % s

        # 通道索引 = 子像素位置 (sy, sx)；空间索引 = 源 cell (cy, cx)
        chan_idx = sy * s + sx  # (n,)
        spatial_idx = cy * w + cx  # (n,)

        off = offset.view(b, 2, s * s, h * w)  # (b, 2, s*s, h*w)
        off_y_all = off[:, 0]  # (b, s*s, h*w)
        off_x_all = off[:, 1]
        gather_idx = (chan_idx * (h * w) + spatial_idx).unsqueeze(0).expand(b, n)
        off_y = torch.gather(off_y_all.reshape(b, s * s * h * w), 1, gather_idx).view(b, n)
        off_x = torch.gather(off_x_all.reshape(b, s * s * h * w), 1, gather_idx).view(b, n)

        # 归一化坐标下，像素偏移需除以输出尺寸的一半
        off_y = off_y * (2.0 / out_h)
        off_x = off_x * (2.0 / out_w)

        grid_y = base_y.reshape(1, n) + off_y
        grid_x = base_x.reshape(1, n) + off_x
        grid = torch.stack([grid_x, grid_y], dim=-1).view(b, out_h, out_w, 2)

        mode = "bilinear" if self.sampling_type in ("bilinear", "bicubic") else "nearest"
        out = F.grid_sample(x, grid, mode=mode, padding_mode="border", align_corners=self.align_corners)

        # ---- 调制掩码：同样的索引映射 ----
        m = mask.view(b, s * s, h * w)
        mask_sel = torch.gather(m.reshape(b, s * s * h * w), 1, gather_idx).view(b, n)
        out = out * mask_sel.view(b, 1, out_h, out_w)
        return out

    def extra_repr(self) -> str:  # pragma: no cover
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, scale={self.scale}"
