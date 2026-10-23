"""ADEA · I2 · DLA —— 密度先验引导的标签分配 (Density-aware Label Assignment)

出处 / 出处依据
--------------
回应 EP-YOLO (Sensors 2026, 26(9):2607) §4.3 第 3 点明确点名的 future work：

    "Future research could consider incorporating Density Map Estimation
     methods or NMS-free, end-to-end detection architectures to tackle this
     crowding dilemma."

以及 §4.2 Fig.12c 的失败案例：

    "the model generated 46 false predictions in a crowded region containing
     172 ground truth targets. The network easily misinterprets localized
     noisy textures as independent micro-targets, making the detection head
     overly sensitive in specific regions."

查重结论：密集计数领域的密度图估计已成熟（DM-Count 等），但**从未用于
"检测的标签分配与抑制调制"**；EP-YOLO 只是把它写进 future work 而未实现。

与 I1 ALA 的分工
----------------
* **ALA 用 GT 级粘连度 β**（局部：两个目标之间的黏连程度）
* **DLA 用图像级密度场 D**（全局：某个区域有多拥挤）

二者正交：ALA 修"粘连对"，DLA 修"拥挤区"。

首版范围（Step 1–3）
--------------------
* Step 1 密度场渲染：训练期 GT 已知 → 直接用 GT 中心点做高斯核渲染，
  **零额外可学习参数、零推理开销**。
* Step 2 密度-配额调制：拥挤处的 GT 获得更高正样本配额。
* Step 3 密度-抑制调制：推理期用密度场空间调制抑制强度（见 ``post/ags.py``）。
* Step 4（密集区敏感度抑制）风险较高，作为可选子项单独消融，未纳入首版。

依赖
----
numpy 可选：有 numpy 走向量路径，无 numpy 走纯 Python 路径（仅供小规模自测）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "DLAConfig",
    "render_density",
    "normalize_density",
    "sample_density",
    "per_gt_density",
    "per_gt_quota",
    "apply_dla_quota",
]

try:  # pragma: no cover - 环境相关
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


@dataclass
class DLAConfig:
    """DLA 超参。

    Attributes
    ----------
    enabled:
        总开关。关闭时 ``per_gt_quota`` 原样返回 ``base_qmin``。
    mu:
        密度调制强度。配额公式 ``k_i = ceil(base_qmin * (1 + mu * D_i))``。
    sigma_scale:
        高斯核半径相对目标尺寸的倍数（crowd counting 常用做法）。
    max_density:
        密度归一化使用的分位数（抑制极端离群点）。
    quantile:
        用于归一化的分位数，默认 0.99。
    """

    enabled: bool = True
    mu: float = 0.5
    sigma_scale: float = 0.5
    max_density: float = 0.0
    quantile: float = 0.99

    def describe(self) -> str:
        return f"DLA(enabled={self.enabled}, mu={self.mu}, sigma_scale={self.sigma_scale})"


def render_density(
    centers: Sequence[Tuple[float, float]],
    radii: Sequence[float],
    shape: Tuple[int, int],
    sigma_scale: float = 0.5,
) -> "List[List[float]] | object":
    """用 GT 中心点渲染密度场 D(x, y)。

    每个实例贡献一个高斯核，核宽随目标尺寸自适应（crowd counting 标准做法）。

    Parameters
    ----------
    centers:
        目标中心点 (x, y)，像素坐标。
    radii:
        对应目标的等效半径（像素）。
    shape:
        密度图尺寸 (H, W)。
    sigma_scale:
        σ = max(radius * sigma_scale, 0.5)，避免 σ→0 造成的奇异。

    Returns
    -------
    形状 (H, W) 的密度图（numpy 数组或纯 Python 嵌套列表）。
    """
    h, w = int(shape[0]), int(shape[1])
    if _np is not None:
        canvas = _np.zeros((h, w), dtype=_np.float32)
        yy, xx = _np.mgrid[0:h, 0:w].astype(_np.float32)
        for (cx, cy), r in zip(centers, radii):
            sigma = max(float(r) * float(sigma_scale), 0.5)
            inv = 1.0 / (2.0 * sigma * sigma)
            dx = xx - float(cx)
            dy = yy - float(cy)
            canvas += _np.exp(-(dx * dx + dy * dy) * inv).astype(_np.float32)
        return canvas

    canvas = [[0.0] * w for _ in range(h)]
    for (cx, cy), r in zip(centers, radii):
        sigma = max(float(r) * float(sigma_scale), 0.5)
        inv = 1.0 / (2.0 * sigma * sigma)
        rad = int(math.ceil(3.0 * sigma))
        x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad + 1)
        y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad + 1)
        for y in range(y0, y1):
            dy = y - cy
            row = canvas[y]
            for x in range(x0, x1):
                dx = x - cx
                row[x] += math.exp(-(dx * dx + dy * dy) * inv)
    return canvas


def normalize_density(density, quantile: float = 0.99) -> "object":
    """把密度场归一化到 [0, 1]（按分位数截断，抑制离群峰值）。"""
    if _np is not None and isinstance(density, _np.ndarray):
        flat = density.reshape(-1)
        pos = flat[flat > 0]
        ref = float(_np.quantile(pos, quantile)) if pos.size else 1.0
        if ref <= 0:
            ref = 1.0
        return _np.clip(density / ref, 0.0, 1.0)
    flat = sorted(v for row in density for v in row if v > 0)
    if not flat:
        return density
    k = min(len(flat) - 1, int(quantile * (len(flat) - 1)))
    ref = flat[k] if flat[k] > 0 else 1.0
    return [[min(1.0, v / ref) for v in row] for row in density]


def sample_density(density, x: float, y: float) -> float:
    """在 (x, y) 处采样归一化密度（最近邻）。"""
    if _np is not None and isinstance(density, _np.ndarray):
        h, w = density.shape
        xi = int(min(max(x, 0), w - 1))
        yi = int(min(max(y, 0), h - 1))
        return float(density[yi, xi])
    h = len(density)
    w = len(density[0]) if h else 0
    if h == 0 or w == 0:
        return 0.0
    xi = int(min(max(x, 0), w - 1))
    yi = int(min(max(y, 0), h - 1))
    return float(density[yi][xi])


def per_gt_density(density, gt_boxes_xywh) -> List[float]:
    """取每个 GT 中心处的归一化密度。"""
    out: List[float] = []
    for b in gt_boxes_xywh:
        cx, cy = float(b[0]), float(b[1])
        out.append(sample_density(density, cx, cy))
    return out


def per_gt_quota(
    gt_boxes_xywh,
    base_qmin: int,
    cfg: DLAConfig,
    density=None,
) -> List[int]:
    """密度调制后的逐 GT 最小正样本配额。

    ``k_i = ceil(base_qmin * (1 + mu * D_i))``

    Parameters
    ----------
    gt_boxes_xywh:
        GT 框 (cx, cy, w, h)。
    base_qmin:
        基础配额（对应 ALA 的 ``q_min``）。
    cfg:
        :class:`DLAConfig`。
    density:
        可选的**已渲染好的**密度场。若为 None 则按需自行渲染。

    Returns
    -------
    长度与 GT 数相同的配额列表。
    """
    n = len(gt_boxes_xywh)
    if not cfg.enabled or n == 0:
        return [int(base_qmin)] * n

    if density is None:
        centers = [(float(b[0]), float(b[1])) for b in gt_boxes_xywh]
        radii = [max(float(b[2]), float(b[3])) * 0.5 for b in gt_boxes_xywh]
        h = int(max((float(b[1]) + float(b[3]) / 2.0) for b in gt_boxes_xywh) + 1.0)
        w = int(max((float(b[0]) + float(b[2]) / 2.0) for b in gt_boxes_xywh) + 1.0)
        density = render_density(centers, radii, (max(h, 1), max(w, 1)), cfg.sigma_scale)
        density = normalize_density(density, cfg.quantile)

    dens = per_gt_density(density, gt_boxes_xywh)
    quotas: List[int] = []
    for d in dens:
        quotas.append(int(math.ceil(base_qmin * (1.0 + cfg.mu * float(d)))))
    return quotas


def apply_dla_quota(gt_boxes_xywh, base_qmin: int, cfg: DLAConfig, density=None) -> List[int]:
    """``per_gt_quota`` 的别名，供上层统一调用。"""
    return per_gt_quota(gt_boxes_xywh, base_qmin, cfg, density=density)
