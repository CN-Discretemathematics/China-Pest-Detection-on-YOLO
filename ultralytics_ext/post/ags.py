"""ADEA · I5 · AGS —— 粘连感知分级抑制 (Adhesion-Graded Suppression)

定位：**工程组件**（不主张为主要创新），与 SG-Det/SG-NMS 明确划界。

与最近工作的关系（必须诚实写明）
--------------------------------
SG-Det (SG-NMS) 用学习式 Semantics-Geometry Embedding 判断两个高重叠框是否
属于同一物理个体，在 KITTI / CityPersons 的**遮挡**行人场景上取得 SOTA
（重遮挡 KITTI car AP 62.08，较 Greedy-NMS +5 pt；CityPersons miss rate 51.1%），
但需要**额外的 SGE 头 + 二阶段精修**，且面向行人/汽车遮挡。

AGS 的差异：
1. **单阶段、anchor-free**，不增加任何额外头，不做二阶段精修；
2. 用**数据集自身的粘连统计 β**（Pest24 论文定义）做标定，而非学习式嵌入；
3. 面向**粘连**（两虫物理黏连、边界融合）而非**遮挡**（被前方物体挡住）。

机制
----
对候选框对 (i, j)，按**重叠程度**分三档处理：

* **高度重合**（同类别且 ``IoU >= same_instance_iou``，默认 0.80）
  -> 大概率是同一个体的重复框 -> **硬抑制**（标准 NMS 行为，Recall 不受损）；
* **中等重合**（``iou_thr < IoU < same_instance_iou``）
  -> 大概率是两个粘连个体 -> **分数按重叠程度平滑衰减而非删除**；
* **不重叠** -> 不处理。

衰减量随重叠程度线性增长::

    score_j *= 1 - decay * (IoU - iou_thr) / (1 - iou_thr)

⚠️ 同一性判据的设计更正（由自测发现）
------------------------------------
初版用「形状相似度」判定实例同一性，实测**失效**：同一物种的个体体型本就相近，
两个不同个体的形状相似度接近 1 -> 被误判为重复框而硬抑制，
恰好抹掉了 AGS 想保留的召回收益。
真正重复的框 IoU 通常 > 0.85，而两个粘连个体的 IoU 多在 0.3~0.75 ——
故改以 **IoU 高低**作为主判据（``same_instance_iou``），
``shape_similarity`` 仅保留为诊断信息（不再参与判定）。

⚠️ 粘连系数 β 的一个事实更正
----------------------------
Pest24 论文定义的 ``β = |A∩B| / |A∪B|`` 与 **IoU 在数值上完全等价**（同一公式）。
二者的差别只在**语义**：IoU 通常被读作「是否重复检出同一体」，
β 在本数据集中被读作「两只虫在捕捉板上物理黏连的程度」。
本项目沿用论文的 β 记号与阈值（0.05），但**不声称** β 与 IoU 数值有别。

依赖：核心 ``ags_core`` 为纯 Python；``apply_ags`` / ``ags_batched_nms`` 为 torch 适配层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "AGSConfig",
    "Detection",
    "shape_similarity",
    "ags_core",
    "apply_ags",
    "ags_batched_nms",
]


@dataclass
class AGSConfig:
    """AGS 超参。

    Attributes
    ----------
    enabled:
        总开关。关闭时完全退化为标准 NMS 行为。
    iou_thr:
        触发配对判定的 IoU 阈值（标准 NMS 的抑制阈值）。
    same_instance_iou:
        **同一性判据**：同类别且 IoU >= 该值 -> 判定为「同一个体的重复框」-> 硬抑制。
        真正重复的框 IoU 通常 > 0.85；两个粘连的不同个体 IoU 多在 0.3~0.75。
    decay:
        软衰减强度 lambda。
    same_class_only:
        是否只对同类别框做判定（与 NMS 的 per-class 行为一致）。
    """

    enabled: bool = True
    iou_thr: float = 0.5
    same_instance_iou: float = 0.80
    decay: float = 0.5
    same_class_only: bool = True

    def describe(self) -> str:
        return (
            f"AGS(enabled={self.enabled}, iou_thr={self.iou_thr}, "
            f"same_instance_iou={self.same_instance_iou}, decay={self.decay})"
        )


class Detection:
    """一个候选框（xyxy 绝对坐标）。"""

    __slots__ = ("box", "score", "cls", "decayed")

    def __init__(self, box: Tuple[float, float, float, float], score: float, cls: int):
        self.box = box
        self.score = float(score)
        self.cls = int(cls)
        self.decayed = False

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"Det(box={tuple(round(v, 1) for v in self.box)}, "
            f"score={self.score:.3f}, cls={self.cls})"
        )


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def iou(a, b) -> float:
    """交并比 IoU（与 Pest24 论文的粘连系数 β 数值等价）。"""
    inter = _inter(a, b)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def adhesion(a, b) -> float:
    """粘连系数 β = |A∩B| / |A∪B|（Pest24 论文定义；数值上等同 IoU）。"""
    return iou(a, b)


def shape_similarity(a, b) -> float:
    """两个框的形状相似度（0~1）—— **仅作诊断信息，不参与抑制判定**。

    同时考察宽高比一致性与尺度比一致性::

        sim = 0.5 * exp(-|log(ar_a/ar_b)|) + 0.5 * min(sa, sb) / max(sa, sb)
    """
    import math

    aw, ah = max(1e-6, a[2] - a[0]), max(1e-6, a[3] - a[1])
    bw, bh = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
    ar_a, ar_b = aw / ah, bw / bh
    ar_term = math.exp(-abs(math.log(ar_a / ar_b)))
    sa, sb = aw * ah, bw * bh
    scale_term = min(sa, sb) / max(sa, sb) if max(sa, sb) > 0 else 0.0
    return 0.5 * ar_term + 0.5 * scale_term


def ags_core(
    dets: Sequence[Detection],
    cfg: AGSConfig,
) -> Tuple[List[Detection], dict]:
    """AGS 核心（纯 Python）。

    判定逻辑（对每对重叠框）：

    * 同类别且 ``IoU >= same_instance_iou`` -> **同一个体的重复框** -> **硬抑制**（丢弃）；
    * 其余重叠（``iou_thr < IoU < same_instance_iou``）-> **两个粘连个体**
      -> **分数按重叠程度平滑衰减**，**保留**；
    * ``IoU <= iou_thr`` -> 不处理。

    Parameters
    ----------
    dets:
        候选框列表（会被就地修改分数）。
    cfg:
        :class:`AGSConfig`。

    Returns
    -------
    (kept, stats)
        ``kept`` 为保留下来的框（按分数降序）；``stats`` 含抑制/衰减计数。
    """
    stats = {"n_input": len(dets), "n_hard_suppressed": 0, "n_soft_decayed": 0}
    if not cfg.enabled or len(dets) == 0:
        return sorted(dets, key=lambda d: -d.score), stats

    order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
    alive = [True] * len(dets)
    span = max(1e-6, 1.0 - cfg.iou_thr)

    for oi in range(len(order)):
        i = order[oi]
        if not alive[i]:
            continue
        for oj in range(oi + 1, len(order)):
            j = order[oj]
            if not alive[j]:
                continue
            di, dj = dets[i], dets[j]
            if cfg.same_class_only and di.cls != dj.cls:
                continue
            ov = iou(di.box, dj.box)
            if ov <= cfg.iou_thr:
                continue
            if ov >= cfg.same_instance_iou:
                alive[j] = False
                stats["n_hard_suppressed"] += 1
            elif cfg.decay > 0.0:
                amount = cfg.decay * (ov - cfg.iou_thr) / span
                new_score = dj.score * (1.0 - amount)
                if new_score < dj.score:
                    dj.score = new_score
                    dj.decayed = True
                    stats["n_soft_decayed"] += 1

    kept = [dets[i] for i in order if alive[i]]
    stats["n_kept"] = len(kept)
    return kept, stats


def apply_ags(dets, cfg: AGSConfig, score_threshold: float = 0.0):
    """对 numpy / torch 形式的检测结果应用 AGS。

    Parameters
    ----------
    dets:
        形如 ``(boxes_xyxy, scores, class_ids)`` 的三元组，
        boxes 形状 (N, 4)，scores/class_ids 形状 (N,)。
    cfg:
        :class:`AGSConfig`。
    score_threshold:
        衰减后低于该阈值的框被丢弃（默认 0，即只做抑制不做阈值过滤）。

    Returns
    -------
    (boxes, scores, class_ids, stats)
    """
    boxes, scores, class_ids = dets
    objs = []
    for b, s, c in zip(boxes, scores, class_ids):
        objs.append(Detection(tuple(float(v) for v in b), float(s), int(c)))
    kept, stats = ags_core(objs, cfg)
    if score_threshold > 0:
        kept = [d for d in kept if d.score >= score_threshold]
        stats["n_kept"] = len(kept)

    out_boxes = [list(d.box) for d in kept]
    out_scores = [d.score for d in kept]
    out_classes = [d.cls for d in kept]
    return out_boxes, out_scores, out_classes, stats


# --------------------------------------------------------------------------- #
# 与 Ultralytics NMS 的对接
# --------------------------------------------------------------------------- #
def ags_batched_nms(boxes, scores, idxs, iou_threshold: float, cfg: AGSConfig | None = None):
    """AGS 版 ``batched_nms``，签名与 ``torchvision.ops.batched_nms`` 一致。

    Ultralytics 的 ``non_max_suppression`` 调用 ``batched_nms``，
    因此替换掉它就能让 AGS 在 ``val`` / ``predict`` 中直接生效。

    与标准 NMS 的差别：
    * 同类别且 ``IoU >= same_instance_iou`` -> 同一个体的重复框 -> **丢弃**；
    * ``iou_threshold < IoU < same_instance_iou`` -> 两个粘连个体
      -> **分数按重叠程度衰减，保留**。

    衰减是**原地**修改 ``scores`` 张量 —— Ultralytics 随后会做
    ``scores = scores[keep]``，因此衰减后的分数会正确传播到最终结果。

    Returns
    -------
    keep:
        保留下来的框的下标（torch.Tensor, long）。
    """
    import torch

    if cfg is None:
        cfg = AGSConfig()
    if boxes.numel() == 0 or scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep_list = []
    span = max(1e-6, 1.0 - cfg.iou_thr)
    for class_id in torch.unique(idxs):
        mask = idxs == class_id
        orig_idx = mask.nonzero(as_tuple=False).flatten()  # 原始下标
        cb = boxes[mask]
        cs = scores[mask]
        if cb.shape[0] == 0:
            continue
        order = cs.argsort(descending=True)
        cb = cb[order]
        cs = cs[order]
        alive = torch.ones(cs.shape[0], dtype=torch.bool, device=boxes.device)

        for i in range(cb.shape[0]):
            if not bool(alive[i]):
                continue
            rest = torch.arange(i + 1, cb.shape[0], device=boxes.device)
            rest = rest[alive[rest]]
            if rest.numel() == 0:
                break
            ious = _torch_iou(cb[i].unsqueeze(0), cb[rest])[0]  # (M,)
            hit_pos = (ious > iou_threshold).nonzero(as_tuple=False).flatten()
            for p in hit_pos.tolist():
                j = int(rest[p])
                if not bool(alive[j]):
                    continue
                ov = float(ious[p])
                if ov >= cfg.same_instance_iou:
                    alive[j] = False
                elif cfg.decay > 0.0:
                    amount = cfg.decay * (ov - cfg.iou_thr) / span
                    new_score = float(cs[j]) * (1.0 - amount)
                    if new_score < float(cs[j]):
                        cs[j] = new_score

        # 原地把（可能已衰减的）分数写回调用方的 scores
        scores[orig_idx[order]] = cs
        keep_list.append(orig_idx[order][alive])

    if not keep_list:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return torch.cat(keep_list)


def _torch_area(b):
    w = (b[2] - b[0]).clamp(min=0)
    h = (b[3] - b[1]).clamp(min=0)
    return w * h


def _torch_inter(a, b):
    ix1 = torch.maximum(a[0], b[0])
    iy1 = torch.maximum(a[1], b[1])
    ix2 = torch.minimum(a[2], b[2])
    iy2 = torch.minimum(a[3], b[3])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    return iw * ih


def _torch_iou(a, b):
    """a: (1,4)，b: (M,4) -> (1, M)。"""
    area_a = _torch_area(a).unsqueeze(1)
    area_b = _torch_area(b).unsqueeze(0)
    inter = _torch_inter(a[0], b)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-7)
