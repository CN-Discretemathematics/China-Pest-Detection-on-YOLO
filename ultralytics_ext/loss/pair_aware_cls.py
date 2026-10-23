"""ADEA · I6（可选）· 混淆对 / 分类群感知分类损失

动机（基于 Pest24 的两条结构事实）
----------------------------------
* **F5 类间相似度极高**：Pest24 原始论文用 aHash 估计类两两相似度，
  范围 0.45~0.7、均值 0.533，**85.5% 的类对 > 0.5**（远高于 VOC/COCO）。
* **F6 非目标害虫干扰**：约 1100 张图含有外观相近的**非目标**害虫
  （论文原文：这类非目标会遮挡目标并被误判）。

→ 误分是 Precision 的主要杀手。

两种做法
--------
1. **混淆对感知损失**（本文件实现）
   用 baseline 的混淆矩阵构建 24 类混淆图；对高混淆类对 (A, B) 增加
   **成对 margin 惩罚**：只惩罚"把 A 判成 B"这一对，
   **不做全局类别重平衡**（避免长尾类被重平衡拖垮，与 I3 DRA 思路互补）。
2. **分类学层级头**
   Pest24 论文 Fig.2 给出 5 目 13 科的分类学 → 先判目再判种。
   ⚠️ 层级分类本身是经典技术，本项目只主张"用目标域真实分类学"的域特化应用。

诚实评价
--------
层级分类与 pair-aware loss **通用版都不新**；此处仅作可选扩展，
不进主线，且必须与主线的 I1/I2/I3/I4 分开消融。

集成方式
--------
Ultralytics 各版本的分类损失计算点不同（``vflLoss.__call__`` 内联计算 BCE，
或 ``self.bce`` 属性）。这里提供**多候选挂载点**的安装函数
:func:`install_pair_aware_cls`，逐一探测并返回实际生效者，
避免版本升级后静默失效。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "PairAwareConfig",
    "build_confusion_graph",
    "pair_penalty",
    "install_pair_aware_cls",
]


@dataclass
class PairAwareConfig:
    """混淆对感知分类损失超参。

    Attributes
    ----------
    enabled:
        总开关。
    conf_threshold:
        构建混淆图时忽略的混淆计数阈值（低于此值不算"高混淆"）。
    top_k:
        每个真值类保留的混淆对数量。
    margin:
        成对 margin 的强度（以 logit 差为单位）。
    penalty_weight:
        该项在总分类损失中的权重。
    class_names:
        类别名列表（顺序须与数据集 yaml 的 ``names`` 一致）。
    """

    enabled: bool = False
    conf_threshold: int = 5
    top_k: int = 3
    margin: float = 2.0
    penalty_weight: float = 0.1
    class_names: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"PairAware(enabled={self.enabled}, thr={self.conf_threshold}, "
            f"top_k={self.top_k}, margin={self.margin}, w={self.penalty_weight})"
        )


def build_confusion_graph(
    confusion_matrix,
    cfg: PairAwareConfig,
) -> Dict[int, List[Tuple[int, float]]]:
    """从混淆矩阵构建"真值类 → 易混淆类列表"。

    Parameters
    ----------
    confusion_matrix:
        形状 (nc, nc) 的混淆矩阵，行=真值，列=预测（Ultralytics 的
        ``ConfusionMatrix.matrix`` 语义）。
    cfg:
        :class:`PairAwareConfig`。

    Returns
    -------
    dict
        ``{gt_class: [(confusable_class, count), ...]}``，按 count 降序，
        每个真值类最多保留 ``cfg.top_k`` 个。
    """
    n = len(confusion_matrix)
    graph: Dict[int, List[Tuple[int, float]]] = {}
    for i in range(n):
        row = confusion_matrix[i]
        pairs: List[Tuple[int, float]] = []
        for j in range(n):
            if i == j:
                continue
            val = row[j]
            cnt = float(val) if not hasattr(val, "item") else float(val.item())
            if cnt >= cfg.conf_threshold:
                pairs.append((j, cnt))
        pairs.sort(key=lambda t: -t[1])
        if pairs:
            graph[i] = pairs[: cfg.top_k]
    return graph


def pair_penalty(
    logits,
    target_classes,
    graph: Dict[int, List[Tuple[int, float]]],
    cfg: PairAwareConfig,
):
    """成对 margin 惩罚。

    对每个样本，其真值类为 ``t``；对 ``t`` 的每个易混淆类 ``c``，
    若 ``logit_c - logit_t > -margin``，则惩罚 ``(logit_c - logit_t + margin)``。

    Parameters
    ----------
    logits:
        (N, nc) 的分类 logit。
    target_classes:
        (N,) 的真值类别下标。
    graph:
        :func:`build_confusion_graph` 的输出。
    cfg:
        :class:`PairAwareConfig`。

    Returns
    -------
    标量损失（None 表示无惩罚项）。
    """
    if not cfg.enabled or not graph or logits is None or logits.numel() == 0:
        return None
    total = logits.new_zeros(())
    count = 0
    for i in range(logits.shape[0]):
        t = int(target_classes[i])
        pairs = graph.get(t)
        if not pairs:
            continue
        lt = logits[i, t]
        for c, _cnt in pairs:
            diff = logits[i, c] - lt + cfg.margin
            if diff > 0:
                total = total + diff
                count += 1
    if count == 0:
        return None
    return total / max(1, logits.shape[0]) * cfg.penalty_weight


def install_pair_aware_cls(
    cfg: Optional[PairAwareConfig] = None,
    confusion_matrix=None,
):
    """把成对 margin 惩罚挂到 Ultralytics 的分类损失上。

    挂载点（依次探测）：

    1. ``ultralytics.utils.loss.vflLoss.__call__`` —— 兜底包裹；
    2. ``ultralytics.utils.loss.vflLoss.bce`` —— 若为可调用属性则包裹之。

    Returns
    -------
    (ok, report)
    """
    cfg = cfg or PairAwareConfig()
    if not cfg.enabled:
        return False, "pair-aware cls disabled"

    graph: Dict[int, List[Tuple[int, float]]] = {}
    if confusion_matrix is not None:
        graph = build_confusion_graph(confusion_matrix, cfg)
    if not graph:
        return False, "confusion graph empty (need baseline confusion matrix)"

    try:
        import ultralytics.utils.loss as loss_mod
    except Exception as exc:  # pragma: no cover
        return False, f"import ultralytics.utils.loss failed: {exc}"

    version = getattr(__import__("ultralytics"), "__version__", "?")
    vfl = getattr(loss_mod, "vflLoss", None)
    if vfl is None:
        return False, f"vflLoss not found (ultralytics {version})"

    orig_call = vfl.__call__

    def patched_call(self, pred_scores, batch_targets, batch_idx, fg_mask, target_gt_idx):
        loss = orig_call(self, pred_scores, batch_targets, batch_idx, fg_mask, target_gt_idx)
        try:
            logits = pred_scores.permute(0, 2, 3, 1).reshape(-1, pred_scores.shape[1])[fg_mask]
            tgt = target_gt_idx[fg_mask]
            pen = pair_penalty(logits, tgt, graph, cfg)
            if pen is not None:
                loss = loss + pen
        except Exception:
            pass
        return loss

    vfl.__call__ = patched_call
    return True, (
        f"hooked ultralytics.utils.loss.vflLoss.__call__ (ultralytics {version}), "
        f"pairs={sum(len(v) for v in graph.values())}"
    )
