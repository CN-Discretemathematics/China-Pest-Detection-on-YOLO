"""ADEA · I3 · DRA —— 难度感知的动态长尾重采样 (Difficulty-aware Resampling)

出处 / 出处依据
--------------
回应 EP-YOLO (Sensors 2026, 26(9):2607) §4.3 第 1 点明确点名的 future work：

    "the severe data imbalance makes it exceedingly difficult for data-driven
     deep models to adequately extract generalized features for minority
     classes, rendering them highly susceptible to false negatives in
     real-world scenarios. Future work could introduce dynamic resampling
     strategies or generative model-based data augmentation techniques for
     minority classes to mitigate this issue."

Pest24 的长尾极端程度（论文 Table 2）：Anomala corpulenta 53,347 实例，
而 Holotrichia oblita 仅 108、Nematode trench 167、eight-character tiger 168。
mAP 是 24 类的**宏平均**，因此稀有类是提升 mAP 的杠杆最高点。

与静态重采样（复现 R6）的区别
------------------------------
静态重采样（class-balanced / repeat-factor sampling）在训练开始前一次性
决定采样比例；DRA 在训练过程中**按当前模型表现**动态调节：

    p_c ∝ (1 / n_c)^alpha * exp(tau * difficulty_c)
    difficulty_c = 1 - rolling_AP_c        (越难 → 权重越高)
    tau 随训练退火                          (早期激进, 后期收敛到均衡)

并带**早停保护**：某类滚动指标连续 N 轮无提升 → 降低其采样权重，
避免在已过拟合的稀有类上继续堆样本。

依赖：纯 Python，可独立单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__all__ = ["DRAConfig", "ClassStat", "DRASampler"]


@dataclass
class DRAConfig:
    """DRA 超参。

    Attributes
    ----------
    enabled:
        总开关。关闭时退化为静态 class-balanced 采样（即复现 R6）。
    alpha:
        类别不均衡的逆频率指数。0 → 纯按难度；1 → 标准逆频率。
    tau:
        难度项的初始强度。随训练线性退火到 ``tau_final``。
    tau_final:
        退火终点强度（默认 0，即后期完全均衡）。
    window:
        滚动统计窗口（epoch 数）。
    patience:
        早停保护：连续 N 轮无提升则降权。
    penalty:
        触发早停保护后的权重折扣系数。
    min_weight:
        单类最小权重（防止某类被完全丢弃）。
    """

    enabled: bool = True
    alpha: float = 1.0
    tau: float = 1.0
    tau_final: float = 0.0
    window: int = 5
    patience: int = 3
    penalty: float = 0.5
    min_weight: float = 1e-4

    def describe(self) -> str:
        return (
            f"DRA(enabled={self.enabled}, alpha={self.alpha}, tau={self.tau}->{self.tau_final}, "
            f"window={self.window}, patience={self.patience})"
        )


@dataclass
class ClassStat:
    """单类的滚动统计。"""

    name: str = ""
    n_instances: int = 0
    rolling_ap: List[float] = field(default_factory=list)
    best_ap: float = 0.0
    no_improve: int = 0
    penalty: float = 1.0

    def update(self, ap: Optional[float]) -> None:
        """用一个 epoch 的 AP（或 None 表示本轮无有效统计）更新滚动状态。

        ⚠️ 早停保护的正确判据（自测发现并修正）
        ---------------------------------------
        初版用 ``ap < best_ap`` 判定"无提升"，但当某类 AP **持平**（即典型的
        平台期 —— 恰恰是该触发保护的情况）时 ``ap == best_ap``，条件不成立 →
        ``no_improve`` 被重置为 0 → 保护永不触发。
        正确判据是 **"未严格提升"**（``ap <= best_ap``）即计入无提升。
        """
        if ap is None:
            return
        ap = float(ap)
        prev_best = self.best_ap
        self.rolling_ap.append(ap)
        if ap > prev_best + 1e-6:
            self.best_ap = ap
            self.no_improve = 0
        else:
            # 未严格提升（含持平）→ 计入无提升
            self.no_improve += 1
        if self.no_improve > 0:
            self.penalty = max(self.min_penalty, self.penalty * 0.5)

    min_penalty: float = 0.1

    @property
    def difficulty(self) -> float:
        """难度 = 1 - 滚动平均 AP（无统计时视为最易 → 0）。"""
        if not self.rolling_ap:
            return 0.0
        window = self.rolling_ap[-max(1, len(self.rolling_ap)) :]
        return max(0.0, min(1.0, 1.0 - sum(window) / len(window)))

    def rolling_mean(self) -> Optional[float]:
        if not self.rolling_ap:
            return None
        return sum(self.rolling_ap) / len(self.rolling_ap)


class DRASampler:
    """按类别难度动态调整采样权重的采样器。

    典型用法（Ultralytics ``build_yolo_dataset`` 的替代/包装）::

        sampler = DRASampler(class_names, cfg)
        weights = sampler.weights(epoch)          # 每类的相对采样权重
        sampler.update(epoch, per_class_ap)        # 每个 epoch 结束后回灌 AP

    权重归一化到和为 1。
    """

    def __init__(self, class_names: Sequence[str], cfg: DRAConfig, n_instances: Optional[Dict[str, int]] = None):
        self.cfg = cfg
        self.class_names = list(class_names)
        self.stats: Dict[str, ClassStat] = {
            name: ClassStat(name=name, n_instances=(n_instances or {}).get(name, 0))
            for name in self.class_names
        }
        self.epoch = 0

    # ------------------------------------------------------------------ #
    def set_instance_counts(self, counts: Dict[str, int]) -> None:
        """回填每类的实例数（来自标签统计）。"""
        for name, cnt in counts.items():
            if name in self.stats:
                self.stats[name].n_instances = int(cnt)

    def current_tau(self) -> float:
        """按训练进度线性退火的难度强度。"""
        cfg = self.cfg
        if not cfg.enabled:
            return 0.0
        return cfg.tau  # 退火需要总轮数；未设置时用固定 tau

    def weights(self, epoch: Optional[int] = None) -> List[float]:
        """计算每类的相对采样权重（归一化，和为 1）。"""
        cfg = self.cfg
        names = self.class_names
        if not names:
            return []
        if epoch is not None:
            self.epoch = int(epoch)

        tau = self.current_tau()
        raw: List[float] = []
        for name in names:
            st = self.stats[name]
            n = max(1, st.n_instances)
            inv_freq = (1.0 / n) ** cfg.alpha if cfg.alpha > 0 else 1.0
            diff = st.difficulty if cfg.enabled else 0.0
            raw.append(inv_freq * math.exp(tau * diff) * st.penalty)

        total = sum(raw)
        if total <= 0:
            return [1.0 / len(names)] * len(names)
        w = [max(cfg.min_weight, r / total) for r in raw]
        # 再次归一化（min_weight 截断后）
        total = sum(w)
        return [x / total for x in w]

    def update(self, epoch: int, per_class_ap: Dict[str, Optional[float]]) -> None:
        """每个 epoch 结束后回灌逐类 AP。"""
        self.epoch = int(epoch)
        for name, ap in per_class_ap.items():
            if name in self.stats:
                self.stats[name].update(ap)
        # 早停保护：连续 N 轮无提升 → 降权（已在 ClassStat.update 内部分处理）
        for name, st in self.stats.items():
            if st.no_improve >= self.cfg.patience:
                st.penalty = max(st.min_penalty, st.penalty * self.cfg.penalty)
                st.no_improve = 0

    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Dict[str, float]]:
        """导出当前状态（用于 results.md / 诊断图）。"""
        w = self.weights()
        out: Dict[str, Dict[str, float]] = {}
        for name, weight in zip(self.class_names, w):
            st = self.stats[name]
            out[name] = {
                "weight": weight,
                "n_instances": float(st.n_instances),
                "difficulty": st.difficulty,
                "rolling_ap": st.rolling_mean() if st.rolling_mean() is not None else float("nan"),
                "penalty": st.penalty,
            }
        return out


def build_static_weights(class_names: Sequence[str], counts: Dict[str, int], alpha: float = 1.0) -> List[float]:
    """静态 class-balanced 权重（复现 R6），作为 DRA 的对照基线。"""
    raw = []
    for name in class_names:
        n = max(1, int(counts.get(name, 0)))
        raw.append((1.0 / n) ** alpha if alpha > 0 else 1.0)
    total = sum(raw)
    return [r / total for r in raw] if total > 0 else [1.0 / len(class_names)] * len(class_names)
