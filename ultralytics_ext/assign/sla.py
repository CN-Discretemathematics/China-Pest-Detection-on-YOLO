"""ADEA · I1 · SLA —— 面积标定的公平正样本分配 (Scale-calibrated fair Label Assignment)

这是本项目的**核心创新**，由本机实测诊断（68,009 个 GT 样本）定型，
而非事先设想。诊断结论见 ``docs/方法设计_ADEA.md`` §2.0b。

诊断事实
--------
1. ``corr(log 相对尺度, n_pos) = +0.852``：正样本数由**目标面积**主导，
   相对尺度 <1e-4 的目标平均只有 2.27 个正样本，59.7% 不足 3 个。
2. 面积标定后，粘连目标仍被剥夺：残差 −0.761 vs 非粘连 0.000，
   ``corr(residual, β) = −0.247``，58.96% 的 GT 被低估。
3. 根因（ultralytics 8.4.184 ``utils/tal.py`` **实测代码事实**）：

   * ``select_topk_candidates`` 末尾的
     ``count_tensor.masked_fill_(count_tensor > 1, 0)`` 其实是**死代码**——
     ``count_tensor`` 按 ``(gt, anchor)`` 索引，同一 GT 的 top-k 下标互不重复，
     因此 ``count`` 恒 ≤ 1（本项目已用随机张量实测确认 ``count.max() == 1``）；
   * 真正的剥夺发生在 ``select_highest_overlaps``：
     一个 anchor 被多个 GT 同时选中（top-k 重叠）时，
     **只有与预测框 IoU 最高的那个 GT 保留它**（源码注释：
     "Anchors assigned to multiple gt_bboxes keep the highest overlap"），
     其余 GT **直接失去**这些 anchor。

   → 粘连簇里，共享区域被"几何上最贴合"的那个目标独占；
     尺寸较小的成员因此被系统性地剥夺了监督预算。

SLA 的两个动作
--------------
**M0 监督密度补偿（候选区扩展）—— 主机制**
    实测（本机 6,589 个 GT 样本）：

    | 事实 | 数值 |
    |---|---|
    | ``n_pos == min(topk, 候选池)`` 的比例 | **80.6%**（分配器基本已到几何上限） |
    | 竞争剥夺（池够却被抢） | **0.24%** —— 冲突剥夺**极其稀疏** |
    | 几何不足（候选池 < 3） | **7.54%** —— 微目标**天生**只有 1~2 个正样本 |
    | ``corr(log 相对面积, log 候选池)`` | **0.965** —— 候选池几乎完全由面积决定 |

    → 真正的瓶颈是**监督密度**（候选点太少），不是"被邻居抢占"。
    因此 SLA 把"公平份额"从"几何上限"提升为"**配额下限**"：

        q_i = max(q_min, min(topk, 候选池_i))

    达不到配额的微目标，其候选框被**按比例**扩张（保持中心，倍数与分辨率无关），
    扩张后的区域同样进入 ultralytics 的 ``align_metric`` 计算 →
    top-k 会自动挑走其中**对齐质量最好**的 anchor（无需额外算 IoU）。

**S1 冲突裁决的度量一致性**
    vanilla 用**纯 IoU** 裁决冲突，却用 **align_metric**（cls^α·IoU^β）选 top-k ——
    同一个 anchor 的"入选"与"归属"用了两套标准。
    SLA 改为**只用提名该 anchor 的 GT 参与裁决**，且默认按 ``align_metric`` 裁决
    （与 top-k 选择同度量）。可切回 ``criterion="overlap"`` 做消融。

**S2 面积标定的公平再分配**
    每个 GT 的**几何份额**定义为

        need_i = min(topk, 框内可用候选 anchor 数)

    这是一个**纯几何量**：候选数只取决于框的尺寸/位置与特征层 stride，
    因此与图像分辨率、模型结构无关（换 imgsz 或加 P2 层会自动跟随）。
    它同时就是"该 GT 若独处时能拿到的正样本数"——即**面积标定的期望份额**。

    分配完成后，若 ``need_i - count_i`` 明显大于 0（被邻簇挤占），
    就把别人手里**同时位于自己框内**的 anchor 转移过来，
    条件是自身对齐度量不低于对方的 ``match_ratio`` 倍（匹配度护栏 ρ）。

    转移是**零和**的（总正样本数不变），改变的只是**分布**：
    监督预算从"欠剥夺"的目标流向"被剥夺"的目标。

**为什么不用"面积表 + 自由参数"**
    早期设计用经验表 ``E[n_pos|a_rel]``（6 档）作为期望。实测发现两个隐患：
    (1) ``a_rel`` 的口径依赖分辨率（letterbox 缩放），表与运行时会错档；
    (2) 加 P2 层（stride 4）后候选池结构改变，同一张表不再成立。
    改用几何量 ``need_i = min(topk, 候选数)`` 后，两个隐患同时消失——
    这也是本文档所说的"面积标定"：**份额由面积（→ 候选数）决定，而非拍脑袋的常数**。

**与 P2 头的关系（重要）**
    候选池是**所有特征层 anchor 点的并集**（assigner 把各层 anchor 拼在一起），
    因此 P3 单独存在时 stride=8 会让微目标框内只有约 1 个候选点；
    引入 P2（stride=4）后，微目标的候选池直接放大 4~6 倍，
    ``need_i`` 随之自动变大 —— **SLA 的公平份额会自动适配新结构**。
    SLA（算法层）与 P2（架构层）互为补充，缺一不可。

不变式
------
* 每个 anchor 至多归属一个 GT（与 vanilla 一致 → **无需改 loss**）；
* 每个 GT 的正样本数 ≤ topk（不突破 vanilla 的候选上限）；
* S2 是零和转移（总正样本数不变）；**M0 会增加**总正样本数（这正是它的目的）；
* 全部在 torch 张量上完成，无 Python 逐样本循环。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SLAConfig",
    "expected_npos",
    "residual_npos",
    "sla_assign",
    "sla_select_topk_candidates",
    "sla_expand_boxes",
]


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class SLAConfig:
    """SLA 超参（全部有安全默认值，默认**关闭**）。

    Attributes
    ----------
    enabled:
        总开关。
    criterion:
        冲突裁决标准：``"align"``（align_metric，与 top-k 选择同度量，默认）
        或 ``"overlap"``（vanilla 的纯 IoU）。
    claim_only:
        冲突裁决是否只在**提名者**中进行（True，默认）。
        False = 严格复刻 vanilla（可在所有 GT 中取 max，可能把 anchor
        塞给 top-k 里没有它的 GT）。
    fairness:
        S2 —— 是否做面积标定的公平再分配。
    expand_enabled:
        是否启用**监督密度补偿**（候选区扩展）：候选池不足 ``q_min`` 的目标，
        其候选框按 ``expand_scale`` **按比例**扩张（保持中心），
        使其有机会获得不少于 ``q_min`` 个正样本。
    q_min:
        每个 GT 的**正样本配额下限**（与尺度无关的"公平下限"）。
        ``0`` 表示直接对齐 assigner 的 ``topk``（推荐：**无自由参数**）。
        实测：候选池不足的 GT 占相当比例，这是微目标监督不足的根源。
    expand_margin:
        候选区的扩张边距，单位是 **stride_val**（网格单位 → 与分辨率无关）。
    expand_mode:
        ``"quota"``（默认，本文）：**只对候选池不足配额的目标**扩张；
        ``"fixed"``：**对所有目标**按固定网格邻域扩张 —— 这是
        **YOLOv7 "allowed area"（粗粒度 5×5 邻域 / 细粒度 3×3 中心网格）**
        思路的对照实现，用于论文中"我们的配额驱动 vs 前人固定网格"的消融。
    fixed_margin_cells:
        ``fixed`` 模式每边扩张的网格数（默认 2 → 区域放大到 ≈5×5，贴近 YOLOv7 粗粒度）。
        ultralytics 会把尺寸小于 ``stride_val`` 的边抬到 ``stride_val``，
        因此乘性扩张对极小框无效，必须用边距。
    match_ratio:
        S2 的匹配度护栏 ρ：接收方的对齐度量至少要达到给出方的
        ``match_ratio`` 倍才允许转移（默认 0.5 = 至少一半）。
    tol:
        残差容差：``need - count > tol`` 才算"饿"（避免为 0.1 个正样本折腾）。
    min_gap:
        转移的最小"不平等度"：给出方的缺口至少要比接收方小 ``min_gap`` 个名额，
        转移才真正缩小了分布差距（Robin Hood 判据）。
    max_rounds:
        S2 最大迭代轮数（每轮每个 batch 元素转移 1 个 anchor）。
    max_transfer:
        单个 GT 在 S2 中最多接收的 anchor 数。
    eps:
        数值阈值（对齐度量 > eps 才算有效候选）。
    expect_table:
        **经验期望表**（仅用于统计/报告与旧路径兼容）。
        由 ``scripts/diagnose_adhesion.py`` 的分桶统计直接给出，无自由参数。
    k0 / c_fit:
        期望表的饱和上限与参数化兜底模型 ``min(k0, c·a_rel)`` 的系数。
    """

    enabled: bool = False
    criterion: str = "align"
    claim_only: bool = True
    fairness: bool = True
    match_ratio: float = 0.5
    tol: float = 0.5
    min_gap: float = 2.0
    max_rounds: int = 8
    max_transfer: int = 8
    eps: float = 1e-9

    # 监督密度补偿（候选区扩展）
    expand_enabled: bool = True
    q_min: int = 0            # 0 = 与 assigner 的 topk 对齐（无自由参数）
    expand_margin: float = 1.0  # 以 stride_val 为单位的扩张边距（quota 模式）
    expand_mode: str = "quota"   # quota = 配额驱动（本文）/ fixed = 固定邻域（YOLOv7 式对照）
    fixed_margin_cells: float = 2.0  # fixed 模式：每边扩张的网格数（2 → 近似 YOLOv7 粗粒度 5×5）

    # 经验期望表（报告用；不参与分配决策）
    expect_table: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.0, 1e-4, 2.27),
            (1e-4, 3e-4, 4.07),
            (3e-4, 1e-3, 6.31),
            (1e-3, 3e-3, 9.84),
            (3e-3, 1e-2, 9.99),
            (1e-2, 1.0, 10.0),
        ]
    )
    k0: int = 11
    c_fit: float = 9476.0
    area_ref: float = 0.0

    def describe(self) -> str:
        return (
            f"SLA(enabled={self.enabled}, criterion={self.criterion}, S2={self.fairness}, "
            f"rho={self.match_ratio}, tol={self.tol}, min_gap={self.min_gap}, "
            f"rounds={self.max_rounds})"
        )


# --------------------------------------------------------------------------- #
# 经验期望模型（报告用）
# --------------------------------------------------------------------------- #
def expected_npos(area_rel: float, cfg: SLAConfig) -> float:
    """期望正样本数 ``E[n_pos | a_rel]``（经验表优先，参数化模型兜底）。"""
    for lo, hi, e in cfg.expect_table:
        if lo <= area_rel < hi:
            return float(max(e, 0.0))
    return float(min(cfg.k0, cfg.c_fit * max(0.0, area_rel)))


def residual_npos(n_pos: float, area_rel: float, cfg: SLAConfig) -> float:
    """残差 ``n_pos - E[n_pos|a]``：负值表示该 GT 被剥夺。"""
    return float(n_pos) - expected_npos(area_rel, cfg)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _exclusive_owner(claims, score, eps: float = 1e-9, claim_only: bool = True):
    """冲突独占：每个被多个 GT 提名的 anchor 归给得分最高者。

    Parameters
    ----------
    claims:
        (b, n_gt, A) bool，True 表示该 GT 提名了该 anchor。
    score:
        (b, n_gt, A) 裁决得分（overlaps 或 align_metric）。
    claim_only:
        True（默认，SLA 规则）：只在**提名者**中裁决 —— 被裁决的 GT 一定
        自己想要这个 anchor，不会出现"平白塞给 top-k 里没有它的 GT"。
        False（vanilla 规则）：冲突 anchor 归给**所有 GT 中**得分最高者
        （ultralytics 原实现，可能把 anchor 塞给 top-k 里没有它的 GT），
        非冲突 anchor 保留原归属。

    Returns
    -------
    owner:
        (b, A) long，anchor 的归属 GT 下标，-1 表示无主。
    """
    import torch

    b, n_gt, n_anchor = claims.shape
    neg = torch.finfo(score.dtype).min / 4 if score.dtype.is_floating_point else -1e9

    # 唯一提名者（无冲突 anchor 的归属）
    has_claim = claims.any(dim=1)
    first = claims.to(torch.long).argmax(dim=1)          # (b, A) 第一个提名者
    sole = torch.where(has_claim, first, torch.full_like(first, -1))

    if claim_only:
        sc = torch.where(claims, score, torch.full_like(score, neg))
        best_idx = sc.argmax(dim=1)
        return torch.where(has_claim, best_idx, torch.full_like(best_idx, -1))

    # vanilla 规则：冲突 anchor → 全局最高分者；非冲突 → 保留原归属
    multi = claims.sum(dim=1) > 1
    global_best = score.argmax(dim=1)
    return torch.where(multi, global_best, sole)


# --------------------------------------------------------------------------- #
# 面积标定的候选区扩展（监督密度补偿）
# --------------------------------------------------------------------------- #
def sla_expand_boxes(gt_bboxes, need_expand, margin: float = 16.0):
    """面积标定的候选区扩展：候选池不足的 GT，其候选框**按边距**扩张（保持中心）。

    为什么需要它（G0 诊断 + 本机实测）
    ----------------------------------
    * 候选池（框内 anchor 数）几乎完全由目标面积决定：
      ``corr(log 相对面积, log 候选池) = 0.965``；
    * 微目标的**几何**候选池很小（受 stride 网格限制，10×10 的框只有 4~9 个 anchor 点），
      其正样本数因此被**几何上限**卡死，与"训练得好不好"无关；
    * 竞争剥夺（池够却被抢）只占 **0.24%** —— 真正的瓶颈是**密度**，不是抢占。

    因此 SLA 把"公平份额"从"几何上限"提升为"**配额下限**"：
    ``q_i = max(q_min, min(topk, 候选池_i))``；
    达不到配额的微目标，其候选区被扩张，
    扩张后的区域同样进入 ultralytics 的 ``align_metric`` 计算 →
    top-k 选择会自动挑走其中**对齐质量最好**的 anchor（无需额外算 IoU）。

    ⚠️ 为什么用**边距**（margin）而不是**倍数**（scale）
    --------------------------------------------------
    ultralytics 的 ``select_candidates_in_gts`` 会把尺寸小于 ``stride_val``
    的边**抬到 stride_val**（见其源码）→ 对极小框做乘性扩张会被这条规则吃掉
    （5×5 框 ×2 = 10×10 → 又被抬回 16×16 → 候选池毫无变化）。
    用**以 stride 为单位**的边距（``margin = m · stride_val``）则：
    * 与分辨率无关（网格单位）；
    * 对任意尺寸都有确定效果（最小也能把框撑到 ``(size + 2m·stride)``）。

    Parameters
    ----------
    gt_bboxes:
        (b, n_gt, 4) xyxy。
    need_expand:
        (b, n_gt) bool，True 表示该 GT 需要扩张。
    margin:
        每边扩张的像素数（调用方按 ``stride_val`` 折算）。
    Returns
    -------
    (b, n_gt, 4) xyxy，扩张后的框（未扩张的行原样返回）。
    """
    import torch

    if not margin or margin <= 0:
        return gt_bboxes

    m = float(margin)
    lt = gt_bboxes[..., :2] - m
    rb = gt_bboxes[..., 2:] + m
    grown = torch.cat((lt, rb), dim=-1)
    cond = need_expand[..., None].bool()
    return torch.where(cond, grown, gt_bboxes)


# --------------------------------------------------------------------------- #
# 核心：SLA 分配
# --------------------------------------------------------------------------- #
def sla_assign(
    mask_pos,
    align_metric,
    overlaps,
    topk: int,
    cfg: Optional[SLAConfig] = None,
) -> Tuple[Any, Dict[str, float]]:
    """执行 SLA 分配（冲突裁决 + 公平再分配）。

    Parameters
    ----------
    mask_pos:
        (b, n_gt, A) —— ultralytics 的 per-GT top-k 掩码
        （已乘 ``mask_in_gts`` 与 ``mask_gt``）。
    align_metric:
        (b, n_gt, A) —— 对齐度量（正值 ⇔ 该 anchor 位于该 GT 框内）。
    overlaps:
        (b, n_gt, A) —— 预测框与 GT 框的 CIoU。
    topk:
        每 GT 的候选上限（ultralytics 的 ``self.topk``，实测 10）。
    cfg:
        SLA 配置。

    Returns
    -------
    (mask_pos_new, info)
        ``mask_pos_new``: (b, n_gt, A) int8，每 anchor 至多一个 GT。
        ``info``: 统计信息。
    """
    import torch

    cfg = cfg or SLAConfig()
    b, n_gt, n_anchor = mask_pos.shape
    dev = mask_pos.device
    kk = max(1, min(int(topk), n_anchor))

    claims = mask_pos.bool()
    usable = align_metric > cfg.eps          # 框内且有匹配度 → 真正有监督价值的候选
    n_claims = claims.sum(dim=1)              # (b, A) 每个 anchor 被几个 GT 提名
    n_contested = int(((n_claims > 1) & (n_claims > 0)).sum())

    # ---- S1：冲突裁决 ----
    crit = align_metric if cfg.criterion == "align" else overlaps
    owner = _exclusive_owner(
        claims, crit.to(claims.device).float(), cfg.eps, claim_only=cfg.claim_only
    )
    own_mask = owner >= 0

    # 可用正样本数（只算有匹配度的：align=0 的 anchor 不产生梯度）
    usable_owned = own_mask & usable.gather(1, owner.clamp(min=0)[:, None, :]).squeeze(1)
    counts = torch.zeros((b, n_gt), dtype=torch.long, device=dev)
    counts.scatter_add_(1, owner.clamp(min=0), usable_owned.long())

    # 公平份额：need = min(topk, 框内可用候选数)
    cand_cnt = usable.sum(dim=-1)                        # (b, n_gt)
    need = torch.clamp(cand_cnt, max=kk).to(torch.long)
    gap_before = (need - counts).to(torch.float32)

    # ---- S2：面积标定的公平再分配 ----
    n_transfer = 0
    if cfg.fairness and cfg.max_rounds > 0:
        per_gt_recv = torch.zeros((b, n_gt), dtype=torch.long, device=dev)
        for _ in range(cfg.max_rounds):
            gap = (need - counts).to(torch.float32)          # ≥ 0，越大越"饿"
            # "最饿" = 缺口最大的那个 GT（论文中写作 most-deficient GT）
            #         缺口 gap_i = 公平份额 need_i - 实际拿到 count_i
            i_star = gap.argmax(dim=1)                      # (b,) 缺口最大的 GT
            gap_star = gap.gather(1, i_star[:, None])
            need_star = need.gather(1, i_star[:, None])
            cnt_star = counts.gather(1, i_star[:, None])
            recv_star = per_gt_recv.gather(1, i_star[:, None])

            # 给出方：缺口至少比接收方小 min_gap（Robin Hood 判据）且手里有 anchor
            donor_ok = (gap <= gap_star - cfg.min_gap) & (counts > 0)
            donor_ok.scatter_(1, i_star[:, None], False)
            can_receive = (
                (need_star - cnt_star > cfg.tol) & (recv_star < cfg.max_transfer)
            )

            own_idx = owner.clamp(min=0)
            owner_is_donor = donor_ok.gather(1, own_idx) & own_mask
            m_recv = align_metric.gather(1, i_star[:, None, None].expand(b, 1, n_anchor)).squeeze(1)
            m_give = align_metric.gather(1, own_idx[:, None, :]).squeeze(1)
            ok = (
                owner_is_donor
                & (m_recv > cfg.eps)                        # 该 anchor 位于接收方框内
                & (m_recv >= cfg.match_ratio * m_give)      # 匹配度护栏
                & can_receive
            )
            score = torch.where(ok, m_recv, torch.full_like(m_recv, -1e9))
            best = score.argmax(dim=1)                     # (b,)
            valid = score.gather(1, best[:, None]).squeeze(1) > -1e8
            if not bool(valid.any()):
                break

            b_idx = torch.arange(b, device=dev)
            old = own_idx[b_idx, best]
            owner[b_idx, best] = torch.where(valid, i_star, old)
            upd = torch.zeros((b, n_gt), dtype=torch.long, device=dev)
            upd.scatter_add_(1, i_star[:, None], valid.long()[:, None])
            # 给出方只在"该 anchor 对它本来就有效"时才扣减（可用正样本口径）
            tmp = usable.gather(1, old[:, None, None].expand(b, 1, n_anchor)).squeeze(1)
            give_usable = tmp.gather(1, best[:, None]).squeeze(1) & valid
            upd.scatter_add_(1, old[:, None], (-give_usable.long())[:, None])
            counts += upd
            per_gt_recv.scatter_add_(1, i_star[:, None], valid.long()[:, None])
            n_transfer += int(valid.sum())

        own_mask = owner >= 0

    # ---- 重建 mask_pos（每个 anchor 至多一个 GT）----
    out = torch.zeros((b, n_anchor, n_gt), dtype=torch.int8, device=dev)
    idx = owner.clamp(min=0).unsqueeze(-1)
    out.scatter_(2, idx, torch.ones_like(idx, dtype=torch.int8))
    out = out * own_mask.unsqueeze(-1).to(torch.int8)
    mask_new = out.permute(0, 2, 1).contiguous()

    # ---- 统计 ----
    usable_owned = own_mask & usable.gather(1, owner.clamp(min=0)[:, None, :]).squeeze(1)
    counts = torch.zeros((b, n_gt), dtype=torch.long, device=dev)
    counts.scatter_add_(1, owner.clamp(min=0), usable_owned.long())
    gap_after = (need - counts).to(torch.float32)
    starved_before = int((gap_before > cfg.tol).sum())
    starved_after = int((gap_after > cfg.tol).sum())

    info: Dict[str, float] = {
        "n_contested_anchors": float(n_contested),
        "n_sla_pos": float(int(usable_owned.sum())),
        "n_transfer": float(n_transfer),
        "starved_before": float(starved_before),
        "starved_after": float(starved_after),
        "mean_gap_before": float(gap_before.mean()),
        "mean_gap_after": float(gap_after.mean()),
        "max_gap_before": float(gap_before.max()),
        "max_gap_after": float(gap_after.max()),
        "n_gt": float(n_gt),
    }
    return mask_new, info


def sla_select_topk_candidates(
    metrics,
    topk: int,
    topk_mask=None,
    cfg: Optional[SLAConfig] = None,
) -> Tuple[Any, Dict[str, float]]:
    """兼容入口：在只有 ``align_metric`` 时模拟 SLA 分配（用于单元自测）。

    真实训练路径请使用 :func:`sla_assign`（它同时拿到 ``overlaps``，
    才能复现 vanilla 的 IoU 独占裁决）。
    """
    import torch

    cfg = cfg or SLAConfig()
    b, n_gt, n_anchor = metrics.shape
    kk = max(1, min(int(topk), n_anchor))

    topk_metrics, topk_idxs = torch.topk(metrics, kk, dim=-1, largest=True)
    if topk_mask is not None and topk_mask.shape[-1] == kk:
        topk_metrics = topk_metrics * topk_mask.expand(-1, -1, kk).to(topk_metrics.dtype)
    claims = torch.zeros((b, n_gt, n_anchor), dtype=torch.int8, device=metrics.device)
    claims.scatter_(2, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
    valid = torch.zeros((b, n_gt, n_anchor), dtype=torch.int8, device=metrics.device)
    valid.scatter_(2, topk_idxs, (topk_metrics > cfg.eps).to(torch.int8))
    claims = claims * valid
    return sla_assign(claims, metrics, metrics, topk=topk, cfg=cfg)
