"""ADEA 接线层：把自定义模块 / 损失 / 标签分配挂进 Ultralytics 训练栈。

职责
----
1. **注册自定义模块**：让 yaml 里能写 ``SPDConv`` / ``DySample``。
2. **打补丁标签分配**：在 ``TaskAlignedAssigner.assign`` 之后施加
   I1 ALA（+ I2 DLA 的密度配额调制）。
3. **安装自定义损失**：I5 无关；R5 的 Scale-Quality Aware EIoU 走
   :func:`ultralytics_ext.loss.scale_eiou.install_scale_quality_eiou`。
4. **配置来源**：环境变量（主）+ 可选 JSON 文件（次）。

为什么用环境变量而不是 CLI 参数
-------------------------------
Ultralytics 的 CLI 会对未知键报错（``yolo detect train model=... ADEA_ALA=1``
会抛 invalid key）。环境变量既能覆盖所有入口（CLI / Python API），
又不需要改动 Ultralytics 源码。示例::

    ADEA_ALA=1 ADEA_DLA=1 ADEA_BETA_THR=0.05 yolo detect train ...

或::

    ADEA_CONFIG=configs/adea.json yolo detect train ...

配置键（全部可选，均有安全默认值）
----------------------------------
============================  ==========================================
环境变量                       含义
============================  ==========================================
``ADEA_SLA``                   **I1 SLA 开关（0/1），默认 0 —— 核心创新**
``ADEA_SLA_CRITERION``         冲突裁决标准：align（默认）/ overlap
``ADEA_SLA_CLAIM_ONLY``        只在提名者中裁决（1，默认）/ 复刻 vanilla（0）
``ADEA_SLA_S2``                SLA 的 S2（公平再分配）开关，默认 1
``ADEA_SLA_MATCH_RATIO``       S2 匹配度护栏 ρ，默认 0.5
``ADEA_SLA_TOL``               S2 残差容差，默认 0.5
``ADEA_SLA_MIN_GAP``           转移的最小不平等度，默认 2.0
``ADEA_SLA_ROUNDS``            S2 迭代轮数，默认 8
``ADEA_SLA_MAX_TRANSFER``      单 GT 最多接收的 anchor 数，默认 8
``ADEA_SLA_EXPAND``            监督密度补偿（候选区扩展）开关，默认 1
``ADEA_SLA_QMIN``              每 GT 的正样本配额下限，默认 0（= 对齐 topk）
``ADEA_SLA_EXPAND_MARGIN``     候选区扩张边距（单位 stride_val），默认 1.0
``ADEA_ALA``                   ALA 开关（0/1），默认 0（降级为消融项）
``ADEA_ALA_BETA_THR``          粘连阈值 β，默认 0.05（论文口径）
``ADEA_ALA_Q_MIN``             每个 GT 的最小正样本数，默认 3
``ADEA_DLA``                   I2 开关（0/1），默认 0
``ADEA_DLA_MU``                密度调制强度 μ，默认 0.5
``ADEA_SFW``                   **I1b SFW 开关（0/1），默认 0 —— 尺度公平监督权重**
``ADEA_SFW_CLIP``              SFW 单样本权重上界（相对倍数），默认 10
``ADEA_SQA_EIOU``              R5 开关（0/1），默认 0
``ADEA_PAIR_CLS``              I6 开关（0/1），默认 0
``ADEA_AGS``                   I5 开关（0/1），默认 0
``ADEA_DRA``                   I3 开关（0/1），默认 0
``ADEA_CONFIG``                JSON 配置文件路径（覆盖上述默认值）
``ADEA_VERBOSE``               打印接线报告（0/1），默认 1
============================  ==========================================
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["ADEAConfig", "load_config", "install_all", "register_modules", "patch_assigner"]


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class ADEAConfig:
    """ADEA 全部开关与超参。

    默认**全部关闭** —— 只有显式设置环境变量/JSON 才会启用，
    保证 baseline 训练不受任何影响（A/B 对照的前提）。
    """

    # I1 SLA（核心创新）
    sla_enabled: bool = False
    sla_criterion: str = "align"
    sla_claim_only: bool = True
    sla_s2: bool = True
    sla_match_ratio: float = 0.5
    sla_tol: float = 0.5
    sla_min_gap: float = 2.0
    sla_rounds: int = 8
    sla_max_transfer: int = 8
    sla_expand: bool = True
    sla_q_min: int = 0
    sla_expand_margin: float = 1.0

    # I1 ALA（已降级为消融项）
    ala_enabled: bool = False
    ala_beta_thr: float = 0.05
    ala_q_min: int = 3
    ala_max_reassign_ratio: float = 0.5
    ala_min_iou: float = 0.05

    # I2 DLA
    dla_enabled: bool = False
    dla_mu: float = 0.5
    dla_sigma_scale: float = 0.5

    # I3 DRA
    dra_enabled: bool = False
    dra_alpha: float = 1.0
    dra_tau: float = 1.0
    dra_window: int = 5
    dra_patience: int = 3

    # I5 AGS
    ags_enabled: bool = False
    ags_iou_thr: float = 0.5
    ags_same_instance_iou: float = 0.80
    ags_decay: float = 0.5

    # I1b SFW（尺度公平的监督权重）
    sfw_enabled: bool = False
    sfw_clip: float = 10.0

    # R5 / I6
    sqa_eiou_enabled: bool = False
    sqa_scale_weight: float = 0.5
    sqa_quality_weight: float = 0.5
    pair_cls_enabled: bool = False
    pair_cls_conf_threshold: int = 5
    pair_cls_top_k: int = 3
    pair_cls_margin: float = 2.0
    pair_cls_weight: float = 0.1

    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def load_config(config_path: Optional[str] = None) -> ADEAConfig:
    """从环境变量 + 可选 JSON 载入配置。"""
    cfg = ADEAConfig()

    path = config_path or os.environ.get("ADEA_CONFIG")
    if path:
        p = Path(path)
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                    else:
                        cfg.extra[k] = v
            except Exception as exc:  # pragma: no cover
                print(f"[ADEA] WARN 读取配置文件失败 {p}: {exc}")

    cfg.sla_enabled = _env_bool("ADEA_SLA", cfg.sla_enabled)
    cfg.sla_criterion = os.environ.get("ADEA_SLA_CRITERION", cfg.sla_criterion).strip() or cfg.sla_criterion
    cfg.sla_claim_only = _env_bool("ADEA_SLA_CLAIM_ONLY", cfg.sla_claim_only)
    cfg.sla_s2 = _env_bool("ADEA_SLA_S2", cfg.sla_s2)
    cfg.sla_match_ratio = _env_float("ADEA_SLA_MATCH_RATIO", cfg.sla_match_ratio)
    cfg.sla_tol = _env_float("ADEA_SLA_TOL", cfg.sla_tol)
    cfg.sla_min_gap = _env_float("ADEA_SLA_MIN_GAP", cfg.sla_min_gap)
    cfg.sla_rounds = _env_int("ADEA_SLA_ROUNDS", cfg.sla_rounds)
    cfg.sla_max_transfer = _env_int("ADEA_SLA_MAX_TRANSFER", cfg.sla_max_transfer)
    cfg.sla_expand = _env_bool("ADEA_SLA_EXPAND", cfg.sla_expand)
    cfg.sla_q_min = _env_int("ADEA_SLA_QMIN", cfg.sla_q_min)
    cfg.sla_expand_margin = _env_float("ADEA_SLA_EXPAND_MARGIN", cfg.sla_expand_margin)

    cfg.ala_enabled = _env_bool("ADEA_ALA", cfg.ala_enabled)
    cfg.ala_beta_thr = _env_float("ADEA_ALA_BETA_THR", cfg.ala_beta_thr)
    cfg.ala_q_min = _env_int("ADEA_ALA_Q_MIN", cfg.ala_q_min)
    cfg.ala_max_reassign_ratio = _env_float("ADEA_ALA_MAX_REASSIGN", cfg.ala_max_reassign_ratio)
    cfg.ala_min_iou = _env_float("ADEA_ALA_MIN_IOU", cfg.ala_min_iou)

    cfg.dla_enabled = _env_bool("ADEA_DLA", cfg.dla_enabled)
    cfg.dla_mu = _env_float("ADEA_DLA_MU", cfg.dla_mu)
    cfg.dla_sigma_scale = _env_float("ADEA_DLA_SIGMA_SCALE", cfg.dla_sigma_scale)

    cfg.dra_enabled = _env_bool("ADEA_DRA", cfg.dra_enabled)
    cfg.dra_alpha = _env_float("ADEA_DRA_ALPHA", cfg.dra_alpha)
    cfg.dra_tau = _env_float("ADEA_DRA_TAU", cfg.dra_tau)
    cfg.dra_window = _env_int("ADEA_DRA_WINDOW", cfg.dra_window)
    cfg.dra_patience = _env_int("ADEA_DRA_PATIENCE", cfg.dra_patience)

    cfg.ags_enabled = _env_bool("ADEA_AGS", cfg.ags_enabled)
    cfg.ags_iou_thr = _env_float("ADEA_AGS_IOU_THR", cfg.ags_iou_thr)
    cfg.ags_same_instance_iou = _env_float("ADEA_AGS_SAME_IOU", cfg.ags_same_instance_iou)
    cfg.ags_decay = _env_float("ADEA_AGS_DECAY", cfg.ags_decay)

    cfg.sfw_enabled = _env_bool("ADEA_SFW", cfg.sfw_enabled)
    cfg.sfw_clip = _env_float("ADEA_SFW_CLIP", cfg.sfw_clip)

    cfg.sqa_eiou_enabled = _env_bool("ADEA_SQA_EIOU", cfg.sqa_eiou_enabled)
    cfg.sqa_scale_weight = _env_float("ADEA_SQA_SCALE_W", cfg.sqa_scale_weight)
    cfg.sqa_quality_weight = _env_float("ADEA_SQA_QUALITY_W", cfg.sqa_quality_weight)

    cfg.pair_cls_enabled = _env_bool("ADEA_PAIR_CLS", cfg.pair_cls_enabled)
    cfg.pair_cls_conf_threshold = _env_int("ADEA_PAIR_CONF_THR", cfg.pair_cls_conf_threshold)
    cfg.pair_cls_top_k = _env_int("ADEA_PAIR_TOP_K", cfg.pair_cls_top_k)
    cfg.pair_cls_margin = _env_float("ADEA_PAIR_MARGIN", cfg.pair_cls_margin)
    cfg.pair_cls_weight = _env_float("ADEA_PAIR_WEIGHT", cfg.pair_cls_weight)

    cfg.verbose = _env_bool("ADEA_VERBOSE", cfg.verbose)
    return cfg


# --------------------------------------------------------------------------- #
# 1) 自定义模块注册
# --------------------------------------------------------------------------- #
def register_modules() -> Dict[str, str]:
    """让 yaml 解析器认识自定义模块名。

    Ultralytics 的 ``parse_model`` 按字符串名在 tasks 模块的 globals 里查表，
    因此需要把自定义类注入 ``ultralytics.nn.tasks``。

    ⚠️ **必须同时注入 ``base_modules``**
    ------------------------------------
    ``parse_model`` 对 ``base_modules`` 中的模块按
    ``[c1, c2_scaled, *args[1:]]`` 传参并正确记账输出通道；
    对**未知**模块则走 ``else`` 分支 —— 那里
    ``c2 = ch[f]``（输出通道被强制等于**输入**通道）且 args 原样传入。
    若不注册进 ``base_modules``：
    * ``SPDConv`` 拿不到正确的 ``c1``/``c2`` → 维度错乱；
    * ``DySample`` 的 ``scale`` 会落到 ``kernel_size`` 参数位。

    ``base_modules`` 是 tasks 模块的全局变量，``parse_model`` 内部按全局名查找，
    因此 ``tasks.base_modules = tasks.base_modules | {...}`` 即可生效。

    Returns
    -------
    dict
        ``{模块名: 来源}``，用于接线报告。
    """
    import ultralytics.nn.tasks as tasks

    from ultralytics_ext.modules.spd_conv import SPDConv
    from ultralytics_ext.modules.dysample import DySample
    from ultralytics_ext.modules.ema_attn import get_ema
    from ultralytics_ext.modules.ema_blocks import C3k2EMA

    registered: Dict[str, str] = {}

    # SPDConv
    if not hasattr(tasks, "SPDConv"):
        setattr(tasks, "SPDConv", SPDConv)
        registered["SPDConv"] = "ultralytics_ext.modules.spd_conv"
    else:
        registered["SPDConv"] = "already present"

    # DySample
    if not hasattr(tasks, "DySample"):
        setattr(tasks, "DySample", DySample)
        registered["DySample"] = "ultralytics_ext.modules.dysample"
    else:
        registered["DySample"] = "already present"

    # EMA：使用**通道守恒**变体（官方版本会翻倍通道，与 parse_model 的记账冲突）
    ema_cls, ema_src = get_ema()
    setattr(tasks, "EMA", ema_cls)
    registered["EMA"] = ema_src

    # C3k2EMA：带 EMA 的 C3k2 子类（**同层替换**，保住预训练参数名）
    setattr(tasks, "C3k2EMA", C3k2EMA)
    registered["C3k2EMA"] = "ultralytics_ext.modules.ema_blocks"

    # ---- 注入 base_modules / repeat_modules，使 parse_model 正确传参与记账 ----
    # ⚠️ 两组必须**分开**：
    #    base_modules 里的模块会按 [c1, c2_scaled, *args[1:]] 传参（通道记账）；
    #    repeat_modules 里的模块会额外执行 args.insert(2, n) —— 把 repeats 数插到第 3 位。
    #    只有"替换 C3k2"的 C3k2EMA 需要 repeat 语义；
    #    若把 DySample/EMA/SPDConv 也加进 repeat_modules，它们的第 3 个位置参数
    #    会被 n 顶掉（例如 DySample 的 scale=2 变成 scale=1）→ 前向形状错乱。
    registered["base_modules"] = _patch_base_modules(
        base_extra=[SPDConv, DySample, ema_cls, C3k2EMA],
        repeat_extra=[C3k2EMA],
    )

    return registered


def _patch_base_modules(base_extra, repeat_extra=None) -> str:
    """把自定义模块注入 ``parse_model`` 的 ``base_modules`` 与 ``repeat_modules``。

    ⚠️ 为什么需要源码级注入
    ----------------------
    在 ultralytics 8.4.184 中，``base_modules`` / ``repeat_modules`` 都是
    ``parse_model`` 的**函数内局部变量**（不是模块全局），无法直接改写。

    而这一注入是**必需**的：``parse_model`` 只对 ``base_modules`` 中的模块按
    ``[c1, c2_scaled, *args[1:]]`` 传参并正确记账输出通道；对**未知**模块则走
    ``else`` 分支 —— 那里 ``c2 = ch[f]``（输出通道被强制等于**输入**通道）
    且 args **原样**传入。若不注入：
    * ``SPDConv`` 拿不到正确的 ``c1``/``c2`` → 维度错乱；
    * ``EMA`` 只会收到 1 个参数 → 构造失败；
    * ``DySample`` 的 ``scale`` 会落到 ``kernel_size`` 参数位；
    * ``C3k2EMA`` 不会被插入 repeats（``n``）→ 层深改变。

    实现：把两处 ``frozenset(...)`` 替换为 ``_adea_frozenset(...)``，
    由该函数返回 ``原集合 | 对应的额外模块`` —— 与源码排版无关。

    Returns
    -------
    str
        人类可读的状态说明。
    """
    import inspect

    import ultralytics.nn.tasks as tasks

    if getattr(tasks, "_adea_parse_model_patched", False):
        return "already patched"

    try:
        src = inspect.getsource(tasks.parse_model)
    except Exception as exc:  # pragma: no cover
        return f"FAILED: cannot read parse_model source ({exc})"

    if "base_modules = frozenset(" not in src or "repeat_modules = frozenset(" not in src:
        return "FAILED: 'base_modules/repeat_modules = frozenset(' marker not found (version changed?)"

    src2 = src.replace("base_modules = frozenset(", "base_modules = _adea_base_frozenset(", 1)
    src2 = src2.replace("repeat_modules = frozenset(", "repeat_modules = _adea_repeat_frozenset(", 1)

    g = vars(tasks)
    g["_ADEA_BASE_EXTRA"] = frozenset(base_extra)
    g["_ADEA_REPEAT_EXTRA"] = frozenset(repeat_extra or [])

    def _adea_base_frozenset(_s):
        """ADEA 注入点（base_modules）：并入需要 (c1, c2, ...) 传参的模块。"""
        return frozenset(_s) | g["_ADEA_BASE_EXTRA"]

    def _adea_repeat_frozenset(_s):
        """ADEA 注入点（repeat_modules）：只并入需要 repeats 语义的模块。"""
        return frozenset(_s) | g["_ADEA_REPEAT_EXTRA"]

    g["_adea_base_frozenset"] = _adea_base_frozenset
    g["_adea_repeat_frozenset"] = _adea_repeat_frozenset

    try:
        exec(compile(src2, "<adea_parse_model>", "exec"), g)  # noqa: S102 - 受控注入
    except Exception as exc:  # pragma: no cover
        return f"FAILED: exec failed ({exc})"

    tasks.parse_model = g["parse_model"]
    tasks._adea_parse_model_patched = True
    return (
        f"source-injected: base=[{', '.join(m.__name__ for m in base_extra)}] "
        f"repeat=[{', '.join(m.__name__ for m in (repeat_extra or []))}]"
    )


# --------------------------------------------------------------------------- #
# 2) 标签分配打补丁（I1 ALA + I2 DLA）
# --------------------------------------------------------------------------- #
def _infer_stride_from_points(points) -> float:
    """从 anchor 网格中心点反推该层的 stride（像素）。

    TAL 传入的 ``anchor_points`` 是该特征层的网格中心（stride 间隔），
    因此相邻网格中心的间距即为 stride。这样无需外部状态即可还原
    anchor 框尺寸，避免依赖 Ultralytics 内部字段名（版本易变）。
    """
    import torch

    if points is None or len(points) == 0:
        return 1.0
    xs = torch.unique(points[:, 0])
    if xs.numel() < 2:
        return 1.0
    diffs = xs[1:] - xs[:-1]
    diffs = diffs[diffs > 0]
    if diffs.numel() == 0:
        return 1.0
    return float(diffs.min())


def patch_assigner(cfg: ADEAConfig) -> Dict[str, str]:
    """给标签分配器打上 ALA / DLA 补丁。

    ⚠️ ultralytics 8.4.x 的 API 变化（实测）
    ----------------------------------------
    * 8.3.x：``TaskAlignedAssigner.assign(...)`` —— 独占分配用 ``mask_pos.argmax(-2)``；
    * **8.4.x**：``TaskAlignedAssigner`` 重构为 ``nn.Module``，分配入口是
      ``forward(...)``，独占分配发生在 ``select_highest_overlaps(...)``
      （源码注释："Anchors assigned to multiple gt_bboxes keep the highest overlap"）——
      这正是 M1 描述的"实例被饿死"机制。
    * 8.4.x 的分配器**已经把 GT×anchor 的 IoU 矩阵**算好（``overlaps``），
      ALA 直接复用，无需自行构造 anchor 框。

    因此本实现打**两个**补丁：
    1. ``get_pos_mask`` —— 暂存 ``gt_bboxes``（后续步骤需要它算 GT 间粘连度）；
    2. ``select_highest_overlaps`` —— 在独占分配**之后**施加 ALA 配额补偿。
       由于 ``_forward`` 随后会用修改后的 ``target_gt_idx`` 去算 targets 与归一化，
       下游全部保持自洽。

    保持"每个 anchor 至多归属一个 GT"这一不变式 → **无需修改 loss**。
    """
    report: Dict[str, str] = {}
    if not (cfg.ala_enabled or cfg.dla_enabled):
        report["assigner"] = "skipped (ADEA_ALA=0 and ADEA_DLA=0)"
        return report

    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as exc:  # pragma: no cover
        report["assigner"] = f"FAILED to import TaskAlignedAssigner: {exc}"
        return report

    from ultralytics_ext.assign.ala import ALAConfig, apply_ala_assign
    from ultralytics_ext.assign.dla import DLAConfig, per_gt_quota, xyxy_to_xywh_list

    ala_cfg = ALAConfig(
        enabled=cfg.ala_enabled,
        beta_thr=cfg.ala_beta_thr,
        q_min=cfg.ala_q_min,
        max_reassign_ratio=cfg.ala_max_reassign_ratio,
        min_iou=cfg.ala_min_iou,
        enable_quota=True,
        enable_soft=False,
    )
    dla_cfg = DLAConfig(enabled=cfg.dla_enabled, mu=cfg.dla_mu, sigma_scale=cfg.dla_sigma_scale)

    if getattr(TaskAlignedAssigner, "_adea_patched", False):
        report["assigner"] = "already patched"
        return report

    # ---- 补丁 1：暂存 gt_bboxes（get_pos_mask 先于 select_highest_overlaps 被调用）----
    orig_get_pos_mask = TaskAlignedAssigner.get_pos_mask

    def patched_get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra):
        self._adea_gt_bboxes = gt_bboxes
        self._adea_mask_gt = mask_gt
        return orig_get_pos_mask(
            self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra
        )

    # ---- 补丁 2：在独占分配之后施加 ALA 配额补偿 ----
    orig_select = TaskAlignedAssigner.select_highest_overlaps

    # 失效计数：补丁异常若被静默吞掉，训练照常跑但创新没生效 —— 必须让它可见
    ade_state = {"errors": 0, "first_tb": None, "reassigned": 0, "starved": 0}

    def patched_select(self, mask_pos, overlaps, n_max_boxes, align_metric):
        target_gt_idx, fg_mask, mask_pos_out = orig_select(
            self, mask_pos, overlaps, n_max_boxes, align_metric
        )
        try:
            gt_bboxes = getattr(self, "_adea_gt_bboxes", None)
            if gt_bboxes is None:
                return target_gt_idx, fg_mask, mask_pos_out

            # 密度配额调制：**按 batch 组织**（每个 batch 元素一个逐 GT 配额 list）
            # ⚠️ 两点必须注意：
            #   1) gt_bboxes 是 **xyxy**，而 per_gt_quota 内部按 **xywh**（中心在 b[0],b[1]）
            #      解释 → 不转换会把密度采样在框的左上角而非中心；
            #   2) gt_bboxes 是**填充后**的（每张图补齐到 n_max_boxes），
            #      零填充行会污染密度场 → 只取 mask_gt 标记的有效 GT。
            per_gt_qmin = None
            if cfg.dla_enabled:
                mask_gt = getattr(self, "_adea_mask_gt", None)
                per_gt_qmin = []
                for b in range(gt_bboxes.shape[0]):
                    gtb = gt_bboxes[b].detach().cpu().tolist()
                    if mask_gt is not None:
                        valid = mask_gt[b].reshape(-1).bool().detach().cpu().tolist()
                        gtb = [g for g, v in zip(gtb, valid) if v]
                    per_gt_qmin.append(
                        per_gt_quota(xyxy_to_xywh_list(gtb), cfg.ala_q_min, dla_cfg)
                    )

            new_tgi, _fg, st = apply_ala_assign(
                target_gt_idx,
                fg_mask,
                gt_bboxes,
                overlaps,
                getattr(self, "_adea_mask_gt", None),
                ala_cfg,
                per_gt_qmin=per_gt_qmin,
            )
            ade_state["reassigned"] += st.n_reassigned
            ade_state["starved"] += st.n_starved_gt
            if st.n_reassigned > 0 and cfg.verbose:
                print(
                    f"[ADEA] ALA/DLA 生效: clusters={st.n_clusters} "
                    f"starved={st.n_starved_gt} reassigned={st.n_reassigned}"
                )
            return new_tgi, fg_mask, mask_pos_out
        except Exception as exc:  # pragma: no cover - 绝不让补丁搞崩训练
            ade_state["errors"] += 1
            if ade_state["first_tb"] is None:
                import traceback

                ade_state["first_tb"] = traceback.format_exc()
                print(f"[ADEA] ERROR 标签分配补丁异常（已回退原始分配），首次堆栈：\n{ade_state['first_tb']}")
            elif cfg.verbose and ade_state["errors"] % 500 == 0:
                print(f"[ADEA] ERROR 标签分配补丁异常累计 {ade_state['errors']} 次")
            return target_gt_idx, fg_mask, mask_pos_out

    TaskAlignedAssigner.get_pos_mask = patched_get_pos_mask
    TaskAlignedAssigner.select_highest_overlaps = patched_select
    TaskAlignedAssigner._adea_patched = True
    TaskAlignedAssigner._adea_state = ade_state
    report["assigner"] = (
        f"patched TaskAlignedAssigner.get_pos_mask + select_highest_overlaps "
        f"(ALA={cfg.ala_enabled}, DLA={cfg.dla_enabled}, "
        f"beta_thr={cfg.ala_beta_thr}, q_min={cfg.ala_q_min})"
    )
    return report


def assigner_health() -> Dict[str, object]:
    """读取标签分配补丁的运行统计（用于训练后确认创新真的生效了）。

    Returns
    -------
    dict
        ALA 部分（``ala_*``）：
        * ``ala_errors > 0`` → 补丁抛过异常并回退，**本次 ALA 结果不可信**；
        * ``ala_reassigned == 0`` → 补丁没找到可回收的 anchor，ALA 没有产生任何作用。

        SLA 部分（``sla_*``）：
        * ``sla_errors > 0`` → SLA 抛过异常并回退，**本次 SLA 结果不可信**；
        * ``sla_batches == 0`` → SLA 根本没被调用（接线失败）；
        * ``sla_n_contested_anchors == 0`` → 这批数据里没有 anchor 冲突，
          SLA 无从发挥（**不等于 SLA 无效**）；
        * ``sla_n_transfer > 0`` → S2 公平再分配确实发生了转移；
        * ``sla_starved_after < sla_starved_before`` → 被剥夺的 GT 变少；
        * ``sla_mean_gap_after < sla_mean_gap_before`` → 面积标定的分配缺口收敛。
    """
    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception:  # pragma: no cover
        return {"errors": -1, "reassigned": -1, "starved": -1, "patched": False}

    out: Dict[str, object] = {}
    st = getattr(TaskAlignedAssigner, "_adea_state", None)
    if st is None:
        out.update({"errors": 0, "reassigned": 0, "starved": 0, "patched": False})
    else:
        out.update(
            {
                "errors": st["errors"],
                "reassigned": st["reassigned"],
                "starved": st["starved"],
                "patched": True,
            }
        )

    sla = getattr(TaskAlignedAssigner, "_adea_sla_state", None)
    if sla is None:
        out.update({"sla_patched": False, "sla_batches": 0, "sla_errors": 0})
    else:
        nb = max(1, sla["batches"])
        out.update(
            {
                "sla_patched": True,
                "sla_batches": sla["batches"],
                "sla_errors": sla["errors"],
                "sla_n_contested_anchors": int(sla["n_contested_anchors"]),
                "sla_n_sla_pos": int(sla["n_sla_pos"]),
                "sla_n_transfer": int(sla["n_transfer"]),
                "sla_starved_before": int(sla["starved_before"]),
                "sla_starved_after": int(sla["starved_after"]),
                "sla_n_expanded": int(sla.get("n_expanded", 0)),
                "sla_mean_gap_before": sla["gap_before"] / nb,
                "sla_mean_gap_after": sla["gap_after"] / nb,
                "sla_max_gap_before": sla["gap_max_before"] / nb,
                "sla_max_gap_after": sla["gap_max_after"] / nb,
            }
        )

    try:
        from ultralytics_ext.loss.scale_fair import scale_fair_health

        out.update({f"sfw_{k}": v for k, v in scale_fair_health().items()})
    except Exception:  # pragma: no cover
        pass
    return out


# --------------------------------------------------------------------------- #
# 2b) SLA 补丁（I1 核心创新）：冲突消解 + 面积标定的公平再分配
# --------------------------------------------------------------------------- #
def patch_sla(cfg: ADEAConfig) -> Dict[str, str]:
    """给标签分配器打上 SLA 补丁（本项目的核心创新）。

    ⚠️ ultralytics 8.4.x 的真实代码事实（``utils/tal.py``，本项目已实测）
    --------------------------------------------------------------------
    * ``select_topk_candidates`` 末尾的
      ``count_tensor.masked_fill_(count_tensor > 1, 0)`` 是**死代码**：
      ``count_tensor`` 按 ``(gt, anchor)`` 索引，同一 GT 的 top-k 下标互不重复，
      因此 ``count`` 恒 ≤ 1（实测 ``count.max() == 1``）；
    * 真正的剥夺发生在 ``select_highest_overlaps``：被多个 GT 同时选中的 anchor
      **只保留给 IoU 最高的那个 GT**，其余 GT 直接失去这些 anchor
      → 粘连簇里的小目标被系统性地剥夺监督预算（G0 诊断的根因）。

    补丁点选择
    ----------
    打在 ``get_pos_mask`` 上（而不是 ``select_topk_candidates``），因为只有这里
    同时拿得到 ``align_metric`` 与 ``overlaps``：
    * 复现/改写 vanilla 的冲突裁决（需要 ``overlaps``）；
    * 面积标定的公平再分配（需要 ``align_metric`` 判断"该 anchor 是否在接收方框内"）。

    SLA 的两个动作
    -------------
    * **S1 冲突裁决的度量一致性**：默认按 ``align_metric`` 裁决（与 top-k 选择
      同一度量），且只在**提名者**中裁决；可切 ``criterion=overlap`` /
      ``claim_only=False`` 复刻 vanilla 以做消融。
    * **S2 面积标定的公平再分配**：每个 GT 的公平份额
      ``need_i = min(topk, 框内可用候选数)``——**纯几何量**，与分辨率/结构无关；
      被挤占的 GT 可从欠剥夺的 GT 手里取回**同时位于自己框内**的 anchor，
      条件是自身对齐度量不低于对方的 ``match_ratio`` 倍（匹配度护栏）。
      转移零和（总正样本数不变），改变的是**分布**。

    为什么可以只改这一个函数
    ----------------------
    * 返回的 mask_pos 保持 ultralytics 的 ``(b, n_max_obj, h*w)`` 0/1 语义，
      下游 ``select_highest_overlaps`` 会照常做 ``max(-2)`` 取归属 → 完全自洽；
    * 每个 anchor 至多归属一个 GT → **"独占分配"不变式保持 → 无需改 loss**；
    * ``topk2 == topk``（ultralytics 默认 ``tal_topk=10``、``topk2=None``），
      ``select_highest_overlaps`` 里的二次过滤被跳过 → 我们的分配不会被改写。

    失效保护：补丁异常一律回退原始实现并计数（``assigner_health()`` 可查），
    绝不让"创新静默失效"污染实验结论。
    """
    if not cfg.sla_enabled:
        return {"sla": "skipped (ADEA_SLA=0)"}

    try:
        from ultralytics.utils.tal import TaskAlignedAssigner
    except Exception as exc:  # pragma: no cover
        return {"sla": f"FAILED to import TaskAlignedAssigner: {exc}"}

    from ultralytics_ext.assign.sla import SLAConfig, sla_assign, sla_expand_boxes

    sla_cfg = SLAConfig(
        enabled=True,
        criterion=cfg.sla_criterion,
        claim_only=cfg.sla_claim_only,
        fairness=cfg.sla_s2,
        match_ratio=cfg.sla_match_ratio,
        tol=cfg.sla_tol,
        min_gap=cfg.sla_min_gap,
        max_rounds=cfg.sla_rounds,
        max_transfer=cfg.sla_max_transfer,
        expand_enabled=cfg.sla_expand,
        q_min=cfg.sla_q_min,
        expand_margin=cfg.sla_expand_margin,
    )

    if getattr(TaskAlignedAssigner, "_adea_sla_patched", False):
        return {"sla": "already patched"}

    orig_get_pos_mask = TaskAlignedAssigner.get_pos_mask
    orig_select_cand = TaskAlignedAssigner.select_candidates_in_gts

    state = {
        "errors": 0,
        "batches": 0,
        "n_contested_anchors": 0,
        "n_sla_pos": 0,
        "n_transfer": 0,
        "starved_before": 0,
        "starved_after": 0,
        "gap_before": 0.0,
        "gap_after": 0.0,
        "gap_max_before": 0.0,
        "gap_max_after": 0.0,
        "n_expanded": 0,
        "first_tb": None,
    }

    def patched_select_cand(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        """面积标定的候选区扩展（监督密度补偿）。

        * 先按原逻辑算出框内候选池 ``pool_i``；
        * 池不足配额（``q_min``，默认对齐 assigner 的 ``topk``）的 GT，
          其候选框按 ``margin · stride_val`` **向四周扩张**（保持中心）；
        * 扩张后的区域同样进入 ``align_metric`` 计算 →
          top-k 会自动挑走其中对齐质量最好的 anchor。

        ⚠️ 只在 ``ADEA_SLA=1`` 且 ``ADEA_SLA_EXPAND=1`` 时启用。
        """
        import torch

        mask = orig_select_cand(self, xy_centers, gt_bboxes, mask_gt, eps)
        if not sla_cfg.expand_enabled:
            return mask
        try:
            pool = mask.sum(-1)
            valid = mask_gt.reshape(mask_gt.shape[0], -1).bool()
            quota = int(getattr(self, "topk", 10)) if sla_cfg.q_min <= 0 else int(sla_cfg.q_min)
            need = (pool < quota) & valid
            if not bool(need.any()):
                return mask
            stride_val = float(getattr(self, "stride_val", 16) or 16)
            margin = sla_cfg.expand_margin * stride_val
            gt2 = sla_expand_boxes(gt_bboxes, need, margin)
            mask2 = orig_select_cand(self, xy_centers, gt2, mask_gt, eps)
            state["n_expanded"] += int(need.sum())
            return torch.where(need[..., None], mask2, mask)
        except Exception:  # pragma: no cover
            state["errors"] += 1
            return mask

    def patched_get_pos_mask(
        self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra
    ):
        mask_pos, align_metric, overlaps = orig_get_pos_mask(
            self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra
        )
        try:
            mask_new, info = sla_assign(
                mask_pos,
                align_metric,
                overlaps,
                topk=int(getattr(self, "topk", 10)),
                cfg=sla_cfg,
            )
            state["batches"] += 1
            state["n_contested_anchors"] += info["n_contested_anchors"]
            state["n_sla_pos"] += info["n_sla_pos"]
            state["n_transfer"] += info["n_transfer"]
            state["starved_before"] += info["starved_before"]
            state["starved_after"] += info["starved_after"]
            state["gap_before"] += info["mean_gap_before"]
            state["gap_after"] += info["mean_gap_after"]
            state["gap_max_before"] += info["max_gap_before"]
            state["gap_max_after"] += info["max_gap_after"]
            if cfg.verbose and state["batches"] % 100 == 0:
                print(
                    f"[ADEA] SLA 生效: batches={state['batches']} "
                    f"conflicts={int(state['n_contested_anchors'])} "
                    f"transfers={int(state['n_transfer'])} "
                    f"starved {int(state['starved_before'])}→{int(state['starved_after'])} "
                    f"max_gap {state['gap_max_before'] / max(1, state['batches']):.3f}"
                    f"→{state['gap_max_after'] / max(1, state['batches']):.3f}"
                )
            return mask_new, align_metric, overlaps
        except Exception as exc:  # pragma: no cover - 绝不让补丁搞崩训练
            state["errors"] += 1
            if state["first_tb"] is None:
                import traceback

                state["first_tb"] = traceback.format_exc()
                print(f"[ADEA] ERROR SLA 补丁异常（已回退原始分配），首次堆栈：\n{state['first_tb']}")
            elif cfg.verbose and state["errors"] % 500 == 0:
                print(f"[ADEA] ERROR SLA 补丁异常累计 {state['errors']} 次")
            return mask_pos, align_metric, overlaps

    TaskAlignedAssigner.get_pos_mask = patched_get_pos_mask
    TaskAlignedAssigner.select_candidates_in_gts = patched_select_cand
    TaskAlignedAssigner._adea_sla_patched = True
    TaskAlignedAssigner._adea_sla_state = state
    return {
        "sla": (
            f"patched TaskAlignedAssigner.get_pos_mask + select_candidates_in_gts "
            f"(criterion={cfg.sla_criterion}, claim_only={cfg.sla_claim_only}, "
            f"S2={cfg.sla_s2}, rho={cfg.sla_match_ratio}, tol={cfg.sla_tol}, "
            f"min_gap={cfg.sla_min_gap}, rounds={cfg.sla_rounds}, "
            f"expand={cfg.sla_expand}, q_min={cfg.sla_q_min}, margin={cfg.sla_expand_margin})"
        )
    }


# --------------------------------------------------------------------------- #
# 3) 损失安装 + NMS 替换
# --------------------------------------------------------------------------- #
def patch_nms(cfg: ADEAConfig) -> Dict[str, str]:
    """把 Ultralytics 的 ``batched_nms`` 换成 AGS 版本。

    Ultralytics 的 ``non_max_suppression`` 调用 ``batched_nms``；
    替换它即可让 I5 AGS 在 ``val`` / ``predict`` 中直接生效，
    无需改动任何训练/推理代码路径。
    """
    if not cfg.ags_enabled:
        return {"nms": "skipped (ADEA_AGS=0)"}

    try:
        import ultralytics.utils.ops as ops_mod
    except Exception as exc:  # pragma: no cover
        return {"nms": f"FAILED to import ultralytics.utils.ops: {exc}"}

    from ultralytics_ext.post.ags import AGSConfig, ags_batched_nms

    ags_cfg = AGSConfig(
        enabled=True,
        iou_thr=cfg.ags_iou_thr,
        same_instance_iou=cfg.ags_same_instance_iou,
        decay=cfg.ags_decay,
    )
    orig = getattr(ops_mod, "batched_nms", None)
    if orig is None:
        return {"nms": "batched_nms not found in ultralytics.utils.ops"}
    if getattr(ops_mod, "_adea_ags_patched", False):
        return {"nms": "already patched"}

    def patched_batched_nms(boxes, scores, idxs, iou_threshold):
        # 仅在 AGS 语义下接管；用调用方传入的 iou_threshold 作为抑制阈值
        return ags_batched_nms(boxes, scores, idxs, iou_threshold, ags_cfg)

    ops_mod.batched_nms = patched_batched_nms
    ops_mod._adea_ags_patched = True
    return {
        "nms": (
            f"patched ultralytics.utils.ops.batched_nms -> AGS "
            f"(iou_thr={cfg.ags_iou_thr}, same_instance_iou={cfg.ags_same_instance_iou}, "
            f"decay={cfg.ags_decay})"
        )
    }


def install_losses(cfg: ADEAConfig) -> Dict[str, str]:
    """安装自定义损失（当前只有 R5 的 Scale-Quality Aware EIoU）。"""
    report: Dict[str, str] = {}
    if cfg.sqa_eiou_enabled:
        from ultralytics_ext.loss.scale_eiou import ScaleQualityConfig, install_scale_quality_eiou

        ok, msg = install_scale_quality_eiou(
            ScaleQualityConfig(
                enabled=True,
                scale_weight=cfg.sqa_scale_weight,
                quality_weight=cfg.sqa_quality_weight,
            )
        )
        report["sqa_eiou"] = f"{'OK' if ok else 'FAILED'} - {msg}"
    else:
        report["sqa_eiou"] = "skipped (ADEA_SQA_EIOU=0)"

    if cfg.sfw_enabled:
        from ultralytics_ext.loss.scale_fair import ScaleFairConfig, install_scale_fair_weighting

        report["scale_fair"] = install_scale_fair_weighting(
            ScaleFairConfig(enabled=True, clip=cfg.sfw_clip, verbose=cfg.verbose)
        )
    else:
        report["scale_fair"] = "skipped (ADEA_SFW=0)"

    if cfg.pair_cls_enabled:
        report["pair_cls"] = (
            "REQUESTED but needs a baseline confusion matrix; "
            "use scripts/diagnose_adhesion.py to export it, then pass it to "
            "install_pair_aware_cls(confusion_matrix=...)"
        )
    else:
        report["pair_cls"] = "skipped (ADEA_PAIR_CLS=0)"
    return report


# --------------------------------------------------------------------------- #
# 4) 总入口
# --------------------------------------------------------------------------- #
def install_all(config_path: Optional[str] = None) -> Dict[str, str]:
    """执行全部接线，返回可打印的报告。

    Returns
    -------
    dict
        各部分的接线状态。
    """
    cfg = load_config(config_path)
    report: Dict[str, str] = {}
    report["config"] = json.dumps(
        {k: v for k, v in cfg.to_dict().items() if k != "extra"}, ensure_ascii=False
    )
    try:
        report["modules"] = json.dumps(register_modules(), ensure_ascii=False)
    except Exception as exc:
        report["modules"] = f"FAILED: {exc}"
    report.update(patch_assigner(cfg))
    report.update(patch_sla(cfg))
    report.update(patch_nms(cfg))
    report.update(install_losses(cfg))
    try:
        from ultralytics_ext.diagnostics import install_diagnostics

        report.update(install_diagnostics())
    except Exception as exc:
        report["diagnostics"] = f"FAILED: {exc}"
    if cfg.verbose:
        print("[ADEA] 接线报告:")
        for k, v in report.items():
            print(f"  - {k}: {v}")
    return report
