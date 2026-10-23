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
``ADEA_ALA``                   I1 开关（0/1），默认 0
``ADEA_ALA_BETA_THR``          粘连阈值 β，默认 0.05（论文口径）
``ADEA_ALA_Q_MIN``             每个 GT 的最小正样本数，默认 3
``ADEA_DLA``                   I2 开关（0/1），默认 0
``ADEA_DLA_MU``                密度调制强度 μ，默认 0.5
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

    # I1 ALA
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

    Ultralytics 的 ``parse_model`` 按字符串名查表，因此只需把自定义类
    注入到 ``ultralytics.nn.tasks`` 的模块命名空间里即可。

    Returns
    -------
    dict
        ``{模块名: 来源}``，用于接线报告。
    """
    import ultralytics.nn.tasks as tasks

    from ultralytics_ext.modules.spd_conv import SPDConv
    from ultralytics_ext.modules.dysample import DySample
    from ultralytics_ext.modules.ema_attn import get_ema

    registered: Dict[str, str] = {}

    # SPDConv
    if not hasattr(tasks, "SPDConv"):
        setattr(tasks, "SPDConv", SPDConv)
        registered["SPDConv"] = "ultralytics_ext.modules.spd_conv"

    # DySample
    if not hasattr(tasks, "DySample"):
        setattr(tasks, "DySample", DySample)
        registered["DySample"] = "ultralytics_ext.modules.dysample"

    # EMA：优先官方实现
    ema_cls, ema_src = get_ema()
    if not hasattr(tasks, "EMA"):
        setattr(tasks, "EMA", ema_cls)
        registered["EMA"] = ema_src
    else:
        registered["EMA"] = "ultralytics.official (already present)"

    return registered


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
    """给 ``TaskAlignedAssigner.assign`` 打上 ALA / DLA 补丁。

    补丁位置在原始分配**之后**：先让 TAL 产出标准的
    ``(gt_idx, fg_mask, target_gt_idx)``，再按粘连度把被"饿死"实例的
    anchor 从粘连簇内的支配者手里回收给它。

    保持"每个 anchor 至多归属一个 GT"这一不变式 → **无需修改 loss**，
    可与未打补丁的 ``vflLoss`` 直接配合（§5.2 降级路径）。
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

    from ultralytics_ext.assign.ala import ALAConfig, apply_ala
    from ultralytics_ext.assign.dla import DLAConfig, per_gt_quota

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

    orig_assign = TaskAlignedAssigner.assign

    def patched_assign(self, *args, **kwargs):
        result = orig_assign(self, *args, **kwargs)
        try:
            gt_idx, fg_mask, target_gt_idx = result[0], result[1], result[2]
            # 从调用参数里取 gt_bboxes 与 anchor_points
            gt_bboxes = None
            anchor_points = None
            for a in args:
                if hasattr(a, "shape") and len(a.shape) == 2 and a.shape[-1] == 4:
                    if gt_bboxes is None:
                        gt_bboxes = a
                elif hasattr(a, "shape") and len(a.shape) == 2 and a.shape[-1] == 2:
                    if anchor_points is None:
                        anchor_points = a
            # 位置参数兜底：TaskAlignedAssigner.assign 的签名是
            # (pred_bboxes, pred_scores, gt_bboxes, gt_scores, anchor_points, mask_gt)
            if gt_bboxes is None and len(args) >= 3:
                gt_bboxes = args[2]
            if anchor_points is None and len(args) >= 5:
                anchor_points = args[4]

            if gt_bboxes is None or len(gt_bboxes) == 0:
                return result

            import torch

            if anchor_points is not None:
                stride = _infer_stride_from_points(anchor_points)
                n = int(anchor_points.shape[0])
                anchors_xywh = torch.zeros(
                    (n, 4), dtype=gt_bboxes.dtype, device=gt_bboxes.device
                )
                anchors_xywh[:, 0] = anchor_points[:, 0]
                anchors_xywh[:, 1] = anchor_points[:, 1]
                anchors_xywh[:, 2] = stride
                anchors_xywh[:, 3] = stride
            else:
                return result

            per_gt_qmin = None
            if cfg.dla_enabled:
                # 密度配额调制：拥挤处的 GT 获得更高配额
                per_gt_qmin = per_gt_quota(gt_bboxes, cfg.ala_q_min, dla_cfg)

            new_tgt, stats = apply_ala(
                target_gt_idx, anchors_xywh, gt_bboxes, ala_cfg, per_gt_qmin=per_gt_qmin
            )
            if stats.n_reassigned > 0 and cfg.verbose:
                print(
                    f"[ADEA] ALA/DLA 生效: clusters={stats.n_clusters} "
                    f"starved={stats.n_starved_gt} reassigned={stats.n_reassigned}"
                )
            out = list(result)
            out[2] = new_tgt
            return tuple(out)
        except Exception as exc:  # pragma: no cover - 绝不让补丁搞崩训练
            if cfg.verbose:
                print(f"[ADEA] WARN 标签分配补丁异常，已回退原始分配: {exc}")
            return result

    TaskAlignedAssigner.assign = patched_assign
    TaskAlignedAssigner._adea_patched = True
    report["assigner"] = (
        f"patched TaskAlignedAssigner.assign (ALA={cfg.ala_enabled}, DLA={cfg.dla_enabled}, "
        f"beta_thr={cfg.ala_beta_thr}, q_min={cfg.ala_q_min})"
    )
    return report


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
