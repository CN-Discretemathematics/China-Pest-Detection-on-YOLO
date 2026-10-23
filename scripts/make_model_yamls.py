#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADEA 模型配置生成器 —— 从**已安装的官方** ``yolo11.yaml`` 程序化派生变体。

为什么不手写 yaml
------------------
Ultralytics 的 ``parse_model`` 会按 ``scales`` 里的 width 对通道做缩放，
手写 yaml 极易在通道数、层索引、Detect 输入列表上出错。
本脚本**读取当前安装版本的官方 yolo11.yaml**，在其基础上做结构化改写，
保证与所用 ultralytics 版本**完全一致**。

变体
----
============================  ==========================================
输出目录                        内容
============================  ==========================================
``yaml/p2/``                   R4：加 P2（stride-4）检测头
``yaml/spd/``                  R1：backbone 前 2 个 stride-2 卷积 → SPD-Conv
``yaml/ema/``                  R2：backbone 插 2 处 EMA 注意力
``yaml/dysample/``             R3：neck 上采样 → DySample
``yaml/spd_ema/``              R1+R2
``yaml/strong/``               **强基线** R1+R2+R3+R4
============================  ==========================================

每个目录下的文件名固定为 ``yolo11n.yaml``（保证 Ultralytics 能从文件名
识别 scale = ``n``），并在 yaml 内显式写入 ``scale: n`` 作为双保险。

用法::

    python scripts/make_model_yamls.py                 # 生成
    python scripts/make_model_yamls.py --validate      # 生成并逐个构建校验
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

Row = List  # [from, number, module, args]


# --------------------------------------------------------------------------- #
# 官方 yaml 定位与基础工具
# --------------------------------------------------------------------------- #
def official_yaml_path() -> Path:
    import ultralytics

    p = Path(ultralytics.__file__).parent / "cfg" / "models" / "11" / "yolo11.yaml"
    if not p.exists():
        raise SystemExit(f"未找到官方 yolo11.yaml：{p}")
    return p


def load_official() -> Dict:
    cfg = yaml.safe_load(official_yaml_path().read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["nc"] = 24  # Pest24 的 24 类
    cfg["scale"] = "n"
    return cfg


def scale_width(cfg: Dict) -> float:
    return float(cfg["scales"][cfg.get("scale", "n")][1])


def make_divisible(x: float, divisor: int = 8) -> int:
    return int((int(x) + divisor - 1) // divisor * divisor)


# --------------------------------------------------------------------------- #
# 通道追踪（用于计算 EMA / DySample 的通道参数）
# --------------------------------------------------------------------------- #
def backbone_channels(cfg: Dict) -> List[int]:
    """返回 backbone 每层**缩放后**的输出通道数。"""
    w = scale_width(cfg)
    max_ch = int(cfg["scales"][cfg.get("scale", "n")][2])
    out: List[int] = []
    c_in = 3
    for f, n, m, args in cfg["backbone"]:
        if f == -1:
            c_in = out[-1] if out else 3
        elif isinstance(f, int):
            c_in = out[f]
        c2 = make_divisible(min(int(args[0]), max_ch) * w, 8)
        out.append(c2)
    return out


def _row_module(row: Row) -> str:
    return str(row[2])


# --------------------------------------------------------------------------- #
# 变换：SPD（R1）
# --------------------------------------------------------------------------- #
def apply_spd(cfg: Dict, n_replace: int = 2) -> Dict:
    """把 backbone 前 n_replace 个 stride-2 卷积换成 SPDConv（1:1 替换，不改索引）。

    SPDConv(c1, c2, k, s) 与 Conv(c1, c2, k, s) 的**输出通道与空间分辨率完全一致**，
    因此后续层的输入通道无需改动 → 预训练权重迁移不受影响。
    """
    cfg = copy.deepcopy(cfg)
    replaced = 0
    for i, row in enumerate(cfg["backbone"]):
        if replaced >= n_replace:
            break
        f, n, m, args = row
        if m == "Conv" and len(args) >= 2 and int(args[1]) == 2:  # stride-2 卷积
            cfg["backbone"][i] = [f, n, "SPDConv", [args[0], args[1], args[2]]]
            replaced += 1
    if replaced == 0:
        raise SystemExit("未找到可替换的 stride-2 卷积（官方 backbone 结构可能已变）")
    return cfg


# --------------------------------------------------------------------------- #
# 变换：EMA（R2）
# --------------------------------------------------------------------------- #
def apply_ema(cfg: Dict, after_backbone_indices: Sequence[int] = (1, 3)) -> Dict:
    """在 backbone 指定层之后插入 EMA 注意力。

    **通道守恒技巧**：yaml 里写 ``EMA, [c1_unscaled // 2]``，
    经 parse_model 的 width 缩放后，EMA 内部再翻倍 → 输出恰为 ``c1``，
    因此**下游各层的输入通道完全不变**（索引会 +1，但通道不变）。
    """
    cfg = copy.deepcopy(cfg)
    chans = backbone_channels(cfg)
    new_backbone: List[Row] = []
    for i, row in enumerate(cfg["backbone"]):
        new_backbone.append(list(row))
        if i in after_backbone_indices and i < len(chans):
            c1_unscaled_scaled = chans[i]  # 缩放后的输出通道
            c1_unscaled = int(row[3][0])   # yaml 里的未缩放通道
            # 需要 c1_scaled = 2 * c2_scaled → c2_unscaled = c1_unscaled / 2
            c2_arg = max(1, c1_unscaled // 2)
            new_backbone.append([-1, 1, "EMA", [c2_arg]])
    cfg["backbone"] = new_backbone
    return cfg


# --------------------------------------------------------------------------- #
# 变换：DySample（R3）
# --------------------------------------------------------------------------- #
def apply_dysample(cfg: Dict) -> Dict:
    """把 head 里的 ``nn.Upsample`` 换成 ``DySample``（输出通道 = 输入通道）。"""
    cfg = copy.deepcopy(cfg)
    for row in cfg["head"]:
        if _row_module(row) == "nn.Upsample":
            # DySample(in_channels=c1, out_channels=c1, scale=2)
            # yaml 参数写 [c2_arg, scale]；parse_model 会按通用分支传入 c1 与缩放后的 c2
            row[2] = "DySample"
            row[3] = [0, 2]  # 0 → 由生成器在下方按通道回填
    # 回填通道参数
    chans = backbone_channels(cfg)
    sfp_c = chans[-1]  # SPPF 输出通道（缩放后）
    width = scale_width(cfg)
    max_ch = int(cfg["scales"][cfg.get("scale", "n")][2])
    # 未缩放的 SPPF 通道
    sfp_c_unscaled = int(cfg["backbone"][-1][3][0])
    for row in cfg["head"]:
        if _row_module(row) == "DySample":
            # args[0] = 目标 c2（未缩放）→ 与 c1 未缩放相同 → 缩放后输出通道 = 输入通道
            row[3] = [sfp_c_unscaled, 2]
    return cfg


# --------------------------------------------------------------------------- #
# 变换：P2 检测头（R4）
# --------------------------------------------------------------------------- #
def apply_p2(cfg: Dict) -> Dict:
    """在 neck 中增加 P2（stride-4）分支。

    做法：
    1. backbone 的 stride-4 输出（layer 1）作为 P2 特征；
    2. head 开头增加：P3/4 级特征上采样到 P2 → 与 backbone P2 拼接 → C3k2；
    3. Detect 的输入列表加入该 P2 分支。
    """
    cfg = copy.deepcopy(cfg)

    backbone = cfg["backbone"]
    head = cfg["head"]

    # --- 找到 backbone 中 stride-4 的层（index 1）与 SPPF（index -1）---
    p2_src = 1
    sppf_src = len(backbone) - 1

    # --- 在 head 最前面插入 P2 分支 ---
    # head 索引（插入后）：backbone 索引不变，head 从 len(backbone) 开始
    b = len(backbone)

    # 1) Upsample：把 SPPF 特征上采样 2 次 → stride 4
    #    为了得到 P2，我们要把 P3(8)→P2(4)，需要两次上采样：
    #    SPPF(32) →P4(16) →P3(8) →P2(4)
    #    这里直接复用 head 中已有的 P3 层输出，向上采样一次。
    #    找到 head 中第一个 C3k2（输出 256 通道，即 P3 分支）
    p3_out = None
    for i, row in enumerate(head):
        if _row_module(row) == "C3k2":
            p3_out = b + i
            break
    if p3_out is None:
        raise SystemExit("head 结构异常：未找到 P3 的 C3k2")

    # 2) 新增层：
    #    up   : Upsample(P3)            -> stride 4
    #    cat  : Concat(up, backbone[1])  -> 与 backbone 的 stride-4 特征拼接
    #    c3k2 : C3k2(cat)               -> P2 输出（通道取 P3 的一半 256→128）
    new_head: List[Row] = []
    up_i = b
    cat_i = b + 1
    p2_i = b + 2

    new_head.append([p3_out, 1, "nn.Upsample", [None, 2, "nearest"]])
    new_head.append([[up_i, p2_src], 1, "Concat", [1]])
    # P2 输出通道：与 P3 同宽的一半（128），保证 nano 规模下开销可控
    p3_c_unscaled = int(head[[i for i, r in enumerate(head) if r is not None][0]][3][0]) if False else None
    # 直接取 head 中第一个 C3k2 的通道作为 P3 通道
    for row in head:
        if _row_module(row) == "C3k2":
            p3_c_unscaled = int(row[3][0])
            break
    new_head.append([cat_i, 2, "C3k2", [p3_c_unscaled // 2, False]])

    # --- 原有 head 的索引需要整体后移 len(new_head) ---
    shift = len(new_head)
    remapped: List[Row] = []
    for row in head:
        r = list(row)
        f = r[0]
        if isinstance(f, int):
            r[0] = f + shift if f >= 0 else f
        elif isinstance(f, list):
            r[0] = [(x + shift if x >= 0 else x) for x in f]
        remapped.append(r)

    cfg["head"] = new_head + remapped

    # --- Detect 输入列表加入 P2 分支 ---
    for row in cfg["head"]:
        if _row_module(row) == "Detect":
            det = list(row[3])
            srcs = list(row[0]) if isinstance(row[0], list) else [row[0]]
            srcs = [p2_i] + srcs
            row[0] = srcs
            break
    else:
        raise SystemExit("head 结构异常：未找到 Detect")

    return cfg


# --------------------------------------------------------------------------- #
# 组合与输出
# --------------------------------------------------------------------------- #
VARIANTS: Dict[str, Dict[str, bool]] = {
    "p2": {"p2": True},
    "spd": {"spd": True},
    "ema": {"ema": True},
    "dysample": {"dysample": True},
    "spd_ema": {"spd": True, "ema": True},
    "strong": {"spd": True, "ema": True, "dysample": True, "p2": True},
}

DESCRIPTIONS = {
    "p2": "R4 —— P2 (stride-4) 检测头（arXiv 2512.16493）",
    "spd": "R1 —— SPD-Conv 替换 backbone 前 2 个 stride-2 卷积（EP-YOLO）",
    "ema": "R2 —— backbone 插入 2 处 EMA 注意力（EP-YOLO）",
    "dysample": "R3 —— neck 上采样替换为 DySample（CCDW-YOLO）",
    "spd_ema": "R1+R2 —— SPD-Conv + EMA（EP-YOLO 已证二者强耦合）",
    "strong": "★强基线 —— R1+R2+R3+R4 组合（本文创新增量的参照系）",
}


def build(base: Dict, flags: Dict[str, bool]) -> Dict:
    cfg = copy.deepcopy(base)
    if flags.get("spd"):
        cfg = apply_spd(cfg)
    if flags.get("ema"):
        cfg = apply_ema(cfg)
    if flags.get("dysample"):
        cfg = apply_dysample(cfg)
    if flags.get("p2"):
        cfg = apply_p2(cfg)
    return cfg


def dump_yaml(cfg: Dict, header: str) -> str:
    body = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"# {header}\n# 本文件由 scripts/make_model_yamls.py 从官方 yolo11.yaml 自动生成，请勿手改。\n{body}"


def validate(paths: Sequence[Path]) -> bool:
    """逐个构建模型，校验结构与参数量。"""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import intersect_dicts

    try:
        base_sd = YOLO("yolo11n.pt").model.state_dict()
    except Exception as exc:
        print(f"  [WARN] 无法加载 yolo11n.pt（{exc}），跳过权重迁移校验")
        base_sd = None

    ok = True
    for p in paths:
        name = p.parent.name
        try:
            m = YOLO(str(p))
            if base_sd is not None:
                m.load("yolo11n.pt")
                tgt = m.model.state_dict()
                matched = intersect_dicts(base_sd, tgt)
                n_matched = sum(v.numel() for v in matched.values())
                total = sum(v.numel() for v in base_sd.values())
                ratio = 100.0 * n_matched / max(1, total)
                n_params = sum(x.numel() for x in m.model.parameters())
                print(
                    f"  ✅ {name:10s} params={n_params/1e6:6.3f}M  "
                    f"迁移={n_matched:4d} 张量 ({ratio:5.1f}%)"
                )
                if n_matched == 0:
                    print(f"     ❌ 权重迁移为 0 —— 等于从零训练，必须修")
                    ok = False
            else:
                n_params = sum(x.numel() for x in m.model.parameters())
                print(f"  ✅ {name:10s} params={n_params/1e6:6.3f}M")
        except Exception as exc:
            print(f"  ❌ {name:10s} 构建失败: {type(exc).__name__}: {exc}")
            ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 ADEA 模型配置变体")
    ap.add_argument("--out", default="ultralytics_ext/yaml")
    ap.add_argument("--validate", action="store_true", help="生成后逐个构建校验")
    ap.add_argument("--only", default=None, help="只生成指定变体（逗号分隔）")
    args = ap.parse_args()

    base = load_official()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    wanted = list(VARIANTS) if not args.only else args.only.split(",")
    written: List[Path] = []
    for name in wanted:
        if name not in VARIANTS:
            print(f"  未知变体：{name}")
            continue
        cfg = build(base, VARIANTS[name])
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        p = d / "yolo11n.yaml"
        p.write_text(dump_yaml(cfg, DESCRIPTIONS[name]), encoding="utf-8")
        written.append(p)
        print(f"  生成 {p}  ({DESCRIPTIONS[name]})")

    if args.validate and written:
        print("\n=== 校验 ===")
        if not validate(written):
            print("❌ 部分变体校验失败，请检查上面的报错")
            sys.exit(1)
        print("✅ 全部变体构建通过")


if __name__ == "__main__":
    main()
