#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADEA 模型配置生成器 —— 从**已安装的官方** ``yolo11.yaml`` 程序化派生变体。

为什么不手写 yaml
------------------
Ultralytics 的 ``parse_model`` 会按 ``scales`` 里的 width 对通道做缩放，
并且自定义模块必须满足严格的签名契约（见 ``ultralytics_ext/register.py``）。
手写 yaml 极易在通道数、层索引、Detect 输入列表上出错。
本脚本**读取当前安装版本的官方 yolo11.yaml**，在其基础上做结构化改写，
保证与所用 ultralytics 版本**完全一致**。

索引安全
--------
内部用 **uid 引用**（每个层一个唯一 id，``f`` 引用 uid），
最后一步才把 uid 解析成最终下标 —— 因此插入层时**不需要**手工修正任何
引用，backbone 引用与 head 引用不会被误改。

变体
----
============================  ==========================================
输出目录                        内容
============================  ==========================================
``yaml/p2/``                   R4：加 P2（stride-4）检测头
``yaml/spd/``                  R1：backbone 前 2 个 stride-2 卷积 → SPD-Conv
``yaml/ema/``                  R2：backbone 插 2 处 EMA 注意力（通道守恒）
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

# 让 `python scripts/make_model_yamls.py` 能 import 到仓库根的 ultralytics_ext
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------- #
# 层表示（uid 引用）
# --------------------------------------------------------------------------- #
class Layer:
    """一层。``f`` 为输入层 uid 列表（``-1`` 表示"上一层"）。"""

    __slots__ = ("uid", "f", "n", "m", "args", "region")

    def __init__(self, uid: int, f, n: int, m: str, args: list, region: str):
        self.uid = uid
        self.f = f
        self.n = n
        self.m = m
        self.args = list(args)
        self.region = region

    def copy(self) -> "Layer":
        return Layer(self.uid, list(self.f), self.n, self.m, list(self.args), self.region)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Layer(uid={self.uid}, f={self.f}, n={self.n}, m={self.m}, args={self.args})"


class Builder:
    """层列表 + uid 分配 + 插入。"""

    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self._next_uid = 0

    def new_uid(self) -> int:
        u = self._next_uid
        self._next_uid += 1
        return u

    def add(self, f, n: int, m: str, args: list, region: str) -> int:
        uid = self.new_uid()
        self.layers.append(Layer(uid, f, n, m, args, region))
        return uid

    def insert(self, pos: int, f, n: int, m: str, args: list, region: str) -> int:
        uid = self.new_uid()
        self.layers.insert(pos, Layer(uid, f, n, m, args, region))
        return uid

    def index_of_uid(self, uid: int) -> int:
        for i, l in enumerate(self.layers):
            if l.uid == uid:
                return i
        raise KeyError(f"uid {uid} not found")

    def head_start(self) -> int:
        for i, l in enumerate(self.layers):
            if l.region == "head":
                return i
        raise KeyError("no head region")


# --------------------------------------------------------------------------- #
# 通道追踪（未缩放空间，线性缩放下守恒性质同样成立）
# --------------------------------------------------------------------------- #
_SCALED_OUT = {"Conv", "C3k2", "SPPF", "C2PSA", "SPDConv", "EMA", "C2f", "C3"}
_PRESERVE = {"nn.Upsample", "DySample"}


def out_channels(b: Builder) -> List[int]:
    """每层的输出通道数（**未缩放**）。"""
    ch: List[int] = []
    for i, l in enumerate(b.layers):
        if l.m == "Concat":
            c = 0
            for ref in l.f:
                c += ch[b.index_of_uid(ref)] if ref != -1 else (ch[-1] if ch else 0)
            ch.append(c)
        elif l.m in _PRESERVE:
            ch.append(ch[-1] if ch else 3)
        elif l.m in _SCALED_OUT:
            ch.append(int(l.args[0]))
        else:
            ch.append(ch[-1] if ch else 3)
    return ch


def in_channels(b: Builder, i: int) -> int:
    """第 i 层的输入通道数（未缩放）。"""
    l = b.layers[i]
    ch = out_channels(b)
    if l.f == -1 or l.f == [-1]:
        return ch[i - 1] if i > 0 else 3
    if isinstance(l.f, int):
        return ch[b.index_of_uid(l.f)]
    if isinstance(l.f, list) and l.f:
        return sum(ch[b.index_of_uid(r)] for r in l.f if r != -1)
    return 3


# --------------------------------------------------------------------------- #
# 官方 yaml 载入
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


def to_builder(cfg: Dict) -> Builder:
    b = Builder()
    for region in ("backbone", "head"):
        for row in cfg[region]:
            f, n, m, args = row[0], row[1], row[2], list(row[3])
            if isinstance(f, int):
                f = [-1] if f == -1 else [f]
            b.add(f, n, m, args, region)
    return b


def to_yaml(b: Builder, cfg: Dict) -> Dict:
    """把 uid 引用解析成最终下标，输出 yaml 字典。"""
    pos = {l.uid: i for i, l in enumerate(b.layers)}
    out = copy.deepcopy(cfg)
    for region in ("backbone", "head"):
        rows = []
        for l in b.layers:
            if l.region != region:
                continue
            if l.f == [-1]:
                f = -1
            else:
                # -1 在列表里表示"上一层"（Ultralytics 的 Concat 写法，如 [[-1, 6], ...]）
                f = [-1 if r == -1 else pos[r] for r in l.f]
                f = f[0] if len(f) == 1 else f
            rows.append([f, l.n, l.m, list(l.args)])
        out[region] = rows
    return out


# --------------------------------------------------------------------------- #
# 变换
# --------------------------------------------------------------------------- #
def apply_spd(b: Builder, n_replace: int = 2) -> None:
    """把 backbone 前 n_replace 个 stride-2 卷积换成 SPD-Conv（1:1 替换，不改层数）。

    SPDConv(c1, c2, k, s) 与 Conv(c1, c2, k, s) 的**输出通道与空间分辨率完全一致**
    （SpaceToDepth 后接 k×s 卷积，stride=1）→ 后续层的输入通道无需改动。
    """
    replaced = 0
    for l in b.layers:
        if l.region != "backbone" or replaced >= n_replace:
            continue
        if l.m == "Conv" and len(l.args) >= 3 and int(l.args[2]) == 2:
            l.m = "SPDConv"
            replaced += 1
    if replaced == 0:
        raise SystemExit("未找到可替换的 stride-2 卷积（官方 backbone 结构可能已变）")


def apply_ema(b: Builder, after_backbone_idx: Sequence[int] = (2, 4)) -> None:
    """在 backbone 的指定 C3k2 层上**同层替换**为带 EMA 的 ``C3k2EMA``。

    ⚠️ 为什么是"同层替换"而不是"插一层"
    ------------------------------------
    Ultralytics 的预训练权重迁移按**参数名精确匹配**。
    插入新层会让其后所有层的参数名整体后移 → 迁移率从 ~99% 掉到 **2.6%**
    （实测），等价于"从零训练"，对 nano 模型是灾难性的。

    ``C3k2EMA`` 继承自 ``C3k2``，原有 ``cv1/cv2/m`` 的参数名**完全不变**，
    只新增 ``ema.*`` → 迁移率保持，且层数/索引/通道记账全部不变。

    默认替换 backbone 的第 1、第 2 个 C3k2（输出 P2/4 与 P3/8 特征）——
    这两处是分辨率最高、微目标信息最关键的层，也符合 EP-YOLO
    "EMA 放 backbone 优于放 neck" 的实证。
    """
    backbone_layers = [l for l in b.layers if l.region == "backbone"]
    c3k2_layers = [l for l in backbone_layers if l.m == "C3k2"]
    if not c3k2_layers:
        raise SystemExit("backbone 中找不到 C3k2 层")
    for k in after_backbone_idx:
        if k >= len(c3k2_layers):
            continue
        c3k2_layers[k].m = "C3k2EMA"  # 同层替换：层数/索引/参数名均不变


def apply_dysample(b: Builder) -> None:
    """把 head 里的 ``nn.Upsample`` 换成 ``DySample``（通道守恒）。"""
    for l in b.layers:
        if l.m == "nn.Upsample":
            l.m = "DySample"
            l.args = [0, 2]  # 通道数稍后回填
    for i, l in enumerate(b.layers):
        if l.m == "DySample":
            l.args = [in_channels(b, i), 2]


def apply_p2(b: Builder) -> None:
    """在 neck 中增加 P2（stride-4）分支。

    ⚠️ 插入位置的两重约束（实现过程中踩过的坑）
    --------------------------------------------
    1. **不能插在 neck 开头**：P2 分支要上采样 P3 特征，而 P3 在 neck 的
       后半段才算出 → 插在前面会形成**前向引用**（from 指向尚未定义的层），
       Ultralytics 的 ``parse_model`` 直接 IndexError。
    2. **应该插在 Detect 之前（neck 末尾）**：这样 neck 的上/下行路径
       **完全不动**（不需要把下行 Conv 的输入从 -1 改成显式引用），
       且只有 Detect 一层的参数名后移 → **预训练权重迁移率保持最高**。
       （若插在 P3 之后，下行路径的层名会整体后移，迁移率会掉到 ~57%。）

    插入后：
        ... P3_C3k2 → P4_C3k2 → P5_C3k2
        → [P2_Upsample(P3) → P2_Concat(+backbone P2/4) → P2_C3k2]
        → Detect(P2, P3, P4, P5)
    """
    # --- backbone 中 stride-4 的源层（第 2 个 stride-2 卷积）---
    p2_src = None
    stride2_seen = 0
    for l in b.layers:
        if l.region != "backbone":
            continue
        if l.m in ("Conv", "SPDConv") and len(l.args) >= 3 and int(l.args[2]) == 2:
            stride2_seen += 1
            if stride2_seen == 2:
                p2_src = l.uid
                break
    if p2_src is None:
        raise SystemExit("backbone 中找不到 stride-4 源层")

    # --- head 中输出 P3（stride-8）的 C3k2 ---
    p3_uid = None
    for l in b.layers:
        if l.region == "head" and l.m in ("C3k2", "C3k2EMA") and int(l.args[0]) == 256:
            p3_uid = l.uid
    if p3_uid is None:
        raise SystemExit("head 中找不到 P3 输出层（C3k2 256）")

    detect = next((l for l in b.layers if l.m == "Detect"), None)
    if detect is None:
        raise SystemExit("head 中找不到 Detect")

    # --- 在 Detect 之前插入 3 层（neck 末尾）---
    pos = b.index_of_uid(detect.uid)
    up_uid = b.insert(pos, [p3_uid], 1, "nn.Upsample", [None, 2, "nearest"], "head")
    cat_uid = b.insert(pos + 1, [up_uid, p2_src], 1, "Concat", [1], "head")
    p2_c3k2_uid = b.insert(pos + 2, [cat_uid], 2, "C3k2", [256, False], "head")

    # --- Detect 输入列表：P2 + 原三尺度（由细到粗）---
    detect.f = [p2_c3k2_uid] + list(detect.f)


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
    "ema": "R2 —— backbone 插入 2 处 EMA 注意力（EP-YOLO，通道守恒变体）",
    "dysample": "R3 —— neck 上采样替换为 DySample（CCDW-YOLO）",
    "spd_ema": "R1+R2 —— SPD-Conv + EMA（EP-YOLO 已证二者强耦合）",
    "strong": "★强基线 —— R1+R2+R3+R4 组合（本文创新增量的参照系）",
}


def build(base: Dict, flags: Dict[str, bool]) -> Dict:
    b = to_builder(base)
    if flags.get("spd"):
        apply_spd(b)
    if flags.get("ema"):
        apply_ema(b)
    if flags.get("dysample"):
        apply_dysample(b)
    if flags.get("p2"):
        apply_p2(b)
    return to_yaml(b, base)


def dump_yaml(cfg: Dict, header: str) -> str:
    body = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return (
        f"# {header}\n"
        f"# 本文件由 scripts/make_model_yamls.py 从官方 yolo11.yaml 自动生成，请勿手改。\n"
        f"{body}"
    )


def validate(paths: Sequence[Path]) -> bool:
    """逐个构建模型，校验结构、参数量与权重迁移率。"""
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import intersect_dicts

    import ultralytics_ext.register as reg

    reg.install_all()  # 注册自定义模块 + patch base_modules

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
            n_params = sum(x.numel() for x in m.model.parameters())
            line = f"  {name:10s} params={n_params/1e6:6.3f}M"
            if base_sd is not None:
                m.load("yolo11n.pt")
                tgt = m.model.state_dict()
                matched = intersect_dicts(base_sd, tgt)
                n_matched = sum(v.numel() for v in matched.values())
                total = sum(v.numel() for v in base_sd.values())
                ratio = 100.0 * n_matched / max(1, total)
                line += f"  迁移={len(matched):4d} 张量 ({ratio:5.1f}%)"
                if n_matched == 0:
                    line += "  [FAIL] 迁移为 0"
                    ok = False
            # 检测尺度
            det = None
            for mod in m.model.modules():
                if type(mod).__name__ == "Detect":
                    det = mod
                    break
            if det is not None:
                line += f"  尺度={getattr(det, 'nl', '?')}"

            # ---- 前向传播校验（eval/batch=1 与 train/batch=4）----
            import torch

            m.model.eval()
            with torch.no_grad():
                m.model(torch.randn(1, 3, 640, 640))
            m.model.train()
            m.model(torch.randn(4, 3, 640, 640))
            line += "  前向=OK"
            print(line)
        except Exception as exc:
            print(f"  [FAIL] {name:10s} 构建/前向失败: {type(exc).__name__}: {exc}")
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
            print("[FAIL] 部分变体校验失败")
            sys.exit(1)
        print("[OK] 全部变体构建通过")


if __name__ == "__main__":
    main()
