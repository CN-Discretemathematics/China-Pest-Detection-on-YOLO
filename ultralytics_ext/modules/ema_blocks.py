"""ADEA · R2（复现）· 带 EMA 的 C3k2 —— **子类包装**，保住预训练权重。

为什么用子类而不是"插一层 EMA"
-------------------------------
Ultralytics 的预训练权重迁移是**按参数名精确匹配**的
（``intersect_dicts(base, target)``）。若在 backbone 中间**插入**一个新层，
其后所有层的参数名都会整体后移（如 ``model.2.cv1.conv.weight`` →
``model.3.cv1.conv.weight``）→ 匹配全部失败，迁移率从 ~99% 掉到 **2.6%**，
等价于"从零训练"，对 nano 模型是灾难性的。

本模块用**子类**解决：``C3k2EMA`` 继承自 ``C3k2``，
原有的 ``cv1 / cv2 / m`` 等子模块**名字完全不变** → 迁移率保持；
只额外新增 ``ema.*`` 参数（无预训练对应项，不影响迁移）。

通道守恒
--------
附加的 EMA 被施加在 C3k2 的**输出**上，输出通道 = C3k2 的输出通道 c2
（``EMA(c2, c2)``）→ 该层的输出通道与原 C3k2 完全一致，
下游各层无需任何改动，``parse_model`` 的通道记账也不受影响。

与官方 EMA 的差异
-----------------
官方 ``ultralytics.nn.modules.attention.EMA`` 内部把 c2 翻倍
（输出 = 2×输入），是"通道扩张型"模块，无法 1:1 插入 backbone
（会与 ``parse_model`` 的通道记账冲突）。
本项目的 EMA 采用**通道守恒变体**，注意力计算结构与官方一致，
仅"末端卷积不翻倍"这一处不同 —— 详见 ``ultralytics_ext/modules/ema_attn.py``。
"""

from __future__ import annotations

from ultralytics.nn.modules.block import C3k2

from ultralytics_ext.modules.ema_attn import EMA

__all__ = ["C3k2EMA"]


class C3k2EMA(C3k2):
    """C3k2 + 通道守恒 EMA 注意力（作为**同层替换**使用，参数名不变）。

    yaml 用法（与 ``C3k2`` 完全同参）::

        - [-1, 2, C3k2EMA, [256, False, 0.25]]
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        **kwargs,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut, **kwargs)
        # 施加在 C3k2 的输出上 → 输入/输出通道均为 c2（通道守恒）
        self.ema = EMA(c2, c2)

    def forward(self, x):
        return self.ema(super().forward(x))
