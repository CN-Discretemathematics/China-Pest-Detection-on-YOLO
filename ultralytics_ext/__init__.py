"""ultralytics_ext —— ADEA 扩展包。

本包把两类东西接入 Ultralytics YOLO11 训练栈：

**复现层（R1–R5）** —— 来自已发表论文的模块移植
    * :mod:`ultralytics_ext.modules.spd_conv`   SPD-Conv（EP-YOLO）
    * :mod:`ultralytics_ext.modules.ema_attn`   EMA 注意力（EP-YOLO）
    * :mod:`ultralytics_ext.modules.dysample`   DySample（CCDW-YOLO）
    * :mod:`ultralytics_ext.loss.scale_eiou`    Scale-Quality Aware EIoU（AEP-PEST）

**创新层（I1–I4）** —— 本项目提出，回应已发表论文点名的 future work
    * :mod:`ultralytics_ext.assign.ala`         I1 粘连感知标签分配
    * :mod:`ultralytics_ext.assign.dla`         I2 密度先验引导分配
    * :mod:`ultralytics_ext.sampling.dra`       I3 难度感知动态重采样
    * :mod:`ultralytics_ext.post.ags`           I5 粘连感知分级抑制（工程组件）

接线入口：:func:`ultralytics_ext.register.install_all`
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
