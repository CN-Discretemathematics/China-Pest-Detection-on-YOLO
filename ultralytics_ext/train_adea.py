#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ADEA 训练入口包装器。

为什么需要它
------------
ADEA 通过**环境变量**配置（Ultralytics 的 CLI 会对未知键报错），
但接线（``parse_model`` 补丁、``TaskAlignedAssigner`` 补丁、``batched_nms`` 替换）
必须发生在**训练所在的同一个进程**里。

因此本模块在调用 Ultralytics 入口点之前先执行
:func:`ultralytics_ext.register.install_all`，再把 ``sys.argv`` 原样交给
``ultralytics.cfg.entrypoint`` —— 所有 CLI 参数与原生 ``yolo`` 命令完全一致。

用法::

    ADEA_ALA=1 python -m ultralytics_ext.train_adea detect train \
        model=ultralytics_ext/yaml/yolo11n-strong.yaml \
        data=datasets/pest24/pest24.yaml epochs=100 imgsz=640 batch=48 seed=0

也可直接用 Python API::

    from ultralytics_ext.train_adea import run
    run(model="ultralytics_ext/yaml/yolo11n-strong.yaml",
        data="datasets/pest24/pest24.yaml", epochs=100, imgsz=640, batch=48, seed=0)
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

from ultralytics_ext.register import install_all

__all__ = ["run", "main"]


def run(**kwargs: Any):
    """安装 ADEA 接线后训练一个模型。

    Parameters
    ----------
    **kwargs:
        透传给 ``ultralytics.YOLO.train`` 的参数（model/data/epochs/imgsz/...）。

    Returns
    -------
    ultralytics.engine.results.Results
    """
    from ultralytics import YOLO

    install_all()
    model = YOLO(kwargs.pop("model"))
    return model.train(**kwargs)


def main() -> None:
    """CLI 入口：接线 → 交给 Ultralytics 的 entrypoint。"""
    report = install_all()
    print("[ADEA] 接线完成，进入训练：", " | ".join(f"{k}={v}" for k, v in report.items()))

    from ultralytics.cfg import entrypoint

    entrypoint()


if __name__ == "__main__":
    main()
