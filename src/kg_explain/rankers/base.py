"""
排序器基础工具

包含共享的评分函数和工具
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hub_penalty(degrees: pd.Series) -> pd.Series:
    """
    Hub惩罚: 降低高连接度节点的权重

    原理: 高连接度的靶点(如常见蛋白)可能不具特异性
    公式: 1 / log(1 + degree)

    Args:
        degrees: 节点的连接度序列

    Returns:
        惩罚系数序列 (0-1之间)
    """
    return 1.0 / np.log1p(degrees.astype(float).clip(lower=1.0))
