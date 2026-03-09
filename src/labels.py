from __future__ import annotations

import pandas as pd


def make_labels(daily: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """构造监督学习目标。

默认：future_return = close(t+h)/close(t)-1
"""
    raise NotImplementedError
