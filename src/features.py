from __future__ import annotations

import pandas as pd


def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    """输入 long-format 日线，输出带特征的 long-format。

约定：
- 必须包含 [date, ticker]
- 特征列以 f_ 开头
"""
    raise NotImplementedError
