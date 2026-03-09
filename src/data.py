"""数据接口占位。

后续会按 config.data_source.name 接入：
- akshare：免费抓取（覆盖面/稳定性一般）
- tushare：更稳，但需要 token
- local_csv：你已有本地K线/因子文件
"""

from __future__ import annotations

import pandas as pd


def load_daily_ohlcv(*args, **kwargs) -> pd.DataFrame:
    """返回 long-format 数据：
    columns: [date, ticker, open, high, low, close, volume, amount]
    """
    raise NotImplementedError
