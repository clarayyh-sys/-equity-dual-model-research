from __future__ import annotations

import pandas as pd


def backtest_topk(
    pred: pd.DataFrame,
    daily: pd.DataFrame,
    topk: int = 30,
    rebalance: str = "W",
    fee_bps: float = 5,
    slippage_bps: float = 5,
) -> pd.DataFrame:
    """TopK 选股回测（占位）。

输入约定：
- pred: [date, ticker, score]
- daily: [date, ticker, close]（至少）

输出：净值曲线、换手、持仓等（后续完善）。
"""
    raise NotImplementedError
