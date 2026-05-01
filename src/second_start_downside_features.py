"""Derive the 22 downside-detection features for 一号 soft_rerank.

Strict no-look-ahead: features come only from
  - T 14:40 snap (already in second_start_dataset)
  - T-1 daily / B-factor / daily_basic
  - T-2 daily / C-factor (mfi/obv/atr from lnewc)
  - T-5..T-1 historical rolling (daily history)
  - pool state at T (window_day_index, days_left_est, pool_share_prev_neg)

Forbidden: T-day close, T-day full volume, T-day full moneyflow, future labels.

Public API:
  enrich_downside_features(df) -> df with 22 new columns (some may already exist)

The 22 features list (also encoded in config/soft_rerank.json):
  Already in second_start_dataset (intra-snap + T-1 daily basic + pool state):
    cum_vol, cum_amt, ret_from_open, drawdown, intraday_range, event_open_gap,
    turnover_rate_f, window_day_index, days_left_est
  Derived here:
    wh_range_t1, wh_ret_5d, wh_vol_t1, wh_vol_5d, wh_vol_5d_rank, wh_dd_5d_rank,
    wh_vol_ratio_t1_t2, cf_mfi_t2, cf_obv_t2, cf_atr_t2,
    cf_atr_delta, cf_mfi_delta, cf_mfi_mean5, cf_mfi_t1_rank
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MONEYFLOW_PATH = ROOT / "data" / "liquidity" / "moneyflow_tminus1_training_watchpool.csv"
LNEWC_PATH = ROOT / "data" / "liquidity" / "lnewc_factors_tminus1.csv"
WPHIST_PATH = ROOT / "data" / "processed" / "watchpool_history.csv"


# Feature whitelist (used as leakage-guard reference; same as productionization audit)
ALLOWED_FEATURE_SOURCES = {
    "T_1440_snap": [
        "cum_vol", "cum_amt", "ret_from_open", "drawdown",
        "intraday_range", "event_open_gap",
    ],
    "T_minus_1_daily_or_basic": [
        "turnover_rate_f", "wh_range_t1", "wh_vol_t1",
        "wh_vol_ratio_t1_t2", "cf_mfi_t1_rank", "cf_mfi_delta", "cf_atr_delta",
    ],
    "T_minus_2_daily_or_C_factor": [
        "cf_mfi_t2", "cf_obv_t2", "cf_atr_t2",
    ],
    "T_minus_5_to_T_minus_1_rolling": [
        "wh_vol_5d_rank", "wh_ret_5d", "wh_vol_5d",
        "wh_dd_5d_rank", "cf_mfi_mean5",
    ],
    "pool_state_at_T": [
        "window_day_index", "days_left_est",
    ],
}
ALL_DOWNSIDE_FEATURES = sum(ALLOWED_FEATURE_SOURCES.values(), [])

FORBIDDEN_LABEL_COLUMNS = {
    "label_down_start_5d", "future_min_ret_5d",
    "label_up_5d", "label_up_2d", "label_up_10d",
    "ret_5d", "ret_2d", "ret_10d",
}


class LeakageError(RuntimeError):
    pass


def _load_lookups() -> dict:
    """Load auxiliary CSVs and return per-ts_code grouped DataFrames."""
    mf = pd.read_csv(MONEYFLOW_PATH, dtype={"trade_date": str})
    mf = mf[["ts_code", "trade_date"]].sort_values(["ts_code", "trade_date"])

    cf = pd.read_csv(LNEWC_PATH, dtype={"prev_trade_date": str})
    cf = cf[["ts_code", "prev_trade_date", "mfi_qfq", "obv_qfq", "atr_qfq"]].dropna(subset=["obv_qfq"])
    cf = cf.sort_values(["ts_code", "prev_trade_date"])

    wh = pd.read_csv(WPHIST_PATH, dtype={"code": str})
    wh["code"] = wh["code"].str.zfill(6)
    wh["trade_date"] = pd.to_datetime(wh["date"], errors="coerce").dt.strftime("%Y%m%d")
    wh = wh.rename(columns={"last": "close", "amount": "amount_d", "volume": "vol"})
    wh = wh[["code", "trade_date", "close", "high", "low", "open", "vol", "amount_d"]].dropna(subset=["close"])
    wh = wh.sort_values(["code", "trade_date"])

    return {
        "cf_by_ts": {ts: g.reset_index(drop=True) for ts, g in cf.groupby("ts_code", sort=False)},
        "wh_by_code": {c: g.reset_index(drop=True) for c, g in wh.groupby("code", sort=False)},
    }


def _per_row_derive(row: pd.Series, cf_by_ts: dict, wh_by_code: dict) -> dict:
    ts = row["ts_code"]
    T = str(row["trade_date"])
    code6 = row["code"] if isinstance(row["code"], str) else f"{int(row['code']):06d}"
    code6 = code6.zfill(6)

    out = {}

    # --- C-factor (uses prev_trade_date < T) ---
    if ts in cf_by_ts:
        sub = cf_by_ts[ts]
        prior = sub[sub["prev_trade_date"] < T]
        if len(prior) >= 1:
            r1 = prior.iloc[-1]
            out["cf_mfi_t1"] = float(r1["mfi_qfq"]) if pd.notna(r1["mfi_qfq"]) else np.nan
            out["cf_atr_t1"] = float(r1["atr_qfq"]) if pd.notna(r1["atr_qfq"]) else np.nan
        else:
            out["cf_mfi_t1"] = np.nan; out["cf_atr_t1"] = np.nan
        if len(prior) >= 2:
            r2 = prior.iloc[-2]
            out["cf_mfi_t2"] = float(r2["mfi_qfq"]) if pd.notna(r2["mfi_qfq"]) else np.nan
            out["cf_obv_t2"] = float(r2["obv_qfq"]) if pd.notna(r2["obv_qfq"]) else np.nan
            out["cf_atr_t2"] = float(r2["atr_qfq"]) if pd.notna(r2["atr_qfq"]) else np.nan
        else:
            out["cf_mfi_t2"] = np.nan; out["cf_obv_t2"] = np.nan; out["cf_atr_t2"] = np.nan
        last5 = prior.tail(5)
        if len(last5) >= 4:
            out["cf_mfi_mean5"] = float(last5["mfi_qfq"].mean())
        else:
            out["cf_mfi_mean5"] = np.nan
        if not np.isnan(out["cf_mfi_t1"]) and not np.isnan(out["cf_mfi_t2"]):
            out["cf_mfi_delta"] = out["cf_mfi_t1"] - out["cf_mfi_t2"]
        else:
            out["cf_mfi_delta"] = np.nan
        if not np.isnan(out["cf_atr_t1"]) and not np.isnan(out["cf_atr_t2"]):
            out["cf_atr_delta"] = out["cf_atr_t1"] - out["cf_atr_t2"]
        else:
            out["cf_atr_delta"] = np.nan
    else:
        for f in ("cf_mfi_t2", "cf_obv_t2", "cf_atr_t2",
                  "cf_mfi_mean5", "cf_mfi_delta", "cf_atr_delta"):
            out[f] = np.nan

    # --- daily history (trade_date < T) ---
    if code6 in wh_by_code:
        sub = wh_by_code[code6]
        prior = sub[sub["trade_date"] < T]
        if len(prior) >= 1:
            t1 = prior.iloc[-1]
            out["wh_vol_t1"] = float(t1["vol"]) if pd.notna(t1["vol"]) else np.nan
            out["wh_range_t1"] = float((t1["high"] - t1["low"]) / t1["close"]) \
                                 if t1["close"] and pd.notna(t1["high"]) else np.nan
        else:
            out["wh_vol_t1"] = np.nan; out["wh_range_t1"] = np.nan
        if len(prior) >= 2:
            t1 = prior.iloc[-1]; t2 = prior.iloc[-2]
            out["wh_vol_ratio_t1_t2"] = float(t1["vol"] / t2["vol"]) if t2["vol"] else np.nan
        else:
            out["wh_vol_ratio_t1_t2"] = np.nan
        last5 = prior.tail(5)
        if len(last5) >= 4 and last5.iloc[0]["close"]:
            out["wh_ret_5d"] = float((last5.iloc[-1]["close"] - last5.iloc[0]["close"])
                                     / last5.iloc[0]["close"])
            rets = last5["close"].pct_change().dropna()
            out["wh_vol_5d"] = float(rets.std()) if len(rets) >= 2 else np.nan
            eq = (1 + rets).cumprod()
            peak = eq.cummax()
            out["wh_dd_5d"] = float(((peak - eq) / peak).max()) if len(eq) else np.nan
        else:
            out["wh_ret_5d"] = np.nan; out["wh_vol_5d"] = np.nan; out["wh_dd_5d"] = np.nan
    else:
        for f in ("wh_vol_t1", "wh_range_t1", "wh_vol_ratio_t1_t2",
                  "wh_ret_5d", "wh_vol_5d", "wh_dd_5d"):
            out[f] = np.nan

    return out


def enrich_downside_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 22 downside-detection features.

    Required existing columns in df:
      ts_code, code, trade_date,
      cum_vol, cum_amt, ret_from_open, drawdown, intraday_range,
      event_open_gap, turnover_rate_f,
      window_day_index, days_left_est, pool_share_prev_neg

    Forbidden columns must NOT be used as features (caller must drop them or
    they will be flagged in assert_no_leakage()).
    """
    if df.empty:
        return df.copy()

    lookups = _load_lookups()
    cf_by_ts = lookups["cf_by_ts"]
    wh_by_code = lookups["wh_by_code"]

    out = df.copy()
    derived_rows = [_per_row_derive(r, cf_by_ts, wh_by_code) for _, r in out.iterrows()]
    derived_df = pd.DataFrame(derived_rows, index=out.index)

    # Add cross-section ranks (per trade_date)
    for col in ("wh_vol_5d", "wh_dd_5d", "cf_mfi_t1"):
        if col not in derived_df.columns:
            continue
        merged = pd.concat([out[["trade_date"]], derived_df[[col]]], axis=1)
        rank = merged.groupby("trade_date")[col].rank(pct=True, method="average")
        derived_df[f"{col}_rank"] = rank

    # Combine
    for c in derived_df.columns:
        out[c] = derived_df[c]

    return out


def assert_no_leakage(df: pd.DataFrame, features_used: list) -> None:
    """Sanity-check that none of features_used is a forbidden label column."""
    bad = [f for f in features_used if f in FORBIDDEN_LABEL_COLUMNS]
    if bad:
        raise LeakageError(f"Forbidden label columns used as features: {bad}")
    missing = [f for f in features_used if f not in df.columns]
    if missing:
        raise LeakageError(f"Required features missing from df: {missing}")
