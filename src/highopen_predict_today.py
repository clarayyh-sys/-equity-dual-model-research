"""Predict next-day high-open candidates.

Data flow:
  - T-1 minute lines  → lag1 features (lag1_px_close/high/low, lag1_cum_vol/amount)
  - T-day snapshot     → snap features (snap_open/high/low/last, snap_volume/amount, snap_pct_chg)
  - T-2 daily/basic/mf → lag2 features + moneyflow features
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tushare as ts

from highopen_multiday_dataset import (
    MINUTE_CUTOFF,
    extract_minute_features,
    init_ts_api,
    load_minute_slice,
    get_daily_map,
    get_daily_basic_map,
    get_moneyflow_map,
)
from highopen_three_agent_pipeline import (
    PRICE_FEATURES,
    LIQUIDITY_FEATURES,
    RISK_FEATURES,
    load_dataset,
    fit_agent_models,
    predict_scores,
)
from tushare_utils import to_ts_code

ROOT = Path(__file__).resolve().parents[1]
REPORT_INFER = ROOT / "reports" / "highopen_three_agent_infer.csv"


def get_recent_trade_dates(ts_api, ref_date: dt.date, n: int = 3) -> List[dt.date]:
    """Return the last *n* trading days up to and including *ref_date*."""
    start = ref_date - dt.timedelta(days=30)
    cal = ts_api.trade_cal(
        start_date=start.strftime("%Y%m%d"),
        end_date=ref_date.strftime("%Y%m%d"),
        is_open=1,
    )
    if cal is None or cal.empty:
        raise RuntimeError("Unable to fetch trade calendar")
    dates = sorted(cal[cal["is_open"] == 1]["cal_date"].tolist(), reverse=True)
    return [dt.datetime.strptime(str(d), "%Y%m%d").date() for d in dates[:n]]


def next_trade_date(ts_api, current: dt.date) -> dt.date:
    start = current + dt.timedelta(days=1)
    end = current + dt.timedelta(days=10)
    cal = ts_api.trade_cal(
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        is_open=1,
    )
    if cal is None or cal.empty:
        raise RuntimeError("Unable to fetch next trade date")
    cal_sorted = cal[cal["is_open"] == 1].sort_values("cal_date")
    return dt.datetime.strptime(str(cal_sorted.iloc[0]["cal_date"]), "%Y%m%d").date()


def _infer_dates_from_local(today: dt.date) -> tuple:
    """Infer T/T-1/T-2/T+1 from local tmp_daily_history files when API is unavailable."""
    import glob
    files = sorted(glob.glob(str(ROOT / "tmp_daily_history_*.csv")))
    if not files:
        raise RuntimeError("No local daily_history files to infer dates")
    # Latest file's date_key is the most recent T-1 we have data for
    latest = Path(files[-1]).stem.replace("tmp_daily_history_", "")
    # Read unique trade dates from that file
    df = pd.read_csv(files[-1], usecols=["trade_date"], dtype={"trade_date": str})
    dates = sorted(df["trade_date"].unique())
    if len(dates) < 3:
        raise RuntimeError(f"Not enough dates in {files[-1]}")
    # T = latest date in file (= T-1 data covers up to this), T-1 = second latest, T-2 = third latest
    t_date = dt.datetime.strptime(dates[-1], "%Y%m%d").date()
    t_minus_1 = dt.datetime.strptime(dates[-2], "%Y%m%d").date()
    t_minus_2 = dt.datetime.strptime(dates[-3], "%Y%m%d").date()
    # Guess T+1 as next weekday after T
    target = t_date + dt.timedelta(days=1)
    while target.weekday() >= 5:
        target += dt.timedelta(days=1)
    print(f"[INFO] Inferred from local: T={t_date}, T-1={t_minus_1}, T-2={t_minus_2}, T+1≈{target}")
    return t_date, t_minus_1, t_minus_2, target


def _load_local_daily(date_key: str) -> Dict[str, dict]:
    """Load T-2 daily data from tmp_daily_history file."""
    path = ROOT / f"tmp_daily_history_{date_key}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"trade_date": str})
    df = df[df["trade_date"] == date_key]
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        code6 = row["ts_code"].split(".")[0]
        out[code6] = row.to_dict()
    return out


def _load_local_daily_basic(date_key: str) -> Dict[str, dict]:
    """Load T-2 daily_basic from tmp file."""
    path = ROOT / f"tmp_daily_basic_{date_key}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"trade_date": str})
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        code6 = row["ts_code"].split(".")[0]
        out[code6] = row.to_dict()
    return out


def _load_local_moneyflow(date_key: str) -> Dict[str, dict]:
    """Load T-2 moneyflow from tmp file."""
    path = ROOT / f"moneyflow_{date_key}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"trade_date": str})
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        code6 = row["ts_code"].split(".")[0]
        out[code6] = row.to_dict()
    return out


def _load_lg_net_ratio(date_key: str) -> Dict[str, float]:
    """Compute 主力净比 (大单+特大单净额/总额) from local moneyflow file."""
    path = ROOT / f"moneyflow_{date_key}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["code"] = df["ts_code"].str[:6]
    buy_cols = ["buy_lg_amount", "buy_elg_amount"]
    sell_cols = ["sell_lg_amount", "sell_elg_amount"]
    all_cols = buy_cols + sell_cols + ["buy_md_amount", "sell_md_amount", "buy_sm_amount", "sell_sm_amount"]
    for c in buy_cols + sell_cols + all_cols:
        if c not in df.columns:
            return {}
    df["lg_net"] = df[buy_cols].sum(axis=1) - df[sell_cols].sum(axis=1)
    df["total"] = df[all_cols].sum(axis=1)
    df["lg_net_ratio"] = df["lg_net"] / df["total"].replace(0, float("nan"))
    return dict(zip(df["code"], df["lg_net_ratio"]))


def _load_flow_structure(date_key: str) -> Dict[str, dict]:
    """Compute main_ratio / retail_ratio from local moneyflow file.

    Returns {code6: {"main": float, "retail": float}}.
    Phase 5 (2026-04-17): flow_signal_flag = (main>0 AND retail<0).
    """
    path = ROOT / f"moneyflow_{date_key}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["code"] = df["ts_code"].str[:6]
    bc = ["buy_lg_amount", "buy_elg_amount"]
    sc = ["sell_lg_amount", "sell_elg_amount"]
    rc_buy = ["buy_sm_amount", "buy_md_amount"]
    rc_sell = ["sell_sm_amount", "sell_md_amount"]
    ac = bc + sc + rc_buy + rc_sell
    for c in ac:
        if c not in df.columns:
            df[c] = 0
    df["main_net"] = df[bc].sum(axis=1) - df[sc].sum(axis=1)
    df["retail_net"] = df[rc_buy].sum(axis=1) - df[rc_sell].sum(axis=1)
    df["total"] = df[ac].sum(axis=1)
    df["main_r"] = df["main_net"] / df["total"].replace(0, float("nan"))
    df["retail_r"] = df["retail_net"] / df["total"].replace(0, float("nan"))
    result = {}
    for _, r in df.iterrows():
        result[r["code"]] = {"main": r["main_r"], "retail": r["retail_r"]}
    return result


def load_rt_k_as_lag1(t_minus_1: dt.date) -> Dict[str, dict]:
    """Fallback: approximate lag1 features from rt_k snapshot when minute data is unavailable."""
    date_str = t_minus_1.strftime("%Y%m%d")
    rt_k_path = ROOT / "data" / "intraday" / date_str / "rt_k_snapshot.csv"
    if not rt_k_path.exists():
        return {}
    df = pd.read_csv(rt_k_path, dtype={"code": str})
    df["code"] = df["code"].astype(str).str.zfill(6)
    lookup: Dict[str, dict] = {}
    for _, row in df.iterrows():
        lookup[row["code"]] = {
            "px_close": row.get("close"),
            "px_open": row.get("open"),
            "px_high": row.get("high"),
            "px_low": row.get("low"),
            "cum_vol": row.get("vol"),
            "cum_amount": row.get("amount"),
        }
    return lookup


def load_snap_features(t_date: dt.date) -> Dict[str, dict]:
    """Load T-day snap features from rt_k_snapshot (primary) + price_snapshot (fallback)."""
    date_str = t_date.strftime("%Y%m%d")
    lookup: Dict[str, dict] = {}

    # Primary: rt_k_snapshot has raw OHLCV + pre_close
    rt_k_path = ROOT / "data" / "intraday" / date_str / "rt_k_snapshot.csv"
    if rt_k_path.exists():
        df = pd.read_csv(rt_k_path, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        for _, row in df.iterrows():
            code = row["code"]
            lookup[code] = {
                "snap_open": row.get("open"),
                "snap_high": row.get("high"),
                "snap_low": row.get("low"),
                "snap_last": row.get("close"),
                "snap_pre_close": row.get("pre_close"),
                "snap_volume": row.get("vol"),
                "snap_amount": row.get("amount"),
            }

    # Enrich with price_snapshot (has cum_vol/cum_amt which may differ from full-day vol)
    snap_path = ROOT / f"tmp_price_snapshot_{date_str}.csv"
    if snap_path.exists():
        df = pd.read_csv(snap_path, dtype={"code": str})
        df["code"] = df["code"].astype(str).str.zfill(6)
        for _, row in df.iterrows():
            code = row["code"]
            if code not in lookup:
                # Fallback: build from price_snapshot (no high/low available)
                lookup[code] = {
                    "snap_open": row.get("open_t"),
                    "snap_last": row.get("price_1440"),
                    "snap_volume": row.get("cum_vol"),
                    "snap_amount": row.get("cum_amt"),
                }
            else:
                # Prefer cum_vol/cum_amt from price_snapshot if available
                if pd.notna(row.get("cum_vol")):
                    lookup[code]["snap_volume"] = row["cum_vol"]
                if pd.notna(row.get("cum_amt")):
                    lookup[code]["snap_amount"] = row["cum_amt"]

    return lookup


def build_inference_df(
    ts_api,
    codes: List[str],
    t_date: dt.date,
    t_minus_1: dt.date,
    t_minus_2: dt.date,
    target_date: dt.date,
) -> pd.DataFrame:
    """Build inference features: T-1 minute + T-day snapshot + T-2 daily."""

    snap_lookup = load_snap_features(t_date)
    if not snap_lookup:
        raise RuntimeError(f"No T-day snapshot for {t_date}")

    # Fallback lag1 data from rt_k snapshot (used when minute files are missing)
    lag1_fallback = load_rt_k_as_lag1(t_minus_1)
    if lag1_fallback:
        print(f"[INFO] Loaded rt_k fallback for T-1={t_minus_1} ({len(lag1_fallback)} stocks)")

    # T-2 data: prefer local tmp files (already fetched by OpenClaw), fallback to TuShare API
    ts_code_set = {to_ts_code(c) for c in codes}
    t2_key = t_minus_2.strftime("%Y%m%d")

    daily_t2 = _load_local_daily(t2_key)
    basic_t2 = _load_local_daily_basic(t2_key)
    mf_t2 = _load_local_moneyflow(t2_key)

    if not daily_t2 or not basic_t2 or not mf_t2:
        print(f"[INFO] Local T-2 data incomplete, fetching from TuShare...")
        daily_cache: Dict[str, Dict[str, dict]] = {}
        basic_cache: Dict[str, Dict[str, dict]] = {}
        mf_cache: Dict[str, Dict[str, dict]] = {}
        if not daily_t2:
            daily_t2 = get_daily_map(ts_api, t2_key, ts_code_set, daily_cache)
        if not basic_t2:
            basic_t2 = get_daily_basic_map(ts_api, t2_key, ts_code_set, basic_cache)
        if not mf_t2:
            mf_t2 = get_moneyflow_map(ts_api, t2_key, ts_code_set, mf_cache)
    else:
        print(f"[INFO] T-2={t2_key} data loaded from local files")

    # T-1 daily OHLCV — needed by enrich_form_features (lag1_close/high/low/pre_close)
    t1_key = t_minus_1.strftime("%Y%m%d")
    daily_t1 = _load_local_daily(t1_key)

    # T-3: 3rd trading day before event (for multi-day accumulation features)
    # Infer T-3 from local daily_history files (treat available dates as trading calendar)
    import glob as _glob
    _avail = sorted({
        p.split("tmp_daily_history_")[1].replace(".csv", "")
        for p in _glob.glob(str(ROOT / "tmp_daily_history_*.csv"))
    })
    t3_key = None
    if t2_key in _avail:
        _idx = _avail.index(t2_key)
        if _idx > 0:
            t3_key = _avail[_idx - 1]

    # 主力净比 (T-1, T-2, T-3)
    lg_t1 = _load_lg_net_ratio(t1_key)
    lg_t2 = _load_lg_net_ratio(t2_key)
    lg_t3 = _load_lg_net_ratio(t3_key) if t3_key else {}
    # Phase 5: load flow structure (main/retail split) at T-1 for flow_signal_flag
    flow_t1 = _load_flow_structure(t1_key)
    if flow_t1:
        print(f"[INFO] 资金结构 T-1={t1_key}: {len(flow_t1)} 只 (flow_signal tracking)")
    if lg_t1:
        print(f"[INFO] 主力净比 T-1={t1_key}: {len(lg_t1)} 只")
    else:
        print(f"[WARN] 主力净比 T-1={t1_key}: 文件不存在")
    if t3_key and lg_t3:
        print(f"[INFO] 主力净比 T-3={t3_key}: {len(lg_t3)} 只 (3日累积信号)")

    # Boundary enforcement: load first_join_date per stock from watchpool
    # Rule: each lag feature date D must satisfy D >= first_join - 5 trading days
    MAX_LOOKBACK = 5  # mirror highopen_multiday_dataset.MAX_LOOKBACK_TRADING_DAYS
    _eh_path = ROOT / "data" / "processed" / "watchpool_effective_history.csv"
    first_join_lookup: Dict[str, dt.date] = {}
    if _eh_path.exists():
        _eh = pd.read_csv(_eh_path, usecols=["code", "first_join_date"])
        _eh["code"] = _eh["code"].astype(str).str.zfill(6)
        _eh["first_join_date"] = pd.to_datetime(_eh["first_join_date"], errors="coerce")
        first_join_lookup = _eh.dropna().groupby("code")["first_join_date"].min().apply(lambda x: x.date()).to_dict()

    # Build trading calendar from available local daily history files (used for "trading day" math)
    _trading_cal = sorted({
        dt.datetime.strptime(p.split("tmp_daily_history_")[1].replace(".csv",""), "%Y%m%d").date()
        for p in _glob.glob(str(ROOT / "tmp_daily_history_*.csv"))
        if "tmp_daily_history_" in p
    })
    _cal_idx = {d: i for i, d in enumerate(_trading_cal)}

    def _within_boundary(code: str, target_date: dt.date) -> bool:
        fj = first_join_lookup.get(code)
        if fj is None: return True
        if fj not in _cal_idx: return True
        bound_idx = max(0, _cal_idx[fj] - MAX_LOOKBACK)
        return target_date >= _trading_cal[bound_idx]

    rows: List[dict] = []
    skipped: List[tuple] = []
    lag1_from_minute = 0
    lag1_from_rtk = 0
    boundary_blocks = 0

    for code in codes:
        # Per-stock boundary checks: which lag dates are within (first_join - 5 trading days)?
        t1_ok = _within_boundary(code, t_minus_1)
        t2_ok = _within_boundary(code, t_minus_2)
        # T-3 date from calendar (best-effort)
        t3_date = None
        if t3_key:
            try:
                t3_date = dt.datetime.strptime(t3_key, "%Y%m%d").date()
            except Exception:
                t3_date = None
        t3_ok = _within_boundary(code, t3_date) if t3_date else True
        if not (t1_ok and t2_ok):
            boundary_blocks += 1
            skipped.append((code, f"out_of_boundary (T-1 ok={t1_ok}, T-2 ok={t2_ok})"))
            continue

        # --- T-1 minute features ---
        minute_df = load_minute_slice(ts_api, code, t_minus_1)
        if minute_df is not None:
            lag1 = extract_minute_features(minute_df)
            lag1_from_minute += 1
        elif code in lag1_fallback:
            lag1 = lag1_fallback[code]
            lag1_from_rtk += 1
        else:
            skipped.append((code, "missing_t1_minute"))
            continue

        # --- T-day snap features ---
        snap = snap_lookup.get(code)
        if snap is None:
            skipped.append((code, "missing_t_snapshot"))
            continue

        pre_close = snap.get("snap_pre_close")
        snap_last = snap.get("snap_last")
        snap_pct_chg = None
        if pre_close and pre_close != 0 and snap_last is not None:
            snap_pct_chg = snap_last / pre_close - 1

        # --- T-2 features ---
        d2 = daily_t2.get(code)
        b2 = basic_t2.get(code)
        mf2 = mf_t2.get(code)

        record = {
            "trade_date": target_date.strftime("%Y-%m-%d"),
            "code": code,
            "ts_code": to_ts_code(code),
            # Placeholders for pipeline compatibility
            "label_high_open": 0,
            "label_low_open": 0,
            "event_prev_close": snap_last or 0,
            "event_open": 0,
            # lag1 from T-1 minute lines
            "lag1_px_close": lag1.get("px_close"),
            "lag1_px_high": lag1.get("px_high"),
            "lag1_px_low": lag1.get("px_low"),
            "lag1_cum_vol": lag1.get("cum_vol"),
            "lag1_cum_amount": lag1.get("cum_amount"),
            # snap from T-day snapshot
            "snap_open": snap.get("snap_open"),
            "snap_high": snap.get("snap_high"),
            "snap_low": snap.get("snap_low"),
            "snap_last": snap_last,
            "snap_volume": snap.get("snap_volume"),
            "snap_amount": snap.get("snap_amount"),
            "snap_pct_chg": snap_pct_chg,
        }

        # T-1 daily OHLCV — for enrich_form_features (lag1_range_pct, lag1_pos)
        d1 = daily_t1.get(code)
        if d1:
            record.update({
                "lag1_open": d1.get("open"),
                "lag1_high": d1.get("high"),
                "lag1_low": d1.get("low"),
                "lag1_close": d1.get("close"),
                "lag1_pre_close": d1.get("pre_close"),
                "lag1_pct_change": d1.get("pct_chg"),
                "lag1_vol": d1.get("vol"),
                "lag1_amount": d1.get("amount"),
            })

        # 主力净比 + 建仓强度特征
        lg1 = lg_t1.get(code)
        lg2 = lg_t2.get(code)
        if lg1 is not None:
            record["lag1_lg_net_ratio"] = lg1
            # 建仓强度 = 主力净比 - 当日涨幅（T-1 pct_change 来自日线）
            pct1 = d1.get("pct_chg") if d1 else None
            if pct1 is not None:
                record["lag1_accum"] = lg1 - float(pct1) / 100
                eps = 1e-4
                record["lag1_kyle_inv"] = lg1 / (abs(float(pct1) / 100) + eps)
        if lg2 is not None:
            record["hist_lg_net_ratio"] = lg2
            pct2 = d2.get("pct_chg") if d2 else None
            if pct2 is not None:
                record["hist_accum"] = lg2 - float(pct2) / 100
        lg3 = lg_t3.get(code) if t3_ok else None
        if lg3 is not None:
            record["lag3_lg_net_ratio"] = lg3
        if lg1 is not None and lg2 is not None:
            record["lg_net_trend"] = lg1 - lg2
            lag1_accum = record.get("lag1_accum")
            hist_accum = record.get("hist_accum")
            if lag1_accum is not None and hist_accum is not None:
                record["accum_combo"] = lag1_accum + 0.5 * hist_accum
            # 3日主力净比均值 (lg_net_ratio_3d) — 建仓持续性
            lg_sum = (lg1 or 0.0) + (lg2 or 0.0) + (lg3 or 0.0)
            n_days = sum(x is not None for x in [lg1, lg2, lg3])
            record["lg_net_ratio_3d"] = lg_sum / n_days if n_days > 0 else None
            # 近3日主力净买入天数 (accum_days_pos_3d)
            record["accum_days_pos_3d"] = sum(
                1 for x in [lg1, lg2, lg3] if x is not None and x > 0
            )
            # 3日主力净比标准差 (lg_3d_std) — 建仓一致性波动
            # 2026-04-16 Phase 4 加入：paired test p=0.002，9胜0负15平
            _vals = [float(x) for x in [lg1, lg2, lg3] if x is not None]
            if len(_vals) >= 2:
                _m = sum(_vals) / len(_vals)
                record["lg_3d_std"] = (sum((v - _m) ** 2 for v in _vals) / len(_vals)) ** 0.5
            else:
                record["lg_3d_std"] = 0.0

        # Phase 5: 资金结构 flow_signal_flag (主力买+散户卖)
        # 作为追踪信号输出，当前不参与模型决策，待数据积累后升级为 decision rule
        fs = flow_t1.get(code)
        if fs:
            mr, rr = fs.get("main", float("nan")), fs.get("retail", float("nan"))
            record["main_ratio"] = mr
            record["retail_ratio"] = rr
            record["flow_signal_flag"] = 1 if (not np.isnan(mr) and mr > 0 and not np.isnan(rr) and rr < 0) else 0
        else:
            record["main_ratio"] = None
            record["retail_ratio"] = None
            record["flow_signal_flag"] = 0

        # lag2 features
        if d2:
            record.update({
                "lag2_open": d2.get("open"),
                "lag2_high": d2.get("high"),
                "lag2_low": d2.get("low"),
                "lag2_close": d2.get("close"),
                "lag2_pre_close": d2.get("pre_close"),
                "lag2_pct_change": d2.get("pct_chg"),
                "lag2_vol": d2.get("vol"),
                "lag2_amount": d2.get("amount"),
            })
        if b2:
            record.update({
                "lag2_turnover_rate": b2.get("turnover_rate_f") or b2.get("turnover_rate"),
                "lag2_volume_ratio": b2.get("volume_ratio"),
            })
        if mf2:
            record.update({
                "hist_mf_net_vol": mf2.get("net_mf_vol"),
                "hist_mf_net_amount": mf2.get("net_mf_amount"),
            })

        rows.append(record)

    if boundary_blocks > 0:
        print(f"[INFO] {boundary_blocks} stocks blocked by 5-day first_join boundary")
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} stocks:")
        for code, reason in skipped[:10]:
            print(f"  {code}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if not rows:
        raise RuntimeError("No inference rows generated")

    print(f"[INFO] Built {len(rows)} inference rows (lag1: {lag1_from_minute} minute, {lag1_from_rtk} rt_k fallback)")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict next-day high-open candidates")
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--risk-threshold", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    ts_api = init_ts_api()

    # Load watchpool codes
    codes_path = ROOT / "data" / "processed" / "master_watchpool_codes.txt"
    codes = [line.strip() for line in codes_path.read_text().splitlines() if line.strip()]

    # Determine dates: try trade calendar API, fallback to local daily_history
    today = dt.date.today()
    try:
        recent = get_recent_trade_dates(ts_api, today, n=3)
        t_date, t_minus_1, t_minus_2 = recent[0], recent[1], recent[2]
        target_date = next_trade_date(ts_api, t_date)
    except Exception as e:
        print(f"[WARN] TuShare calendar unavailable ({e}), inferring dates from local data")
        t_date, t_minus_1, t_minus_2, target_date = _infer_dates_from_local(today)

    print(f"T={t_date}, T-1={t_minus_1}, T-2={t_minus_2}")
    print(f"Predicting T+1={target_date}")

    # Build inference features
    infer_df = build_inference_df(ts_api, codes, t_date, t_minus_1, t_minus_2, target_date)

    # Fill missing feature columns with NaN
    feature_cols = set(PRICE_FEATURES + LIQUIDITY_FEATURES + RISK_FEATURES)
    for col in feature_cols:
        if col not in infer_df.columns:
            infer_df[col] = np.nan

    # Train on full history, then predict
    dataset = load_dataset(ROOT / "data" / "processed" / "highopen_multiday_dataset.csv")
    models = fit_agent_models(dataset)
    scored = predict_scores(models, infer_df)

    # Filter
    scored = scored[
        (scored["risk_score"] < args.risk_threshold)
        & (scored["final_score"] >= args.score_threshold)
    ]
    scored = scored.sort_values("final_score", ascending=False)

    # Industry dedup
    for col in ("industry", "t2_industry"):
        if col in scored.columns:
            scored = scored.drop_duplicates(subset=col, keep="first")
            break

    top = scored.head(args.top_k)

    # Save
    REPORT_INFER.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(REPORT_INFER, index=False)
    dated_path = REPORT_INFER.parent / f"highopen_three_agent_infer_{target_date.strftime('%Y%m%d')}.csv"
    top.to_csv(dated_path, index=False)

    # Display
    display_cols = [c for c in ["code", "final_score", "price_score", "liq_score", "risk_score"] if c in top.columns]
    print(f"\n=== Top {len(top)} candidates for {target_date} ===")
    if not top.empty:
        print(top[display_cols].to_string(index=False))
    else:
        print("(no candidates passed filters)")


if __name__ == "__main__":
    main()
