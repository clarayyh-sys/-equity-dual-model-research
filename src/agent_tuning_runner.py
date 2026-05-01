from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "highopen_multiday_dataset.csv"
INTRADAY_DIR = ROOT / "data" / "intraday" / "minute_full"
REPORT_SIGNALS = ROOT / "reports" / "highopen_three_agent_signals.csv"
REPORT_METRICS = ROOT / "reports" / "highopen_three_agent_metrics.json"
REPORT_RISK = ROOT / "reports" / "highopen_three_agent_risk_alerts.csv"
MONEYFLOW_CACHE = ROOT / "data" / "processed" / "moneyflow_cache_20260302_20260312.csv"

PRICE_FEATURES = [
    "lag1_px_close",
    "lag1_px_high",
    "lag1_px_low",
    "snap_open",
    "snap_high",
    "snap_low",
    "snap_last",
    "snap_pct_chg",
    "lag2_open",
    "lag2_high",
    "lag2_low",
    "lag2_close",
    "lag2_pre_close",
    "lag2_pct_change",
]

LIQUIDITY_FEATURES = [
    "lag1_cum_vol",
    "lag1_cum_amount",
    "snap_volume",
    "snap_amount",
    "lag2_vol",
    "lag2_amount",
    "lag2_turnover_rate",
    "lag2_volume_ratio",
    "hist_mf_net_vol",
    "hist_mf_net_amount",
]

RISK_FEATURES = [
    "snap_pct_chg",
    "lag1_px_low",
    "lag1_px_high",
    "lag2_pct_change",
    "lag2_volume_ratio",
    "hist_mf_net_vol",
    "hist_mf_net_amount",
]


@dataclass
class AgentModels:
    price_scaler: StandardScaler
    price_model: Optional[LogisticRegression]
    liq_scaler: StandardScaler
    liq_model: Optional[LogisticRegression]
    risk_scaler: StandardScaler
    risk_model: Optional[LogisticRegression]


@dataclass
class Config:
    dataset: Path
    top_k: int
    score_threshold: float
    risk_threshold: float
    warmup_days: int
    liq_min: float
    industry_cap: Optional[float]
    price_weight: float
    liq_weight: float
    risk_weight: float
    mid_pm_slope_gate: bool
    gap_residual_adjust: bool
    dryness_threshold: float
    adv_threshold: float
    adv_bonus: float
    low_liq_threshold: float
    low_liq_per_industry: int
    low_gap_override: Optional[float]
    mainflow_window: int
    mainflow_min_sum: Optional[float]
    require_mainflow_nonneg: bool
    filter_fake_flow: bool


@functools.lru_cache(maxsize=2048)
def load_intraday_frame(date_key: str, code6: str) -> Optional[pd.DataFrame]:
    path = INTRADAY_DIR / date_key / f"{code6}_1m_full.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "trade_time" not in df.columns:
        return None
    df = df.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    target_date = dt.datetime.strptime(date_key, "%Y%m%d").date()
    df = df[df["trade_time"].dt.date == target_date]
    if df.empty:
        return None
    return df


def mid_pm_slope_positive(code6: str, trade_date: dt.date) -> bool:
    date_key = trade_date.strftime("%Y%m%d")
    df = load_intraday_frame(date_key, code6)
    if df is None:
        return False
    start_time = dt.time(13, 0)
    end_time = dt.time(14, 30)
    segment = df[(df["trade_time"].dt.time >= start_time) & (df["trade_time"].dt.time <= end_time)]
    if segment.empty:
        return False
    start_val = segment.iloc[0]["close"]
    end_val = segment.iloc[-1]["close"]
    if start_val in (None, 0):
        return False
    return (end_val - start_val) / start_val > 0


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df.sort_values(["trade_date", "code"], inplace=True)
    return df


def load_moneyflow_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df


def augment_with_moneyflow(df: pd.DataFrame, window: int) -> pd.DataFrame:
    cache = load_moneyflow_cache(MONEYFLOW_CACHE)
    if cache.empty:
        df["mf_main_net"] = 0.0
        df["mf_small_net"] = 0.0
        df["mf_mid_net"] = 0.0
        df["mf_main_sum"] = 0.0
        df["mf_fake_flow_flag"] = False
        return df
    merged = df.merge(cache, on=["ts_code", "trade_date"], how="left")
    merged.rename(columns={"main_net": "mf_main_net", "small_net": "mf_small_net", "mid_net": "mf_mid_net"}, inplace=True)
    for col in ["mf_main_net", "mf_small_net", "mf_mid_net"]:
        merged[col] = merged[col].fillna(0.0)
    merged["mf_fake_flow_flag"] = (merged["mf_small_net"] > 0) & (merged["mf_main_net"] < 0)
    if window <= 0:
        merged["mf_main_sum"] = 0.0
    else:
        merged = merged.sort_values(["ts_code", "trade_date"])
        min_periods = min(2, window)
        rolling = merged.groupby("ts_code")["mf_main_net"].transform(
            lambda s: s.rolling(window, min_periods=min_periods).sum().shift(1)
        )
        merged["mf_main_sum"] = rolling.fillna(-1e9)
    return merged.sort_values(["trade_date", "code"])


def prepare_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError("No feature columns available")
    return df[cols].fillna(0.0).to_numpy()


def fit_agent_models(train_df: pd.DataFrame, label_low_col: str) -> AgentModels:
    price_X = prepare_features(train_df, PRICE_FEATURES)
    liq_X = prepare_features(train_df, LIQUIDITY_FEATURES)
    risk_X = prepare_features(train_df, RISK_FEATURES)

    price_scaler = StandardScaler().fit(price_X)
    liq_scaler = StandardScaler().fit(liq_X)
    risk_scaler = StandardScaler().fit(risk_X)

    has_high = train_df["label_high_open"].nunique() >= 2
    has_low = train_df[label_low_col].nunique() >= 2

    price_model = None
    liq_model = None
    risk_model = None
    if has_high:
        price_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            price_scaler.transform(price_X), train_df["label_high_open"]
        )
        liq_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            liq_scaler.transform(liq_X), train_df["label_high_open"]
        )
    if has_low:
        risk_model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
            risk_scaler.transform(risk_X), train_df[label_low_col]
        )

    return AgentModels(price_scaler, price_model, liq_scaler, liq_model, risk_scaler, risk_model)


def assign_scores(models: AgentModels, df: pd.DataFrame) -> pd.DataFrame:
    price_X = prepare_features(df, PRICE_FEATURES)
    liq_X = prepare_features(df, LIQUIDITY_FEATURES)
    risk_X = prepare_features(df, RISK_FEATURES)

    if models.price_model is None:
        price_scores = np.zeros(len(df))
    else:
        price_scores = models.price_model.predict_proba(models.price_scaler.transform(price_X))[:, 1]
    if models.liq_model is None:
        liq_scores = np.zeros(len(df))
    else:
        liq_scores = models.liq_model.predict_proba(models.liq_scaler.transform(liq_X))[:, 1]
    if models.risk_model is None:
        risk_scores = np.zeros(len(df))
    else:
        risk_scores = models.risk_model.predict_proba(models.risk_scaler.transform(risk_X))[:, 1]

    heur = np.zeros(len(df))
    if "t1_drawdown_to_1440" in df.columns:
        drawdown = df["t1_drawdown_to_1440"].fillna(0).to_numpy(dtype=float)
        heur = np.maximum(heur, np.clip(drawdown / 0.2, 0, 1))
    if "t1_pct_change" in df.columns:
        pct_change = df["t1_pct_change"].fillna(0).to_numpy(dtype=float)
        heur = np.maximum(heur, np.clip(-pct_change / 5.0, 0, 1))
    risk_scores = np.maximum(risk_scores, heur)

    out = df.copy()
    out["price_score"] = price_scores
    out["liq_score"] = liq_scores
    out["risk_score"] = risk_scores
    return out


def enforce_low_liq_limit(df: pd.DataFrame, threshold: float, limit: int) -> pd.DataFrame:
    if threshold <= 0 or limit <= 0:
        return df
    kept_rows = []
    counts: Dict[str, int] = {}
    for _, row in df.iterrows():
        industry = row.get("industry") or "Unknown"
        score = row.get("liq_score", 0.0)
        if score < threshold:
            cnt = counts.get(industry, 0)
            if cnt >= limit:
                continue
            counts[industry] = cnt + 1
        kept_rows.append(row)
    if len(kept_rows) == len(df):
        return df
    return pd.DataFrame(kept_rows)


def apply_industry_cap_rows(rows: List[pd.Series], cap: Optional[float]) -> List[pd.Series]:
    if cap is None or cap <= 0 or not rows:
        return rows
    limit = max(1, math.floor(len(rows) * cap))
    buckets: Dict[str, List[tuple[int, pd.Series]]] = {}
    for idx, row in enumerate(rows):
        industry = row.get("industry") or "Unknown"
        buckets.setdefault(industry, []).append((idx, row))
    keep_flags = [True] * len(rows)
    for industry, items in buckets.items():
        if len(items) <= limit:
            continue
        items_sorted = sorted(items, key=lambda x: x[1].get("final_score", 0), reverse=True)
        keep_indices = {idx for idx, _ in items_sorted[:limit]}
        for idx, _ in items:
            if idx not in keep_indices:
                keep_flags[idx] = False
    return [row for idx, row in enumerate(rows) if keep_flags[idx]]


def dedup_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    col = "t2_industry" if "t2_industry" in df.columns else ("industry" if "industry" in df.columns else None)
    if col is None:
        return df
    return df.sort_values("final_score", ascending=False).drop_duplicates(col, keep="first")


def run_backtest(df: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if cfg.low_gap_override is not None:
        df["label_low_custom"] = (df["event_open_gap"] <= cfg.low_gap_override).astype(int)
    else:
        df["label_low_custom"] = df["label_low_open"]

    trades = []
    alerts = []
    dates = sorted(df["trade_date"].dt.date.unique())
    pre_dedup_total = 0

    for idx in range(cfg.warmup_days, len(dates)):
        date = dates[idx]
        train_mask = df["trade_date"].dt.date < date
        test_mask = df["trade_date"].dt.date == date
        train_df = df[train_mask]
        day_df = df[test_mask]
        if train_df.empty or day_df.empty:
            continue

        models = fit_agent_models(train_df, "label_low_custom")
        train_scored = assign_scores(models, train_df)
        train_scored["_industry"] = train_scored.get("industry", "Unknown")
        industry_means = {}
        if cfg.gap_residual_adjust:
            industry_means = train_scored.groupby("_industry")["price_score"].mean().to_dict()

        scored = assign_scores(models, day_df)
        scored["_industry"] = scored.get("industry", "Unknown")
        price_component = scored["price_score"]
        if cfg.gap_residual_adjust:
            price_component = price_component - scored["_industry"].map(industry_means).fillna(0.0)
            price_component = price_component.clip(lower=0.0)
        scored["price_component"] = price_component
        scored["final_score"] = (
            cfg.price_weight * price_component
            + cfg.liq_weight * scored["liq_score"]
            + cfg.risk_weight * scored["risk_score"]
        )

        risk_alert_rows = scored[scored["risk_score"] >= cfg.risk_threshold]
        for _, row in risk_alert_rows.iterrows():
            alerts.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "code": row["code"],
                    "risk_score": float(row["risk_score"]),
                    "actual_low": int(row["label_low_custom"]),
                    "open_gap": float(row.get("event_open_gap", 0.0)),
                }
            )

        candidates = scored[scored["final_score"] >= cfg.score_threshold]
        if cfg.liq_min > 0:
            effective_liq = candidates["liq_score"].copy()
            if cfg.adv_threshold > 0 and cfg.adv_bonus != 0:
                adv_mask = candidates["lag1_cum_amount"] >= cfg.adv_threshold
                effective_liq = effective_liq + cfg.adv_bonus * adv_mask.astype(float)
            candidates = candidates[effective_liq >= cfg.liq_min]
        if cfg.adv_threshold > 0 and cfg.adv_bonus == 0:
            candidates = candidates[candidates["lag1_cum_amount"] >= cfg.adv_threshold]
        if cfg.dryness_threshold > 0 and "snap_amount" in candidates.columns and "lag1_cum_amount" in candidates.columns:
            ratio = candidates["snap_amount"] / candidates["lag1_cum_amount"].replace(0, np.nan)
            candidates = candidates[ratio >= cfg.dryness_threshold]
        if cfg.mainflow_min_sum is not None:
            candidates = candidates[candidates["mf_main_sum"] >= cfg.mainflow_min_sum]
        if cfg.require_mainflow_nonneg:
            candidates = candidates[candidates["mf_main_net"] >= 0]
        if cfg.filter_fake_flow:
            candidates = candidates[~candidates["mf_fake_flow_flag"]]
        if cfg.mid_pm_slope_gate:
            mask = []
            for _, row in candidates.iterrows():
                try:
                    date_obj = pd.to_datetime(row["trade_date"]).date()
                except Exception:
                    date_obj = date
                code6 = str(row["code"]).zfill(6)
                mask.append(mid_pm_slope_positive(code6, date_obj))
            if mask:
                candidates = candidates[mask]
        candidates = candidates[candidates["risk_score"] < cfg.risk_threshold]
        pre_dedup_total += len(candidates)
        candidates = candidates.sort_values("final_score", ascending=False)
        candidates = dedup_by_industry(candidates)
        candidates = enforce_low_liq_limit(candidates, cfg.low_liq_threshold, cfg.low_liq_per_industry)
        candidates = candidates.head(cfg.top_k)

        candidate_rows = [row for _, row in candidates.iterrows()]
        candidate_rows = apply_industry_cap_rows(candidate_rows, cfg.industry_cap)

        for row in candidate_rows:
            entry = float(row["event_prev_close"])
            exit_price = float(row["event_open"])
            if entry == 0:
                continue
            trades.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "code": row["code"],
                    "industry": row.get("industry", ""),
                    "final_score": float(row["final_score"]),
                    "price_score": float(row["price_score"]),
                    "liq_score": float(row["liq_score"]),
                    "risk_score": float(row["risk_score"]),
                    "return_pct": exit_price / entry - 1,
                    "label_high": int(row["label_high_open"]),
                    "label_low": int(row["label_low_custom"]),
                    "event_prev_close": entry,
                    "event_open": exit_price,
                }
            )

    metrics = compute_metrics(trades)
    risk_metrics = compute_risk_metrics(alerts, df, "label_low_custom")
    metrics.update(risk_metrics)
    metrics["signal_count_pre_dedup"] = pre_dedup_total

    save_outputs(trades, alerts, metrics)
    return metrics


def compute_metrics(trades: List[dict]) -> Dict[str, object]:
    if not trades:
        return {
            "trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": [],
        }
    returns = np.array([t["return_pct"] for t in trades])
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(returns[returns < 0].sum())
    net_profit = float(returns.sum())
    win_rate = float((returns > 0).mean())

    daily_returns: Dict[str, List[float]] = {}
    for t in trades:
        daily_returns.setdefault(t["trade_date"], []).append(t["return_pct"])
    equity = []
    value = 1.0
    peak = 1.0
    max_dd = 0.0
    for date in sorted(daily_returns.keys()):
        day_ret = np.mean(daily_returns[date])
        value *= 1 + day_ret
        peak = max(peak, value)
        dd = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        equity.append({"date": date, "equity": value})

    return {
        "trades": len(trades),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "equity_curve": equity,
    }


def compute_risk_metrics(alerts: List[dict], df: pd.DataFrame, label_col: str) -> Dict[str, object]:
    actual_low = df[df[label_col] == 1]
    actual_pairs = set(zip(actual_low["trade_date"].dt.strftime("%Y-%m-%d"), actual_low["code"]))
    alert_pairs = set((a["trade_date"], a["code"]) for a in alerts)
    hits = sum(1 for pair in actual_pairs if pair in alert_pairs)
    recall = hits / len(actual_pairs) if actual_pairs else 0.0
    precision = hits / len(alert_pairs) if alert_pairs else 0.0
    return {
        "risk_alerts": len(alerts),
        "risk_true_events": len(actual_pairs),
        "risk_recall": recall,
        "risk_precision": precision,
    }


def save_outputs(trades: List[dict], alerts: List[dict], metrics: Dict[str, object]) -> None:
    REPORT_SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(REPORT_SIGNALS, index=False)
    pd.DataFrame(alerts).to_csv(REPORT_RISK, index=False)
    REPORT_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent tuning runner")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--risk-threshold", type=float, default=0.65)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--liq-min", type=float, default=0.0)
    parser.add_argument("--industry-cap", type=float, default=0.0)
    parser.add_argument("--price-weight", type=float, default=0.6)
    parser.add_argument("--liq-weight", type=float, default=0.3)
    parser.add_argument("--risk-weight", type=float, default=-0.1)
    parser.add_argument("--mid-pm-slope", action="store_true")
    parser.add_argument("--gap-residual", action="store_true")
    parser.add_argument("--dryness-threshold", type=float, default=0.0)
    parser.add_argument("--adv-threshold", type=float, default=0.0)
    parser.add_argument("--adv-bonus", type=float, default=0.0)
    parser.add_argument("--low-liq-threshold", type=float, default=0.0)
    parser.add_argument("--low-liq-per-industry", type=int, default=0)
    parser.add_argument("--low-gap-override", type=float, default=None)
    parser.add_argument("--mainflow-window", type=int, default=3)
    parser.add_argument("--mainflow-min-sum", type=float, default=None)
    parser.add_argument("--require-mainflow-nonneg", action="store_true")
    parser.add_argument("--fakeflow-filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        dataset=Path(args.dataset),
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        risk_threshold=args.risk_threshold,
        warmup_days=args.warmup,
        liq_min=args.liq_min,
        industry_cap=(args.industry_cap if args.industry_cap > 0 else None),
        price_weight=args.price_weight,
        liq_weight=args.liq_weight,
        risk_weight=args.risk_weight,
        mid_pm_slope_gate=args.mid_pm_slope,
        gap_residual_adjust=args.gap_residual,
        dryness_threshold=args.dryness_threshold,
        adv_threshold=args.adv_threshold,
        adv_bonus=args.adv_bonus,
        low_liq_threshold=args.low_liq_threshold,
        low_liq_per_industry=args.low_liq_per_industry,
        low_gap_override=args.low_gap_override,
        mainflow_window=args.mainflow_window,
        mainflow_min_sum=args.mainflow_min_sum,
        require_mainflow_nonneg=args.require_mainflow_nonneg,
        filter_fake_flow=args.fakeflow_filter,
    )
    df = load_dataset(cfg.dataset)
    df = augment_with_moneyflow(df, cfg.mainflow_window)
    metrics = run_backtest(df, cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
