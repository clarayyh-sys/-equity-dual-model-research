from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import second_start_signal as ss
from tushare_utils import get_tushare_token
import tushare as ts

ROOT = Path(__file__).resolve().parents[1]


def load_snapshot(trade_date: str) -> pd.DataFrame:
    path = ROOT / "data" / "intraday" / trade_date / "rt_k_snapshot.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df


def fetch_moneyflow(trade_date: str, ts_codes: List[str]) -> pd.DataFrame:
    return ss.fetch_moneyflow(trade_date, ts_codes)


def load_watchpool_history() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "watchpool_history.csv"
    return pd.read_csv(path, dtype={"code": str})


def compute_daily_features(date: str, codes: List[str], hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist[hist["code"].isin(codes)].copy()
    hist["date"] = hist["date"].astype(str)
    hist.sort_values(["code", "date"], inplace=True)
    recent = hist[hist["date"] <= date]
    feats = []
    for code in codes:
        sub = recent[recent["code"] == code].tail(10)
        today = sub[sub["date"] == date]
        if today.empty:
            feats.append(
                {
                    "code": code,
                    "pct_change_prev": 0.0,
                    "pct_change_5d": 0.0,
                    "limit_up_count_5d": 0,
                    "vol_ratio_daily": 0.0,
                    "turnover_pct_prev": 0.0,
                }
            )
            continue
        last = today.iloc[-1]
        prev = sub[sub["date"] < date].tail(5)
        pct_change_prev = prev["pct_change"].iloc[-1] if not prev.empty else 0.0
        pct_change_5d = prev["pct_change"].mean() if not prev.empty else 0.0
        limit_up_count = int((prev["pct_change"] >= 9).sum())
        vol = last.get("volume", 0.0)
        avg_vol = prev["volume"].mean() if not prev.empty else vol
        vol_ratio = vol / avg_vol if avg_vol not in (0, None) else 0.0
        turnover = last.get("turnover_pct", 0.0)
        feats.append(
            {
                "code": code,
                "pct_change_prev": pct_change_prev,
                "pct_change_5d": pct_change_5d,
                "limit_up_count_5d": limit_up_count,
                "vol_ratio_daily": vol_ratio,
                "turnover_pct_prev": turnover,
            }
        )
    return pd.DataFrame(feats)


def build_training(prev_date: str, next_date: str, hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    snapshot = load_snapshot(prev_date)
    ts_codes = snapshot["ts_code"].dropna().unique().tolist()
    moneyflow = fetch_moneyflow(prev_date, ts_codes)
    factors = ss.compute_factors(snapshot.merge(moneyflow, on="ts_code", how="left"))
    factors = factors.rename(columns={"vol": "vol_prev", "amount": "amount_prev"})
    factors["turnover_proxy"] = factors["amount_prev"] / (factors["pre_close"] * 1e6)

    daily_feats = compute_daily_features(prev_date, snapshot["code"].tolist(), hist)
    df = factors.merge(daily_feats, on="code", how="left")

    label_path = ROOT / "data" / "processed" / f"highopen_{prev_date}_to_{next_date}.csv"
    label_df = pd.read_csv(label_path, dtype={"code": str})[["code", "open_gap_today"]]
    label_df.rename(columns={"open_gap_today": "label"}, inplace=True)
    df = df.merge(label_df, on="code", how="inner")
    df["target"] = (df["label"] > 0.02).astype(int)
    feature_cols = [
        "ret_pct",
        "amp_pct",
        "pullback_pct",
        "close_pos",
        "net_main_inflow_ratio",
        "turnover_proxy",
        "vol_prev",
        "amount_prev",
        "pct_change_prev",
        "pct_change_5d",
        "limit_up_count_5d",
        "vol_ratio_daily",
        "turnover_pct_prev",
    ]
    df = df.fillna(0)
    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols


def predict_for_date(prev_date: str, hist: pd.DataFrame, scaler: StandardScaler, model: LogisticRegression, feature_cols: List[str]) -> pd.DataFrame:
    snapshot = load_snapshot(prev_date)
    ts_codes = snapshot["ts_code"].dropna().unique().tolist()
    moneyflow = fetch_moneyflow(prev_date, ts_codes)
    factors = ss.compute_factors(snapshot.merge(moneyflow, on="ts_code", how="left"))
    factors = factors.rename(columns={"vol": "vol_prev", "amount": "amount_prev"})
    factors["turnover_proxy"] = factors["amount_prev"] / (factors["pre_close"] * 1e6)
    daily_feats = compute_daily_features(prev_date, snapshot["code"].tolist(), hist)
    df = factors.merge(daily_feats, on="code", how="left").fillna(0)
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    df["daily_score"] = model.predict_proba(X_scaled)[:, 1]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="High-open daily agent")
    parser.add_argument("--train-prev-date", default="20260309")
    parser.add_argument("--train-next-date", default="20260310")
    parser.add_argument("--predict-prev-date", default="20260310")
    args = parser.parse_args()

    hist = load_watchpool_history()
    X, y, feature_cols = build_training(args.train_prev_date, args.train_next_date, hist)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    df_pred = predict_for_date(args.predict_prev_date, hist, scaler, model, feature_cols)
    out_path = ROOT / "reports" / f"highopen_daily_agent_{args.predict_prev_date}_to_next.csv"
    df_pred.to_csv(out_path, index=False)

    model_path = ROOT / "reports" / "highopen_daily_agent_model.json"
    model_data = {
        "feature_cols": feature_cols,
        "coef": model.coef_[0].tolist(),
        "intercept": model.intercept_[0],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_prev": args.train_prev_date,
        "train_next": args.train_next_date,
    }
    model_path.write_text(json.dumps(model_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved daily agent predictions: {out_path}")
    print(f"Saved model params: {model_path}")


if __name__ == "__main__":
    main()
