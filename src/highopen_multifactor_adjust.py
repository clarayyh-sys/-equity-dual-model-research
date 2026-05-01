from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tushare as ts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import second_start_signal as ss
from tushare_utils import get_tushare_token


def load_snapshot(prev_date: str, root: Path) -> pd.DataFrame:
    path = root / "data" / "intraday" / prev_date / "rt_k_snapshot.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df


def fetch_moneyflow(trade_date: str, ts_codes: List[str]) -> pd.DataFrame:
    return ss.fetch_moneyflow(trade_date, ts_codes, chunk_size=20)


def load_today_open(trade_date: str, root: Path, ts_codes: List[str]) -> pd.DataFrame:
    csv_path = root / "data" / "processed" / f"highopen_{trade_date_minus_one(trade_date)}_to_{trade_date}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        return df[["ts_code", "code", "open_today", "pre_close_today", "open_gap_today"]]

    ts.set_token(get_tushare_token())
    pro = ts.pro_api()
    chunks = [ts_codes[i : i + 40] for i in range(0, len(ts_codes), 40)]
    frames = []
    for chunk in chunks:
        df = pro.query("rt_k", ts_code=",".join(chunk))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["ts_code"] = data["ts_code"].astype(str)
    data["code"] = data["ts_code"].str[:6]
    data = data.rename(
        columns={
            "open": "open_today",
            "pre_close": "pre_close_today",
        }
    )
    data["open_gap_today"] = data["open_today"] / data["pre_close_today"] - 1
    return data[["ts_code", "code", "open_today", "pre_close_today", "open_gap_today"]]


def trade_date_minus_one(trade_date: str) -> str:
    from datetime import datetime, timedelta

    ts.set_token(get_tushare_token())
    pro = ts.pro_api()
    end = datetime.strptime(trade_date, "%Y%m%d")
    start = end - timedelta(days=10)
    cal = pro.trade_cal(
        start_date=start.strftime("%Y%m%d"),
        end_date=(end - timedelta(days=1)).strftime("%Y%m%d"),
        is_open=1,
    )
    if cal is not None and not cal.empty:
        return cal.sort_values("cal_date").iloc[-1]["cal_date"]
    # fallback: simple subtraction for weekdays
    d = end - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime("%Y%m%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="High-open multi-factor adjustment")
    parser.add_argument("--trade-date", default="20260310")
    parser.add_argument("--prev-date", default="20260309")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    snapshot = load_snapshot(args.prev_date, root)
    ts_codes = snapshot["ts_code"].dropna().unique().tolist()

    moneyflow = fetch_moneyflow(args.prev_date, ts_codes)
    merged = snapshot.merge(moneyflow, on="ts_code", how="left")

    factors = ss.compute_factors(merged)
    rename_cols = {
        "pre_close": "pre_close_prev",
        "high": "high_prev",
        "open": "open_prev",
        "low": "low_prev",
        "close": "close_prev",
        "vol": "vol_prev",
        "amount": "amount_prev",
    }
    factors = factors.rename(columns=rename_cols)

    today_open = load_today_open(args.trade_date, root, ts_codes)
    df = factors.merge(today_open, on=["ts_code", "code"], how="inner")

    # feature blocks
    df["turnover_proxy"] = df["amount_prev"] / (df["pre_close_prev"] * 1e6)
    feature_cols = [
        "ret_pct",
        "amp_pct",
        "pullback_pct",
        "close_pos",
        "turnover_proxy",
        "vol_prev",
        "amount_prev",
        "net_main_inflow_ratio",
    ]
    feat_df = df[feature_cols].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df)

    # regression to open gap
    y = df["open_gap_today"].fillna(0)
    reg = LinearRegression()
    reg.fit(X, y)
    df["pred_open_gap"] = reg.predict(X)

    # agent scores
    idx = {col: i for i, col in enumerate(feature_cols)}
    X_df = pd.DataFrame(X, columns=feature_cols)
    df["agent_price"] = X_df[["ret_pct", "amp_pct", "pullback_pct", "close_pos"]].mean(axis=1)
    df["agent_liquidity"] = X_df[["turnover_proxy", "vol_prev", "amount_prev"]].mean(axis=1)
    df["agent_flow"] = X_df[["net_main_inflow_ratio"]].mean(axis=1)

    df["agent_score"] = (
        0.5 * df["agent_price"]
        + 0.3 * df["agent_liquidity"]
        + 0.2 * df["agent_flow"]
    )

    df.sort_values("agent_score", ascending=False, inplace=True)

    out_cols = [
        "code",
        "name",
        "open_today",
        "pre_close_today",
        "open_gap_today",
        "pred_open_gap",
        "agent_price",
        "agent_liquidity",
        "agent_flow",
        "agent_score",
    ]
    out_path = root / "reports" / f"highopen_3agent_adjust_{args.trade_date}.csv"
    df[out_cols].to_csv(out_path, index=False)
    summary_path = root / "reports" / f"highopen_3agent_adjust_{args.trade_date}.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"# 3-Agent 高开调整 — {args.prev_date} → {args.trade_date}\n\n")
        f.write(f"样本数: {len(df)}\n\n")
        f.write("## Top10 (按 agent_score 排序)\n\n")
        f.write("code | name | open_gap | pred_gap | agent_score\n")
        f.write("---|---|---:|---:|---:\n")
        for _, row in df.head(10).iterrows():
            f.write(
                f"{row['code']} | {row.get('name','')} | {row['open_gap_today']:.4f} | "
                f"{row['pred_open_gap']:.4f} | {row['agent_score']:.3f}\n"
            )
        f.write("\n特征权重 (LinearRegression 系数):\n\n")
        for col, coef in zip(feature_cols, reg.coef_):
            f.write(f"- {col}: {coef:.4f}\n")

    print(f"Saved scores: {out_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
