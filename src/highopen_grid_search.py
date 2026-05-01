"""
超参 grid search：在训练区间 (2026-03-02 ~ 2026-03-23) 内 walk-forward。
对 B0' (旧36维) 和 B1 (新52维) 各跑 4×3×3 = 36 组，按 final_equity 选优。
hold-out (03-24~03-27) 不动。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import highopen_three_agent_pipeline as pipe

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "highopen_multiday_dataset.csv"
TRAIN_END = "2026-03-23"
RESULT_PATH = Path(__file__).resolve().parents[1] / "reports" / "highopen_grid_results.csv"

PRICE_OLD = [
    "lag1_px_close", "lag1_px_high", "lag1_px_low",
    "snap_open", "snap_high", "snap_low", "snap_last", "snap_pct_chg",
    "lag2_open", "lag2_high", "lag2_low", "lag2_close", "lag2_pre_close", "lag2_pct_change",
]
LIQ_OLD = [
    "lag1_cum_vol", "lag1_cum_amount", "snap_volume", "snap_amount",
    "lag2_vol", "lag2_amount", "lag2_turnover_rate", "lag2_volume_ratio",
    "hist_mf_net_vol", "hist_mf_net_amount",
]
RISK_OLD = [
    "snap_pct_chg", "lag1_px_low", "lag1_px_high",
    "lag2_pct_change", "lag2_volume_ratio", "hist_mf_net_vol", "hist_mf_net_amount",
    "cum_drop_t_t2", "consecutive_drop", "severe_drop", "t_drop_abs", "t2_drop_abs",
]

PRICE_NEW = list(pipe.PRICE_FEATURES)
LIQ_NEW = list(pipe.LIQUIDITY_FEATURES)
RISK_NEW = list(pipe.RISK_FEATURES)

C_GRID = [0.1, 0.3, 1.0, 3.0]
SCORE_GRID = [0.40, 0.45, 0.50]
RISK_GRID = [0.50, 0.55, 0.60]


def set_features(price, liq, risk):
    pipe.PRICE_FEATURES[:] = price
    pipe.LIQUIDITY_FEATURES[:] = liq
    pipe.RISK_FEATURES[:] = risk


def run_one(df, price, liq, risk, C, score_th, risk_th):
    set_features(price, liq, risk)
    pipe.LR_C = C
    cfg = pipe.Config(
        dataset=DATASET_PATH,
        top_k=10,
        score_threshold=score_th,
        risk_threshold=risk_th,
        warmup_days=2,
    )
    metrics = pipe.run_backtest(df.copy(), cfg)
    eq = metrics.get("equity_curve", [])
    final_eq = eq[-1]["equity"] if eq else 1.0
    trades_df = pd.read_csv(pipe.REPORT_SIGNALS) if pipe.REPORT_SIGNALS.exists() else pd.DataFrame()
    n_trades = len(trades_df)
    if n_trades:
        hit = (trades_df["return_pct"] >= 0.03).mean()
        avg_ret = trades_df["return_pct"].mean()
    else:
        hit = avg_ret = 0.0
    return {
        "trades": n_trades,
        "hit_rate": hit,
        "avg_return": avg_ret,
        "final_equity": final_eq,
        "max_drawdown": metrics.get("max_drawdown", 0.0),
    }


def main():
    df = pipe.load_dataset(DATASET_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    train = df[df["trade_date"] <= pd.Timestamp(TRAIN_END)].copy()
    print(f"Train rows: {len(train)}, dates: {train['trade_date'].nunique()}")

    rows = []
    variants = [
        ("B0prime", PRICE_OLD, LIQ_OLD, RISK_OLD),
        ("B1",      PRICE_NEW, LIQ_NEW, RISK_NEW),
    ]
    total = len(variants) * len(C_GRID) * len(SCORE_GRID) * len(RISK_GRID)
    n = 0
    for vname, price, liq, risk in variants:
        for C, sth, rth in itertools.product(C_GRID, SCORE_GRID, RISK_GRID):
            n += 1
            r = run_one(train, price, liq, risk, C, sth, rth)
            row = {"variant": vname, "C": C, "score_th": sth, "risk_th": rth, **r}
            rows.append(row)
            print(f"[{n:>3}/{total}] {vname} C={C:<4} sth={sth} rth={rth} | "
                  f"trades={r['trades']:>3} hit={r['hit_rate']*100:>5.1f}% "
                  f"avg={r['avg_return']*100:>+5.2f}% eq={r['final_equity']:.4f}")

    res = pd.DataFrame(rows)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULT_PATH, index=False)

    print("\n" + "=" * 70)
    print("TOP 10 by final_equity (across all variants)")
    print("=" * 70)
    top = res.sort_values("final_equity", ascending=False).head(10)
    print(top.to_string(index=False))

    print("\n" + "=" * 70)
    print("BEST per variant")
    print("=" * 70)
    for v in res["variant"].unique():
        sub = res[res["variant"] == v]
        best = sub.sort_values("final_equity", ascending=False).iloc[0]
        print(f"\n{v}:")
        print(f"  C={best['C']}, score_th={best['score_th']}, risk_th={best['risk_th']}")
        print(f"  trades={int(best['trades'])}, hit={best['hit_rate']*100:.1f}%, "
              f"avg={best['avg_return']*100:+.2f}%, eq={best['final_equity']:.4f}, "
              f"max_dd={best['max_drawdown']*100:.2f}%")

    print(f"\nResults saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
