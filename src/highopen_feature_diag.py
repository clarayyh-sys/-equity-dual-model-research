"""
特征诊断：在训练集 (03-02~03-23) 上分析每个特征的价值。
- LR 归一化系数（绝对值大 = 模型依赖）
- 单变量 AUC（点二列相关近似）
- 互相关（发现共线冗余）
合规：只用训练集，hold-out 不碰。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import highopen_three_agent_pipeline as pipe

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "highopen_multiday_dataset.csv"
TRAIN_END = "2026-03-23"


def agent_diag(name, features, X_df, y, C=3.0):
    X = X_df[features].fillna(0).to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=C).fit(Xs, y)
    coefs = lr.coef_[0]

    rows = []
    for i, f in enumerate(features):
        col = X_df[f].fillna(0).to_numpy()
        try:
            auc = roc_auc_score(y, col)
            auc = max(auc, 1 - auc)  # 方向无关
        except Exception:
            auc = 0.5
        rows.append({
            "feature": f,
            "coef_abs": abs(coefs[i]),
            "coef": coefs[i],
            "univar_auc": auc,
        })
    diag = pd.DataFrame(rows).sort_values("coef_abs", ascending=False)
    print(f"\n=== {name} Agent ({len(features)} features, label positives={int(y.sum())}/{len(y)}) ===")
    print(diag.to_string(index=False, formatters={
        "coef_abs": "{:.4f}".format, "coef": "{:+.4f}".format, "univar_auc": "{:.3f}".format,
    }))

    # 冗余检测
    corr = X_df[features].fillna(0).corr().abs()
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr.iloc[i, j]
            if c > 0.90:
                pairs.append((features[i], features[j], c))
    if pairs:
        print(f"\n  High-correlation pairs (|r|>0.90):")
        for a, b, c in sorted(pairs, key=lambda x: -x[2]):
            print(f"    {a:<25} <-> {b:<25}  r={c:.3f}")
    return diag, pairs


def main():
    df = pipe.load_dataset(DATASET_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    train = df[df["trade_date"] <= pd.Timestamp(TRAIN_END)].copy()
    print(f"Train: {len(train)} rows, label_high_open_10m positives={int(train['label_high_open_10m'].sum())}")

    y_high = train["label_high_open_10m"].to_numpy()
    y_low = train["label_low_open"].to_numpy()

    agent_diag("Price", list(pipe.PRICE_FEATURES), train, y_high)
    agent_diag("Liquidity", list(pipe.LIQUIDITY_FEATURES), train, y_high)
    agent_diag("Risk", list(pipe.RISK_FEATURES), train, y_low)


if __name__ == "__main__":
    main()
