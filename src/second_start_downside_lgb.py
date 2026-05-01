"""LightGBM downside detector for 一号 soft_rerank gray rollout.

Trains on label_down_start_5d using 22 selected features (see config/soft_rerank.json).
Designed to be called from src/second_start_restart_predict.py at inference time.

Public API:
  predict_downside_prob(train_df, today_df, features) -> np.ndarray of probs in [0,1]

Fallback safety: any failure in this module should raise an exception so the caller
can downgrade to BASE_12d behavior. NEVER silently return zeros or arbitrary defaults.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_LGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

LABEL_COL = "label_down_start_5d"


class DownsideModelError(RuntimeError):
    pass


def predict_downside_prob(
    train_df: pd.DataFrame,
    today_df: pd.DataFrame,
    features: Sequence[str],
    params: dict | None = None,
) -> np.ndarray:
    """Train LGB on train_df, predict probability of label_down_start_5d on today_df.

    Args:
        train_df: must contain `label_down_start_5d` and all `features`.
        today_df: must contain all `features`. Length must match returned array.
        features: 22-feature whitelist from config.
        params: optional LGB hyperparams override.

    Returns:
        np.ndarray shape (len(today_df),) with values in [0, 1].

    Raises:
        DownsideModelError: any failure (including NaN ratio > 0.20 in output,
                            constant predictions, etc.). Caller should fallback.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise DownsideModelError(f"lightgbm not installed: {e}")

    if LABEL_COL not in train_df.columns:
        raise DownsideModelError(f"missing label column {LABEL_COL} in train_df")

    missing_train = [f for f in features if f not in train_df.columns]
    missing_today = [f for f in features if f not in today_df.columns]
    if missing_train or missing_today:
        raise DownsideModelError(
            f"missing features: train={missing_train} today={missing_today}"
        )

    train_clean = train_df.dropna(subset=[LABEL_COL]).copy()
    if len(train_clean) < 100:
        raise DownsideModelError(f"insufficient labeled train rows: {len(train_clean)}")

    y_train = train_clean[LABEL_COL].astype(int).values
    if y_train.sum() < 30 or (len(y_train) - y_train.sum()) < 30:
        raise DownsideModelError(
            f"degenerate class balance: pos={y_train.sum()}, neg={len(y_train)-y_train.sum()}"
        )

    X_train_raw = train_clean[list(features)].values
    X_today_raw = today_df[list(features)].values

    imp = SimpleImputer(strategy="median").fit(X_train_raw)
    X_train_imp = imp.transform(X_train_raw)
    X_today_imp = imp.transform(X_today_raw)

    sc = StandardScaler().fit(X_train_imp)
    X_train = sc.transform(X_train_imp)
    X_today = sc.transform(X_today_imp)

    use_params = {**DEFAULT_LGB_PARAMS, **(params or {})}
    model = lgb.LGBMClassifier(**use_params)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_today)[:, 1]

    nan_ratio = np.isnan(proba).mean()
    if nan_ratio > 0.20:
        raise DownsideModelError(f"NaN ratio in proba = {nan_ratio:.3f} > 0.20")
    if np.nanstd(proba) < 1e-6:
        raise DownsideModelError(
            f"degenerate output: std={np.nanstd(proba):.2e} (constant predictions)"
        )

    return proba
