# A-stock-equity-dual-model-research

A research project for short-horizon equity selection on the China A-share market,
combining two complementary supervised models with a strict out-of-sample
validation framework. **Code structure is public; trained parameters, factor
thresholds, training data, and live signals are kept private.**

---

## Research Question

Two short-horizon events on China A-share watchpools have known statistical edge
but are hard to capture with a single model:

1. **second start** — stocks rebounding after a pullback or
   consolidation, where an additional 5-day uptrend is likely.
2. **next-day high open** — stocks likely to gap up by ≥3% in the first
   ten minutes of the following trading session.

Single models on either event are noisy and overfit easily. This project asks:

> Can two specialised models, each tuned for one event, be combined via an
> ensemble layer to produce a strictly more robust signal than either alone,
> while satisfying production-grade out-of-sample (OOS) validation gates?

---

## Methodology

### Data architecture

- **Universe**: rolling 30-trading-day watchpool of ~220 names
- **Sources**: TuShare Pro (daily / 1min / moneyflow / Level-2 factor),
  Tongdaxin watchpool exports
- **Time slicing**: T-5 → T-3 → T-2 → T-1 → T-14:40 snapshot → T+1 label
- **Anti-leakage**: feature-source whitelist + temporal assert
  `train.trade_date.max() < today.trade_date.min()`

### Model 1 — Second Start

- BASE: `sklearn.linear_model.Ridge` over a small set of price/flow/breadth
  features, trained against forward-5-day return (clipped)
- Downside detector: `LightGBMClassifier` over a wider rolling-window feature
  set (5d realized vol/dd ranks, multi-day moneyflow & MFI/OBV/ATR)
- Soft rerank: `score = pred_ret_5d × (1 − downside_prob)`
- Optional Model-2 boost: `× (1 + α × highopen_picked)`
- Graceful fallback to BASE when LightGBM fails or feature coverage drops

### Model 2 — High Open

- Three-agent Logistic Regression (price / liquidity / risk),
  `class_weight=balanced`, per-agent `StandardScaler` fit on training data only
- Liquidity agent emphasises multi-day large-order net-ratio dynamics and a
  3-day large-order net-ratio standard deviation (the only liquidity feature
  to pass paired significance testing during Phase 4)
- Risk agent uses a heuristic floor:
  `risk = max(model_risk, max(drawdown / k1, −pct_change / k2))`
- Output: per-name `final_score`, with industry de-duplication and a top-K cut

### Joint layer

- **soft_rerank**: rank-rerank Model 1 candidates using its own downside
  detector, optionally scaled by Model 2's overlap
- **weighted_rerank**: cross-sectional pct-rank blend of Model 1 / Model 2
  scores, only active when Model 2's T+1 inference file is present

### Anti-overfit framework

Five hard rules (see `docs/model_tuning_rules.md` once published):

1. No look-ahead — feature dates strictly precede sample dates
2. Time-based train/val/test split — no random shuffles
3. Cross-validation lives only inside the training fold
4. Test set is touched once and only once for the final number
5. No partial-data runs — gaps are filled before any inference fires

Above the per-model gates, candidate strategy upgrades pass through:

- Multi-window paired tests (≥3 disjoint validation windows)
- An OOS Shadow framework: a candidate runs *byte-identically* alongside the
  current production line for ~10 trading days, evaluated against seven hard
  gates (signal-count parity, Δ avg-ret, Δ max-dd, paired win rate,
  loser-add ≤ loser-save, killed-winner ≤ found-winner + 1, single-window
  contribution share)
- 7/7 PASS → promote to gray; otherwise stay in shadow or roll back

---

## Repository Structure

```
src/                       core models and inference
  highopen_*               Model 2 (high-open three-agent)
  second_start_restart_*   Model 1 (second-start)
  build_master_watchpool   universe construction
  fetch_intraday_today     T-day snapshot fetcher
  tushare_utils            API wrapper (token via env var)
scripts/                   experiment runners, daily orchestrators,
                           data-patch utilities, OOS shadow harness
docs/                      methodology / data-pipeline / tuning rules
tests/                     unit + smoke tests
```

Files **deliberately excluded from this repository** (see `.gitignore`):

- All datasets (`data/`, any `*.csv` / `*.parquet` / `*.pkl`)
- All trained model artifacts (`artifacts/`, `models/`, `*.joblib`)
- All live signals and reports (`reports/`, `results/`, `logs/`)
- All strategy configurations carrying tuned parameters
  (`config/`, `configs/`, any `*.yaml` / `*.json`)
- Credentials (`.env`, anything matching `*token*` or `*secret*`)

The thresholds, hyperparameters, and trained weights that turn this code from
a generic skeleton into a working alpha live in those private artifacts. **For
research collaboration or recruiting evaluation, please contact the author for
the full configuration and reproduction package.**

---

## Tech Stack

`Python 3.12` · `pandas` · `numpy` · `scikit-learn` · `LightGBM` ·
`TuShare Pro` · bash + cron orchestration

---

## Contact

严予晗 · clara_yyh@outlook.com
