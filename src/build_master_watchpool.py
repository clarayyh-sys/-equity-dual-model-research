from __future__ import annotations

import csv
import datetime as dt
import pathlib
from collections import defaultdict
from typing import Dict, List, Set

import tushare as ts

from tdx_watchpool import read_tdx_watchpool_xls_tsv
from tushare_utils import get_tushare_token, to_ts_code

WINDOW = 30
WATCHPOOL_PATTERNS = (
    "观察池*.xls",
    "观察池*.xlsx",
    "临时条件股*.xls",
    "临时条件股*.xlsx",
)


def fetch_trade_days(start_date: str, end_date: str) -> List[str]:
    """Return sorted open trading days [start, end] in ISO format."""
    if not start_date or not end_date:
        return []
    ts.set_token(get_tushare_token())
    pro = ts.pro_api()
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    if cal is None or cal.empty:
        return []
    days: List[str] = []
    for _, row in cal.iterrows():
        if row.get("is_open") == 1:
            cal_date = str(row.get("cal_date"))
            try:
                date_obj = dt.datetime.strptime(cal_date, "%Y%m%d").date()
            except ValueError:
                continue
            days.append(date_obj.isoformat())
    days = sorted(days)
    return days


def load_tdx_exports(raw_dir: pathlib.Path) -> tuple[Dict[str, Set[str]], Dict[str, Dict[str, dict]]]:
    daily_codes: Dict[str, Set[str]] = {}
    daily_rows: Dict[str, Dict[str, dict]] = {}
    files: list[pathlib.Path] = []
    for pattern in WATCHPOOL_PATTERNS:
        files.extend(raw_dir.glob(pattern))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No watchpool exports found: {raw_dir}")
    for p in files:
        rows = read_tdx_watchpool_xls_tsv(p)
        if not rows:
            continue
        file_date = rows[0].get("date")
        if file_date is None:
            continue
        date_str = file_date.isoformat()
        codes: Set[str] = set()
        row_map: Dict[str, dict] = {}
        for r in rows:
            code = r.get("code")
            if not code:
                continue
            codes.add(code)
            row_map[code] = r
        if not codes:
            continue
        daily_codes[date_str] = codes
        daily_rows[date_str] = row_map
    if not daily_codes:
        raise SystemExit("No dated rows parsed from TDX exports")
    return daily_codes, daily_rows


def build_effective_history(
    trade_days: List[str],
    daily_codes: Dict[str, Set[str]],
    daily_rows: Dict[str, Dict[str, dict]],
) -> List[dict[str, object]]:
    states: Dict[str, dict] = {}
    rows: List[dict[str, object]] = []
    prev_observed_codes: Set[str] = set()

    for idx, day in enumerate(trade_days):
        observed_codes = daily_codes.get(day, set())
        if observed_codes:
            new_codes = observed_codes - prev_observed_codes
        else:
            new_codes = set()

        for code in new_codes:
            state = states.setdefault(code, {"join_round": 0})
            if state.get("first_join_idx") is None:
                state["first_join_idx"] = idx
                state["first_join_date"] = day
            state["current_join_idx"] = idx
            state["current_join_date"] = day
            state["last_join_idx"] = idx
            state["last_join_date"] = day
            state["active_until_idx"] = idx + (WINDOW - 1)
            state["join_round"] = state.get("join_round", 0) + 1

        for code in observed_codes:
            state = states.setdefault(code, {})
            state["last_seen_date"] = day
            row = daily_rows.get(day, {}).get(code)
            if row:
                if row.get("name"):
                    state["name"] = row["name"]
                if row.get("industry"):
                    state["industry"] = row.get("industry")

        # expire finished windows
        for state in states.values():
            active_until = state.get("active_until_idx")
            if active_until is not None and idx > active_until:
                state["current_join_idx"] = None
                state["current_join_date"] = None

        # emit active rows
        for code, state in states.items():
            current_idx = state.get("current_join_idx")
            active_until = state.get("active_until_idx")
            if current_idx is None or active_until is None or idx > active_until:
                continue
            window_idx = idx - current_idx
            rows.append(
                {
                    "trade_date": day,
                    "code": code,
                    "ts_code": to_ts_code(code),
                    "join_date": state.get("current_join_date"),
                    "first_join_date": state.get("first_join_date"),
                    "last_join_date": state.get("last_join_date"),
                    "window_day_index": window_idx,
                    "days_left_est": max(active_until - idx, 0),
                    "is_effective": 1,
                    "join_round": state.get("join_round"),
                    "last_seen_date": state.get("last_seen_date"),
                    "name": state.get("name"),
                    "industry": state.get("industry"),
                }
            )

        if observed_codes:
            prev_observed_codes = observed_codes

    rows.sort(key=lambda r: (r["trade_date"], r["code"]))
    return rows


def write_csv(path: pathlib.Path, rows: List[dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "tdx_export"
    daily_codes, daily_rows = load_tdx_exports(raw_dir)

    dates = sorted(daily_codes.keys())
    original_trading_days = len(dates)

    last_date = dt.date.fromisoformat(dates[-1])
    end_date = (last_date + dt.timedelta(days=5)).isoformat()
    trade_days = fetch_trade_days(dates[0], end_date)
    effective_rows = build_effective_history(trade_days, daily_codes, daily_rows)

    effective_path = root / "data" / "processed" / "watchpool_effective_history.csv"
    cols = [
        "trade_date",
        "code",
        "ts_code",
        "join_date",
        "first_join_date",
        "last_join_date",
        "window_day_index",
        "days_left_est",
        "is_effective",
        "join_round",
        "last_seen_date",
        "name",
        "industry",
    ]
    write_csv(effective_path, effective_rows, cols)

    # maintain old master outputs for compatibility
    first_join_date: dict[str, str] = {}
    last_join_date: dict[str, str] = {}
    last_join_idx: dict[str, int] = {}
    last_seen: dict[str, str] = {}
    prev_codes: Set[str] = set()
    for i, d in enumerate(dates):
        codes = daily_codes[d]
        new_codes = codes - prev_codes
        for c in new_codes:
            if c not in first_join_date:
                first_join_date[c] = d
            last_join_date[c] = d
            last_join_idx[c] = i
        for c in codes:
            last_seen[c] = d
        prev_codes = codes

    current_i = len(dates) - 1
    current_date = dates[current_i]

    state_rows = []
    for c in sorted(first_join_date.keys() | last_join_date.keys() | last_seen.keys()):
        j = last_join_idx.get(c)
        expire_i = (j + (WINDOW - 1)) if j is not None else None
        expire_date = dates[expire_i] if (expire_i is not None and expire_i < len(dates)) else None
        days_left = (expire_i - current_i) if (expire_i is not None and expire_i >= current_i) else None
        state_rows.append(
            {
                "code": c,
                "first_join_date": first_join_date.get(c),
                "last_join_date": last_join_date.get(c),
                "last_seen": last_seen.get(c),
                "expire_date_est": expire_date,
                "days_left_est": days_left,
            }
        )

    out_state = root / "data" / "processed" / "master_watchpool_state.csv"
    out_state.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_state,
        state_rows,
        ["code", "first_join_date", "last_join_date", "last_seen", "expire_date_est", "days_left_est"],
    )

    active_rows = []
    for c, j in last_join_idx.items():
        expire_i = j + (WINDOW - 1)
        if current_i <= expire_i:
            expire_date = dates[expire_i] if expire_i < len(dates) else None
            active_rows.append(
                {
                    "code": c,
                    "first_join_date": first_join_date.get(c),
                    "last_join_date": last_join_date.get(c),
                    "last_seen": last_seen.get(c),
                    "expire_date_est": expire_date,
                    "days_left_est": (expire_i - current_i),
                }
            )

    active_rows.sort(key=lambda x: (x["last_join_date"] or "", x["code"]))
    out_master = root / "data" / "processed" / "master_watchpool_current.csv"
    write_csv(
        out_master,
        active_rows,
        ["code", "first_join_date", "last_join_date", "last_seen", "expire_date_est", "days_left_est"],
    )

    out_codes = root / "data" / "processed" / "master_watchpool_codes.txt"
    out_codes.write_text("\n".join([r["code"] for r in active_rows]) + "\n", encoding="utf-8")

    effective_days = sorted({row["trade_date"] for row in effective_rows})
    print(f"TDX export days: {original_trading_days}")
    print(f"Effective trading days reconstructed: {len(effective_days)} ({effective_days[0]} -> {effective_days[-1]})")
    print(f"Effective membership rows: {len(effective_rows)} -> {effective_path}")
    print(f"Master watchpool size: {len(active_rows)} on {current_date}")


if __name__ == "__main__":
    main()
