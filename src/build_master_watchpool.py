from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

from tdx_watchpool import read_tdx_watchpool_xls_tsv

# Rule: each stock stays in master pool for 20 trading days since it is ADDED (new vs previous day).
WINDOW = 20


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "tdx_export"
    files = sorted(raw_dir.glob("观察池*.xls"))
    if not files:
        raise SystemExit(f"No watchpool exports found: {raw_dir}")

    # date -> codes
    daily = {}
    for p in files:
        rows = read_tdx_watchpool_xls_tsv(p)
        if not rows or rows[0].get("date") is None:
            continue
        d = rows[0]["date"].isoformat()
        daily[d] = {r["code"] for r in rows if r.get("code")}

    dates = sorted(daily.keys())
    if not dates:
        raise SystemExit("No dated rows parsed")

    # state: code -> join_index
    join_idx: dict[str, int] = {}
    join_date: dict[str, str] = {}
    last_seen: dict[str, str] = {}

    prev_codes: set[str] = set()
    for i, d in enumerate(dates):
        codes = daily[d]
        new_codes = codes - prev_codes

        # expire and re-add logic: if code existed long ago and expired, treat as new when it reappears.
        for c in new_codes:
            old_i = join_idx.get(c)
            if old_i is None:
                join_idx[c] = i
                join_date[c] = d
            else:
                # if previously expired before today, reset join
                if i > old_i + (WINDOW - 1):
                    join_idx[c] = i
                    join_date[c] = d

        for c in codes:
            last_seen[c] = d

        prev_codes = codes

    current_i = len(dates) - 1
    current_date = dates[current_i]

    active_codes = []
    for c, j in join_idx.items():
        expire_i = j + (WINDOW - 1)
        if current_i <= expire_i:
            expire_date = dates[expire_i] if expire_i < len(dates) else None
            active_codes.append(
                {
                    "code": c,
                    "join_date": join_date.get(c),
                    "last_seen": last_seen.get(c),
                    "expire_date_est": expire_date,
                    "days_left_est": (expire_i - current_i) if expire_i >= current_i else -1,
                }
            )

    active_codes.sort(key=lambda x: (x["join_date"] or "", x["code"]))

    out_master = root / "data" / "processed" / "master_watchpool_current.csv"
    out_master.parent.mkdir(parents=True, exist_ok=True)
    with out_master.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["code", "join_date", "last_seen", "expire_date_est", "days_left_est"],
        )
        w.writeheader()
        w.writerows(active_codes)

    out_codes = root / "data" / "processed" / "master_watchpool_codes.txt"
    out_codes.write_text("\n".join([r["code"] for r in active_codes]) + "\n", encoding="utf-8")

    print(f"Trading days loaded: {len(dates)} ({dates[0]} -> {dates[-1]})")
    print(f"Current date: {current_date}")
    print(f"Master watchpool (20 trading days window) size: {len(active_codes)}")
    print(f"Wrote: {out_master}")
    print(f"Wrote: {out_codes}")


if __name__ == "__main__":
    main()
