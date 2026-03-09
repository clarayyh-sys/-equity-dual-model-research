from __future__ import annotations

import csv
import pathlib
from typing import Any

from tdx_watchpool import read_tdx_watchpool_xls_tsv


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k) for k in fieldnames}
            # dates as iso
            if rr.get("date") is not None:
                rr["date"] = rr["date"].isoformat()
            w.writerow(rr)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "tdx_export"
    files = sorted(raw_dir.glob("观察池*.xls"))
    if not files:
        raise SystemExit(f"No files found in {raw_dir}")

    rows: list[dict[str, Any]] = []
    for p in files:
        rows.extend(read_tdx_watchpool_xls_tsv(p))

    # stable column order
    cols = [
        "date",
        "code",
        "name",
        "pct_change",
        "last",
        "open",
        "high",
        "low",
        "prev_close",
        "turnover_pct",
        "amount",
        "volume",
        "volume_cur",
        "speed_pct",
        "pe_ttm",
        "vol_ratio",
        "industry",
    ]

    out_hist = root / "data" / "processed" / "watchpool_history.csv"
    write_csv(out_hist, rows, cols)

    # latest snapshot (by filename date inside content)
    rows_sorted = sorted([r for r in rows if r.get("date") is not None], key=lambda x: (x["date"], x["code"]))
    latest_date = rows_sorted[-1]["date"]
    latest = [r for r in rows_sorted if r["date"] == latest_date]
    out_latest = root / "data" / "processed" / "watchpool_latest.csv"
    write_csv(out_latest, latest, cols)

    print(f"Integrated {len(files)} files, {len(rows)} rows")
    print(f"Latest date: {latest_date}, rows: {len(latest)}")
    print(f"Wrote: {out_hist}")
    print(f"Wrote: {out_latest}")


if __name__ == "__main__":
    main()
