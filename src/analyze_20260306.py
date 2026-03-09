from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass


def _to_float(x: str):
    if x is None:
        return None
    x = str(x).strip()
    if x in {"", "None"}:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_watchpool_history(path: pathlib.Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def idx(rows, date: str):
    return {r["code"]: r for r in rows if r.get("date") == date and r.get("code")}


def feat(prev):
    # compute features on previous day snapshot
    last = _to_float(prev.get("last"))
    open_ = _to_float(prev.get("open"))
    high = _to_float(prev.get("high"))
    low = _to_float(prev.get("low"))
    pc = _to_float(prev.get("prev_close"))

    def safe_div(a, b):
        if a is None or b in (None, 0):
            return None
        return a / b

    day_ret = None if (last is None or pc in (None, 0)) else last / pc - 1
    open_gap = None if (open_ is None or pc in (None, 0)) else open_ / pc - 1
    amp = None
    if high is not None and low is not None and pc not in (None, 0):
        amp = (high - low) / pc

    pullback_from_high = None
    if high not in (None, 0) and last is not None:
        pullback_from_high = (high - last) / high

    spike_from_open = None
    if open_ not in (None, 0) and high is not None:
        spike_from_open = (high - open_) / open_

    return {
        "prev_day_ret": day_ret,
        "prev_open_gap": open_gap,
        "prev_amp": amp,
        "prev_pullback_from_high": pullback_from_high,
        "prev_spike_from_open": spike_from_open,
        "prev_turnover_pct": _to_float(prev.get("turnover_pct")),
        "prev_vol_ratio": _to_float(prev.get("vol_ratio")),
    }


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    hist_path = root / "data" / "processed" / "watchpool_history.csv"
    rows = load_watchpool_history(hist_path)

    d_winner = "2026-03-06"
    d_prev = "2026-03-05"

    m6 = [r for r in rows if r.get("date") == d_winner]
    winners = [r for r in m6 if _to_float(r.get("pct_change")) is not None and _to_float(r.get("pct_change")) > 5]

    prev_map = idx(rows, d_prev)

    out_rows = []
    for w in winners:
        code = w["code"]
        prev = prev_map.get(code)
        rec = {
            "code": code,
            "name": w.get("name"),
            "d_20260306_pct": _to_float(w.get("pct_change")),
            "d_20260306_last": _to_float(w.get("last")),
        }
        if prev:
            rec.update(feat(prev))
            rec["d_20260305_last"] = _to_float(prev.get("last"))
            rec["d_20260305_pct"] = _to_float(prev.get("pct_change"))
        out_rows.append(rec)

    out_csv = root / "reports" / "analysis_20260306_winners_gt5.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "code",
        "name",
        "d_20260306_pct",
        "d_20260306_last",
        "d_20260305_pct",
        "d_20260305_last",
        "prev_day_ret",
        "prev_open_gap",
        "prev_amp",
        "prev_pullback_from_high",
        "prev_spike_from_open",
        "prev_turnover_pct",
        "prev_vol_ratio",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Simple markdown summary
    out_md = root / "reports" / "analysis_20260306_summary.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# 2026-03-06 涨幅>5%（观察池内）复盘\n\n")
        f.write(f"样本数：{len(winners)}\n\n")
        f.write("说明：这里的 3/5 指标来自 TDX 导出的当日快照（通常接近收盘，但以你的导出时间为准）。\n\n")
        f.write("输出明细：`analysis_20260306_winners_gt5.csv`\n")

    print(f"Winners (>{5}%): {len(winners)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
