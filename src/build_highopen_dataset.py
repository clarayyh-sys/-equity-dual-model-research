from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

THRESH = 0.03  # >3% counts as high-open


def _f(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in {"", "None"}:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def features(r):
    last = _f(r.get("last"))
    open_ = _f(r.get("open"))
    high = _f(r.get("high"))
    low = _f(r.get("low"))
    pc = _f(r.get("prev_close"))

    day_ret = None if (last is None or pc in (None, 0)) else last / pc - 1
    open_gap = None if (open_ is None or pc in (None, 0)) else open_ / pc - 1
    amp = None
    if high is not None and low is not None and pc not in (None, 0):
        amp = (high - low) / pc

    pullback = None
    if high not in (None, 0) and last is not None:
        pullback = (high - last) / high

    spike = None
    if open_ not in (None, 0) and high is not None:
        spike = (high - open_) / open_

    close_pos = None
    if high is not None and low is not None and high != low and last is not None:
        close_pos = (last - low) / (high - low)

    return {
        "x_day_ret": day_ret,
        "x_open_gap": open_gap,
        "x_amp": amp,
        "x_pullback": pullback,
        "x_spike": spike,
        "x_close_pos": close_pos,
        "x_turnover_pct": _f(r.get("turnover_pct")),
        "x_vol_ratio": _f(r.get("vol_ratio")),
    }


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    hist_path = root / "data" / "processed" / "watchpool_history.csv"

    with hist_path.open("r", encoding="utf-8", newline="") as f:
        hist = list(csv.DictReader(f))

    byd = defaultdict(dict)
    dates = set()
    for r in hist:
        d = r.get("date")
        c = r.get("code")
        if not d or not c:
            continue
        dates.add(d)
        byd[d][c] = r

    dates = sorted(dates)

    out = []
    for d0, d1 in zip(dates, dates[1:]):
        m0, m1 = byd[d0], byd[d1]
        for code in sorted(m0.keys() & m1.keys()):
            r0, r1 = m0[code], m1[code]
            o1 = _f(r1.get("open"))
            pc1 = _f(r1.get("prev_close"))
            y_gap = None if (o1 is None or pc1 in (None, 0)) else o1 / pc1 - 1
            y_cls = None if y_gap is None else int(y_gap > THRESH)

            rec = {
                "date_t": d0,
                "date_t1": d1,
                "code": code,
                "name": r0.get("name"),
                "y_next_open_gap": y_gap,
                f"y_highopen_gt_{int(THRESH*100)}pct": y_cls,
            }
            rec.update(features(r0))
            out.append(rec)

    out_path = root / "reports" / f"highopen_dataset_allpairs_gt{int(THRESH*100)}pct.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = list(out[0].keys()) if out else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)

    pos = [r for r in out if r.get(f"y_highopen_gt_{int(THRESH*100)}pct") == "1" or r.get(f"y_highopen_gt_{int(THRESH*100)}pct") == 1]
    print(f"pairs: {len(out)}")
    print(f"positives (>{THRESH:.0%}): {len(pos)}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
