from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from statistics import mean

THRESH = 0.03  # >3% high-open


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


def load_csv(path: pathlib.Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    root = pathlib.Path(__file__).resolve().parents[1]

    hist = load_csv(root / "data" / "processed" / "watchpool_history.csv")
    master = load_csv(root / "data" / "processed" / "master_watchpool_current.csv")
    master_codes = {r["code"].strip() for r in master if r.get("code")}

    byd = defaultdict(dict)
    dates = set()
    for r in hist:
        d = r.get("date")
        c = r.get("code")
        if not d or not c or c not in master_codes:
            continue
        dates.add(d)
        byd[d][c] = r

    dates = sorted(dates)

    rows = []
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
                "y_highopen": y_cls,
            }
            rec.update(features(r0))
            rows.append(rec)

    out_csv = root / "reports" / "highopen_dataset_masterpool_gt3pct.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    ys = [r["y_next_open_gap"] for r in rows if r.get("y_next_open_gap") is not None]
    p_pos = mean([y > THRESH for y in ys]) if ys else 0

    out_md = root / "reports" / "highopen_analysis_masterpool.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# 总池（20交易日窗口）高开分析（>3%）\n\n")
        f.write(f"总池股票数：{len(master_codes)}\n\n")
        f.write(f"可用样本对（date_t->date_t1）：{len(rows)}\n\n")
        f.write(f"P(次日高开>3%)：{p_pos:.3f}\n\n")
        f.write("说明：样本来自你导出的观察池快照（按导出日期串联），缺少的交易日不会被自动补齐。\n")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
