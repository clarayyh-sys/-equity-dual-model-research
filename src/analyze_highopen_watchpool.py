from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from statistics import mean


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


def load_hist(path: pathlib.Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def by_date(rows):
    m = defaultdict(list)
    for r in rows:
        d = r.get("date")
        if d:
            m[d].append(r)
    return dict(m)


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
        "day_ret": day_ret,
        "open_gap": open_gap,
        "amp": amp,
        "pullback": pullback,
        "spike": spike,
        "close_pos": close_pos,
        "turnover_pct": _f(r.get("turnover_pct")),
        "vol_ratio": _f(r.get("vol_ratio")),
    }


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    hist = load_hist(root / "data" / "processed" / "watchpool_history.csv")
    m = by_date(hist)

    # focus: 3/4 -> 3/5 high-open
    d0, d1 = "2026-03-04", "2026-03-05"
    r0 = {r["code"]: r for r in m.get(d0, []) if r.get("code")}
    r1 = {r["code"]: r for r in m.get(d1, []) if r.get("code")}

    overlap = sorted(r0.keys() & r1.keys())
    if not overlap:
        raise SystemExit(f"No overlap between {d0} and {d1}")

    rows = []
    for code in overlap:
        a = r0[code]
        b = r1[code]
        # label: next-day open gap
        b_open = _f(b.get("open"))
        b_pc = _f(b.get("prev_close"))
        y = None if (b_open is None or b_pc in (None, 0)) else b_open / b_pc - 1
        rec = {
            "code": code,
            "name": a.get("name"),
            "y_next_open_gap": y,
        }
        rec.update({f"x_{k}": v for k, v in features(a).items()})
        rows.append(rec)

    # summary stats
    ys = [r["y_next_open_gap"] for r in rows if r["y_next_open_gap"] is not None]

    thresholds = [0.0, 0.01, 0.02, 0.03]
    hit = {t: mean([y > t for y in ys]) if ys else 0 for t in thresholds}

    # heuristic: "冲高回落" score from d0
    # spike high, pullback large, close not too weak
    def score(r):
        sp = r.get("x_spike")
        pb = r.get("x_pullback")
        cp = r.get("x_close_pos")
        if sp is None or pb is None or cp is None:
            return None
        return 0.6 * sp + 0.6 * pb + 0.2 * (1 - cp)

    scored = [(score(r), r) for r in rows]
    scored = [(s, r) for s, r in scored if s is not None and r.get("y_next_open_gap") is not None]
    scored.sort(key=lambda x: x[0], reverse=True)

    out_csv = root / "reports" / "highopen_dataset_20260304_to_20260305.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["code", "name", "y_next_open_gap"] + [k for k in rows[0].keys() if k.startswith("x_")]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_md = root / "reports" / "highopen_analysis_20260305.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# 观察池：{d0} -> {d1} 次日高开分析\n\n")
        f.write(f"样本数（两日都有数据）：{len(rows)}\n\n")
        f.write("## 次日开盘缺口分布\n")
        for t in thresholds:
            f.write(f"- P(y_next_open_gap > {t:.0%}) = {hit[t]:.3f}\n")
        f.write("\n")
        if ys:
            f.write(f"- 平均 y_next_open_gap：{mean(ys):.4f}\n\n")

        f.write("## 冲高回落候选（按启发式 score 排序，Top10）\n")
        f.write("说明：score=0.6*spike+0.6*pullback+0.2*(1-close_pos)，用于找“冲高后回落”的形态，不等于最终模型。\n\n")
        f.write("code | name | score | next_open_gap\n")
        f.write("---|---|---:|---:\n")
        for s, r in scored[:10]:
            f.write(f"{r['code']} | {r.get('name','')} | {s:.4f} | {r['y_next_open_gap']:.4f}\n")

        f.write("\n输出数据集：`highopen_dataset_20260304_to_20260305.csv`\n")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
