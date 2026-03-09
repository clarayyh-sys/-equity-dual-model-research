from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

# Heuristic scoring for "冲高回落后二次启动" + next-day high-open tendency.
# NOTE: With very limited labeled data, this is a rule-based ranker (not a trained model).


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


def load_csv(path: pathlib.Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def score(feat: dict, staleness_penalty: float = 1.0) -> float | None:
    sp = feat.get("spike")
    pb = feat.get("pullback")
    amp = feat.get("amp")
    cp = feat.get("close_pos")
    vr = feat.get("vol_ratio")

    if sp is None or pb is None or amp is None or cp is None:
        return None

    # Prefer: big spike, meaningful pullback (fell from high), decent amplitude, close not at the very top.
    s = 0.7 * sp + 0.7 * pb + 0.3 * amp + 0.2 * (1 - cp)
    if vr is not None:
        s += 0.05 * vr

    return s * staleness_penalty


def main():
    root = pathlib.Path(__file__).resolve().parents[1]

    hist = load_csv(root / "data" / "processed" / "watchpool_history.csv")
    master = load_csv(root / "data" / "processed" / "master_watchpool_current.csv")

    # Build date index for staleness
    dates = sorted({r["date"] for r in hist if r.get("date")})
    date_idx = {d: i for i, d in enumerate(dates)}
    latest_date = dates[-1]

    # last row per (code) based on latest available date in history
    last_row = {}
    for r in hist:
        c = r.get("code")
        d = r.get("date")
        if not c or not d:
            continue
        if c not in last_row or date_idx[d] > date_idx[last_row[c]["date"]]:
            last_row[c] = r

    master_codes = [r["code"].strip() for r in master if r.get("code")]

    ranked = []
    for code in master_codes:
        r = last_row.get(code)
        if not r:
            continue
        d = r.get("date")
        f = features(r)

        # penalize staleness by export-date steps (not calendar days)
        steps_old = date_idx[latest_date] - date_idx.get(d, date_idx[latest_date])
        penalty = 1.0 / (1.0 + 0.6 * steps_old)

        s = score(f, staleness_penalty=penalty)
        if s is None:
            continue

        ranked.append(
            {
                "code": code,
                "name": r.get("name"),
                "last_seen_date": d,
                "score": s,
                "spike": f.get("spike"),
                "pullback": f.get("pullback"),
                "amp": f.get("amp"),
                "close_pos": f.get("close_pos"),
                "open_gap": f.get("open_gap"),
                "day_ret": f.get("day_ret"),
                "vol_ratio": f.get("vol_ratio"),
                "turnover_pct": f.get("turnover_pct"),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    top10 = ranked[:10]

    out_csv = root / "reports" / f"today_top10_hf_watch.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "code",
        "name",
        "last_seen_date",
        "score",
        "spike",
        "pullback",
        "amp",
        "close_pos",
        "open_gap",
        "day_ret",
        "vol_ratio",
        "turnover_pct",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(top10)

    out_md = root / "reports" / "today_top10_hf_watch.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# 今日重点高频观察 Top10（基于总池 + 冲高回落启发式打分）\n\n")
        f.write(f"最新导出日期：{latest_date}\n\n")
        f.write("说明：这是规则打分（非训练模型），用于从总池里挑“更像冲高回落后二次启动”的标的优先盯盘。\n\n")
        f.write("code | name | last_seen | score\n")
        f.write("---|---|---|---:\n")
        for r in top10:
            f.write(f"{r['code']} | {r.get('name','')} | {r['last_seen_date']} | {r['score']:.4f}\n")

    print(f"Latest date: {latest_date}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
