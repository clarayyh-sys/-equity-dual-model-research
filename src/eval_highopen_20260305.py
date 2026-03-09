from __future__ import annotations

import csv
import pathlib
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


def load_csv(p: pathlib.Path):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    root = pathlib.Path(__file__).resolve().parents[1]

    hist = load_csv(root / "data" / "processed" / "watchpool_history.csv")
    d = [r for r in hist if r.get("date") == "2026-03-05"]

    # Actual open gap on 3/5 from TDX snapshot: (open/prev_close - 1)
    actual = {}
    for r in d:
        code = r.get("code")
        o = _f(r.get("open"))
        pc = _f(r.get("prev_close"))
        if not code or o is None or pc in (None, 0):
            continue
        actual[code] = o / pc - 1

    pred_path = root / "data" / "raw" / "highopen_pred_20260304.csv"
    if not pred_path.exists():
        raise SystemExit(
            "Missing prediction file: data/raw/highopen_pred_20260304.csv\n"
            "Please provide a CSV with columns: code,pred_open_gap (predicted next-day open gap)."
        )

    pred_rows = load_csv(pred_path)
    preds = {r["code"].strip(): _f(r.get("pred_open_gap")) for r in pred_rows if r.get("code")}

    pairs = [(preds[c], actual[c]) for c in preds.keys() & actual.keys() if preds[c] is not None]
    if not pairs:
        raise SystemExit("No overlap between predictions and watchpool actuals on 2026-03-05")

    yhat, y = zip(*pairs)

    # Basic metrics
    mae = mean(abs(a - b) for a, b in pairs)

    # Directional hit rate: predicted positive vs actual positive
    hit = mean(((a > 0) == (b > 0)) for a, b in pairs)

    out_md = root / "reports" / "eval_highopen_20260305.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# 2026-03-05 高开预测评估（基于 2026-03-04 的预测）\n\n")
        f.write(f"样本数（预测∩观察池）：{len(pairs)}\n\n")
        f.write(f"MAE（pred_open_gap vs actual_open_gap）：{mae:.4f}\n\n")
        f.write(f"方向命中率（预测正/负 vs 实际正/负）：{hit:.3f}\n\n")
        f.write("说明：实际值来自 TDX 导出快照的今开/昨收计算，导出时间会影响一致性。\n")

    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
