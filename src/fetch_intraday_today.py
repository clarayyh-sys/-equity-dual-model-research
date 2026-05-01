from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import time

import pandas as pd
import tushare as ts

from tushare_utils import get_tushare_token, to_ts_code

CHUNK = 40  # number of codes per rt_k call (API allows comma-separated list)


def load_pool(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.strip().str.zfill(6)
    df = df[df["code"].str.fullmatch(r"\d{6}")].copy()
    df["ts_code"] = df["code"].apply(to_ts_code)
    return df


def chunk_codes(codes: list[str], size: int) -> list[list[str]]:
    return [codes[i : i + size] for i in range(0, len(codes), size)]


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    pool_path = root / "data" / "processed" / "master_watchpool_current.csv"
    if not pool_path.exists():
        raise SystemExit("Missing master_watchpool_current.csv; run build_master_watchpool first")

    pool = load_pool(pool_path)
    if pool.empty:
        raise SystemExit("No valid codes in master_watchpool_current.csv")

    ts_codes = pool["ts_code"].tolist()
    trade_date = dt.datetime.now().strftime("%Y%m%d")
    fetch_time = dt.datetime.now().isoformat(timespec="seconds")

    ts.set_token(get_tushare_token())
    pro = ts.pro_api()

    frames: list[pd.DataFrame] = []
    chunks = chunk_codes(ts_codes, CHUNK)
    for idx, group in enumerate(chunks, 1):
        query_codes = ",".join(group)
        try:
            df = pro.query("rt_k", ts_code=query_codes)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR chunk {idx}/{len(chunks)}: {exc}")
            time.sleep(0.5)
            continue
        if df is None or df.empty:
            print(f"WARN chunk {idx}/{len(chunks)}: empty result")
            time.sleep(0.2)
            continue
        df["fetch_time"] = fetch_time
        frames.append(df)
        time.sleep(0.2)  # courtesy pause

    if not frames:
        raise SystemExit("No rt_k data retrieved. Check permissions or params.")

    data = pd.concat(frames, ignore_index=True)
    merged = pool.merge(data, on="ts_code", how="left")

    out_dir = root / "data" / "intraday" / trade_date
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "rt_k_snapshot.csv"
    merged.to_csv(out_csv, index=False)

    summary = {
        "trade_date": trade_date,
        "fetch_time": fetch_time,
        "pool_size": int(len(pool)),
        "retrieved": int(data["ts_code"].nunique()),
        "missing": sorted(set(ts_codes) - set(data["ts_code"].unique())),
        "chunks": len(chunks),
        "output": str(out_csv.relative_to(root)),
    }
    (out_dir / "fetch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
