from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import pathlib
import re
from typing import Any, Iterable


_DATE_RE = re.compile(r"(\d{8})")


def _parse_date_from_name(name: str) -> dt.date | None:
    m = _DATE_RE.search(name)
    if not m:
        return None
    s = m.group(1)
    return dt.date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def _clean_code(x: str) -> str:
    # examples: ="600028"  / 600028
    x = (x or "").strip()
    x = x.removeprefix("=")
    x = x.strip().strip('"').strip()
    return x


def _to_float(x: str) -> float | None:
    x = (x or "").strip()
    if x in {"--", ""}:
        return None
    x = x.replace(",", "")
    if x.endswith("%"):
        x = x[:-1]
    try:
        return float(x)
    except ValueError:
        return None


def read_tdx_watchpool_xls_tsv(path: str | pathlib.Path, encoding: str = "gbk") -> list[dict[str, Any]]:
    """Read TDX-exported watchpool file.

    Many TDX "*.xls" exports are actually GBK-encoded tab-separated text.

    Returns rows as dictionaries with normalized keys.
    """
    path = pathlib.Path(path)
    txt = path.read_bytes().decode(encoding, errors="replace")
    rows = list(csv.reader(txt.splitlines(), delimiter="\t"))
    if not rows:
        return []

    header_raw = [h.strip() for h in rows[0] if h.strip()]

    # Normalize headers like 名称(35) -> 名称
    header = [re.sub(r"\(\d+\)$", "", h) for h in header_raw]

    out: list[dict[str, Any]] = []
    file_date = _parse_date_from_name(path.name)

    for r in rows[1:]:
        if not any(cell.strip() for cell in r):
            continue
        r = r[: len(header)]
        rec = {header[i]: (r[i].strip() if i < len(r) else "") for i in range(len(header))}

        code = _clean_code(rec.get("代码", ""))
        if not code:
            continue

        # Standardize a subset of commonly used fields
        last = _to_float(rec.get("现价", ""))
        prev_close = _to_float(rec.get("昨收", ""))
        pct_change = _to_float(rec.get("涨幅%", ""))
        # Some exports have "涨幅%" as "--"; compute from 现价/昨收 when possible.
        if pct_change is None and last is not None and prev_close not in (None, 0):
            pct_change = (last / prev_close - 1) * 100

        out.append(
            {
                "date": file_date,
                "code": code,
                "name": rec.get("名称", ""),
                "pct_change": pct_change,
                "last": last,
                "open": _to_float(rec.get("今开", "")),
                "high": _to_float(rec.get("最高", "")),
                "low": _to_float(rec.get("最低", "")),
                "prev_close": prev_close,
                "turnover_pct": _to_float(rec.get("换手%", "")),
                "amount": _to_float(rec.get("总金额", "")),
                "volume": _to_float(rec.get("总量", "")),
                "volume_cur": _to_float(rec.get("现量", "")),
                "speed_pct": _to_float(rec.get("涨速%", "")),
                "pe_ttm": _to_float(rec.get("市盈(动)", "")),
                "vol_ratio": _to_float(rec.get("量比", "")),
                "industry": rec.get("细分行业", ""),
                "_raw": rec,
            }
        )

    return out


def iter_watchpool_files(paths: Iterable[str | pathlib.Path]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in paths:
        pth = pathlib.Path(p)
        if pth.is_dir():
            out.extend(sorted(pth.glob("观察池*.xls")))
        else:
            out.append(pth)
    return out
