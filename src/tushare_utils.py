from __future__ import annotations

import os


def get_tushare_token() -> str:
    tok = os.getenv("TUSHARE_TOKEN")
    if tok:
        return tok.strip()
    # fallback: read from .env if present (simple parse)
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("TUSHARE_TOKEN="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError("Missing TUSHARE_TOKEN (set env var or create .env in project root)")


def to_ts_code(code6: str) -> str:
    code6 = code6.strip()
    if len(code6) != 6 or not code6.isdigit():
        raise ValueError(f"Invalid 6-digit code: {code6}")
    if code6.startswith("6"):
        return f"{code6}.SH"
    if code6.startswith(("0", "3")):
        return f"{code6}.SZ"
    if code6.startswith(("4", "8")):
        return f"{code6}.BJ"
    # default to SZ
    return f"{code6}.SZ"
