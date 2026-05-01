"""One-click daily pipeline.

Steps:
1) Sync new TDX exports from /mnt/c/tdx_export -> data/raw/tdx_export
2) Integrate watchpool history
3) Build master pool with 30 trading-day retention (added-since-prev-day)
4) Re-run high-open analysis (master pool, >3%)
5) Pick today's Top10 HF watch list (heuristic)

Usage:
  python3 run_daily_watchpool.py
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess


def run(cmd: list[str], cwd: pathlib.Path):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def main():
    root = pathlib.Path(__file__).resolve().parent

    src_dir = pathlib.Path("/mnt/c/tdx_export")
    dst_dir = root / "data" / "raw" / "tdx_export"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Sync: copy any new 观察池*.xls into project raw dir
    if src_dir.exists():
        for p in src_dir.glob("观察池*.xls"):
            tgt = dst_dir / p.name
            if not tgt.exists() or p.stat().st_mtime > tgt.stat().st_mtime:
                shutil.copy2(p, tgt)
    else:
        print(f"WARN: source export dir not found: {src_dir}")

    run(["python3", "src/integrate_watchpool.py"], cwd=root)
    run(["python3", "src/build_master_watchpool.py"], cwd=root)
    run(["python3", "src/analyze_highopen_masterpool.py"], cwd=root)
    run(["python3", "src/pick_top10_today.py"], cwd=root)


if __name__ == "__main__":
    main()
