from __future__ import annotations

import pathlib
import yaml


def load_config(path: str | pathlib.Path) -> dict:
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(pathlib.Path(__file__).resolve().parents[1] / "configs" / "config.yaml")
    print("Loaded config:")
    for k, v in cfg.items():
        print(f"- {k}: {v}")

    # TODO: 1) load data
    # TODO: 2) build features
    # TODO: 3) build labels
    # TODO: 4) train model
    # TODO: 5) backtest & report


if __name__ == "__main__":
    main()
