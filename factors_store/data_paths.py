from __future__ import annotations

import os
from pathlib import Path


def _env_path(env_name: str, default: str) -> Path:
    raw = str(os.getenv(env_name, "") or "").strip()
    return Path(raw).expanduser() if raw else Path(default)


PANEL_PATH_ENV = "ALPHAREFINERY_PANEL_PATH"
BENCHMARK_PATH_ENV = "ALPHAREFINERY_BENCHMARK_PATH"
INDUSTRY_CSV_PATH_ENV = "ALPHAREFINERY_INDUSTRY_CSV_PATH"

DEFAULT_PANEL_PATH = _env_path(PANEL_PATH_ENV, "/root/dmd/BaoStock/panel.parquet")
DEFAULT_BENCHMARK_PATH = _env_path(BENCHMARK_PATH_ENV, "/root/dmd/BaoStock/Index/sh.000001.csv")
DEFAULT_INDUSTRY_CSV_PATH = _env_path(INDUSTRY_CSV_PATH_ENV, "/root/dmd/BaoStock/Industry/stock_industry.csv")


__all__ = [
    "PANEL_PATH_ENV",
    "BENCHMARK_PATH_ENV",
    "INDUSTRY_CSV_PATH_ENV",
    "DEFAULT_PANEL_PATH",
    "DEFAULT_BENCHMARK_PATH",
    "DEFAULT_INDUSTRY_CSV_PATH",
]
