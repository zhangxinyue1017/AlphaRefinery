'''Normalization, validation, and storage for AlphaRefinery daily panels.'''

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..contract import CORE_FIELDS, EXPECTED_INDEX_NAMES


PANEL_FIELDS = CORE_FIELDS


def normalize_alpaca_bars(frame: pd.DataFrame) -> pd.DataFrame:
    '''Convert an Alpaca ``BarSet.df`` frame to the AlphaRefinery contract.'''

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Alpaca bars must be a non-empty DataFrame")

    work = frame.reset_index()
    symbol_col = next((name for name in ("symbol", "instrument") if name in work.columns), None)
    timestamp_col = next((name for name in ("timestamp", "datetime") if name in work.columns), None)
    if symbol_col is None or timestamp_col is None:
        raise ValueError("Alpaca bars must provide symbol and timestamp index levels or columns")

    missing_fields = [field for field in PANEL_FIELDS if field not in work.columns]
    if missing_fields:
        raise ValueError(f"Alpaca bars missing required fields: {missing_fields}")

    timestamps = pd.to_datetime(work[timestamp_col], utc=True, errors="raise")
    trade_dates = timestamps.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    normalized = work.loc[:, list(PANEL_FIELDS)].copy()
    for field in PANEL_FIELDS:
        normalized[field] = pd.to_numeric(normalized[field], errors="coerce").astype(float)
    normalized["datetime"] = trade_dates
    normalized["instrument"] = work[symbol_col].astype(str).str.strip().str.upper()
    normalized = normalized.set_index(list(EXPECTED_INDEX_NAMES)).sort_index()
    normalized = normalized.loc[:, list(PANEL_FIELDS)]
    validate_panel(normalized)
    return normalized


def validate_panel(
    panel: pd.DataFrame,
    *,
    expected_symbols: Iterable[str] | None = None,
) -> dict[str, object]:
    '''Validate the daily OHLCV panel and return compact coverage diagnostics.'''

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel must use MultiIndex(datetime, instrument)")
    if tuple(panel.index.names) != EXPECTED_INDEX_NAMES:
        raise TypeError(f"panel index names must be {EXPECTED_INDEX_NAMES}")
    if panel.empty:
        raise ValueError("panel is empty")
    if not panel.index.is_unique:
        raise ValueError("panel index contains duplicate datetime/instrument rows")
    if not panel.index.equals(panel.index.sort_values()):
        raise ValueError("panel index must be sorted")

    missing_fields = [field for field in PANEL_FIELDS if field not in panel.columns]
    if missing_fields:
        raise ValueError(f"panel missing required fields: {missing_fields}")

    values = panel.loc[:, list(PANEL_FIELDS)].apply(pd.to_numeric, errors="coerce")
    missing_values = {field: int(values[field].isna().sum()) for field in PANEL_FIELDS}
    if any(missing_values.values()):
        raise ValueError(f"panel contains missing required values: {missing_values}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("panel contains non-finite OHLCV values")

    price_fields = ["open", "high", "low", "close", "vwap"]
    if (values[price_fields] <= 0).any().any():
        raise ValueError("OHLC and VWAP values must be positive")
    if (values["volume"] < 0).any():
        raise ValueError("volume must be non-negative")

    tolerance = 1e-8
    row_max = values[["open", "low", "close", "vwap"]].max(axis=1)
    row_min = values[["open", "high", "close", "vwap"]].min(axis=1)
    if (values["high"] + tolerance < row_max).any():
        raise ValueError("high is below another price field")
    if (values["low"] - tolerance > row_min).any():
        raise ValueError("low is above another price field")

    datetimes = panel.index.get_level_values("datetime")
    if getattr(datetimes, "tz", None) is not None:
        raise ValueError("daily panel datetimes must be timezone-naive trade dates")
    if any(timestamp != timestamp.normalize() for timestamp in datetimes.unique()):
        raise ValueError("daily panel datetimes must be normalized to midnight")

    actual_symbols = sorted(str(value) for value in panel.index.get_level_values("instrument").unique())
    if any(not symbol for symbol in actual_symbols):
        raise ValueError("instrument symbols must be non-empty")
    all_dates = panel.index.get_level_values("datetime").unique()
    observed_counts = panel.groupby(level="instrument").size()
    missing_observations = {
        str(symbol): int(len(all_dates) - count)
        for symbol, count in observed_counts.items()
        if int(len(all_dates) - count) > 0
    }
    expected = sorted({str(symbol).strip().upper() for symbol in (expected_symbols or ()) if str(symbol).strip()})
    missing_symbols = sorted(set(expected) - set(actual_symbols))
    return {
        "row_count": int(len(panel)),
        "date_count": int(len(all_dates)),
        "symbol_count": int(len(actual_symbols)),
        "symbols": actual_symbols,
        "start": str(datetimes.min().date()),
        "end": str(datetimes.max().date()),
        "missing_values": missing_values,
        "missing_observations_by_symbol": missing_observations,
        "missing_symbols": missing_symbols,
    }


def save_panel(panel: pd.DataFrame, path: str | Path) -> dict[str, object]:
    '''Validate and save one Parquet panel.'''

    report = validate_panel(panel)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output)
    return report


def file_sha256(path: str | Path) -> str:
    '''Return the SHA-256 checksum for a saved data snapshot.'''

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["PANEL_FIELDS", "file_sha256", "normalize_alpaca_bars", "save_panel", "validate_panel"]
