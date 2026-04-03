from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

EXPECTED_INDEX_NAMES = ("datetime", "instrument")

CORE_FIELDS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
)

EXTENDED_DAILY_FIELDS: tuple[str, ...] = (
    "pre_close",
    "amount",
    "turnover",
    "pct_chg",
    "is_st",
    "trade_status",
)

DERIVED_FIELDS: tuple[str, ...] = (
    "returns",
)

OPTIONAL_CONTEXT_FIELDS: tuple[str, ...] = (
    "benchmark_open",
    "benchmark_close",
    "market_return",
    "cap",
    "size",
    "float_market_cap",
    "smb",
    "hml",
)

LIBRARY_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "alpha158": ("open", "high", "low", "close", "volume", "vwap"),
    "alpha360": ("open", "high", "low", "close", "volume", "vwap"),
    "alpha101": ("open", "high", "low", "close", "volume", "vwap", "returns"),
    "alpha191": (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "amount",
        "returns",
        "benchmark_open",
        "benchmark_close",
    ),
}

LIBRARY_OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "alpha101": ("cap",),
    "alpha191": ("cap", "market_return", "smb", "hml"),
}


def ensure_multiindex_series(series: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be pandas Series")
    if not isinstance(series.index, pd.MultiIndex):
        raise TypeError(f"{name}.index must be MultiIndex(datetime, instrument)")
    if series.index.nlevels != 2:
        raise TypeError(f"{name}.index must have exactly 2 levels")
    if tuple(series.index.names) != EXPECTED_INDEX_NAMES:
        raise TypeError(f"{name}.index names must be {EXPECTED_INDEX_NAMES}")
    return series


def validate_data(
    data: Mapping[str, pd.Series],
    *,
    required_fields: tuple[str, ...] | list[str] | None = None,
) -> None:
    fields = tuple(required_fields or ())
    for field in fields:
        if field not in data:
            raise KeyError(f"missing required field: {field}")
    for key, value in data.items():
        ensure_multiindex_series(value, name=f"data[{key!r}]")


def summarize_library_requirements() -> dict[str, dict[str, tuple[str, ...]]]:
    names = sorted(set(LIBRARY_REQUIRED_FIELDS) | set(LIBRARY_OPTIONAL_FIELDS))
    return {
        name: {
            "required": LIBRARY_REQUIRED_FIELDS.get(name, ()),
            "optional": LIBRARY_OPTIONAL_FIELDS.get(name, ()),
        }
        for name in names
    }
