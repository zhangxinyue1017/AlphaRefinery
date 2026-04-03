from __future__ import annotations

"""Pattern-retrieval factors migrated from /root/365factors."""

import pandas as pd

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..registry import FactorRegistry
from ._pattern_retrieval import similar_low_volatility_frame, similar_reverse_frame

FACTOR365_PATTERN_SOURCE = "factor365_pattern_v1"
FACTOR365_PATTERN_V2_SOURCE = "factor365_pattern_v2"
FACTOR365_SIMILAR_LOW_VOL_FIELDS: tuple[str, ...] = ("close",)
FACTOR365_SIMILAR_REVERSE_FIELDS: tuple[str, ...] = ("close", "benchmark_close")


def _prepare_frames(
    data: dict[str, pd.Series],
    required_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, ...]:
    validate_data(data, required_fields=required_fields)
    frames = tuple(data[field].unstack(level="instrument").sort_index() for field in required_fields)
    common_index = frames[0].index
    common_cols = frames[0].columns
    for frame in frames[1:]:
        common_index = common_index.intersection(frame.index)
        common_cols = common_cols.intersection(frame.columns)
    return tuple(frame.loc[common_index, common_cols].sort_index().sort_index(axis=1) for frame in frames)


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def factor365_similar_low_volatility_6_120_04_h6_hl20(data: dict[str, pd.Series]) -> pd.Series:
    (close_df,) = _prepare_frames(data, FACTOR365_SIMILAR_LOW_VOL_FIELDS)
    frame = similar_low_volatility_frame(
        close_df,
        rw=6,
        history_window=120,
        threshold=0.4,
        holding_time=6,
        min_matches=5,
        use_exponential_weight=True,
        half_life=20.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_low_volatility_6_120_04_h6_hl20")


def factor365_similar_reverse_6_120_04_h6_hl6(data: dict[str, pd.Series]) -> pd.Series:
    close_df, benchmark_df = _prepare_frames(data, FACTOR365_SIMILAR_REVERSE_FIELDS)
    benchmark_close = benchmark_df.iloc[:, 0]
    frame = similar_reverse_frame(
        close_df,
        benchmark_close,
        rw=6,
        history_window=120,
        threshold=0.4,
        holding_time=6,
        half_life=6.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_reverse_6_120_04_h6_hl6")


def factor365_similar_low_volatility_6_120_05_h6_hl20(data: dict[str, pd.Series]) -> pd.Series:
    (close_df,) = _prepare_frames(data, FACTOR365_SIMILAR_LOW_VOL_FIELDS)
    frame = similar_low_volatility_frame(
        close_df,
        rw=6,
        history_window=120,
        threshold=0.5,
        holding_time=6,
        min_matches=5,
        use_exponential_weight=True,
        half_life=20.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_low_volatility_6_120_05_h6_hl20")


def factor365_similar_low_volatility_6_120_04_h10_hl20(data: dict[str, pd.Series]) -> pd.Series:
    (close_df,) = _prepare_frames(data, FACTOR365_SIMILAR_LOW_VOL_FIELDS)
    frame = similar_low_volatility_frame(
        close_df,
        rw=6,
        history_window=120,
        threshold=0.4,
        holding_time=10,
        min_matches=5,
        use_exponential_weight=True,
        half_life=20.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_low_volatility_6_120_04_h10_hl20")


def factor365_similar_low_volatility_10_120_04_h6_hl20(data: dict[str, pd.Series]) -> pd.Series:
    (close_df,) = _prepare_frames(data, FACTOR365_SIMILAR_LOW_VOL_FIELDS)
    frame = similar_low_volatility_frame(
        close_df,
        rw=10,
        history_window=120,
        threshold=0.4,
        holding_time=6,
        min_matches=5,
        use_exponential_weight=True,
        half_life=20.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_low_volatility_10_120_04_h6_hl20")


def factor365_similar_reverse_6_120_05_h6_hl6(data: dict[str, pd.Series]) -> pd.Series:
    close_df, benchmark_df = _prepare_frames(data, FACTOR365_SIMILAR_REVERSE_FIELDS)
    benchmark_close = benchmark_df.iloc[:, 0]
    frame = similar_reverse_frame(
        close_df,
        benchmark_close,
        rw=6,
        history_window=120,
        threshold=0.5,
        holding_time=6,
        half_life=6.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_reverse_6_120_05_h6_hl6")


def factor365_similar_reverse_6_120_04_h10_hl6(data: dict[str, pd.Series]) -> pd.Series:
    close_df, benchmark_df = _prepare_frames(data, FACTOR365_SIMILAR_REVERSE_FIELDS)
    benchmark_close = benchmark_df.iloc[:, 0]
    frame = similar_reverse_frame(
        close_df,
        benchmark_close,
        rw=6,
        history_window=120,
        threshold=0.4,
        holding_time=10,
        half_life=6.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_reverse_6_120_04_h10_hl6")


def factor365_similar_reverse_10_120_04_h6_hl6(data: dict[str, pd.Series]) -> pd.Series:
    close_df, benchmark_df = _prepare_frames(data, FACTOR365_SIMILAR_REVERSE_FIELDS)
    benchmark_close = benchmark_df.iloc[:, 0]
    frame = similar_reverse_frame(
        close_df,
        benchmark_close,
        rw=10,
        history_window=120,
        threshold=0.4,
        holding_time=6,
        half_life=6.0,
    )
    return _series_from_frame(frame, factor_name="factor365.similar_reverse_10_120_04_h6_hl6")


def register_factor365_pattern(registry: FactorRegistry) -> int:
    factor_specs: tuple[tuple[str, object, tuple[str, ...], str, str], ...] = (
        (
            "factor365.similar_low_volatility_6_120_04_h6_hl20",
            factor365_similar_low_volatility_6_120_04_h6_hl20,
            FACTOR365_SIMILAR_LOW_VOL_FIELDS,
            FACTOR365_PATTERN_SOURCE,
            "Pattern retrieval factor: inverse weighted volatility of future 6-day returns after historically similar 6-day close paths over a 120-day lookback.",
        ),
        (
            "factor365.similar_reverse_6_120_04_h6_hl6",
            factor365_similar_reverse_6_120_04_h6_hl6,
            FACTOR365_SIMILAR_REVERSE_FIELDS,
            FACTOR365_PATTERN_SOURCE,
            "Pattern retrieval factor: negative weighted mean of future 6-day excess returns after historically similar 6-day close paths over a 120-day lookback.",
        ),
        (
            "factor365.similar_low_volatility_6_120_05_h6_hl20",
            factor365_similar_low_volatility_6_120_05_h6_hl20,
            FACTOR365_SIMILAR_LOW_VOL_FIELDS,
            FACTOR365_PATTERN_V2_SOURCE,
            "Pattern retrieval variant: higher path-similarity threshold for inverse future-volatility stability.",
        ),
        (
            "factor365.similar_low_volatility_6_120_04_h10_hl20",
            factor365_similar_low_volatility_6_120_04_h10_hl20,
            FACTOR365_SIMILAR_LOW_VOL_FIELDS,
            FACTOR365_PATTERN_V2_SOURCE,
            "Pattern retrieval variant: longer post-match holding window for inverse future-volatility stability.",
        ),
        (
            "factor365.similar_low_volatility_10_120_04_h6_hl20",
            factor365_similar_low_volatility_10_120_04_h6_hl20,
            FACTOR365_SIMILAR_LOW_VOL_FIELDS,
            FACTOR365_PATTERN_V2_SOURCE,
            "Pattern retrieval variant: longer 10-day path length for inverse future-volatility stability.",
        ),
        (
            "factor365.similar_reverse_6_120_05_h6_hl6",
            factor365_similar_reverse_6_120_05_h6_hl6,
            FACTOR365_SIMILAR_REVERSE_FIELDS,
            FACTOR365_PATTERN_V2_SOURCE,
            "Pattern retrieval variant: higher path-similarity threshold for weighted future excess-return reversal.",
        ),
        (
            "factor365.similar_reverse_6_120_04_h10_hl6",
            factor365_similar_reverse_6_120_04_h10_hl6,
            FACTOR365_SIMILAR_REVERSE_FIELDS,
            FACTOR365_PATTERN_V2_SOURCE,
            "Pattern retrieval variant: longer holding window for weighted future excess-return reversal.",
        ),
        (
            "factor365.similar_reverse_10_120_04_h6_hl6",
            factor365_similar_reverse_10_120_04_h6_hl6,
            FACTOR365_SIMILAR_REVERSE_FIELDS,
            FACTOR365_PATTERN_V2_SOURCE,
            "Pattern retrieval variant: longer 10-day path length for weighted future excess-return reversal.",
        ),
    )
    for name, func, required_fields, source, notes in factor_specs:
        registry.register(
            name,
            func,
            source=source,
            required_fields=required_fields,
            notes=notes,
        )
    return len(factor_specs)


def factor365_pattern_source_info() -> dict[str, object]:
    return {
        "source": FACTOR365_PATTERN_SOURCE,
        "status": "first_pass_pattern_retrieval",
        "implemented_factors": (
            "factor365.similar_low_volatility_6_120_04_h6_hl20",
            "factor365.similar_reverse_6_120_04_h6_hl6",
        ),
        "notes": (
            "First-pass migration of path-retrieval factors from /root/365factors. "
            "These factors are high-order local computations and are intended for AlphaRefinery-local "
            "research and backtesting rather than operators_pro-compatible expression replay."
        ),
    }


def factor365_pattern_v2_source_info() -> dict[str, object]:
    return {
        "source": FACTOR365_PATTERN_V2_SOURCE,
        "status": "pattern_retrieval_variants",
        "implemented_factors": (
            "factor365.similar_low_volatility_6_120_05_h6_hl20",
            "factor365.similar_low_volatility_6_120_04_h10_hl20",
            "factor365.similar_low_volatility_10_120_04_h6_hl20",
            "factor365.similar_reverse_6_120_05_h6_hl6",
            "factor365.similar_reverse_6_120_04_h10_hl6",
            "factor365.similar_reverse_10_120_04_h6_hl6",
        ),
        "notes": (
            "Second-pass family expansion for path-retrieval factors. "
            "This batch varies similarity threshold, path length, and holding horizon "
            "around the first-pass similar_low_volatility and similar_reverse definitions."
        ),
    }
