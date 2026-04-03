from __future__ import annotations

"""Portable daily factors adapted from a CICC-style research snippet."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..registry import FactorRegistry

CICC_DAILY_SOURCE = "cicc_daily_v1"
EPS = 1e-12

CICC_PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "pre_close")
CICC_AMOUNT_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "amount")


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


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)


def _daily_return_frame(close_df: pd.DataFrame, preclose_df: pd.DataFrame) -> pd.DataFrame:
    return close_df / preclose_df.replace(0.0, np.nan) - 1.0


def _amplitude_frame(high_df: pd.DataFrame, low_df: pd.DataFrame, preclose_df: pd.DataFrame) -> pd.DataFrame:
    return (high_df - low_df) / preclose_df.replace(0.0, np.nan)


def _rolling_std(frame: pd.DataFrame, *, window: int) -> pd.DataFrame:
    return frame.rolling(window=window, min_periods=window).std()


def _mmt_intraday_frame(data: dict[str, pd.Series], *, window: int = 21) -> pd.DataFrame:
    open_df, high_df, low_df, close_df, preclose_df = _prepare_frames(data, CICC_PRICE_FIELDS)
    del open_df, high_df, low_df
    daily_ret = _daily_return_frame(close_df, preclose_df)
    return daily_ret.rolling(window=window, min_periods=window).sum()


def _mmt_range_frame(
    data: dict[str, pd.Series],
    *,
    window: int = 21,
    ratio: float = 0.2,
) -> pd.DataFrame:
    open_df, high_df, low_df, close_df, preclose_df = _prepare_frames(data, CICC_PRICE_FIELDS)
    del open_df
    return_df = _daily_return_frame(close_df, preclose_df)
    amplitude_df = _amplitude_frame(high_df, low_df, preclose_df)
    out = _empty_like(close_df)
    if len(close_df) < window:
        return out

    ret_view = sliding_window_view(return_df.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    amp_view = sliding_window_view(amplitude_df.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    valid = ~np.isnan(ret_view) & ~np.isnan(amp_view)
    full_valid = valid.all(axis=2)

    day_num = max(int(ratio * window), 1)
    result_matrix = np.zeros_like(amp_view, dtype=float)
    sorted_indices = np.argsort(amp_view, axis=2, kind="mergesort")
    np.put_along_axis(result_matrix, sorted_indices[:, :, :day_num], -1.0, axis=2)
    np.put_along_axis(result_matrix, sorted_indices[:, :, -day_num:], 1.0, axis=2)

    tail = np.full((ret_view.shape[0], ret_view.shape[1]), np.nan, dtype=float)
    tail[full_valid] = np.sum(ret_view[full_valid] * result_matrix[full_valid], axis=1)
    out.iloc[window - 1 :] = tail
    return out


def _vol_std_frame(data: dict[str, pd.Series], *, window: int = 63) -> pd.DataFrame:
    open_df, high_df, low_df, close_df, preclose_df = _prepare_frames(data, CICC_PRICE_FIELDS)
    del open_df, high_df, low_df
    daily_ret = _daily_return_frame(close_df, preclose_df)
    return _rolling_std(daily_ret, window=window)


def _vol_highlow_std_frame(data: dict[str, pd.Series], *, window: int = 63) -> pd.DataFrame:
    open_df, high_df, low_df, close_df, preclose_df = _prepare_frames(data, CICC_PRICE_FIELDS)
    del open_df, close_df, preclose_df
    highlow = high_df / low_df.replace(0.0, np.nan)
    return _rolling_std(highlow, window=window)


def _vol_upshadow_std_frame(data: dict[str, pd.Series], *, window: int = 126) -> pd.DataFrame:
    open_df, high_df, low_df, close_df, preclose_df = _prepare_frames(data, CICC_PRICE_FIELDS)
    del low_df, preclose_df
    maxv = np.maximum(open_df, close_df)
    upshadow = (high_df - maxv) / high_df.replace(0.0, np.nan)
    return _rolling_std(upshadow, window=window)


def _vol_downshadow_std_frame(data: dict[str, pd.Series], *, window: int = 63) -> pd.DataFrame:
    open_df, high_df, low_df, close_df, preclose_df = _prepare_frames(data, CICC_PRICE_FIELDS)
    del open_df, high_df, preclose_df
    downshadow = (close_df - low_df) / low_df.replace(0.0, np.nan)
    return _rolling_std(downshadow, window=window)


def _liq_shortcut_frame(data: dict[str, pd.Series], *, window: int = 21) -> pd.DataFrame:
    open_df, high_df, low_df, close_df, amount_df = _prepare_frames(data, CICC_AMOUNT_FIELDS)
    shortcut = (2.0 * (high_df - low_df) - (open_df - close_df).abs()) / amount_df.replace(0.0, np.nan)
    return shortcut.rolling(window=window, min_periods=window).mean()


def cicc_daily_mmt_intraday_m_21(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _mmt_intraday_frame(data, window=21),
        factor_name="cicc_daily.mmt_intraday_m_21",
    )


def cicc_daily_mmt_range_m_21_q20(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _mmt_range_frame(data, window=21, ratio=0.2),
        factor_name="cicc_daily.mmt_range_m_21_q20",
    )


def cicc_daily_vol_std_3m(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _vol_std_frame(data, window=63),
        factor_name="cicc_daily.vol_std_3m",
    )


def cicc_daily_vol_highlow_std_3m(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _vol_highlow_std_frame(data, window=63),
        factor_name="cicc_daily.vol_highlow_std_3m",
    )


def cicc_daily_vol_upshadow_std_6m(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _vol_upshadow_std_frame(data, window=126),
        factor_name="cicc_daily.vol_upshadow_std_6m",
    )


def cicc_daily_vol_w_downshadow_std_3m(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _vol_downshadow_std_frame(data, window=63),
        factor_name="cicc_daily.vol_w_downshadow_std_3m",
    )


def cicc_daily_liq_shortcut_avg_m_21(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _liq_shortcut_frame(data, window=21),
        factor_name="cicc_daily.liq_shortcut_avg_m_21",
    )


def register_cicc_daily(registry: FactorRegistry) -> int:
    factor_specs: tuple[tuple[str, object, tuple[str, ...], str], ...] = (
        (
            "cicc_daily.mmt_intraday_m_21",
            cicc_daily_mmt_intraday_m_21,
            CICC_PRICE_FIELDS,
            "21-day sum of daily close-to-preclose returns.",
        ),
        (
            "cicc_daily.mmt_range_m_21_q20",
            cicc_daily_mmt_range_m_21_q20,
            CICC_PRICE_FIELDS,
            "21-day amplitude-sliced return sum: lowest 20% amplitude days negative, highest 20% positive.",
        ),
        (
            "cicc_daily.vol_std_3m",
            cicc_daily_vol_std_3m,
            CICC_PRICE_FIELDS,
            "63-day standard deviation of daily close-to-preclose returns.",
        ),
        (
            "cicc_daily.vol_highlow_std_3m",
            cicc_daily_vol_highlow_std_3m,
            CICC_PRICE_FIELDS,
            "63-day standard deviation of daily high/low ratio.",
        ),
        (
            "cicc_daily.vol_upshadow_std_6m",
            cicc_daily_vol_upshadow_std_6m,
            CICC_PRICE_FIELDS,
            "126-day standard deviation of normalized upper shadow.",
        ),
        (
            "cicc_daily.vol_w_downshadow_std_3m",
            cicc_daily_vol_w_downshadow_std_3m,
            CICC_PRICE_FIELDS,
            "63-day standard deviation of normalized lower shadow proxy.",
        ),
        (
            "cicc_daily.liq_shortcut_avg_m_21",
            cicc_daily_liq_shortcut_avg_m_21,
            CICC_AMOUNT_FIELDS,
            "21-day mean of shortcut liquidity proxy based on price range and amount.",
        ),
    )
    for name, func, required_fields, notes in factor_specs:
        registry.register(
            name,
            func,
            source=CICC_DAILY_SOURCE,
            required_fields=required_fields,
            notes=notes,
        )
    return len(factor_specs)


def cicc_daily_source_info() -> dict[str, object]:
    return {
        "source": CICC_DAILY_SOURCE,
        "status": "first_pass_portable",
        "implemented_factors": (
            "cicc_daily.mmt_intraday_m_21",
            "cicc_daily.mmt_range_m_21_q20",
            "cicc_daily.vol_std_3m",
            "cicc_daily.vol_highlow_std_3m",
            "cicc_daily.vol_upshadow_std_6m",
            "cicc_daily.vol_w_downshadow_std_3m",
            "cicc_daily.liq_shortcut_avg_m_21",
        ),
        "notes": (
            "Portable first-pass batch adapted from a CICC-style daily research snippet. "
            "This batch focuses on amplitude-sliced momentum, shadow volatility, range volatility, "
            "and a shortcut liquidity proxy."
        ),
    }
