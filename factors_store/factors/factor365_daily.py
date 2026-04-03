from __future__ import annotations

"""Portable daily factors migrated from /root/365factors."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..registry import FactorRegistry

FACTOR365_DAILY_SOURCE = "factor365_daily_v1"
EPS = 1e-12

FACTOR365_VOLUME_FIELDS: tuple[str, ...] = ("volume",)
FACTOR365_REVERSAL_FIELDS: tuple[str, ...] = ("open", "close", "pre_close")
FACTOR365_SHADOW_FIELDS: tuple[str, ...] = ("high", "open", "close", "pre_close")
FACTOR365_AMPLITUDE_FIELDS: tuple[str, ...] = ("high", "low", "close")


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


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def _abnvold_frame(
    data: dict[str, pd.Series],
    *,
    avg_volume_window: int = 242,
    max_window: int = 20,
) -> pd.DataFrame:
    (volume_df,) = _prepare_frames(data, FACTOR365_VOLUME_FIELDS)
    avg_vol = volume_df.rolling(avg_volume_window, min_periods=avg_volume_window).mean()
    abnvol = volume_df / avg_vol.replace(0.0, np.nan)
    return abnvol.rolling(max_window, min_periods=max_window).max()


def _abnormal_volume_average_frame(
    data: dict[str, pd.Series],
    *,
    annual_window: int = 242,
    month_window: int = 20,
) -> pd.DataFrame:
    (volume_df,) = _prepare_frames(data, FACTOR365_VOLUME_FIELDS)
    avg_vol = volume_df.rolling(annual_window, min_periods=annual_window).mean()
    abnvol = volume_df / avg_vol.replace(0.0, np.nan)
    return abnvol.rolling(month_window, min_periods=month_window).mean()


def _positive_intraday_reversal_frequency_frame(
    data: dict[str, pd.Series],
    *,
    window: int = 20,
) -> pd.DataFrame:
    open_df, close_df, preclose_df = _prepare_frames(data, FACTOR365_REVERSAL_FIELDS)
    ret_co = open_df / preclose_df.replace(0.0, np.nan) - 1.0
    ret_oc = close_df / open_df.replace(0.0, np.nan) - 1.0
    valid = open_df.notna() & close_df.notna() & preclose_df.notna()
    event = ((ret_co < 0.0) & (ret_oc > 0.0)).astype(float).where(valid)
    return event.rolling(window, min_periods=window).mean()


def _abnormal_positive_intraday_reversal_frequency_frame(
    data: dict[str, pd.Series],
    *,
    min_months: int = 12,
) -> pd.DataFrame:
    open_df, close_df, preclose_df = _prepare_frames(data, FACTOR365_REVERSAL_FIELDS)
    ret_co = open_df / preclose_df.replace(0.0, np.nan) - 1.0
    ret_oc = close_df / open_df.replace(0.0, np.nan) - 1.0
    valid = open_df.notna() & close_df.notna() & preclose_df.notna()
    event = ((ret_co < 0.0) & (ret_oc > 0.0)).astype(float).where(valid)

    event_count_monthly = event.resample("ME").sum(min_count=1)
    trade_days_monthly = valid.astype(float).resample("ME").sum(min_count=1)
    pr = event_count_monthly / trade_days_monthly.replace(0.0, np.nan)
    pr_mean_12m = pr.rolling(window=12, min_periods=min_months).mean()
    return pr / pr_mean_12m.replace(0.0, np.nan)


def _exp_decay_weights(window: int, half_life: float) -> np.ndarray:
    age = np.arange(window - 1, -1, -1, dtype=float)
    return 0.5 ** (age / float(half_life))


def _rolling_weighted_sum_div_const(
    x: np.ndarray,
    *,
    weights: np.ndarray,
    divisor: float,
) -> np.ndarray:
    t_count, n_inst = x.shape
    window = len(weights)
    out = np.full((t_count, n_inst), np.nan, dtype=float)
    for t in range(window - 1, t_count):
        window_x = x[t - window + 1 : t + 1]
        valid = np.isfinite(window_x)
        out[t] = np.nansum(window_x * weights[:, None] * valid, axis=0) / float(divisor)
    return out


def _weighted_upper_shadow_frequency_frame(
    data: dict[str, pd.Series],
    *,
    window: int = 40,
    threshold: float = 0.01,
    half_life: float = 10.0,
) -> pd.DataFrame:
    high_df, open_df, close_df, preclose_df = _prepare_frames(data, FACTOR365_SHADOW_FIELDS)
    body_top = np.maximum(open_df.to_numpy(dtype=float), close_df.to_numpy(dtype=float))
    preclose_np = preclose_df.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        upper_shadow = (high_df.to_numpy(dtype=float) - body_top) / preclose_np
    upper_shadow[~np.isfinite(upper_shadow)] = np.nan
    signal = (upper_shadow > float(threshold)).astype(float)
    signal[~np.isfinite(upper_shadow)] = np.nan
    weights = _exp_decay_weights(window, half_life)
    factor_arr = _rolling_weighted_sum_div_const(signal, weights=weights, divisor=float(window))
    return pd.DataFrame(factor_arr, index=high_df.index, columns=high_df.columns)


def _ideal_amplitude_frame(
    data: dict[str, pd.Series],
    *,
    window: int = 20,
    lambda_ratio: float = 0.25,
) -> pd.DataFrame:
    high_df, low_df, close_df = _prepare_frames(data, FACTOR365_AMPLITUDE_FIELDS)
    amplitude_df = high_df / low_df.replace(0.0, np.nan) - 1.0
    out = _empty_like(close_df)
    if len(close_df) < window:
        return out

    close_view = sliding_window_view(close_df.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    amp_view = sliding_window_view(amplitude_df.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    valid = ~np.isnan(close_view) & ~np.isnan(amp_view)
    full_valid = valid.all(axis=2)

    order = np.argsort(close_view, axis=2, kind="mergesort")
    ranks = np.argsort(order, axis=2, kind="mergesort") + 1
    pct_rank = ranks / float(window)
    low_mask = pct_rank <= float(lambda_ratio)
    high_mask = pct_rank >= float(1.0 - lambda_ratio)

    low_sum = np.where(low_mask & valid, amp_view, 0.0).sum(axis=2)
    high_sum = np.where(high_mask & valid, amp_view, 0.0).sum(axis=2)
    low_cnt = np.where(low_mask & valid, 1.0, 0.0).sum(axis=2)
    high_cnt = np.where(high_mask & valid, 1.0, 0.0).sum(axis=2)

    low_mean = np.full(low_sum.shape, np.nan, dtype=float)
    high_mean = np.full(high_sum.shape, np.nan, dtype=float)
    np.divide(low_sum, low_cnt, out=low_mean, where=full_valid & (low_cnt > 0))
    np.divide(high_sum, high_cnt, out=high_mean, where=full_valid & (high_cnt > 0))

    out.iloc[window - 1 :] = high_mean - low_mean
    return out


def factor365_abnvold_20(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(_abnvold_frame(data), factor_name="factor365.abnvold_20")


def factor365_abnormal_volume_average_20(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _abnormal_volume_average_frame(data),
        factor_name="factor365.abnormal_volume_average_20",
    )


def factor365_positive_intraday_reversal_frequency_20(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _positive_intraday_reversal_frequency_frame(data),
        factor_name="factor365.positive_intraday_reversal_frequency_20",
    )


def factor365_abnormal_positive_intraday_reversal_frequency_12m(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _abnormal_positive_intraday_reversal_frequency_frame(data),
        factor_name="factor365.abnormal_positive_intraday_reversal_frequency_12m",
    )


def factor365_weighted_upper_shadow_frequency_40_hl10(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _weighted_upper_shadow_frequency_frame(data),
        factor_name="factor365.weighted_upper_shadow_frequency_40_hl10",
    )


def factor365_ideal_amplitude_20_q25(data: dict[str, pd.Series]) -> pd.Series:
    return _series_from_frame(
        _ideal_amplitude_frame(data),
        factor_name="factor365.ideal_amplitude_20_q25",
    )


def register_factor365_daily(registry: FactorRegistry) -> int:
    factor_specs: tuple[tuple[str, object, tuple[str, ...], str], ...] = (
        (
            "factor365.abnvold_20",
            factor365_abnvold_20,
            FACTOR365_VOLUME_FIELDS,
            "Daily abnormal volume relative to 242-day mean, then 20-day rolling max.",
        ),
        (
            "factor365.abnormal_volume_average_20",
            factor365_abnormal_volume_average_20,
            FACTOR365_VOLUME_FIELDS,
            "Daily abnormal volume relative to 242-day mean, then 20-day rolling mean.",
        ),
        (
            "factor365.positive_intraday_reversal_frequency_20",
            factor365_positive_intraday_reversal_frequency_20,
            FACTOR365_REVERSAL_FIELDS,
            "20-day frequency of negative overnight followed by positive intraday reversal.",
        ),
        (
            "factor365.abnormal_positive_intraday_reversal_frequency_12m",
            factor365_abnormal_positive_intraday_reversal_frequency_12m,
            FACTOR365_REVERSAL_FIELDS,
            "Monthly positive intraday reversal frequency normalized by trailing 12-month mean.",
        ),
        (
            "factor365.weighted_upper_shadow_frequency_40_hl10",
            factor365_weighted_upper_shadow_frequency_40_hl10,
            FACTOR365_SHADOW_FIELDS,
            "40-day half-life weighted frequency of upper shadows larger than 1% of pre-close.",
        ),
        (
            "factor365.ideal_amplitude_20_q25",
            factor365_ideal_amplitude_20_q25,
            FACTOR365_AMPLITUDE_FIELDS,
            "20-day ideal amplitude: mean amplitude of top-close quartile minus bottom-close quartile.",
        ),
    )
    for name, func, required_fields, notes in factor_specs:
        registry.register(
            name,
            func,
            source=FACTOR365_DAILY_SOURCE,
            required_fields=required_fields,
            notes=notes,
        )
    return len(factor_specs)


def factor365_daily_source_info() -> dict[str, object]:
    return {
        "source": FACTOR365_DAILY_SOURCE,
        "status": "first_pass_portable",
        "implemented_factors": (
            "factor365.abnvold_20",
            "factor365.abnormal_volume_average_20",
            "factor365.positive_intraday_reversal_frequency_20",
            "factor365.abnormal_positive_intraday_reversal_frequency_12m",
            "factor365.weighted_upper_shadow_frequency_40_hl10",
            "factor365.ideal_amplitude_20_q25",
        ),
        "notes": (
            "First-pass migration of daily-portable themes from /root/365factors. "
            "This batch focuses on abnormal volume, overnight-vs-intraday reversal, "
            "upper-shadow event frequency, and ideal amplitude."
        ),
    }
