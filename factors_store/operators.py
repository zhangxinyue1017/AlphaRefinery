'''Expression operator implementations for alpha-like factor definitions.

Contains cross-sectional, time-series, arithmetic, and helper operators used by factor formulas.
'''

from __future__ import annotations

"""Shared operators for factors_store.

This module intentionally keeps two operator families side by side:

1. Series operators
   Input/output are ``pd.Series`` with
   ``MultiIndex(datetime, instrument)``. These are the normalized
   operators aligned with the broader gp_factor_qlib contract.

2. Wide operators
   Input/output are ``pd.DataFrame`` in the classic
   ``date x instrument`` layout. These are used internally by
   source-style libraries such as Alpha101/Alpha191, where keeping the
   original formula shape makes maintenance much easier.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    from numba import njit, prange

    HAS_NUMBA = True
except Exception:  # pragma: no cover - optional acceleration dependency
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    def prange(*args):
        return range(*args)

from ._bootstrap import ensure_project_roots

ensure_project_roots()

from ._vendor.gpqlib_runtime.core.operators_pro import (  # noqa: E402
    add,
    cs_multi_reg_resid as _series_cs_multi_reg_resid,
    cs_reg_resid as _series_cs_reg_resid,
    delta,
    delay,
    div,
    ema,
    log,
    mul,
    rolling_cs_spearman_mean as _series_rolling_cs_spearman_mean,
    resi,
    rsquare,
    sign,
    slope,
    sub,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_max,
    ts_mean,
    ts_min,
    ts_quantile,
    ts_rank,
    ts_sorted_mean_spread as _series_ts_sorted_mean_spread,
    ts_std,
    ts_sum,
    ts_turnover_ref_price as _series_ts_turnover_ref_price,
)


# Series operators

def abs_(x: pd.Series) -> pd.Series:
    return x.abs()


def cs_rank(x: pd.Series) -> pd.Series:
    return x.groupby(level="datetime", group_keys=False).rank(method="min", pct=True)


def cs_zscore(x: pd.Series) -> pd.Series:
    def _one_day(series: pd.Series) -> pd.Series:
        std = series.std()
        if pd.isna(std) or std <= 1e-12:
            return series * np.nan
        return (series - series.mean()) / std

    return x.groupby(level="datetime", group_keys=False).apply(_one_day)


def safe_div(x: pd.Series, y: pd.Series | float) -> pd.Series:
    return div(x, y)


# Wide operators

_WIDE_EPS = 1e-12


def _as_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("wide operator input must be DataFrame")
    return df.astype(float, copy=False)


def _wide_to_series(df: pd.DataFrame) -> pd.Series:
    frame = _as_float_frame(df)
    try:
        series = frame.stack(future_stack=True).astype(float)
    except TypeError:
        series = frame.stack(dropna=False).astype(float)
    series.index = series.index.set_names(["datetime", "instrument"])
    return series


def _series_to_wide(series: pd.Series, *, like: pd.DataFrame) -> pd.DataFrame:
    wide = series.unstack(level="instrument")
    return wide.reindex(index=like.index, columns=like.columns).astype(float)


def _full_window_valid(view: np.ndarray) -> np.ndarray:
    return ~np.isnan(view).any(axis=2)


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)


def _rolling_view(df: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    base = _as_float_frame(df).to_numpy(dtype=float, copy=False)
    if int(window) <= 0:
        raise ValueError("window must be positive")
    if window > len(df):
        return base, np.empty((0, base.shape[1], int(window)), dtype=float)
    return base, sliding_window_view(base, window_shape=int(window), axis=0)


def _fill_tail_result(df: pd.DataFrame, tail: np.ndarray, window: int) -> pd.DataFrame:
    out = _empty_like(df)
    if len(tail):
        out.iloc[window - 1 :] = tail
    return out


def _weighted_tail(view: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(view * weights[None, None, :], axis=2) / (np.sum(weights) + _WIDE_EPS)


@njit(cache=True, parallel=True)
def _wide_rank_numba(arr: np.ndarray) -> np.ndarray:
    n_rows, n_cols = arr.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in prange(n_rows):
        row = arr[i]
        valid_count = 0
        for j in range(n_cols):
            if np.isnan(row[j]):
                out[i, j] = np.nan
            else:
                valid_count += 1
        if valid_count == 0:
            continue
        denom = float(valid_count)
        for j in range(n_cols):
            if np.isnan(row[j]):
                continue
            rank = 1.0
            v = row[j]
            for k in range(n_cols):
                if not np.isnan(row[k]) and row[k] < v:
                    rank += 1.0
            out[i, j] = rank / denom
    return out


@njit(cache=True, parallel=True)
def _wide_ts_rank_numba(arr: np.ndarray, window: int) -> np.ndarray:
    n_rows, n_cols = arr.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    out[:] = np.nan
    if window <= 0 or window > n_rows:
        return out
    for i in prange(window - 1, n_rows):
        start = i - window + 1
        for j in range(n_cols):
            last = arr[i, j]
            if np.isnan(last):
                continue
            valid = True
            rank = 1.0
            for t in range(start, i + 1):
                value = arr[t, j]
                if np.isnan(value):
                    valid = False
                    break
                if value < last:
                    rank += 1.0
            if valid:
                out[i, j] = rank
    return out


@njit(cache=True, parallel=True)
def _wide_regbeta_numba(arr: np.ndarray, x_centered: np.ndarray, x_mean: float, denom: float) -> np.ndarray:
    n_rows, n_cols = arr.shape
    window = len(x_centered)
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    out[:] = np.nan
    if window <= 0 or window > n_rows:
        return out
    for j in prange(n_cols):
        for i in range(window - 1, n_rows):
            start = i - window + 1
            y_sum = 0.0
            valid = True
            for t in range(window):
                value = arr[start + t, j]
                if np.isnan(value):
                    valid = False
                    break
                y_sum += value
            if not valid:
                continue
            y_mean = y_sum / window
            cov = 0.0
            for t in range(window):
                cov += (arr[start + t, j] - y_mean) * x_centered[t]
            out[i, j] = cov / denom
    return out


@njit(cache=True, parallel=True)
def _wide_trend_stats_numba(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows, n_cols = arr.shape
    slope = np.empty((n_rows, n_cols), dtype=np.float64)
    rsq = np.empty((n_rows, n_cols), dtype=np.float64)
    resi = np.empty((n_rows, n_cols), dtype=np.float64)
    slope[:] = np.nan
    rsq[:] = np.nan
    resi[:] = np.nan
    if window <= 0 or window > n_rows:
        return slope, rsq, resi

    x_mean = 0.5 * (window - 1)
    x_last = float(window - 1)
    denom = 0.0
    for t in range(window):
        diff = t - x_mean
        denom += diff * diff
    denom += _WIDE_EPS

    for j in prange(n_cols):
        for i in range(window - 1, n_rows):
            start = i - window + 1
            y_sum = 0.0
            valid = True
            for t in range(window):
                value = arr[start + t, j]
                if np.isnan(value):
                    valid = False
                    break
                y_sum += value
            if not valid:
                continue

            y_mean = y_sum / window
            cov = 0.0
            ss_tot = 0.0
            for t in range(window):
                y_val = arr[start + t, j]
                y_diff = y_val - y_mean
                x_diff = t - x_mean
                cov += y_diff * x_diff
                ss_tot += y_diff * y_diff

            beta = cov / denom
            alpha = y_mean - beta * x_mean
            ss_res = 0.0
            for t in range(window):
                y_val = arr[start + t, j]
                y_hat = alpha + beta * t
                diff = y_val - y_hat
                ss_res += diff * diff

            slope[i, j] = beta
            rsq[i, j] = 1.0 - ss_res / (ss_tot + _WIDE_EPS)
            resi[i, j] = arr[i, j] - (alpha + beta * x_last)

    return slope, rsq, resi


def wide_returns(df: pd.DataFrame) -> pd.DataFrame:
    frame = _as_float_frame(df)
    out = _empty_like(frame)
    if len(frame) < 2:
        return out
    arr = frame.to_numpy(dtype=float, copy=False)
    prev = arr[:-1, :]
    curr = arr[1:, :]
    tail = np.full_like(curr, np.nan, dtype=float)
    valid = (~np.isnan(prev)) & (~np.isnan(curr))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = curr / prev - 1.0
    tail[valid] = raw[valid]
    out.iloc[1:, :] = tail
    return out


def wide_ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.sum(view, axis=2)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_sma(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.mean(view, axis=2)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_stddev(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.std(view, axis=2, ddof=1)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_correlation(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    x_frame = _as_float_frame(x)
    y_frame = _as_float_frame(y).reindex_like(x_frame)
    x_arr, x_view = _rolling_view(x_frame, window)
    _, y_view = _rolling_view(y_frame, window)
    if len(x_view) == 0:
        return _empty_like(x_frame)
    valid = _full_window_valid(x_view) & _full_window_valid(y_view)
    mx = np.mean(x_view, axis=2)
    my = np.mean(y_view, axis=2)
    xc = x_view - mx[:, :, None]
    yc = y_view - my[:, :, None]
    cov = np.sum(xc * yc, axis=2) / max(window - 1, 1)
    sx = np.sqrt(np.sum(xc * xc, axis=2) / max(window - 1, 1))
    sy = np.sqrt(np.sum(yc * yc, axis=2) / max(window - 1, 1))
    tail = cov / (sx * sy + _WIDE_EPS)
    tail[~valid] = 0.0
    bad = ~np.isfinite(tail)
    tail[bad & valid] = 0.0
    out = _fill_tail_result(x_frame, tail, window)
    out.iloc[: window - 1, :] = np.nan
    return out


def wide_covariance(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    x_frame = _as_float_frame(x)
    y_frame = _as_float_frame(y).reindex_like(x_frame)
    _, x_view = _rolling_view(x_frame, window)
    _, y_view = _rolling_view(y_frame, window)
    if len(x_view) == 0:
        return _empty_like(x_frame)
    valid = _full_window_valid(x_view) & _full_window_valid(y_view)
    mx = np.mean(x_view, axis=2)
    my = np.mean(y_view, axis=2)
    tail = np.sum((x_view - mx[:, :, None]) * (y_view - my[:, :, None]), axis=2) / max(window - 1, 1)
    tail[~valid] = np.nan
    return _fill_tail_result(x_frame, tail, window)


def wide_ts_rank(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if HAS_NUMBA:
        tail = _wide_ts_rank_numba(frame.to_numpy(dtype=np.float64, copy=False), int(window))
        return pd.DataFrame(tail, index=frame.index, columns=frame.columns)
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    valid = _full_window_valid(view)
    last = view[:, :, -1][:, :, None]
    tail = np.sum(view < last, axis=2).astype(float) + 1.0
    tail[~valid] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_product(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.prod(view, axis=2)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_ts_min(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.min(view, axis=2)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_ts_max(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.max(view, axis=2)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_quantile(df: pd.DataFrame, window: int, q: float) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.quantile(view, float(q), axis=2)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_delta(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return df.diff(period)


def wide_delay(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return df.shift(period)


def wide_rank(df: pd.DataFrame) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if HAS_NUMBA:
        out = _wide_rank_numba(frame.to_numpy(dtype=np.float64, copy=False))
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)
    arr = frame.to_numpy(dtype=float, copy=False)
    n_rows, n_cols = arr.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    if n_cols == 0:
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)

    order = np.argsort(arr, axis=1, kind="mergesort")
    sorted_vals = np.take_along_axis(arr, order, axis=1)
    valid_sorted = ~np.isnan(sorted_vals)
    counts = valid_sorted.sum(axis=1)
    positions = np.broadcast_to(np.arange(1, n_cols + 1, dtype=float), (n_rows, n_cols))
    group_start = np.ones((n_rows, n_cols), dtype=bool)
    group_start[:, 1:] = sorted_vals[:, 1:] != sorted_vals[:, :-1]
    group_start &= valid_sorted
    min_rank_sorted = np.maximum.accumulate(np.where(group_start, positions, 0.0), axis=1)
    min_rank_sorted[~valid_sorted] = np.nan

    inverse = np.empty_like(order)
    row_idx = np.arange(n_rows)[:, None]
    inverse[row_idx, order] = np.arange(n_cols)
    out = np.take_along_axis(min_rank_sorted, inverse, axis=1)
    denom = np.where(counts > 0, counts, np.nan)[:, None]
    out = out / denom
    out[np.isnan(arr)] = np.nan
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def wide_abs(df):
    return df.abs()


def wide_sign(df):
    return np.sign(df)


def wide_log(df):
    return np.log(np.clip(df, 1e-12, None))


def wide_slog1p(df: pd.DataFrame) -> pd.DataFrame:
    frame = _as_float_frame(df)
    arr = frame.to_numpy(dtype=float, copy=False)
    out = np.sign(arr) * np.log1p(np.abs(arr))
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def wide_inv(df: pd.DataFrame) -> pd.DataFrame:
    frame = _as_float_frame(df)
    base = frame.where(frame.abs() > _WIDE_EPS, np.nan)
    return 1.0 / base


def wide_sqrt(df: pd.DataFrame) -> pd.DataFrame:
    frame = _as_float_frame(df)
    return pd.DataFrame(
        np.sqrt(np.clip(frame.to_numpy(dtype=float, copy=False), 0.0, None)),
        index=frame.index,
        columns=frame.columns,
    )


def wide_power(df: pd.DataFrame, p: pd.DataFrame | float) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if isinstance(p, pd.DataFrame):
        exponent = _as_float_frame(p).reindex_like(frame)
        out = np.power(frame.to_numpy(dtype=float, copy=False), exponent.to_numpy(dtype=float, copy=False))
    else:
        out = np.power(frame.to_numpy(dtype=float, copy=False), float(p))
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def wide_ts_var(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    std = wide_stddev(df, window)
    return std * std


def wide_ts_ir(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    mean = wide_sma(df, window)
    std = wide_stddev(df, window).replace(0.0, np.nan)
    return mean / std


def wide_ts_skew(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if int(window) < 3:
        raise ValueError("window must be >= 3 for wide_ts_skew")
    return frame.rolling(int(window)).skew()


def wide_ts_kurt(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if int(window) < 5:
        raise ValueError("window must be >= 5 for wide_ts_kurt")
    return frame.rolling(int(window)).kurt()


def wide_ts_med(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)
    return frame.rolling(int(window)).median()


def wide_ts_mad(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)

    def _mad(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        vals = values.astype(float)
        return float(np.mean(np.abs(vals - vals.mean())))

    return frame.rolling(int(window)).apply(_mad, raw=True)


def wide_ts_count(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)
    return frame.rolling(int(window)).count().astype(float)


def wide_ts_pct_change(df: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    frame = _as_float_frame(df)
    ref = wide_delay(frame, period).replace(0.0, np.nan)
    return frame / ref - 1.0


def wide_ts_max_diff(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)
    return frame - wide_ts_max(frame, window)


def wide_ts_min_diff(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame = _as_float_frame(df)
    return frame - wide_ts_min(frame, window)


def wide_ts_min_max_diff(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return wide_ts_max(df, window) - wide_ts_min(df, window)


def wide_cs_sum(df: pd.DataFrame) -> pd.DataFrame:
    frame = _as_float_frame(df)
    sums = frame.sum(axis=1)
    out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for col in frame.columns:
        out[col] = sums
    return out


def wide_cs_std(df: pd.DataFrame, ddof: int = 0) -> pd.DataFrame:
    frame = _as_float_frame(df)
    std = frame.std(axis=1, ddof=int(ddof))
    out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for col in frame.columns:
        out[col] = std
    return out


def wide_cs_skew(df: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    frame = _as_float_frame(df)
    min_count = int(min_count)

    def _calc(row: pd.Series) -> float:
        arr = row.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_count:
            return np.nan
        mu = float(arr.mean())
        xc = arr - mu
        m2 = float(np.mean(xc ** 2))
        if m2 <= _WIDE_EPS:
            return 0.0
        m3 = float(np.mean(xc ** 3))
        return float(m3 / (m2 ** 1.5))

    skew = frame.apply(_calc, axis=1)
    out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for col in frame.columns:
        out[col] = skew
    return out


def wide_upper_shadow(high: pd.DataFrame, open_: pd.DataFrame, close: pd.DataFrame, preclose: pd.DataFrame) -> pd.DataFrame:
    high_frame = _as_float_frame(high)
    open_frame = _as_float_frame(open_).reindex_like(high_frame)
    close_frame = _as_float_frame(close).reindex_like(high_frame)
    preclose_frame = _as_float_frame(preclose).reindex_like(high_frame).replace(0.0, np.nan)
    body_top = np.maximum(open_frame, close_frame)
    return (high_frame - body_top) / preclose_frame


def wide_ts_turnover_ref_price(close: pd.DataFrame, turn: pd.DataFrame, window: int) -> pd.DataFrame:
    close_frame = _as_float_frame(close)
    turn_frame = _as_float_frame(turn).reindex_like(close_frame)
    result = _series_ts_turnover_ref_price(
        _wide_to_series(close_frame),
        _wide_to_series(turn_frame),
        int(window),
    )
    return _series_to_wide(result, like=close_frame)


def wide_ts_sorted_mean_spread(
    value: pd.DataFrame,
    sort_key: pd.DataFrame,
    window: int,
    ratio: float,
) -> pd.DataFrame:
    value_frame = _as_float_frame(value)
    sort_key_frame = _as_float_frame(sort_key).reindex_like(value_frame)
    result = _series_ts_sorted_mean_spread(
        _wide_to_series(value_frame),
        _wide_to_series(sort_key_frame),
        int(window),
        float(ratio),
    )
    return _series_to_wide(result, like=value_frame)


def wide_rolling_cs_spearman_mean(
    value: pd.DataFrame,
    window: int,
    *,
    min_obs: int = 10,
) -> pd.DataFrame:
    value_frame = _as_float_frame(value)
    result = _series_rolling_cs_spearman_mean(
        _wide_to_series(value_frame),
        int(window),
        int(min_obs),
    )
    return _series_to_wide(result, like=value_frame)


def wide_cs_reg_resid(y: pd.DataFrame, x: pd.DataFrame, *, min_obs: int = 20) -> pd.DataFrame:
    y_frame = _as_float_frame(y)
    x_frame = _as_float_frame(x).reindex_like(y_frame)
    result = _series_cs_reg_resid(
        _wide_to_series(y_frame),
        _wide_to_series(x_frame),
        int(min_obs),
    )
    return _series_to_wide(result, like=y_frame)


def wide_cs_multi_reg_resid(y: pd.DataFrame, *xs: pd.DataFrame, min_obs: int = 20) -> pd.DataFrame:
    y_frame = _as_float_frame(y)
    x_frames = [_as_float_frame(item).reindex_like(y_frame) for item in xs]
    result = _series_cs_multi_reg_resid(
        _wide_to_series(y_frame),
        *[_wide_to_series(item) for item in x_frames],
        int(min_obs),
    )
    return _series_to_wide(result, like=y_frame)


def wide_ts_linear_decay_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return wide_decay_linear(df, window)


def wide_ts_exp_weighted_mean_lagged(df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = _as_float_frame(df)
    out = _empty_like(frame)
    if int(window) <= 0:
        raise ValueError("window must be positive")
    if len(frame) < 2:
        return out
    alpha = 2.0 / (1.0 + float(window))
    weights = np.array([(1.0 - alpha) ** i for i in range(int(window))], dtype=float)
    arr = frame.to_numpy(dtype=float, copy=False)
    tail = np.full_like(arr, np.nan, dtype=float)
    for i in range(1, len(frame)):
        hist = arr[max(0, i - int(window)) : i][::-1]
        valid = np.isfinite(hist)
        if hist.shape[0] == 0:
            continue
        w = weights[: hist.shape[0]][:, None]
        numer = np.where(valid, hist * w, 0.0).sum(axis=0)
        valid_count = valid.sum(axis=0)
        vals = np.full(hist.shape[1], np.nan, dtype=float)
        mask = valid_count > 0
        vals[mask] = numer[mask] / float(window)
        tail[i, :] = vals
    return pd.DataFrame(tail, index=frame.index, columns=frame.columns)


def wide_macd(df: pd.DataFrame, fast_window: int, slow_window: int) -> pd.DataFrame:
    fast = _wide_ema_like(df, fast_window, min_periods=int(fast_window))
    slow = _wide_ema_like(df, slow_window, min_periods=int(slow_window))
    return fast - slow


def _wide_ema_like(df: pd.DataFrame, window: int, *, min_periods: int | None = None) -> pd.DataFrame:
    frame = _as_float_frame(df)
    span = max(int(window), 1)
    min_periods = span if min_periods is None else int(min_periods)
    return frame.ewm(span=span, adjust=False, min_periods=min_periods).mean()


def _wide_rolling_regression(
    y: pd.DataFrame,
    x: pd.DataFrame,
    window: int,
    mode: str,
) -> pd.DataFrame:
    y_frame = _as_float_frame(y)
    x_frame = _as_float_frame(x).reindex_like(y_frame)
    _, y_view = _rolling_view(y_frame, window)
    _, x_view = _rolling_view(x_frame, window)
    if len(y_view) == 0:
        return _empty_like(y_frame)
    valid = _full_window_valid(y_view) & _full_window_valid(x_view)
    x_mean = np.mean(x_view, axis=2)
    y_mean = np.mean(y_view, axis=2)
    x_centered = x_view - x_mean[:, :, None]
    y_centered = y_view - y_mean[:, :, None]
    var_x = np.sum(x_centered * x_centered, axis=2)
    cov_xy = np.sum(x_centered * y_centered, axis=2)
    beta = cov_xy / (var_x + _WIDE_EPS)
    alpha = y_mean - beta * x_mean

    if mode == "slope":
        tail = beta
    else:
        y_hat = alpha[:, :, None] + beta[:, :, None] * x_view
        ss_res = np.sum((y_view - y_hat) ** 2, axis=2)
        ss_tot = np.sum((y_view - y_mean[:, :, None]) ** 2, axis=2)
        if mode == "rsq":
            tail = 1.0 - ss_res / (ss_tot + _WIDE_EPS)
        elif mode == "residual":
            tail = y_view[:, :, -1] - (alpha + beta * x_view[:, :, -1])
        else:
            raise ValueError(f"unsupported regression mode: {mode}")
    tail = tail.astype(float, copy=False)
    tail[~valid] = np.nan
    return _fill_tail_result(y_frame, tail, window)


def wide_regression_slope(y: pd.DataFrame, x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _wide_rolling_regression(y, x, window, mode="slope")


def wide_regression_rsq(y: pd.DataFrame, x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _wide_rolling_regression(y, x, window, mode="rsq")


def wide_regression_residual(y: pd.DataFrame, x: pd.DataFrame, window: int) -> pd.DataFrame:
    return _wide_rolling_regression(y, x, window, mode="residual")


def wide_scale(df: pd.DataFrame, k: float = 1) -> pd.DataFrame:
    return df.mul(k).div(np.abs(df).sum())


def wide_ts_argmax(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.argmax(view, axis=2).astype(float) + 1.0
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_ts_argmin(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.argmin(view, axis=2).astype(float) + 1.0
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_decay_linear(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    frame, view = _rolling_view(df, period)
    if len(view) == 0:
        return _empty_like(df)
    weights = np.arange(1, period + 1, dtype=float)
    tail = np.sum(view * weights[None, None, :], axis=2) / (weights.sum() + _WIDE_EPS)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, period)


def wide_max(left, right):
    return np.maximum(left, right)


def wide_min(left, right):
    return np.minimum(left, right)


def wide_greater(left, right):
    return np.maximum(left, right)


def wide_less(left, right):
    return np.minimum(left, right)


def wide_rowmax(df: pd.DataFrame) -> pd.Series:
    return df.max(axis=1)


def wide_rowmin(df: pd.DataFrame) -> pd.Series:
    return df.min(axis=1)


def wide_sma_ewm(df: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    return df.ewm(alpha=m / n, adjust=False).mean()


def wide_sequence(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


def wide_regbeta(df: pd.DataFrame, x: np.ndarray) -> pd.DataFrame:
    frame = _as_float_frame(df)
    window = len(x)
    x_axis = np.asarray(x, dtype=float)
    x_centered = x_axis - x_axis.mean()
    denom = float(np.dot(x_centered, x_centered)) + _WIDE_EPS
    if HAS_NUMBA:
        out = _wide_regbeta_numba(
            frame.to_numpy(dtype=np.float64, copy=False),
            x_centered.astype(np.float64),
            float(x_axis.mean()),
            denom,
        )
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)
    _, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    y_mean = np.mean(view, axis=2)
    tail = np.sum((view - y_mean[:, :, None]) * x_centered[None, None, :], axis=2) / denom
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_lowday(df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = (window - np.argmin(view, axis=2)).astype(float)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_highday(df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = (window - np.argmax(view, axis=2)).astype(float)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_wma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    weights = np.array(range(window - 1, -1, -1))
    weights = np.power(0.9, weights)
    frame, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    tail = np.sum(view * weights[None, None, :], axis=2) / (weights.sum() + _WIDE_EPS)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def _trend_slope(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    window = len(values)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered)) + 1e-12
    y = values.astype(float)
    y_centered = y - y.mean()
    return float(np.dot(x_centered, y_centered) / denom)


def _trend_rsquare(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    window = len(values)
    x = np.arange(window, dtype=float)
    beta, alpha = np.polyfit(x, values.astype(float), deg=1)
    y_hat = alpha + beta * x
    ss_res = float(np.sum((values - y_hat) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _trend_resi(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    window = len(values)
    x = np.arange(window, dtype=float)
    beta, alpha = np.polyfit(x, values.astype(float), deg=1)
    return float(values[-1] - (alpha + beta * x[-1]))


def wide_slope(df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if HAS_NUMBA:
        slope_tail, _, _ = _wide_trend_stats_numba(frame.to_numpy(dtype=np.float64, copy=False), int(window))
        return pd.DataFrame(slope_tail, index=frame.index, columns=frame.columns)
    _, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered)) + _WIDE_EPS
    y_mean = np.mean(view, axis=2)
    tail = np.sum((view - y_mean[:, :, None]) * x_centered[None, None, :], axis=2) / denom
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_rsquare(df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if HAS_NUMBA:
        _, rsq_tail, _ = _wide_trend_stats_numba(frame.to_numpy(dtype=np.float64, copy=False), int(window))
        return pd.DataFrame(rsq_tail, index=frame.index, columns=frame.columns)
    _, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered)) + _WIDE_EPS
    y_mean = np.mean(view, axis=2)
    beta = np.sum((view - y_mean[:, :, None]) * x_centered[None, None, :], axis=2) / denom
    alpha = y_mean - beta * x.mean()
    y_hat = alpha[:, :, None] + beta[:, :, None] * x[None, None, :]
    ss_res = np.sum((view - y_hat) ** 2, axis=2)
    ss_tot = np.sum((view - y_mean[:, :, None]) ** 2, axis=2)
    tail = 1.0 - ss_res / (ss_tot + _WIDE_EPS)
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_resi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = _as_float_frame(df)
    if HAS_NUMBA:
        _, _, resi_tail = _wide_trend_stats_numba(frame.to_numpy(dtype=np.float64, copy=False), int(window))
        return pd.DataFrame(resi_tail, index=frame.index, columns=frame.columns)
    _, view = _rolling_view(df, window)
    if len(view) == 0:
        return _empty_like(df)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.dot(x_centered, x_centered)) + _WIDE_EPS
    y_mean = np.mean(view, axis=2)
    beta = np.sum((view - y_mean[:, :, None]) * x_centered[None, None, :], axis=2) / denom
    alpha = y_mean - beta * x.mean()
    tail = view[:, :, -1] - (alpha + beta * x[-1])
    tail[~_full_window_valid(view)] = np.nan
    return _fill_tail_result(df, tail, window)


def wide_count(cond: pd.DataFrame, window: int) -> pd.DataFrame:
    return wide_ts_sum(_as_float_frame(cond.astype(float, copy=False)), window)


def wide_sumif(df: pd.DataFrame, window: int, cond: pd.DataFrame) -> pd.DataFrame:
    masked = _as_float_frame(df).copy()
    masked[~cond] = 0
    return wide_ts_sum(masked, window)
