from __future__ import annotations

"""Reusable path-retrieval helpers for high-order pattern factors."""

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration dependency
    def njit(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    def prange(*args):
        return range(*args)


_EPS = 1e-12
_STAT_MEAN = 1
_STAT_INV_STD = 2
_WEIGHT_NONE = 0
_WEIGHT_AGE = 1
_WEIGHT_RECENCY_RANK = 2


def _coerce_close_frame(close: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(close, pd.DataFrame):
        raise TypeError("close must be a DataFrame")
    return close.astype(float, copy=False).sort_index().sort_index(axis=1)


def _coerce_benchmark_series(
    benchmark_close: pd.Series | pd.DataFrame | None,
    *,
    target_index: pd.Index,
) -> np.ndarray:
    if benchmark_close is None:
        return np.array([], dtype=np.float64)
    if isinstance(benchmark_close, pd.DataFrame):
        if benchmark_close.shape[1] == 0:
            return np.array([], dtype=np.float64)
        series = benchmark_close.iloc[:, 0]
    elif isinstance(benchmark_close, pd.Series):
        series = benchmark_close
    else:
        raise TypeError("benchmark_close must be Series, DataFrame, or None")
    return series.reindex(target_index).to_numpy(dtype=np.float64)


@njit(cache=True)
def _calc_ret_1d(close_arr: np.ndarray) -> np.ndarray:
    n = len(close_arr)
    ret = np.empty(n, dtype=np.float64)
    ret[:] = np.nan
    for i in range(1, n):
        prev_close = close_arr[i - 1]
        curr_close = close_arr[i]
        if np.isfinite(prev_close) and np.isfinite(curr_close) and prev_close > 0.0 and curr_close > 0.0:
            ret[i] = curr_close / prev_close - 1.0
    return ret


@njit(cache=True)
def _weighted_std(values: np.ndarray, weights: np.ndarray, count: int) -> float:
    weight_sum = 0.0
    weighted_mean = 0.0
    for i in range(count):
        weight_sum += weights[i]
        weighted_mean += weights[i] * values[i]
    if weight_sum <= 0.0:
        return np.nan
    weighted_mean /= weight_sum

    variance = 0.0
    for i in range(count):
        diff = values[i] - weighted_mean
        variance += weights[i] * diff * diff
    variance /= weight_sum
    if variance <= 0.0:
        return 0.0
    return np.sqrt(variance)


@njit(cache=True)
def _simple_std(values: np.ndarray, count: int) -> float:
    if count <= 0:
        return np.nan
    mean = 0.0
    for i in range(count):
        mean += values[i]
    mean /= count

    variance = 0.0
    for i in range(count):
        diff = values[i] - mean
        variance += diff * diff
    variance /= count
    if variance <= 0.0:
        return 0.0
    return np.sqrt(variance)


@njit(cache=True)
def _selection_sort_pairs(values: np.ndarray, ages: np.ndarray, count: int) -> None:
    for i in range(count - 1):
        min_idx = i
        min_age = ages[i]
        for j in range(i + 1, count):
            if ages[j] < min_age:
                min_idx = j
                min_age = ages[j]
        if min_idx != i:
            age_tmp = ages[i]
            ages[i] = ages[min_idx]
            ages[min_idx] = age_tmp
            val_tmp = values[i]
            values[i] = values[min_idx]
            values[min_idx] = val_tmp


@njit(cache=True)
def _aggregate_values(
    values: np.ndarray,
    ages: np.ndarray,
    count: int,
    stat_mode: int,
    weight_mode: int,
    half_life: float,
    negate: bool,
) -> float:
    if count <= 0:
        return np.nan

    lam = 0.0
    use_weights = weight_mode != _WEIGHT_NONE and half_life > 0.0
    if use_weights:
        lam = np.log(2.0) / half_life

    if stat_mode == _STAT_MEAN:
        if not use_weights:
            mean = 0.0
            for i in range(count):
                mean += values[i]
            mean /= count
            return -mean if negate else mean

        weights = np.empty(count, dtype=np.float64)
        if weight_mode == _WEIGHT_AGE:
            for i in range(count):
                weights[i] = np.exp(-lam * ages[i])
        else:
            _selection_sort_pairs(values, ages, count)
            for i in range(count):
                rank_from_oldest = count - i
                weights[i] = np.exp(-lam * rank_from_oldest)

        weighted_sum = 0.0
        weight_sum = 0.0
        for i in range(count):
            weighted_sum += weights[i] * values[i]
            weight_sum += weights[i]
        if weight_sum <= 0.0:
            return np.nan
        mean = weighted_sum / weight_sum
        return -mean if negate else mean

    if stat_mode == _STAT_INV_STD:
        std_val = np.nan
        if not use_weights:
            std_val = _simple_std(values, count)
        else:
            weights = np.empty(count, dtype=np.float64)
            if weight_mode == _WEIGHT_AGE:
                for i in range(count):
                    weights[i] = np.exp(-lam * ages[i])
            else:
                _selection_sort_pairs(values, ages, count)
                for i in range(count):
                    rank_from_oldest = count - i
                    weights[i] = np.exp(-lam * rank_from_oldest)
            std_val = _weighted_std(values, weights, count)
        if np.isfinite(std_val) and std_val > _EPS:
            return 1.0 / std_val
        return np.nan

    return np.nan


@njit(cache=True, parallel=True)
def _path_match_future_stat_2d(
    close_arr: np.ndarray,
    benchmark_close_arr: np.ndarray,
    rw: int,
    history_window: int,
    threshold: float,
    holding_time: int,
    min_matches: int,
    use_excess_return: bool,
    stat_mode: int,
    weight_mode: int,
    half_life: float,
    negate: bool,
) -> np.ndarray:
    n_rows, n_cols = close_arr.shape
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    out[:] = np.nan
    if rw <= 1 or history_window <= 0 or holding_time <= 0 or n_rows == 0 or n_cols == 0:
        return out

    benchmark_ret = np.empty(0, dtype=np.float64)
    if use_excess_return:
        benchmark_ret = _calc_ret_1d(benchmark_close_arr)

    first_valid_t = history_window + rw - 1
    max_matches = history_window + 1

    for col in prange(n_cols):
        stock_close = close_arr[:, col]
        stock_ret = _calc_ret_1d(stock_close)
        current_seq = np.empty(rw, dtype=np.float64)
        matched_values = np.empty(max_matches, dtype=np.float64)
        matched_ages = np.empty(max_matches, dtype=np.float64)

        for t in range(first_valid_t, n_rows):
            current_start = t - rw + 1
            hist_start = current_start - history_window
            hist_end = current_start - holding_time - rw
            if hist_start < 0 or hist_end < hist_start:
                continue

            current_mean = 0.0
            current_ss = 0.0
            valid_current = True
            for k in range(rw):
                value = stock_close[current_start + k]
                if not np.isfinite(value):
                    valid_current = False
                    break
                current_seq[k] = value
                current_mean += value
            if not valid_current:
                continue
            current_mean /= rw
            for k in range(rw):
                diff = current_seq[k] - current_mean
                current_ss += diff * diff
            if current_ss <= _EPS:
                continue

            match_count = 0
            for s in range(hist_start, hist_end + 1):
                hist_mean = 0.0
                valid_hist = True
                for k in range(rw):
                    hist_value = stock_close[s + k]
                    if not np.isfinite(hist_value):
                        valid_hist = False
                        break
                    hist_mean += hist_value
                if not valid_hist:
                    continue
                hist_mean /= rw

                hist_ss = 0.0
                cov = 0.0
                for k in range(rw):
                    hist_diff = stock_close[s + k] - hist_mean
                    curr_diff = current_seq[k] - current_mean
                    hist_ss += hist_diff * hist_diff
                    cov += hist_diff * curr_diff
                if hist_ss <= _EPS:
                    continue
                corr = cov / np.sqrt(hist_ss * current_ss)
                if not np.isfinite(corr) or np.abs(corr) < threshold:
                    continue

                cum_ret = 1.0
                valid_future = True
                future_start = s + rw
                future_end = future_start + holding_time
                if future_end > n_rows:
                    continue
                for u in range(future_start, future_end):
                    value = stock_ret[u]
                    if use_excess_return:
                        bench_value = benchmark_ret[u]
                        if not np.isfinite(value) or not np.isfinite(bench_value):
                            valid_future = False
                            break
                        value = value - bench_value
                    elif not np.isfinite(value):
                        valid_future = False
                        break
                    cum_ret *= 1.0 + value
                if not valid_future:
                    continue

                matched_values[match_count] = cum_ret - 1.0
                matched_ages[match_count] = current_start - s
                match_count += 1

            if match_count < min_matches:
                continue

            out[t, col] = _aggregate_values(
                matched_values,
                matched_ages,
                match_count,
                stat_mode,
                weight_mode,
                half_life,
                negate,
            )

    return out


def _run_path_match(
    close_df: pd.DataFrame,
    *,
    benchmark_close: pd.Series | pd.DataFrame | None = None,
    rw: int,
    history_window: int,
    threshold: float,
    holding_time: int,
    min_matches: int,
    use_excess_return: bool,
    stat_mode: int,
    weight_mode: int,
    half_life: float,
    negate: bool,
) -> pd.DataFrame:
    close_df = _coerce_close_frame(close_df)
    benchmark_arr = _coerce_benchmark_series(benchmark_close, target_index=close_df.index)
    close_arr = close_df.to_numpy(dtype=np.float64, copy=False)
    out_arr = _path_match_future_stat_2d(
        close_arr,
        benchmark_arr,
        int(rw),
        int(history_window),
        float(threshold),
        int(holding_time),
        int(min_matches),
        bool(use_excess_return),
        int(stat_mode),
        int(weight_mode),
        float(half_life),
        bool(negate),
    )
    return pd.DataFrame(out_arr, index=close_df.index, columns=close_df.columns)


def similar_low_volatility_frame(
    close_df: pd.DataFrame,
    *,
    rw: int = 6,
    history_window: int = 120,
    threshold: float = 0.4,
    holding_time: int = 6,
    min_matches: int = 5,
    use_exponential_weight: bool = True,
    half_life: float = 20.0,
) -> pd.DataFrame:
    return _run_path_match(
        close_df,
        rw=rw,
        history_window=history_window,
        threshold=threshold,
        holding_time=holding_time,
        min_matches=min_matches,
        use_excess_return=False,
        stat_mode=_STAT_INV_STD,
        weight_mode=_WEIGHT_AGE if use_exponential_weight else _WEIGHT_NONE,
        half_life=half_life,
        negate=False,
    )


def similar_reverse_frame(
    close_df: pd.DataFrame,
    benchmark_close: pd.Series | pd.DataFrame,
    *,
    rw: int = 6,
    history_window: int = 120,
    threshold: float = 0.4,
    holding_time: int = 6,
    half_life: float = 6.0,
) -> pd.DataFrame:
    return _run_path_match(
        close_df,
        benchmark_close=benchmark_close,
        rw=rw,
        history_window=history_window,
        threshold=threshold,
        holding_time=holding_time,
        min_matches=1,
        use_excess_return=True,
        stat_mode=_STAT_MEAN,
        weight_mode=_WEIGHT_RECENCY_RANK,
        half_life=half_life,
        negate=True,
    )
