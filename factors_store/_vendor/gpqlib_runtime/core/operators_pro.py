from __future__ import annotations

import math
import os
from functools import lru_cache
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    njit = None  # type: ignore[assignment]
    _NUMBA_AVAILABLE = False

EPS = 1e-12
# 可用 GP_USE_NUMBA=0 禁用 JIT，回退 numpy 向量化路径。
_USE_NUMBA = _NUMBA_AVAILABLE and os.getenv("GP_USE_NUMBA", "1").strip().lower() not in {"0", "false", "no"}


if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=False)
    def _nb_rolling_weighted_mean_2d(arr: np.ndarray, window: int, w: np.ndarray, eps: float) -> np.ndarray:
        t, n = arr.shape
        out = np.empty((t, n), dtype=np.float64)
        out[:, :] = np.nan
        if window > t:
            return out
        for j in range(n):
            for i in range(window - 1, t):
                start = i - window + 1
                cnt = 0
                sum_w = 0.0
                sum_x = 0.0
                for k in range(window):
                    v = arr[start + k, j]
                    if not np.isnan(v):
                        cnt += 1
                        wk = w[k]
                        sum_w += wk
                        sum_x += wk * v
                if cnt < window:
                    continue
                out[i, j] = sum_x / (sum_w + eps)
        return out

    @njit(cache=True, fastmath=False)
    def _nb_rolling_weighted_std_2d(arr: np.ndarray, window: int, w: np.ndarray, eps: float) -> np.ndarray:
        t, n = arr.shape
        out = np.empty((t, n), dtype=np.float64)
        out[:, :] = np.nan
        if window > t:
            return out
        for j in range(n):
            for i in range(window - 1, t):
                start = i - window + 1
                cnt = 0
                sum_w = 0.0
                sum_x = 0.0
                for k in range(window):
                    v = arr[start + k, j]
                    if not np.isnan(v):
                        cnt += 1
                        wk = w[k]
                        sum_w += wk
                        sum_x += wk * v
                if cnt < window:
                    continue
                mean = sum_x / (sum_w + eps)
                var = 0.0
                for k in range(window):
                    v = arr[start + k, j]
                    if not np.isnan(v):
                        d = v - mean
                        var += w[k] * d * d
                var = var / (sum_w + eps)
                if var < 0.0:
                    var = 0.0
                out[i, j] = np.sqrt(var)
        return out

    @njit(cache=True, fastmath=False)
    def _nb_rolling_weighted_cov_2d(x_arr: np.ndarray, y_arr: np.ndarray, window: int, w: np.ndarray, eps: float) -> np.ndarray:
        t, n = x_arr.shape
        out = np.empty((t, n), dtype=np.float64)
        out[:, :] = np.nan
        if window > t:
            return out
        for j in range(n):
            for i in range(window - 1, t):
                start = i - window + 1
                sum_w = 0.0
                sum_x = 0.0
                sum_y = 0.0
                cnt = 0
                for k in range(window):
                    xv = x_arr[start + k, j]
                    yv = y_arr[start + k, j]
                    if not np.isnan(xv) and not np.isnan(yv):
                        cnt += 1
                        wk = w[k]
                        sum_w += wk
                        sum_x += wk * xv
                        sum_y += wk * yv
                if cnt == 0:
                    continue
                mx = sum_x / (sum_w + eps)
                my = sum_y / (sum_w + eps)
                cov = 0.0
                for k in range(window):
                    xv = x_arr[start + k, j]
                    yv = y_arr[start + k, j]
                    if not np.isnan(xv) and not np.isnan(yv):
                        cov += w[k] * (xv - mx) * (yv - my)
                out[i, j] = cov / (sum_w + eps)
        return out

    @njit(cache=True, fastmath=False)
    def _nb_rolling_weighted_regression_2d(
        y_arr: np.ndarray,
        x_arr: np.ndarray,
        window: int,
        w: np.ndarray,
        eps: float,
        mode: int,
    ) -> np.ndarray:
        # mode: 0 slope, 1 rsq, 2 residual
        t, n = y_arr.shape
        out = np.empty((t, n), dtype=np.float64)
        out[:, :] = np.nan
        if window > t:
            return out

        for j in range(n):
            for i in range(window - 1, t):
                start = i - window + 1
                cnt = 0
                sum_w = 0.0
                sx = 0.0
                sy = 0.0
                sxx = 0.0
                sxy = 0.0
                last_x = 0.0
                last_y = 0.0
                has_last = False
                for k in range(window):
                    xv = x_arr[start + k, j]
                    yv = y_arr[start + k, j]
                    if not np.isnan(xv) and not np.isnan(yv):
                        cnt += 1
                        wk = w[k]
                        sum_w += wk
                        sx += wk * xv
                        sy += wk * yv
                        sxx += wk * xv * xv
                        sxy += wk * xv * yv
                        last_x = xv
                        last_y = yv
                        has_last = True
                if cnt < 2:
                    continue

                mx = sx / (sum_w + eps)
                my = sy / (sum_w + eps)
                cov = sxy - (sx * sy) / (sum_w + eps)
                var = sxx - (sx * sx) / (sum_w + eps)
                beta = cov / (var + eps)
                alpha = my - beta * mx

                if mode == 0:
                    out[i, j] = beta
                    continue
                if mode == 2:
                    if has_last:
                        out[i, j] = last_y - (alpha + beta * last_x)
                    continue

                ss_res = 0.0
                ss_tot = 0.0
                for k in range(window):
                    xv = x_arr[start + k, j]
                    yv = y_arr[start + k, j]
                    if not np.isnan(xv) and not np.isnan(yv):
                        wk = w[k]
                        y_hat = alpha + beta * xv
                        dy = yv - y_hat
                        ss_res += wk * dy * dy
                        dt = yv - my
                        ss_tot += wk * dt * dt
                out[i, j] = 1.0 - (ss_res / (ss_tot + eps))
        return out



def _validate_mi(s: pd.Series) -> None:
    if not isinstance(s.index, pd.MultiIndex) or s.index.nlevels < 2:
        raise ValueError("Series index must be MultiIndex [datetime, instrument]")


def _reindex_like(a: pd.Series, b: pd.Series) -> pd.Series:
    return b if a.index.equals(b.index) else b.reindex(a.index)


def _exp_weights(window: int, half_life: int | None) -> np.ndarray:
    if half_life is None:
        # 未设置半衰期时采用等权。
        return np.ones(window, dtype=float)
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    decay = np.log(2) / half_life
    # 从最旧到最新
    offsets = np.arange(window - 1, -1, -1)
    w = np.exp(-decay * offsets)
    return w / (w.sum() + EPS)


def _validate_window(window: int) -> None:
    if int(window) <= 0:
        raise ValueError("window must be positive")


def _to_wide_2d(series: pd.Series) -> tuple[np.ndarray, pd.Index, pd.Index]:
    _validate_mi(series)
    wide = series.unstack(level=1)
    return wide.to_numpy(dtype=float, copy=False), wide.index, wide.columns


def _stack_wide(wide: pd.DataFrame) -> pd.Series:
    try:
        return wide.stack(future_stack=True)
    except TypeError:
        return wide.stack(dropna=False)


def _from_wide_2d(
    values: np.ndarray,
    dates: pd.Index,
    instruments: pd.Index,
    target_index: pd.MultiIndex,
    names: list[str | None],
) -> pd.Series:
    out = pd.DataFrame(values, index=dates, columns=instruments)
    s = _stack_wide(out)
    s.index = s.index.set_names(names)
    return s.reindex(target_index)


@lru_cache(maxsize=None)
def _hp_system_matrix(n: int, lamb: float) -> np.ndarray:
    n = int(n)
    if n < 4:
        return np.eye(n, dtype=float)
    I = np.eye(n, dtype=float)
    D = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return I + float(lamb) * (D.T @ D)


def _rolling_weighted_mean_std_2d(arr: np.ndarray, window: int, w: np.ndarray, stat: str) -> np.ndarray:
    if _USE_NUMBA:
        if stat == "mean":
            return _nb_rolling_weighted_mean_2d(arr, int(window), w, float(EPS))
        if stat == "std":
            return _nb_rolling_weighted_std_2d(arr, int(window), w, float(EPS))
        raise ValueError(f"Unknown stat: {stat}")

    out = np.full(arr.shape, np.nan, dtype=float)
    t, _ = arr.shape
    if window > t:
        return out

    v = sliding_window_view(arr, window_shape=window, axis=0)  # (T-W+1, N, W)
    mask = ~np.isnan(v)
    # 兼容旧实现 rolling(..., min_periods=window)：窗口内必须全非空才计算。
    valid = mask.sum(axis=2) >= window
    den = (mask * w).sum(axis=2)
    sum_x = (np.where(mask, v, 0.0) * w).sum(axis=2)
    mean = sum_x / (den + EPS)
    mean[~valid] = np.nan

    if stat == "mean":
        tail = mean
    elif stat == "std":
        var = (np.where(mask, v - mean[..., None], 0.0) ** 2 * w).sum(axis=2) / (den + EPS)
        var[~valid] = np.nan
        tail = np.sqrt(np.clip(var, 0.0, None))
    else:
        raise ValueError(f"Unknown stat: {stat}")

    out[window - 1 :, :] = tail
    return out


def _rolling_weighted_cov_2d(x_arr: np.ndarray, y_arr: np.ndarray, window: int, w: np.ndarray) -> np.ndarray:
    if _USE_NUMBA:
        return _nb_rolling_weighted_cov_2d(x_arr, y_arr, int(window), w, float(EPS))

    out = np.full(x_arr.shape, np.nan, dtype=float)
    t, _ = x_arr.shape
    if window > t:
        return out

    vx = sliding_window_view(x_arr, window_shape=window, axis=0)  # (T-W+1, N, W)
    vy = sliding_window_view(y_arr, window_shape=window, axis=0)
    mask = (~np.isnan(vx)) & (~np.isnan(vy))
    den = (mask * w).sum(axis=2)

    x0 = np.where(mask, vx, 0.0)
    y0 = np.where(mask, vy, 0.0)
    mx = (x0 * w).sum(axis=2) / (den + EPS)
    my = (y0 * w).sum(axis=2) / (den + EPS)

    cov = (np.where(mask, (vx - mx[..., None]) * (vy - my[..., None]), 0.0) * w).sum(axis=2) / (den + EPS)
    cov[den <= 0] = np.nan
    out[window - 1 :, :] = cov
    return out


def _rolling_weighted_regression_2d(
    y_arr: np.ndarray,
    x_arr: np.ndarray,
    window: int,
    w: np.ndarray,
    mode: str,
) -> np.ndarray:
    if _USE_NUMBA:
        mode_map = {"slope": 0, "rsq": 1, "residual": 2}
        if mode not in mode_map:
            raise ValueError(f"Unknown mode: {mode}")
        return _nb_rolling_weighted_regression_2d(
            y_arr,
            x_arr,
            int(window),
            w,
            float(EPS),
            int(mode_map[mode]),
        )

    out = np.full(y_arr.shape, np.nan, dtype=float)
    t, _ = y_arr.shape
    if window > t:
        return out

    vx = sliding_window_view(x_arr, window_shape=window, axis=0)  # (T-W+1, N, W)
    vy = sliding_window_view(y_arr, window_shape=window, axis=0)
    mask = (~np.isnan(vx)) & (~np.isnan(vy))
    cnt = mask.sum(axis=2)
    den = (mask * w).sum(axis=2)
    valid = cnt >= 2

    x0 = np.where(mask, vx, 0.0)
    y0 = np.where(mask, vy, 0.0)
    sx = (x0 * w).sum(axis=2)
    sy = (y0 * w).sum(axis=2)
    sxx = (x0 * x0 * w).sum(axis=2)
    sxy = (x0 * y0 * w).sum(axis=2)

    mx = sx / (den + EPS)
    my = sy / (den + EPS)
    cov = sxy - (sx * sy) / (den + EPS)
    var = sxx - (sx * sx) / (den + EPS)
    beta = cov / (var + EPS)
    alpha = my - beta * mx

    tail = np.full_like(beta, np.nan, dtype=float)
    if mode == "slope":
        tail[valid] = beta[valid]
    elif mode == "rsq":
        y_hat = alpha[..., None] + beta[..., None] * vx
        ss_res = (np.where(mask, (vy - y_hat) ** 2, 0.0) * w).sum(axis=2) / (den + EPS)
        ss_tot = (np.where(mask, (vy - my[..., None]) ** 2, 0.0) * w).sum(axis=2) / (den + EPS)
        rsq = 1.0 - ss_res / (ss_tot + EPS)
        tail[valid] = rsq[valid]
    elif mode == "residual":
        # 与旧实现一致：取窗口内“最后一个有效样本对”的残差。
        last_pos = window - 1 - np.argmax(mask[..., ::-1], axis=2)
        x_last = np.take_along_axis(vx, last_pos[..., None], axis=2).squeeze(axis=2)
        y_last = np.take_along_axis(vy, last_pos[..., None], axis=2).squeeze(axis=2)
        resid = y_last - (alpha + beta * x_last)
        tail[valid] = resid[valid]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out[window - 1 :, :] = tail
    return out


def _rolling_weighted(series: pd.Series, window: int, half_life: int | None, stat: str) -> pd.Series:
    _validate_window(window)
    _validate_mi(series)
    w = _exp_weights(window, half_life)
    arr, dates, instruments = _to_wide_2d(series)
    values = _rolling_weighted_mean_std_2d(arr, window, w, stat=stat)
    return _from_wide_2d(values, dates, instruments, series.index, list(series.index.names[:2]))


def ts_mean(x: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """时间序列滚动均值，可选半衰期加权。"""
    _validate_window(window)
    if half_life is None:
        _validate_mi(x)
        out = x.groupby(level=1, group_keys=False).rolling(window).mean().droplevel(0)
        return out.reindex(x.index)
    return _rolling_weighted(x, window, half_life, stat="mean")


def ts_std(x: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """时间序列滚动标准差，可选半衰期加权。"""
    _validate_window(window)
    if half_life is None:
        _validate_mi(x)
        out = x.groupby(level=1, group_keys=False).rolling(window).std().droplevel(0)
        return out.reindex(x.index)
    return _rolling_weighted(x, window, half_life, stat="std")


def ts_var(x: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """时间序列滚动方差。"""
    std = ts_std(x, window, half_life=half_life)
    return pd.Series(np.square(std.to_numpy(dtype=float)), index=std.index)


def ts_ir(x: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """时间序列信息比率，等于均值除以标准差。"""
    mean = ts_mean(x, window, half_life=half_life)
    std = ts_std(x, window, half_life=half_life)
    return mean / (std + EPS)


def ts_skew(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动偏度。"""
    _validate_window(window)
    if int(window) < 3:
        raise ValueError("window must be >= 3 for ts_skew")
    _validate_mi(x)
    out = x.groupby(level=1, group_keys=False).rolling(window).skew().droplevel(0)
    return out.reindex(x.index)


def ts_kurt(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动峰度。"""
    _validate_window(window)
    if int(window) < 5:
        raise ValueError("window must be >= 5 for ts_kurt")
    _validate_mi(x)
    out = x.groupby(level=1, group_keys=False).rolling(window).kurt().droplevel(0)
    return out.reindex(x.index)


def ts_max(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动最大值。"""
    _validate_window(window)
    _validate_mi(x)
    out = x.groupby(level=1, group_keys=False).rolling(window).max().droplevel(0)
    return out.reindex(x.index)


def ts_min(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动最小值。"""
    _validate_window(window)
    _validate_mi(x)
    out = x.groupby(level=1, group_keys=False).rolling(window).min().droplevel(0)
    return out.reindex(x.index)


def ts_max_diff(x: pd.Series, window: int) -> pd.Series:
    """当前值减去窗口内最大值。"""
    return x - ts_max(x, window)


def ts_min_diff(x: pd.Series, window: int) -> pd.Series:
    """当前值减去窗口内最小值。"""
    return x - ts_min(x, window)


def ts_min_max_diff(x: pd.Series, window: int) -> pd.Series:
    """窗口内最大值与最小值之差。"""
    return ts_max(x, window) - ts_min(x, window)


def ts_sum(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动求和。"""
    _validate_window(window)
    _validate_mi(x)
    out = x.groupby(level=1, group_keys=False).rolling(window).sum().droplevel(0)
    return out.reindex(x.index)


def ts_count(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动有效值计数。"""
    _validate_window(window)
    _validate_mi(x)
    out = x.groupby(level=1, group_keys=False).rolling(window).count().droplevel(0)
    return out.reindex(x.index).astype(float)


def ts_rank(x: pd.Series, window: int) -> pd.Series:
    """当前值在滚动窗口中的分位排名。"""
    _validate_window(window)
    _validate_mi(x)

    def _last_rank(arr: np.ndarray) -> float:
        if np.all(np.isnan(arr)):
            return np.nan
        return float(pd.Series(arr).rank(pct=True).iloc[-1])

    out = x.groupby(level=1, group_keys=False).rolling(window).apply(_last_rank, raw=True).droplevel(0)
    return out.reindex(x.index)


def ts_quantile(x: pd.Series, window: int, q: float) -> pd.Series:
    """时间序列滚动分位数。"""
    _validate_window(window)
    _validate_mi(x)
    qv = float(q)
    if not 0.0 <= qv <= 1.0:
        raise ValueError("q must be in [0, 1]")
    out = x.groupby(level=1, group_keys=False).rolling(window).quantile(qv).droplevel(0)
    return out.reindex(x.index)


def ts_med(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动中位数。"""
    _validate_window(window)
    _validate_mi(x)
    out = x.groupby(level=1, group_keys=False).rolling(window).median().droplevel(0)
    return out.reindex(x.index)


def ts_mad(x: pd.Series, window: int) -> pd.Series:
    """时间序列滚动平均绝对离差。"""
    _validate_window(window)
    _validate_mi(x)

    def _mad(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        vals = arr.astype(float)
        return float(np.mean(np.abs(vals - vals.mean())))

    out = x.groupby(level=1, group_keys=False).rolling(window).apply(_mad, raw=True).droplevel(0)
    return out.reindex(x.index)


def _rolling_argextreme(x: pd.Series, window: int, mode: str) -> pd.Series:
    _validate_window(window)
    _validate_mi(x)

    def _arg(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        if mode == "max":
            # 与 Alpha158 的 IdxMax/IdxMin 用法对齐，返回 1-based 窗口位置。
            return float(np.argmax(arr) + 1)
        if mode == "min":
            return float(np.argmin(arr) + 1)
        raise ValueError(f"Unknown argextreme mode: {mode}")

    out = x.groupby(level=1, group_keys=False).rolling(window).apply(_arg, raw=True).droplevel(0)
    return out.reindex(x.index)


def ts_argmax(x: pd.Series, window: int) -> pd.Series:
    """窗口内最大值出现的位置，返回 1-based 索引。"""
    return _rolling_argextreme(x, window, mode="max")


def ts_argmin(x: pd.Series, window: int) -> pd.Series:
    """窗口内最小值出现的位置，返回 1-based 索引。"""
    return _rolling_argextreme(x, window, mode="min")


def ts_cov(x: pd.Series, y: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """两个序列的滚动协方差，可选半衰期加权。"""
    _validate_window(window)
    _validate_mi(x)
    y = _reindex_like(x, y)
    if half_life is None:
        xw = x.unstack(level=1)
        yw = y.unstack(level=1)
        cov = xw.rolling(window).cov(yw)
        out = _stack_wide(cov)
        out.index = out.index.set_names(list(x.index.names[:2]))
        return out.reindex(x.index)

    # 加权分支使用 numpy 向量化，保留“窗口长度满足后允许部分缺失”的旧逻辑。
    w = _exp_weights(window, half_life)
    x_arr, dates, instruments = _to_wide_2d(x)
    y_arr, _, _ = _to_wide_2d(y)
    cov_vals = _rolling_weighted_cov_2d(x_arr, y_arr, window, w)
    return _from_wide_2d(cov_vals, dates, instruments, x.index, list(x.index.names[:2]))


def ts_corr(x: pd.Series, y: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """两个序列的滚动相关系数，可选半衰期加权。"""
    _validate_window(window)
    if half_life is None:
        _validate_mi(x)
        y = _reindex_like(x, y)
        xw = x.unstack(level=1)
        yw = y.unstack(level=1)
        corr = xw.rolling(window).corr(yw)
        out = _stack_wide(corr)
        out.index = out.index.set_names(list(x.index.names[:2]))
        return out.reindex(x.index)
    cov = ts_cov(x, y, window, half_life=half_life)
    sx = ts_std(x, window, half_life=half_life)
    sy = ts_std(_reindex_like(x, y), window, half_life=half_life)
    return cov / (sx * sy + EPS)


def rank(x: pd.Series) -> pd.Series:
    """按截面做百分位排名。"""
    _validate_mi(x)
    return x.groupby(level=0, group_keys=False).rank(method="average", pct=True)


def zscore(x: pd.Series) -> pd.Series:
    """按截面做标准化。"""
    _validate_mi(x)
    mean = x.groupby(level=0, group_keys=False).transform("mean")
    std = x.groupby(level=0, group_keys=False).transform(lambda s: s.std(ddof=0))
    std = std.where(std.abs() > EPS, np.nan)
    return (x - mean) / std


def cs_mean(x: pd.Series) -> pd.Series:
    """按截面计算均值并广播回原索引。"""
    _validate_mi(x)
    return x.groupby(level=0, group_keys=False).transform("mean")


def cs_sum(x: pd.Series) -> pd.Series:
    """按截面计算求和并广播回原索引。"""
    _validate_mi(x)
    return x.groupby(level=0, group_keys=False).transform("sum")


def cs_std(x: pd.Series, ddof: int = 0) -> pd.Series:
    """按截面计算标准差并广播回原索引。"""
    _validate_mi(x)
    ddof = int(ddof)
    if ddof < 0:
        raise ValueError("cs_std requires ddof >= 0")
    return x.groupby(level=0, group_keys=False).transform(lambda s: s.std(ddof=ddof))


def cs_skew(x: pd.Series, min_count: int = 3) -> pd.Series:
    """按截面计算偏度并广播回原索引。"""
    _validate_mi(x)
    min_count = int(min_count)
    if min_count < 3:
        raise ValueError("cs_skew requires min_count >= 3")

    def _calc(s: pd.Series) -> float:
        arr = s.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_count:
            return np.nan
        mu = float(arr.mean())
        xc = arr - mu
        m2 = float(np.mean(xc ** 2))
        if m2 <= EPS:
            return 0.0
        m3 = float(np.mean(xc ** 3))
        return float(m3 / (m2 ** 1.5))

    stat = x.groupby(level=0, group_keys=False).transform(_calc)
    return stat


def cs_zscore(x: pd.Series) -> pd.Series:
    """按截面计算 z-score。"""
    return zscore(x)


def _group_frame(x: pd.Series, group_map: pd.Series) -> pd.DataFrame:
    _validate_mi(x)
    gm = group_map.reindex(x.index)
    return pd.DataFrame({"x": x, "g": gm})


def group_mean(x: pd.Series, group_map: pd.Series, min_group_size: int = 2) -> pd.Series:
    """按日分组后计算组内均值。"""
    df = _group_frame(x, group_map)
    valid = df["x"].notna() & df["g"].notna()
    grp_mean = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].transform("mean")
    grp_count = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].transform("count")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[valid] = grp_mean.where(grp_count >= int(min_group_size), np.nan).to_numpy(dtype=float)
    return out


def group_sum(x: pd.Series, group_map: pd.Series, min_group_size: int = 1) -> pd.Series:
    """按日分组后计算组内求和。"""
    df = _group_frame(x, group_map)
    valid = df["x"].notna() & df["g"].notna()
    grp_sum = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].transform("sum")
    grp_count = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].transform("count")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[valid] = grp_sum.where(grp_count >= int(min_group_size), np.nan).to_numpy(dtype=float)
    return out


def group_rank(x: pd.Series, group_map: pd.Series, min_group_size: int = 1) -> pd.Series:
    """按日分组后计算组内百分位排名。"""
    df = _group_frame(x, group_map)
    valid = df["x"].notna() & df["g"].notna()
    ranked = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].rank(method="average", pct=True)
    grp_count = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].transform("count")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[valid] = ranked.where(grp_count >= int(min_group_size), np.nan).to_numpy(dtype=float)
    return out


def group_bucket(
    x: pd.Series,
    group_map: pd.Series,
    q_low: float = 0.3,
    q_high: float = 0.7,
    min_group_size: int = 5,
) -> pd.Series:
    """按日分组后分成低中高三个桶。"""
    if not (0.0 < float(q_low) < float(q_high) < 1.0):
        raise ValueError("group_bucket requires 0 < q_low < q_high < 1")
    df = _group_frame(x, group_map)
    valid = df["x"].notna() & df["g"].notna()
    ranked = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].rank(method="average", pct=True)
    grp_count = df.loc[valid].groupby([df.loc[valid].index.get_level_values(0), "g"])["x"].transform("count")
    bucket = pd.Series(np.nan, index=ranked.index, dtype=float)
    enough = grp_count >= int(min_group_size)
    bucket.loc[enough & (ranked <= float(q_low))] = 0.0
    bucket.loc[enough & (ranked > float(q_low)) & (ranked <= float(q_high))] = 1.0
    bucket.loc[enough & (ranked > float(q_high))] = 2.0
    out = pd.Series(np.nan, index=x.index, dtype=float)
    out.loc[valid] = bucket.to_numpy(dtype=float)
    return out


def group_combine(group_a: pd.Series, group_b: pd.Series, base: int = 10000) -> pd.Series:
    """组合两个分组标签，生成联合分组。"""
    _validate_mi(group_a)
    group_b = _reindex_like(group_a, group_b)
    mask = group_a.notna() & group_b.notna()

    out: pd.Series
    a_non_na = group_a[mask]
    b_non_na = group_b[mask]
    numeric_ok = not a_non_na.empty and not b_non_na.empty
    if numeric_ok:
        numeric_ok = pd.api.types.is_numeric_dtype(a_non_na) and pd.api.types.is_numeric_dtype(b_non_na)

    if numeric_ok:
        out = pd.Series(np.nan, index=group_a.index, dtype=float)
        out.loc[mask] = a_non_na.astype(float) * float(base) + b_non_na.astype(float)
        return out

    out = pd.Series(np.nan, index=group_a.index, dtype=object)
    out.loc[mask] = a_non_na.astype(str) + "|" + b_non_na.astype(str)
    return out


def neighbor_mean(x: pd.Series, n: int = 10, exclude_self: bool = True) -> pd.Series:
    """按代码邻近关系计算邻居均值。"""
    _validate_mi(x)
    n = int(n)
    if n <= 0:
        raise ValueError("neighbor_mean requires n > 0")
    wide = x.unstack(level=1).sort_index()
    cols_sorted = sorted(wide.columns)
    wide = wide.reindex(columns=cols_sorted)
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns, dtype=float)

    m = len(cols_sorted)
    for j, col in enumerate(cols_sorted):
        neighbor_idx: list[int] = []
        step = 1
        while len(neighbor_idx) < n:
            left = j - step
            right = j + step
            if left >= 0:
                neighbor_idx.append(left)
                if len(neighbor_idx) >= n:
                    break
            if right < m:
                neighbor_idx.append(right)
                if len(neighbor_idx) >= n:
                    break
            if left < 0 and right >= m:
                break
            step += 1

        if not bool(exclude_self) and j not in neighbor_idx:
            if len(neighbor_idx) >= n:
                neighbor_idx = [j] + neighbor_idx[:-1]
            else:
                neighbor_idx = [j] + neighbor_idx

        if neighbor_idx:
            neighbor_cols = [cols_sorted[k] for k in neighbor_idx]
            out.loc[:, col] = wide.loc[:, neighbor_cols].mean(axis=1, skipna=True)

    return _stack_wide(out).reindex(x.index)


def cs_reg_resid(y: pd.Series, x: pd.Series, min_obs: int = 20) -> pd.Series:
    """按日做截面回归并返回残差。"""
    _validate_mi(y)
    x = _reindex_like(y, x)
    min_obs = int(min_obs)
    df = pd.DataFrame({"y": y, "x": x})
    out = pd.Series(np.nan, index=y.index, dtype=float)

    for dt, g in df.groupby(level=0, sort=False):
        mask = g["y"].notna() & g["x"].notna()
        if int(mask.sum()) < min_obs:
            continue
        xx = g.loc[mask, "x"].to_numpy(dtype=float)
        yy = g.loc[mask, "y"].to_numpy(dtype=float)
        x_mean = xx.mean()
        y_mean = yy.mean()
        denom = float(np.sum((xx - x_mean) ** 2))
        if denom <= EPS:
            continue
        beta = float(np.sum((xx - x_mean) * (yy - y_mean)) / denom)
        alpha = float(y_mean - beta * x_mean)
        out.loc[g.index[mask]] = yy - (alpha + beta * xx)

    return out


def cs_multi_reg_resid(y: pd.Series, *xs_and_min_obs: pd.Series | int | float) -> pd.Series:
    """按日做多变量截面回归并返回残差。"""
    _validate_mi(y)
    if not xs_and_min_obs:
        raise ValueError("cs_multi_reg_resid requires at least one regressor")

    args = list(xs_and_min_obs)
    min_obs = 20
    if not isinstance(args[-1], pd.Series):
        min_obs = int(args.pop())
    xs = [_reindex_like(y, x) for x in args]
    if not xs:
        raise ValueError("cs_multi_reg_resid requires at least one regressor")
    min_obs = int(min_obs)

    cols = {"y": y}
    for i, x in enumerate(xs, start=1):
        cols[f"x{i}"] = x
    df = pd.DataFrame(cols)
    out = pd.Series(np.nan, index=y.index, dtype=float)

    for _, g in df.groupby(level=0, sort=False):
        mask = g.notna().all(axis=1)
        if int(mask.sum()) < min_obs:
            continue
        valid = g.loc[mask]
        yy = valid["y"].to_numpy(dtype=float)
        xx = valid.drop(columns=["y"]).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(valid), dtype=float), xx])
        try:
            beta = np.linalg.lstsq(X, yy, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        out.loc[valid.index] = yy - X @ beta

    return out


def csad_daily(ret: pd.Series) -> pd.Series:
    """计算每日横截面绝对偏离均值。"""
    _validate_mi(ret)
    market_ret = cs_mean(ret)
    abs_dev = (ret - market_ret).abs()
    return abs_dev.groupby(level=0, group_keys=False).transform("mean")


def cs_corr_mean(x: pd.Series, window: int, min_obs: int = 15) -> pd.Series:
    """计算窗口内个股与其他个股的平均相关性。"""
    _validate_mi(x)
    _validate_window(window)
    min_obs = int(min_obs)
    if min_obs <= 1:
        raise ValueError("cs_corr_mean requires min_obs > 1")

    wide = x.unstack(level=1).sort_index()
    arr = wide.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)

    for t in range(window - 1, len(wide.index)):
        seg = arr[t - window + 1 : t + 1]
        valid_cols = np.sum(np.isfinite(seg), axis=0) >= min_obs
        if int(valid_cols.sum()) <= 1:
            continue
        seg_valid = seg[:, valid_cols]
        corr = np.corrcoef(seg_valid, rowvar=False)
        if np.ndim(corr) == 0:
            continue
        np.fill_diagonal(corr, np.nan)
        pbar = np.nanmean(corr, axis=1)
        row = np.full(wide.shape[1], np.nan, dtype=float)
        row[np.where(valid_cols)[0]] = pbar
        out[t] = row

    return _stack_wide(pd.DataFrame(out, index=wide.index, columns=wide.columns)).reindex(x.index)


def cssd_daily(ret: pd.Series, market_ret: pd.Series, ddof: int = 1) -> pd.Series:
    """计算每日横截面收益离散度。"""
    _validate_mi(ret)
    market_ret = _reindex_like(ret, market_ret)
    ddof = int(ddof)
    if ddof < 0:
        raise ValueError("cssd_daily requires ddof >= 0")
    sq_dev = (ret - market_ret) ** 2
    grp_mean = sq_dev.groupby(level=0, group_keys=False).transform("mean")
    grp_count = sq_dev.groupby(level=0, group_keys=False).transform("count")
    if ddof == 0:
        return grp_mean.pow(0.5)
    scale = grp_count / (grp_count - ddof)
    scale = scale.where(grp_count > ddof, np.nan)
    return (grp_mean * scale).pow(0.5)


def rolling_herding_beta(y: pd.Series, x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """按个股计算滚动单变量回归 beta。"""
    _validate_mi(y)
    x = _reindex_like(y, x)
    window = int(window)
    if window <= 0:
        raise ValueError("rolling_herding_beta requires window > 0")
    min_periods = window if min_periods is None else int(min_periods)
    if min_periods <= 1:
        raise ValueError("rolling_herding_beta requires min_periods > 1")

    def _rolling_beta(df: pd.DataFrame) -> pd.Series:
        yy = df["y"].to_numpy(dtype=float)
        xx = df["x"].to_numpy(dtype=float)
        out = np.full(len(df), np.nan, dtype=float)

        for i in range(len(df)):
            start = max(0, i - window + 1)
            sub_y = yy[start : i + 1]
            sub_x = xx[start : i + 1]
            valid = np.isfinite(sub_y) & np.isfinite(sub_x)
            if int(valid.sum()) < min_periods:
                continue
            sv_y = sub_y[valid]
            sv_x = sub_x[valid]
            x_mean = float(sv_x.mean())
            denom = float(np.sum((sv_x - x_mean) ** 2))
            if denom <= EPS:
                continue
            y_mean = float(sv_y.mean())
            beta = float(np.sum((sv_x - x_mean) * (sv_y - y_mean)) / denom)
            out[i] = beta

        return pd.Series(out, index=df.index)

    result = (
        pd.concat([y.rename("y"), x.rename("x")], axis=1)
        .groupby(level=1, group_keys=False, sort=False)
        .apply(_rolling_beta)
    )
    return result.reindex(y.index)


def rolling_peer_csad_ratio(
    ret: pd.Series,
    lookback_days: int,
    peer_num: int,
    recent_window: int,
    max_inactive_days: int = 20,
    month_end_only: int = 1,
) -> pd.Series:
    """按滚动收益相关性选取 top peers，并输出近期/长期 CSAD 波动率比值之比。"""
    _validate_mi(ret)
    lookback_days = int(lookback_days)
    peer_num = int(peer_num)
    recent_window = int(recent_window)
    max_inactive_days = int(max_inactive_days)
    month_end_only = int(month_end_only)
    if lookback_days <= 1:
        raise ValueError("rolling_peer_csad_ratio requires lookback_days > 1")
    if peer_num <= 0:
        raise ValueError("rolling_peer_csad_ratio requires peer_num > 0")
    if recent_window <= 1:
        raise ValueError("rolling_peer_csad_ratio requires recent_window > 1")
    if max_inactive_days < 0:
        raise ValueError("rolling_peer_csad_ratio requires max_inactive_days >= 0")

    wide = ret.unstack(level=1).sort_index()
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns, dtype=float)
    eval_dates = wide.index
    if month_end_only:
        eval_dates = wide.index.to_series().groupby(wide.index.to_period("M")).tail(1).values

    for dt in eval_dates:
        loc = int(wide.index.get_loc(dt))
        if loc < lookback_days - 1:
            continue

        win = wide.iloc[loc - lookback_days + 1 : loc + 1]
        valid_cols = win.columns[win.isna().sum(axis=0) <= max_inactive_days]
        if len(valid_cols) < peer_num + 1:
            continue

        win_valid = win[valid_cols]
        min_periods = max(20, lookback_days - max_inactive_days)
        corr = win_valid.corr(min_periods=min_periods)

        for center in valid_cols:
            c = corr[center].drop(index=center).dropna()
            if len(c) < peer_num:
                continue
            peers = c.nlargest(peer_num).index.tolist()
            block = [center] + peers
            block_ret = win_valid[block]
            center_ret = block_ret[center]
            csad = (block_ret.sub(center_ret, axis=0)).abs().mean(axis=1)

            recent_csad = csad.iloc[-recent_window:]
            recent_r = center_ret.iloc[-recent_window:]
            num = float(recent_csad.std(ddof=1))
            den = float(recent_r.std(ddof=1))
            long_num = float(csad.std(ddof=1))
            long_den = float(center_ret.std(ddof=1))
            if min(abs(den), abs(long_den)) <= EPS:
                continue
            ratio_long = long_num / long_den
            if abs(ratio_long) <= EPS:
                continue
            out.at[dt, center] = -float((num / den) / ratio_long)

    return _from_wide_2d(out.to_numpy(dtype=float), out.index, out.columns, ret.index, list(ret.index.names[:2]))


def rolling_regression_beta_sum(
    y: pd.Series,
    x1: pd.Series,
    x2: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """按个股计算双变量滚动回归的 beta 之和。"""
    _validate_mi(y)
    x1 = _reindex_like(y, x1)
    x2 = _reindex_like(y, x2)
    window = int(window)
    if window <= 0:
        raise ValueError("rolling_regression_beta_sum requires window > 0")
    min_periods = max(3, min(window, 20)) if min_periods is None else int(min_periods)
    if min_periods < 3:
        raise ValueError("rolling_regression_beta_sum requires min_periods >= 3")

    def _rolling_beta_sum(df: pd.DataFrame) -> pd.Series:
        yy = df["y"].to_numpy(dtype=float)
        xx1 = df["x1"].to_numpy(dtype=float)
        xx2 = df["x2"].to_numpy(dtype=float)
        out = np.full(len(df), np.nan, dtype=float)

        for i in range(len(df)):
            start = max(0, i - window + 1)
            sub_y = yy[start : i + 1]
            sub_x1 = xx1[start : i + 1]
            sub_x2 = xx2[start : i + 1]
            valid = np.isfinite(sub_y) & np.isfinite(sub_x1) & np.isfinite(sub_x2)
            if int(valid.sum()) < min_periods:
                continue
            Y = sub_y[valid]
            X = np.column_stack([
                np.ones(int(valid.sum()), dtype=float),
                sub_x1[valid],
                sub_x2[valid],
            ])
            try:
                beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            out[i] = float(beta[1] + beta[2])

        return pd.Series(out, index=df.index)

    result = (
        pd.concat([y.rename("y"), x1.rename("x1"), x2.rename("x2")], axis=1)
        .groupby(level=1, group_keys=False, sort=False)
        .apply(_rolling_beta_sum)
    )
    return result.reindex(y.index)


def ts_sorted_mean_spread(value: pd.Series, sort_key: pd.Series, window: int, ratio: float) -> pd.Series:
    """按排序键分层后，计算高低组均值差。"""
    _validate_mi(value)
    sort_key = _reindex_like(value, sort_key)
    _validate_window(window)
    ratio = float(ratio)
    if not (0.0 < ratio <= 0.5):
        raise ValueError("ts_sorted_mean_spread requires 0 < ratio <= 0.5")

    k = max(1, int(np.ceil(window * ratio)))
    out = pd.Series(np.nan, index=value.index, dtype=float)

    for inst, g in pd.DataFrame({"v": value, "s": sort_key}).groupby(level=1, sort=False):
        vv = g["v"].to_numpy(dtype=float)
        ss = g["s"].to_numpy(dtype=float)
        idx = g.index
        for i in range(window - 1, len(g)):
            sub_v = vv[i - window + 1 : i + 1]
            sub_s = ss[i - window + 1 : i + 1]
            valid = np.isfinite(sub_v) & np.isfinite(sub_s)
            if int(valid.sum()) < k:
                continue
            sv = sub_v[valid]
            sk = sub_s[valid]
            order = np.argsort(sk)
            low_mean = float(np.mean(sv[order[:k]]))
            high_mean = float(np.mean(sv[order[-k:]]))
            out.loc[idx[i]] = high_mean - low_mean

    return out


def group_day_sum(x: pd.Series, min_count: int = 1) -> pd.Series:
    """按日内数据聚合为日频求和。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    grouped = x.groupby([dt, inst], sort=True).sum(min_count=int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_bar_sum(x: pd.Series, start_bar: int, window_bars: int, min_count: int = 1) -> pd.Series:
    """按日内 bar 位置窗口聚合为日频求和。"""
    _validate_mi(x)
    start_bar = int(start_bar)
    window_bars = int(window_bars)
    min_count = int(min_count)
    if start_bar < 0:
        raise ValueError("group_day_bar_sum requires start_bar >= 0")
    if window_bars <= 0:
        raise ValueError("group_day_bar_sum requires window_bars > 0")
    if min_count <= 0:
        raise ValueError("group_day_bar_sum requires min_count > 0")

    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    slot = x.groupby([dt, inst], sort=False).cumcount()
    end_bar = start_bar + window_bars
    selected = x.where((slot >= start_bar) & (slot < end_bar))
    grouped = selected.groupby([dt, inst], sort=True).sum(min_count=min_count)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_mean(x: pd.Series, min_count: int = 1) -> pd.Series:
    """按日内数据聚合为日频均值。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    grouped = x.groupby([dt, inst], sort=True).mean()
    counts = x.groupby([dt, inst], sort=True).count()
    grouped = grouped.where(counts >= int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_last(x: pd.Series, min_count: int = 1) -> pd.Series:
    """按日内数据聚合为日频最后一个值。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    grouped = x.groupby([dt, inst], sort=True).last()
    counts = x.groupby([dt, inst], sort=True).count()
    grouped = grouped.where(counts >= int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_bucket_entropy(x: pd.Series, maxbin: int = 10, min_count: int = 1) -> pd.Series:
    """按日内数据聚合为日频分桶熵。"""
    _validate_mi(x)
    maxbin = int(maxbin)
    min_count = int(min_count)
    if maxbin <= 0:
        raise ValueError("group_day_bucket_entropy requires maxbin > 0")
    if min_count <= 0:
        raise ValueError("group_day_bucket_entropy requires min_count > 0")

    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_count:
            return np.nan
        if len(arr) == 0:
            return np.nan
        if np.all(arr == arr[0]):
            return 0.0
        bins = min(maxbin, len(arr))
        if bins <= 1:
            return 0.0
        hist, _ = np.histogram(arr, bins=bins, range=(arr.min(), arr.max()))
        total = hist.sum()
        if total <= 0:
            return np.nan
        p = hist.astype(float) / float(total)
        p = p[p > 0]
        if len(p) == 0:
            return np.nan
        return float(-np.sum(p * np.log(p)))

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_kurt(x: pd.Series, min_count: int = 2, excess: bool = False) -> pd.Series:
    """按日内数据聚合为日频峰度。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < int(min_count):
            return np.nan
        mu = float(arr.mean())
        xc = arr - mu
        m2 = float(np.mean(xc ** 2))
        if m2 <= EPS:
            return np.nan
        m4 = float(np.mean(xc ** 4))
        out = m4 / (m2 ** 2)
        return out - 3.0 if bool(excess) else out

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_skew(x: pd.Series, min_count: int = 2) -> pd.Series:
    """按日内数据聚合为日频偏度。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < int(min_count):
            return np.nan
        mu = float(arr.mean())
        xc = arr - mu
        m2 = float(np.mean(xc ** 2))
        if m2 <= EPS:
            return 0.0
        m3 = float(np.mean(xc ** 3))
        return float(m3 / (m2 ** 1.5))

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_std(x: pd.Series, min_count: int = 2, ddof: int = 0) -> pd.Series:
    """按日内数据聚合为日频标准差。"""
    _validate_mi(x)
    ddof = int(ddof)
    if ddof < 0:
        raise ValueError("group_day_std requires ddof >= 0")
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < int(min_count) or len(arr) <= ddof:
            return np.nan
        return float(np.std(arr, ddof=ddof))

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_count(x: pd.Series, min_count: int = 1) -> pd.Series:
    """按日内数据聚合为日频有效值个数。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    grouped = x.groupby([dt, inst], sort=True).count()
    grouped = grouped.where(grouped >= int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index().astype(float)


def group_day_tripower_iv(x: pd.Series, k: int = 3, q: float = 2.0 / 3.0, min_count: int = 3) -> pd.Series:
    """按日内对数收益率估计日频 tripower 型连续波动 IV。"""
    _validate_mi(x)
    k = int(k)
    min_count = int(min_count)
    q = float(q)
    if k <= 0:
        raise ValueError("group_day_tripower_iv requires k > 0")
    if min_count <= 0:
        raise ValueError("group_day_tripower_iv requires min_count > 0")
    if q <= 0:
        raise ValueError("group_day_tripower_iv requires q > 0")

    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    mu_q = (2.0 ** (q / 2.0)) * math.gamma((q + 1.0) / 2.0) / math.sqrt(math.pi)
    scale = mu_q ** (-3.0)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        if len(arr) < k:
            return np.nan
        abs_r_q = np.power(np.abs(arr), q)
        a = abs_r_q[k - 1 :]
        b = abs_r_q[k - 2 : len(arr) - 1]
        c = abs_r_q[: len(arr) - k + 1]
        trip = a * b * c
        valid = np.isfinite(trip)
        if int(valid.sum()) < min_count:
            return np.nan
        return float(scale * np.nansum(trip))

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_bipower_iv(x: pd.Series, min_count: int = 2) -> pd.Series:
    """按日内对数收益率估计日频 bipower variation 型连续波动 IV。"""
    _validate_mi(x)
    min_count = int(min_count)
    if min_count <= 0:
        raise ValueError("group_day_bipower_iv requires min_count > 0")

    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    scale = np.pi / 2.0

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            return np.nan
        bp = np.abs(arr[1:]) * np.abs(arr[:-1])
        valid = np.isfinite(bp)
        if int(valid.sum()) < min_count:
            return np.nan
        return float(scale * np.nansum(bp))

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_vwap(price: pd.Series, volume: pd.Series, min_count: int = 1) -> pd.Series:
    """按日内价格和成交量计算日频 VWAP。"""
    _validate_mi(price)
    volume = _reindex_like(price, volume)
    dt = pd.to_datetime(price.index.get_level_values(0)).normalize()
    inst = price.index.get_level_values(1)
    valid = price.notna() & volume.notna() & (volume >= 0)
    pv = (price * volume).where(valid)
    vol = volume.where(valid)
    pv_sum = pv.groupby([dt, inst], sort=True).sum(min_count=int(min_count))
    vol_sum = vol.groupby([dt, inst], sort=True).sum(min_count=int(min_count))
    counts = valid.astype(int).groupby([dt, inst], sort=True).sum()
    grouped = pv_sum / vol_sum.where(vol_sum.abs() > EPS, np.nan)
    grouped = grouped.where(counts >= int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def rolling_score_top_volume_vwap_ratio(
    score: pd.Series,
    volume: pd.Series,
    amount: pd.Series,
    lookback_days: int,
    top_volume_ratio: float,
    min_count: int = 1,
) -> pd.Series:
    """按滚动日窗对日内评分排序，取累计成交量前若干比例 bar 的 VWAP / 全部 VWAP。"""
    _validate_mi(score)
    volume = _reindex_like(score, volume)
    amount = _reindex_like(score, amount)
    lookback_days = int(lookback_days)
    top_volume_ratio = float(top_volume_ratio)
    min_count = int(min_count)
    if lookback_days <= 0:
        raise ValueError("rolling_score_top_volume_vwap_ratio requires lookback_days > 0")
    if not (0.0 < top_volume_ratio <= 1.0):
        raise ValueError("rolling_score_top_volume_vwap_ratio requires 0 < top_volume_ratio <= 1")
    if min_count <= 0:
        raise ValueError("rolling_score_top_volume_vwap_ratio requires min_count > 0")

    frame = pd.DataFrame({"score": score, "volume": volume, "amount": amount}, index=score.index)
    results: list[tuple[pd.Timestamp, str, float]] = []

    for inst, g in frame.groupby(level=1, sort=False):
        dt = pd.to_datetime(g.index.get_level_values(0))
        work = pd.DataFrame(
            {
                "trade_date": dt.normalize(),
                "score": g["score"].to_numpy(dtype=float),
                "volume": g["volume"].to_numpy(dtype=float),
                "amount": g["amount"].to_numpy(dtype=float),
            },
            index=g.index,
        )
        unique_dates = pd.Index(work["trade_date"].drop_duplicates().sort_values())

        for i, current_date in enumerate(unique_dates):
            hist_dates = unique_dates[max(0, i - lookback_days + 1) : i + 1]
            window = work[work["trade_date"].isin(hist_dates)].copy()
            window = window.replace([np.inf, -np.inf], np.nan)
            window = window.dropna(subset=["score", "volume", "amount"])
            window = window[window["volume"] > 0]

            if len(window) < min_count:
                results.append((pd.Timestamp(current_date), inst, np.nan))
                continue

            total_volume = float(window["volume"].sum())
            total_amount = float(window["amount"].sum())
            if total_volume <= EPS or total_amount <= EPS:
                results.append((pd.Timestamp(current_date), inst, np.nan))
                continue

            window = window.sort_values("score", ascending=False)
            threshold = total_volume * top_volume_ratio
            window["cum_volume"] = window["volume"].cumsum()
            smart = window[window["cum_volume"] <= threshold]
            if smart.empty:
                smart = window.iloc[[0]]

            smart_volume = float(smart["volume"].sum())
            smart_amount = float(smart["amount"].sum())
            if smart_volume <= EPS:
                results.append((pd.Timestamp(current_date), inst, np.nan))
                continue

            vwap_smart = smart_amount / smart_volume
            vwap_all = total_amount / total_volume
            factor_value = np.nan if abs(vwap_all) <= EPS else float(vwap_smart / vwap_all)
            results.append((pd.Timestamp(current_date), inst, factor_value))

    out = pd.Series(
        [v for _, _, v in results],
        index=pd.MultiIndex.from_tuples(
            [(dt, inst) for dt, inst, _ in results],
            names=["datetime", "instrument"],
        ),
        dtype=float,
    )
    return out.sort_index()


def group_day_top_prod(x: pd.Series, top_pct: float, min_count: int = 1) -> pd.Series:
    """按日内数据取前 top_pct 比例最大值并连乘。"""
    _validate_mi(x)
    top_pct = float(top_pct)
    min_count = int(min_count)
    if not (0.0 < top_pct <= 1.0):
        raise ValueError("group_day_top_prod requires 0 < top_pct <= 1")
    if min_count <= 0:
        raise ValueError("group_day_top_prod requires min_count > 0")

    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_count:
            return np.nan
        k = max(1, int(np.ceil(len(arr) * top_pct)))
        top = np.partition(arr, len(arr) - k)[-k:]
        return float(np.prod(top))

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_top_mean(
    value: pd.Series,
    sort_key: pd.Series,
    top_n: int,
    min_count: int = 1,
) -> pd.Series:
    """按日内排序键取前 top_n 个 bar，对 value 求均值。"""
    _validate_mi(value)
    sort_key = _reindex_like(value, sort_key)
    top_n = int(top_n)
    min_count = int(min_count)
    if top_n <= 0:
        raise ValueError("group_day_top_mean requires top_n > 0")
    if min_count <= 0:
        raise ValueError("group_day_top_mean requires min_count > 0")

    dt = pd.to_datetime(value.index.get_level_values(0)).normalize()
    inst = value.index.get_level_values(1)

    def _calc(g: pd.DataFrame) -> float:
        work = g.dropna()
        need = max(top_n, min_count)
        if len(work) < need:
            return np.nan
        top = work.nlargest(top_n, "s")
        return float(top["v"].mean())

    frame = pd.DataFrame({"v": value, "s": sort_key})
    grouped = frame.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_cvar(x: pd.Series, alpha: float = 0.05, min_count: int = 1) -> pd.Series:
    """按日内数据聚合为左尾 CVaR。"""
    _validate_mi(x)
    alpha = float(alpha)
    min_count = int(min_count)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("group_day_cvar requires 0 < alpha <= 1")
    if min_count <= 0:
        raise ValueError("group_day_cvar requires min_count > 0")

    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_count:
            return np.nan
        k = max(1, int(np.ceil(alpha * len(arr))))
        tail = np.partition(arr, k - 1)[:k]
        return float(np.mean(tail))

    grouped = x.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_sorted_skew(
    value: pd.Series,
    sort_key: pd.Series,
    top_ratio: float,
    min_count: int = 3,
) -> pd.Series:
    """按日内排序键选前 top_ratio 比例后，计算取值序列的偏度。"""
    _validate_mi(value)
    sort_key = _reindex_like(value, sort_key)
    top_ratio = float(top_ratio)
    min_count = int(min_count)
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError("group_day_sorted_skew requires 0 < top_ratio <= 1")
    if min_count < 3:
        raise ValueError("group_day_sorted_skew requires min_count >= 3")

    dt = pd.to_datetime(value.index.get_level_values(0)).normalize()
    inst = value.index.get_level_values(1)
    df = pd.DataFrame({"v": value, "s": sort_key}, index=value.index)

    def _calc(g: pd.DataFrame) -> float:
        work = g.dropna()
        if len(work) < min_count:
            return np.nan
        k = max(int(np.ceil(len(work) * top_ratio)), min_count)
        top = work.nlargest(k, "s")
        arr = top["v"].to_numpy(dtype=float)
        mu = float(arr.mean())
        xc = arr - mu
        m2 = float(np.mean(xc ** 2))
        if m2 <= EPS:
            return np.nan
        m3 = float(np.mean(xc ** 3))
        return float(m3 / (m2 ** 1.5))

    grouped = df.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_day_weighted_skew(
    value: pd.Series,
    weight: pd.Series,
    min_count: int = 1,
) -> pd.Series:
    """按日内权重计算日频加权偏度。"""
    _validate_mi(value)
    weight = _reindex_like(value, weight)
    min_count = int(min_count)
    if min_count <= 0:
        raise ValueError("group_day_weighted_skew requires min_count > 0")

    dt = pd.to_datetime(value.index.get_level_values(0)).normalize()
    inst = value.index.get_level_values(1)
    df = pd.DataFrame({"v": value, "w": weight}, index=value.index)

    def _calc(g: pd.DataFrame) -> float:
        work = g.dropna()
        if len(work) < min_count:
            return np.nan
        v = work["v"].to_numpy(dtype=float)
        w = work["w"].to_numpy(dtype=float)
        w_sum = float(np.nansum(w))
        if not np.isfinite(w_sum) or w_sum <= EPS:
            return np.nan
        w = w / w_sum
        mean_v = float(np.mean(v))
        std_v = float(np.std(v, ddof=0))
        if not np.isfinite(std_v) or std_v <= EPS:
            return 0.0
        centered = v - mean_v
        return float(np.nansum(w * (centered ** 3)) / (std_v ** 3))

    grouped = df.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def day_trend_ratio(price: pd.Series, min_count: int = 2) -> pd.Series:
    """衡量日内净涨跌相对总波动的趋势强度。"""
    _validate_mi(price)
    dt = pd.to_datetime(price.index.get_level_values(0)).normalize()
    inst = price.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        valid = np.isfinite(arr)
        arr = arr[valid]
        if len(arr) < int(min_count):
            return np.nan
        denom = float(np.abs(np.diff(arr)).sum())
        if denom <= EPS:
            return np.nan
        return float((arr[-1] - arr[0]) / denom)

    grouped = price.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def intraday_maxdrawdown(price: pd.Series, min_count: int = 2) -> pd.Series:
    """计算单日内的最大回撤。"""
    _validate_mi(price)
    dt = pd.to_datetime(price.index.get_level_values(0)).normalize()
    inst = price.index.get_level_values(1)

    def _calc(g: pd.Series) -> float:
        arr = g.to_numpy(dtype=float)
        valid = np.isfinite(arr)
        arr = arr[valid]
        if len(arr) < int(min_count):
            return np.nan
        future_min = np.minimum.accumulate(arr[::-1])[::-1]
        cur = arr[:-1]
        fut = future_min[1:]
        good = cur > 0
        if not bool(np.any(good)):
            return np.nan
        dd = fut[good] / cur[good] - 1.0
        return float(np.min(dd))

    grouped = price.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def upper_shadow(high: pd.Series, open_: pd.Series, close: pd.Series, preclose: pd.Series) -> pd.Series:
    """计算以上一收盘归一化的上影线长度。"""
    _validate_mi(high)
    open_ = _reindex_like(high, open_)
    close = _reindex_like(high, close)
    preclose = _reindex_like(high, preclose)
    body_top = pd.concat([open_, close], axis=1).max(axis=1)
    denom = preclose.where(preclose.abs() > EPS, np.nan)
    return (high - body_top) / denom


def same_bar_rolling_zscore(
    x: pd.Series,
    lookback_days: int,
    min_history_days: int = 10,
    ddof: int = 1,
) -> pd.Series:
    """同一日内时点相对历史同位置数据做滚动标准化。"""
    _validate_mi(x)
    lookback_days = int(lookback_days)
    min_history_days = int(min_history_days)
    ddof = int(ddof)
    if lookback_days <= 0:
        raise ValueError("same_bar_rolling_zscore requires lookback_days > 0")
    if min_history_days <= 0:
        raise ValueError("same_bar_rolling_zscore requires min_history_days > 0")

    dt = pd.to_datetime(x.index.get_level_values(0))
    day = dt.normalize()
    slot = dt - day
    out = pd.Series(np.nan, index=x.index, dtype=float)

    for inst, g in pd.DataFrame({"x": x, "day": day, "slot": slot}, index=x.index).groupby(level=1, sort=False):
        for cur_slot, sg in g.groupby("slot", sort=False):
            hist = sg.sort_values("day").copy()
            series = hist["x"].astype(float)
            mu = series.shift(1).rolling(window=lookback_days, min_periods=min_history_days).mean()
            sd = series.shift(1).rolling(window=lookback_days, min_periods=min_history_days).std(ddof=ddof)
            z = (series - mu) / sd.where(sd.abs() > EPS, np.nan)
            out.loc[hist.index] = z.to_numpy(dtype=float)

    return out


def same_bar_rolling_std(
    x: pd.Series,
    lookback_days: int,
    min_history_days: int = 10,
    ddof: int = 1,
) -> pd.Series:
    """同一日内时点相对历史同位置数据做滚动标准差。"""
    _validate_mi(x)
    lookback_days = int(lookback_days)
    min_history_days = int(min_history_days)
    ddof = int(ddof)
    if lookback_days <= 0:
        raise ValueError("same_bar_rolling_std requires lookback_days > 0")
    if min_history_days <= 0:
        raise ValueError("same_bar_rolling_std requires min_history_days > 0")
    if ddof < 0:
        raise ValueError("same_bar_rolling_std requires ddof >= 0")

    dt = pd.to_datetime(x.index.get_level_values(0))
    day = dt.normalize()
    slot = dt - day
    out = pd.Series(np.nan, index=x.index, dtype=float)

    for _, g in pd.DataFrame({"x": x, "day": day, "slot": slot}, index=x.index).groupby(level=1, sort=False):
        for _, sg in g.groupby("slot", sort=False):
            hist = sg.sort_values("day").copy()
            series = hist["x"].astype(float)
            sd = series.shift(1).rolling(window=lookback_days, min_periods=min_history_days).std(ddof=ddof)
            out.loc[hist.index] = sd.to_numpy(dtype=float)

    return out


def same_bar_rolling_mean(
    x: pd.Series,
    lookback_days: int,
    min_history_days: int = 1,
) -> pd.Series:
    """同一日内时点相对历史同位置数据做滚动均值。"""
    _validate_mi(x)
    lookback_days = int(lookback_days)
    min_history_days = int(min_history_days)
    if lookback_days <= 0:
        raise ValueError("same_bar_rolling_mean requires lookback_days > 0")
    if min_history_days <= 0:
        raise ValueError("same_bar_rolling_mean requires min_history_days > 0")

    dt = pd.to_datetime(x.index.get_level_values(0))
    day = dt.normalize()
    slot = dt - day
    out = pd.Series(np.nan, index=x.index, dtype=float)

    for _, g in pd.DataFrame({"x": x, "day": day, "slot": slot}, index=x.index).groupby(level=1, sort=False):
        for _, sg in g.groupby("slot", sort=False):
            hist = sg.sort_values("day").copy()
            series = hist["x"].astype(float)
            mu = series.shift(1).rolling(window=lookback_days, min_periods=min_history_days).mean()
            out.loc[hist.index] = mu.to_numpy(dtype=float)

    return out


def intraday_peak_signal(signal: pd.Series) -> pd.Series:
    """标记孤立出现的日内峰值信号。"""
    _validate_mi(signal)
    dt = pd.to_datetime(signal.index.get_level_values(0)).normalize()
    inst = signal.index.get_level_values(1)
    out = pd.Series(0.0, index=signal.index, dtype=float)

    for _, g in signal.groupby([dt, inst], sort=False):
        arr = g.astype(bool).to_numpy()
        prev_arr = np.zeros_like(arr, dtype=bool)
        next_arr = np.zeros_like(arr, dtype=bool)
        prev_arr[1:] = arr[:-1]
        next_arr[:-1] = arr[1:]
        peak = arr & (~prev_arr) & (~next_arr)
        out.loc[g.index] = peak.astype(float)

    return out


def intraday_ridge_signal(signal: pd.Series) -> pd.Series:
    """标记连续出现的日内脊状信号。"""
    _validate_mi(signal)
    dt = pd.to_datetime(signal.index.get_level_values(0)).normalize()
    inst = signal.index.get_level_values(1)
    out = pd.Series(0.0, index=signal.index, dtype=float)

    for _, g in signal.groupby([dt, inst], sort=False):
        arr = g.astype(bool).to_numpy()
        prev_arr = np.zeros_like(arr, dtype=bool)
        next_arr = np.zeros_like(arr, dtype=bool)
        prev_arr[1:] = arr[:-1]
        next_arr[:-1] = arr[1:]
        ridge = arr & (prev_arr | next_arr)
        out.loc[g.index] = ridge.astype(float)

    return out


def intraday_broadcast(x: pd.Series, like: pd.Series) -> pd.Series:
    """把低频日度值广播到目标日内索引。"""
    _validate_mi(x)
    _validate_mi(like)
    x_dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    x_inst = pd.Index(x.index.get_level_values(1), name="instrument")
    lookup = pd.Series(x.to_numpy(dtype=float), index=pd.MultiIndex.from_arrays([x_dt, x_inst], names=["date", "instrument"]))

    like_dt = pd.to_datetime(like.index.get_level_values(0)).normalize()
    like_inst = pd.Index(like.index.get_level_values(1), name="instrument")
    target = pd.MultiIndex.from_arrays([like_dt, like_inst], names=["date", "instrument"])
    out = lookup.reindex(target)
    return pd.Series(out.to_numpy(dtype=float), index=like.index)


def month_broadcast(x: pd.Series, like: pd.Series) -> pd.Series:
    """把月频值广播到目标日频或日内索引。"""
    _validate_mi(x)
    _validate_mi(like)
    x_dt = pd.to_datetime(x.index.get_level_values(0)).to_period("M").to_timestamp("M")
    x_inst = pd.Index(x.index.get_level_values(1), name="instrument")
    lookup = pd.Series(x.to_numpy(dtype=float), index=pd.MultiIndex.from_arrays([x_dt, x_inst], names=["month", "instrument"]))

    like_dt = pd.to_datetime(like.index.get_level_values(0)).to_period("M").to_timestamp("M")
    like_inst = pd.Index(like.index.get_level_values(1), name="instrument")
    target = pd.MultiIndex.from_arrays([like_dt, like_inst], names=["month", "instrument"])
    out = lookup.reindex(target)
    return pd.Series(out.to_numpy(dtype=float), index=like.index)


def intraday_downsample_last(x: pd.Series, block_size: int = 2) -> pd.Series:
    """按固定块降采样，保留每块最后一个值。"""
    _validate_mi(x)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("intraday_downsample_last requires block_size > 0")
    dt = pd.to_datetime(x.index.get_level_values(0))
    day = dt.normalize()
    slot = x.groupby([day, x.index.get_level_values(1)], sort=False).cumcount()
    block = slot // block_size
    grouped = x.groupby([day, x.index.get_level_values(1), block], sort=False).last()
    grouped.index = grouped.index.set_names(["date", "instrument", "block"])
    out_index = []
    for date_val, inst_val, block_val in grouped.index:
        ts = pd.Timestamp(date_val) + pd.Timedelta(minutes=5 * (int(block_val) + 1) * block_size)
        out_index.append((ts, inst_val))
    out = pd.Series(grouped.to_numpy(dtype=float), index=pd.MultiIndex.from_tuples(out_index, names=["datetime", "instrument"]))
    return out.sort_index()


def intraday_downsample_sum(x: pd.Series, block_size: int = 2) -> pd.Series:
    """按固定块降采样，对每块做求和。"""
    _validate_mi(x)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("intraday_downsample_sum requires block_size > 0")
    dt = pd.to_datetime(x.index.get_level_values(0))
    day = dt.normalize()
    slot = x.groupby([day, x.index.get_level_values(1)], sort=False).cumcount()
    block = slot // block_size
    grouped = x.groupby([day, x.index.get_level_values(1), block], sort=False).sum(min_count=1)
    grouped.index = grouped.index.set_names(["date", "instrument", "block"])
    out_index = []
    for date_val, inst_val, block_val in grouped.index:
        ts = pd.Timestamp(date_val) + pd.Timedelta(minutes=5 * (int(block_val) + 1) * block_size)
        out_index.append((ts, inst_val))
    out = pd.Series(grouped.to_numpy(dtype=float), index=pd.MultiIndex.from_tuples(out_index, names=["datetime", "instrument"]))
    return out.sort_index()


def group_day_delay(x: pd.Series, periods: int = 1) -> pd.Series:
    _validate_mi(x)
    periods = int(periods)
    if periods == 0:
        return x.copy()
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)
    return x.groupby([dt, inst], group_keys=False, sort=False).shift(periods).reindex(x.index)


def group_day_log_return(x: pd.Series, periods: int = 1) -> pd.Series:
    """按日内序列计算对数收益率。"""
    _validate_mi(x)
    periods = int(periods)
    if periods <= 0:
        raise ValueError("group_day_log_return requires periods > 0")
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> pd.Series:
        prev = g.shift(periods)
        out = np.log(g / prev)
        return out.replace([np.inf, -np.inf], np.nan)

    return x.groupby([dt, inst], group_keys=False, sort=False).apply(_calc).reindex(x.index)


def intraday_jump_strength(x: pd.Series, trim_head: int = 1, trim_tail: int = 1) -> pd.Series:
    """计算日内单利与复利偏差定义的 bar 级跳跃强度。"""
    _validate_mi(x)
    trim_head = int(trim_head)
    trim_tail = int(trim_tail)
    if trim_head < 0 or trim_tail < 0:
        raise ValueError("intraday_jump_strength requires trim_head/trim_tail >= 0")
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = x.index.get_level_values(1)

    def _calc(g: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=g.index, dtype=float)
        end = len(g) - trim_tail if trim_tail > 0 else len(g)
        work = g.iloc[trim_head:end]
        if len(work) <= 1:
            return out
        prev = work.shift(1)
        simple_ret = work / prev - 1.0
        log_ret = np.log(work / prev)
        jump_bar = 2.0 * (simple_ret - log_ret) - np.square(log_ret)
        jump_bar = jump_bar.replace([np.inf, -np.inf], np.nan)
        out.loc[work.index] = jump_bar.to_numpy(dtype=float)
        return out

    return x.groupby([dt, inst], group_keys=False, sort=False).apply(_calc).reindex(x.index)


def intraday_jump_count(close: pd.Series, alpha: float = 0.05) -> pd.Series:
    """基于 SwV 检验输出单日是否发生跳跃的 0/1 指示。"""
    _validate_mi(close)
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("intraday_jump_count requires 0 < alpha < 1")

    dt = pd.to_datetime(close.index.get_level_values(0)).normalize()
    inst = close.index.get_level_values(1)
    crit = 1.6448536269514722 if abs(alpha - 0.05) <= 1e-12 else float(__import__("scipy.stats").stats.norm.ppf(1.0 - alpha))
    mu6 = 15.0

    def _calc(g: pd.Series) -> float:
        px = g.astype(float)
        R = px.pct_change().dropna().to_numpy(dtype=float)
        r = np.log(px / px.shift(1)).dropna().to_numpy(dtype=float)
        n = len(r)
        if n < 6:
            return np.nan
        rv = float(np.sum(r ** 2))
        bv = float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
        swv = float(2.0 * np.sum(R - r))
        if not np.isfinite(swv) or abs(swv) <= EPS:
            return np.nan
        omega_terms = []
        for i in range(0, n - 3):
            prod4 = float(np.prod(np.abs(r[i : i + 4])))
            omega_terms.append(prod4 ** 1.5)
        if not omega_terms:
            return np.nan
        omega_hat = (mu6 / 9.0) * (n ** 3) / max(n - 3, 1) * float(np.sum(omega_terms))
        if not np.isfinite(omega_hat) or omega_hat <= EPS:
            return np.nan
        t_swv = (bv / ((n ** -1) * np.sqrt(omega_hat))) * (1.0 - rv / swv)
        return float(t_swv > crit)

    grouped = close.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def intraday_jump_return(open_: pd.Series, close: pd.Series, z_threshold: float = 1.96) -> pd.Series:
    """基于 Jiang-Oomen 检验输出单日总跳跃收益。"""
    _validate_mi(open_)
    close = _reindex_like(open_, close)
    z_threshold = float(z_threshold)
    dt = pd.to_datetime(open_.index.get_level_values(0)).normalize()
    inst = open_.index.get_level_values(1)
    df = pd.DataFrame({"open": open_, "close": close}, index=open_.index)
    mu1 = math.sqrt(2.0 / math.pi)
    mu6 = 15.0

    def _calc_js(r: np.ndarray) -> float:
        n = len(r)
        if n < 7:
            return np.nan
        R = np.exp(r) - 1.0
        rv_n = np.sum(r ** 2)
        swv_n = 2.0 * np.sum(R - r)
        if not np.isfinite(swv_n) or abs(swv_n) <= EPS:
            return np.nan
        vhat = np.sum(np.abs(r[1:]) * np.abs(r[:-1])) / (mu1 ** 2)
        prod_sum = 0.0
        for k in range(0, n - 5):
            prod_sum += float(np.prod(np.abs(r[k : k + 6])))
        if not np.isfinite(prod_sum) or prod_sum <= EPS or n <= 5:
            return np.nan
        omega = (mu6 / 9.0) * (n ** 3) * (mu1 ** (-6)) / (n - 5) * prod_sum
        if not np.isfinite(omega) or omega <= EPS:
            return np.nan
        return float(n * vhat / math.sqrt(omega) * (1.0 - rv_n / swv_n))

    def _calc(g: pd.DataFrame) -> float:
        o = g["open"].to_numpy(dtype=float)
        c = g["close"].to_numpy(dtype=float)
        n = len(g)
        if n < 7:
            return np.nan
        if o[0] <= 0 or c[0] <= 0:
            return np.nan
        r = np.empty(n, dtype=float)
        r[0] = math.log(c[0] / o[0])
        for i in range(1, n):
            if c[i] <= 0 or c[i - 1] <= 0:
                return np.nan
            r[i] = math.log(c[i] / c[i - 1])
        r_work = r.copy()
        jump_ret = 0.0
        while True:
            js0 = _calc_js(r_work)
            if not np.isfinite(js0) or abs(js0) <= z_threshold:
                break
            med_r = float(np.median(r_work))
            best_idx = None
            best_gain = -np.inf
            for i in range(n):
                r_tmp = r_work.copy()
                r_tmp[i] = med_r
                jsi = _calc_js(r_tmp)
                if not np.isfinite(jsi):
                    continue
                gain = abs(js0) - abs(jsi)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx is None:
                break
            jump_ret += float(r_work[best_idx])
            r_work[best_idx] = med_r
        return jump_ret

    grouped = df.groupby([dt, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def student_t_cdf(x: pd.Series, df: int = 5) -> pd.Series:
    """逐元素计算 Student t 分布 CDF。"""
    _validate_mi(x)
    df = int(df)
    if df <= 0:
        raise ValueError("student_t_cdf requires df > 0")
    try:
        from scipy.stats import t as _student_t
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("student_t_cdf requires scipy to be installed") from exc

    arr = x.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    valid = np.isfinite(arr)
    if bool(np.any(valid)):
        out[valid] = _student_t.cdf(arr[valid], df=df)
    return pd.Series(out, index=x.index)


def ts_sorted_weighted_spread(value: pd.Series, sort_key: pd.Series, window: int, ratio: float) -> pd.Series:
    """按排序键分层后，计算加权高低组价差。"""
    _validate_mi(value)
    sort_key = _reindex_like(value, sort_key)
    _validate_window(window)
    ratio = float(ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError("ts_sorted_weighted_spread requires 0 < ratio < 1")
    k = max(1, int(np.floor(window * ratio)))
    out = pd.Series(np.nan, index=value.index, dtype=float)

    for _, g in pd.DataFrame({"v": value, "s": sort_key}, index=value.index).groupby(level=1, sort=False):
        vv = g["v"].to_numpy(dtype=float)
        ss = g["s"].to_numpy(dtype=float)
        idx = g.index
        for i in range(window - 1, len(g)):
            sub_v = vv[i - window + 1 : i + 1]
            sub_s = ss[i - window + 1 : i + 1]
            valid = np.isfinite(sub_v) & np.isfinite(sub_s) & (sub_s > 0)
            if int(valid.sum()) < k + 1:
                continue
            sv = sub_v[valid]
            sk = sub_s[valid]
            order = np.argsort(sk)
            low_v = sv[order[:k]]
            low_s = sk[order[:k]]
            high_v = sv[order[k:]]
            high_s = sk[order[k:]]
            inv_low = 1.0 / np.clip(low_s, EPS, None)
            w_low = inv_low / (inv_low.sum() + EPS)
            w_high = high_s / (high_s.sum() + EPS)
            rev_mom = float(np.sum(w_low * low_v))
            rev_rev = float(np.sum(w_high * high_v))
            out.loc[idx[i]] = rev_rev - rev_mom

    return out


def rolling_cs_spearman_mean(x: pd.Series, window: int, min_obs: int = 10) -> pd.Series:
    """计算窗口内截面 Spearman 相关性的平均强度。"""
    _validate_mi(x)
    _validate_window(window)
    min_obs = int(min_obs)
    if min_obs < 2:
        raise ValueError("rolling_cs_spearman_mean requires min_obs >= 2")

    wide = x.unstack(level=1).sort_index()
    arr = wide.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)

    for t in range(window - 1, len(wide.index)):
        seg = arr[t - window + 1 : t + 1]
        valid_cols = np.sum(np.isfinite(seg), axis=0) >= min_obs
        if int(valid_cols.sum()) <= 1:
            continue
        seg_valid = seg[:, valid_cols]
        ranks = np.full_like(seg_valid, np.nan, dtype=float)
        for j in range(seg_valid.shape[1]):
            col = seg_valid[:, j]
            mask = np.isfinite(col)
            if int(mask.sum()) >= 2:
                ranks[mask, j] = pd.Series(col[mask]).rank(method="average").to_numpy(dtype=float)
        fill = np.nanmean(ranks, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        ranks = np.where(np.isfinite(ranks), ranks, fill)
        corr = np.corrcoef(ranks, rowvar=False)
        if np.ndim(corr) == 0:
            continue
        np.fill_diagonal(corr, np.nan)
        score = np.nanmean(np.abs(corr), axis=1)
        row = np.full(wide.shape[1], np.nan, dtype=float)
        row[np.where(valid_cols)[0]] = score
        out[t] = row

    return _stack_wide(pd.DataFrame(out, index=wide.index, columns=wide.columns)).reindex(x.index)


def group_month_mean(x: pd.Series, min_count: int = 1) -> pd.Series:
    """按月聚合并计算月均值。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0))
    inst = x.index.get_level_values(1)
    month_end = dt.to_period("M").to_timestamp("M")
    grouped = x.groupby([month_end, inst], sort=True).mean()
    counts = x.groupby([month_end, inst], sort=True).count()
    grouped = grouped.where(counts >= int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_month_sum(x: pd.Series, min_count: int = 1) -> pd.Series:
    """按月聚合并计算月求和。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0))
    inst = x.index.get_level_values(1)
    month_end = dt.to_period("M").to_timestamp("M")
    grouped = x.groupby([month_end, inst], sort=True).sum(min_count=int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_month_regression_coef(
    y: pd.Series,
    x1: pd.Series,
    x2: pd.Series,
    lead: int = 1,
    coef_pos: int = 2,
    min_obs: int = 10,
) -> pd.Series:
    """按月对单只股票做回归，返回指定解释变量系数。"""
    _validate_mi(y)
    x1 = _reindex_like(y, x1)
    x2 = _reindex_like(y, x2)
    lead = int(lead)
    coef_pos = int(coef_pos)
    min_obs = int(min_obs)
    if lead <= 0:
        raise ValueError("group_month_regression_coef requires lead > 0")
    if coef_pos not in {1, 2}:
        raise ValueError("group_month_regression_coef requires coef_pos in {1, 2}")
    if min_obs <= 0:
        raise ValueError("group_month_regression_coef requires min_obs > 0")

    dt = pd.to_datetime(y.index.get_level_values(0))
    inst = y.index.get_level_values(1)
    month_end = dt.to_period("M").to_timestamp("M")
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2}, index=y.index)

    def _calc(g: pd.DataFrame) -> float:
        g = g.sort_index()
        if len(g) <= lead:
            return np.nan
        yy = g["y"].to_numpy(dtype=float)[lead:]
        xx1 = g["x1"].to_numpy(dtype=float)[:-lead]
        xx2 = g["x2"].to_numpy(dtype=float)[:-lead]
        valid = np.isfinite(yy) & np.isfinite(xx1) & np.isfinite(xx2)
        if int(valid.sum()) < min_obs:
            return np.nan
        Y = yy[valid]
        X = np.column_stack([np.ones(int(valid.sum()), dtype=float), xx1[valid], xx2[valid]])
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.nan
        return float(beta[coef_pos])

    grouped = df.groupby([month_end, inst], sort=True).apply(_calc)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_month_last(x: pd.Series, min_count: int = 1) -> pd.Series:
    """按月聚合并取月末最后一个值。"""
    _validate_mi(x)
    dt = pd.to_datetime(x.index.get_level_values(0))
    inst = x.index.get_level_values(1)
    month_end = dt.to_period("M").to_timestamp("M")
    grouped = x.groupby([month_end, inst], sort=True).last()
    counts = x.groupby([month_end, inst], sort=True).count()
    grouped = grouped.where(counts >= int(min_count))
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def group_ex_self_weighted_mean(
    x: pd.Series,
    weight: pd.Series,
    group_map: pd.Series,
    min_group_size: int = 2,
) -> pd.Series:
    """按组计算剔除自身后的加权均值。"""
    _validate_mi(x)
    weight = _reindex_like(x, weight)
    group_map = _reindex_like(x, group_map)
    df = pd.DataFrame({"x": x, "w": weight, "g": group_map})
    out = pd.Series(np.nan, index=x.index, dtype=float)

    for dt, gdf in df.groupby(level=0, sort=False):
        valid = gdf["x"].notna() & gdf["w"].notna() & gdf["g"].notna() & (gdf["w"] > 0)
        if not bool(valid.any()):
            continue
        work = gdf.loc[valid].copy()
        grp_count = work.groupby("g")["x"].transform("count")
        weighted = work["x"] * work["w"]
        grp_weight_sum = work.groupby("g")["w"].transform("sum")
        grp_weighted_sum = weighted.groupby(work["g"]).transform("sum")

        num = grp_weighted_sum - weighted
        den = grp_weight_sum - work["w"]
        values = num / den
        values = values.where((grp_count >= int(min_group_size)) & (den > EPS), np.nan)
        out.loc[work.index] = values.to_numpy(dtype=float)

    return out


def group_day_corr(x: pd.Series, y: pd.Series, min_count: int = 2) -> pd.Series:
    """按日内数据聚合为日频相关系数。"""
    _validate_mi(x)
    y = _reindex_like(x, y)
    dt = pd.to_datetime(x.index.get_level_values(0)).normalize()
    inst = pd.Index(x.index.get_level_values(1), name="instrument")
    df = pd.DataFrame({"x": x, "y": y})

    def _corr(g: pd.DataFrame) -> float:
        xv = g["x"].astype(float)
        yv = g["y"].astype(float)
        valid = xv.notna() & yv.notna()
        if int(valid.sum()) < int(min_count):
            return np.nan
        xv = xv[valid]
        yv = yv[valid]
        if float(xv.std(ddof=0)) <= EPS or float(yv.std(ddof=0)) <= EPS:
            return np.nan
        return float(xv.corr(yv))

    grouped = df.groupby([dt, inst], sort=True).apply(_corr)
    grouped.index = grouped.index.set_names(["datetime", "instrument"])
    return grouped.sort_index()


def add(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素相加。"""
    return x + y


def sub(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素相减。"""
    return x - y


def mul(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素相乘。"""
    return x * y


def div(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素相除，并规避除零。"""
    if isinstance(y, pd.Series):
        y = y.where(y.abs() > EPS, np.nan)
    elif abs(y) <= EPS:
        return pd.Series(np.nan, index=x.index)
    return x / y


def log(x: pd.Series) -> pd.Series:
    """对数变换，输入会裁剪到正数区间。"""
    return np.log(np.clip(x, EPS, None))


def slog1p(x: pd.Series) -> pd.Series:
    """保留符号的 log1p 变换。"""
    arr = x.to_numpy(dtype=float)
    return pd.Series(np.sign(arr) * np.log1p(np.abs(arr)), index=x.index)


def inv(x: pd.Series) -> pd.Series:
    """逐元素取倒数，并规避除零。"""
    base = x.where(x.abs() > EPS, np.nan)
    return 1.0 / base


def exp(x: pd.Series) -> pd.Series:
    """指数变换，并限制输入范围避免溢出。"""
    return pd.Series(np.exp(np.clip(x.to_numpy(dtype=float), -50.0, 50.0)), index=x.index)


def sqrt(x: pd.Series) -> pd.Series:
    """平方根变换，负值会先裁剪为 0。"""
    return pd.Series(np.sqrt(np.clip(x.to_numpy(dtype=float), 0.0, None)), index=x.index)


def power(x: pd.Series, p: pd.Series | float) -> pd.Series:
    """逐元素幂运算。"""
    if isinstance(p, pd.Series):
        p = _reindex_like(x, p)
        return pd.Series(np.power(x.to_numpy(dtype=float), p.to_numpy(dtype=float)), index=x.index)
    return pd.Series(np.power(x.to_numpy(dtype=float), float(p)), index=x.index)


def abs_op(x: pd.Series) -> pd.Series:
    """逐元素取绝对值。"""
    return x.abs()


def sign(x: pd.Series) -> pd.Series:
    """逐元素取符号。"""
    return pd.Series(np.sign(x.to_numpy(dtype=float)), index=x.index)


def not_op(x: pd.Series) -> pd.Series:
    """逐元素逻辑非。"""
    return pd.Series(~x.astype(bool).to_numpy(), index=x.index)


def greater(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素取两者中的较大值。"""
    if isinstance(y, pd.Series):
        y = _reindex_like(x, y)
        return x.where(x >= y, y)
    return x.where(x >= float(y), float(y))


def less(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素取两者中的较小值。"""
    if isinstance(y, pd.Series):
        y = _reindex_like(x, y)
        return x.where(x <= y, y)
    return x.where(x <= float(y), float(y))


def gt(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素判断是否大于。"""
    return x.gt(_reindex_like(x, y) if isinstance(y, pd.Series) else float(y))


def ge(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素判断是否大于等于。"""
    return x.ge(_reindex_like(x, y) if isinstance(y, pd.Series) else float(y))


def lt(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素判断是否小于。"""
    return x.lt(_reindex_like(x, y) if isinstance(y, pd.Series) else float(y))


def le(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素判断是否小于等于。"""
    return x.le(_reindex_like(x, y) if isinstance(y, pd.Series) else float(y))


def eq(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素判断是否相等。"""
    return x.eq(_reindex_like(x, y) if isinstance(y, pd.Series) else float(y))


def ne(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素判断是否不相等。"""
    return x.ne(_reindex_like(x, y) if isinstance(y, pd.Series) else float(y))


def and_op(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素逻辑与。"""
    rhs = _reindex_like(x, y) if isinstance(y, pd.Series) else bool(y)
    return x.astype(bool) & rhs


def or_op(x: pd.Series, y: pd.Series | float) -> pd.Series:
    """逐元素逻辑或。"""
    rhs = _reindex_like(x, y) if isinstance(y, pd.Series) else bool(y)
    return x.astype(bool) | rhs


def delay(x: pd.Series, periods: int) -> pd.Series:
    """按个股时序滞后若干期。"""
    _validate_mi(x)
    return x.groupby(level=1, group_keys=False).shift(int(periods))


def ref(x: pd.Series, periods: int) -> pd.Series:
    """`delay` 的别名。"""
    return delay(x, periods)


def delta(x: pd.Series, periods: int) -> pd.Series:
    """当前值减去滞后若干期的值。"""
    return x - delay(x, periods)


def ts_pct_change(x: pd.Series, periods: int) -> pd.Series:
    """按个股计算若干期收益率。"""
    ref_x = delay(x, periods)
    return (x / ref_x) - 1.0


def ema(x: pd.Series, window: int) -> pd.Series:
    """按个股计算指数移动平均。"""
    _validate_window(window)
    _validate_mi(x)

    def _one_inst(s: pd.Series) -> pd.Series:
        base = s.droplevel(1)
        out = base.ewm(span=window, adjust=False, min_periods=window).mean()
        out.index = s.index
        return out

    out = x.groupby(level=1, group_keys=False).apply(_one_inst)
    return out.reindex(x.index)


def ts_linear_decay_mean(x: pd.Series, window: int) -> pd.Series:
    """按个股计算线性衰减均值，最近一期权重最高。"""
    _validate_window(window)
    _validate_mi(x)
    weights = np.arange(1, int(window) + 1, dtype=float)
    weights /= weights.sum()

    def _apply(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return float(np.dot(arr.astype(float), weights))

    out = x.groupby(level=1, group_keys=False).rolling(window).apply(_apply, raw=True).droplevel(0)
    return out.reindex(x.index)


def ts_exp_weighted_mean_lagged(x: pd.Series, window: int) -> pd.Series:
    """按个股计算使用 t-1..t-window 的指数衰减加权均值。"""
    _validate_window(window)
    _validate_mi(x)
    alpha = 2.0 / (1.0 + float(window))
    weights = np.array([(1.0 - alpha) ** i for i in range(int(window))], dtype=float)

    def _one_inst(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float)
        out = np.full(len(s), np.nan, dtype=float)
        for i in range(len(s)):
            if i < 1:
                continue
            hist = vals[max(0, i - int(window)) : i][::-1]
            w = weights[: len(hist)]
            valid = np.isfinite(hist)
            if int(valid.sum()) == 0:
                continue
            out[i] = float(np.sum(hist[valid] * w[valid]) / float(window))
        return pd.Series(out, index=s.index)

    out = x.groupby(level=1, group_keys=False).apply(_one_inst)
    return out.reindex(x.index)


def ts_turnover_ref_price(close: pd.Series, turn: pd.Series, window: int) -> pd.Series:
    """按个股利用换手率递推历史持仓权重，计算参考价格。"""
    _validate_window(window)
    _validate_mi(close)
    turn = _reindex_like(close, turn)
    window = int(window)

    out = pd.Series(np.nan, index=close.index, dtype=float)

    def _one_inst(df: pd.DataFrame) -> pd.Series:
        p = df["close"].to_numpy(dtype=float)
        v = df["turn"].to_numpy(dtype=float)
        valid_turn = v[np.isfinite(v)]
        if len(valid_turn) and float(np.nanmedian(valid_turn)) > 1.0:
            v = v / 100.0
        v = np.clip(v, 0.0, 1.0)
        out = np.full(len(df), np.nan, dtype=float)

        for t in range(len(df)):
            start = max(0, t - window)
            if t - start < 1:
                continue
            survive = 1.0
            weights: list[float] = []
            prices: list[float] = []
            for j in range(t - 1, start - 1, -1):
                if not np.isfinite(p[j]) or not np.isfinite(v[j]):
                    continue
                w = float(v[j]) * survive
                weights.append(w)
                prices.append(float(p[j]))
                survive *= (1.0 - float(v[j]))
            if not weights:
                continue
            w_arr = np.asarray(weights, dtype=float)
            p_arr = np.asarray(prices, dtype=float)
            den = float(w_arr.sum())
            if den <= EPS:
                continue
            out[t] = float(np.sum(w_arr * p_arr) / den)

        return pd.Series(out, index=df.index, dtype=float)

    frame = pd.concat([close.rename("close"), turn.rename("turn")], axis=1)
    for _, g in frame.groupby(level=1, sort=False):
        inst_out = _one_inst(g)
        out.loc[inst_out.index] = inst_out.to_numpy(dtype=float)
    return out.reindex(close.index)


def resiliency(x: pd.Series, lookback_days: int, bars_per_day: int, hp_lambda: float = 1600.0) -> pd.Series:
    """按个股对滚动日内 log 价格做 HP 分解与频谱分析，输出恢复速度均值。"""
    _validate_mi(x)
    lookback_days = int(lookback_days)
    bars_per_day = int(bars_per_day)
    hp_lambda = float(hp_lambda)
    if lookback_days <= 0:
        raise ValueError("resiliency requires lookback_days > 0")
    if bars_per_day <= 0:
        raise ValueError("resiliency requires bars_per_day > 0")
    if hp_lambda < 0.0:
        raise ValueError("resiliency requires hp_lambda >= 0")

    window = lookback_days * bars_per_day
    results: list[tuple[pd.Timestamp, str, float]] = []

    for inst, s in x.groupby(level=1, sort=False):
        base = s.droplevel(1).sort_index()
        base = base.replace([np.inf, -np.inf], np.nan)
        dt = pd.to_datetime(base.index)
        day = dt.normalize()
        values = np.log(base.to_numpy(dtype=float))
        day_vals = day.to_numpy()
        day_end_idx = np.where(pd.Series(day_vals).ne(pd.Series(day_vals).shift(-1)).to_numpy())[0]

        for idx in day_end_idx:
            current_day = pd.Timestamp(day_vals[idx])
            if idx + 1 < window:
                results.append((current_day, inst, np.nan))
                continue
            y = values[idx - window + 1 : idx + 1]
            if np.any(~np.isfinite(y)):
                results.append((current_day, inst, np.nan))
                continue

            n = len(y)
            if n < 4:
                results.append((current_day, inst, np.nan))
                continue
            A = _hp_system_matrix(n, hp_lambda)
            try:
                trend = np.linalg.solve(A, y)
            except np.linalg.LinAlgError:
                results.append((current_day, inst, np.nan))
                continue
            cycle = y - trend
            fft_vals = np.fft.fft(cycle)
            half = n // 2
            amps = np.abs(fft_vals[1 : half + 1]) / float(n)
            if len(amps) == 0:
                results.append((current_day, inst, np.nan))
                continue
            ks = np.arange(1, len(amps) + 1, dtype=float)
            periods = n / ks
            fk = 2.0 * amps / periods
            results.append((current_day, inst, float(np.mean(fk))))

    out = pd.Series(
        [v for _, _, v in results],
        index=pd.MultiIndex.from_tuples(
            [(dt, inst) for dt, inst, _ in results],
            names=["datetime", "instrument"],
        ),
        dtype=float,
    )
    return out.sort_index()


def _calc_ret_1d_arr(close_arr: np.ndarray) -> np.ndarray:
    ret = np.full(close_arr.shape, np.nan, dtype=np.float64)
    if len(close_arr) <= 1:
        return ret
    valid = (
        np.isfinite(close_arr[1:])
        & np.isfinite(close_arr[:-1])
        & (close_arr[1:] > 0.0)
        & (close_arr[:-1] > 0.0)
    )
    ret[1:][valid] = close_arr[1:][valid] / close_arr[:-1][valid] - 1.0
    return ret


def _exp_decay_weights_ranked(n: int, half_life: float) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float64)
    lam = np.log(2.0) / float(half_life)
    k = np.arange(n, 0, -1, dtype=np.float64)
    w = np.exp(-lam * k)
    den = float(w.sum())
    return w / den if den > EPS else np.full(n, 1.0 / n, dtype=np.float64)


def _rowwise_corr_arr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_mean = X.mean(axis=1, keepdims=True)
    y_mean = float(y.mean())
    Xc = X - X_mean
    yc = y - y_mean
    num = np.sum(Xc * yc, axis=1)
    den = np.sqrt(np.sum(Xc * Xc, axis=1) * np.sum(yc * yc))
    out = np.full(X.shape[0], np.nan, dtype=np.float64)
    valid = den > 0.0
    out[valid] = num[valid] / den[valid]
    return out


def _similar_path_samples_1d(
    path_arr: np.ndarray,
    future_ret_arr: np.ndarray,
    rw: int,
    history_window: int,
    threshold: float,
    holding_time: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = len(path_arr)
    samples: list[np.ndarray] = [np.array([], dtype=np.float64) for _ in range(n)]
    ages: list[np.ndarray] = [np.array([], dtype=np.float64) for _ in range(n)]
    if n < rw:
        return samples, ages

    all_windows = sliding_window_view(path_arr, rw)
    for t in range(rw - 1, n):
        current_start = t - rw + 1
        current_seq = path_arr[current_start : t + 1]
        if np.any(~np.isfinite(current_seq)):
            continue

        hist_start = max(0, current_start - history_window)
        hist_end = current_start - holding_time - rw
        if hist_end < hist_start:
            continue

        candidate_starts = np.arange(hist_start, hist_end + 1, dtype=int)
        candidate_windows = all_windows[candidate_starts]
        valid_rows = np.isfinite(candidate_windows).all(axis=1)
        if not bool(valid_rows.any()):
            continue

        valid_starts = candidate_starts[valid_rows]
        valid_windows = candidate_windows[valid_rows]
        corr = _rowwise_corr_arr(valid_windows, current_seq)
        matched_starts = valid_starts[np.abs(corr) >= threshold]
        if matched_starts.size == 0:
            continue

        future_vals: list[float] = []
        age_vals: list[float] = []
        for s in matched_starts:
            fut_start = s + rw
            fut_end = s + rw + holding_time
            if fut_end > n:
                continue
            seg = future_ret_arr[fut_start:fut_end]
            if np.any(~np.isfinite(seg)):
                continue
            cum_ret = float(np.prod(1.0 + seg) - 1.0)
            future_vals.append(cum_ret)
            age_vals.append(float(current_start - s))

        if future_vals:
            samples[t] = np.asarray(future_vals, dtype=np.float64)
            ages[t] = np.asarray(age_vals, dtype=np.float64)

    return samples, ages


def similar_path_reversal(
    close: pd.Series,
    benchmark_close: pd.Series,
    rw: int,
    history_window: int,
    threshold: float,
    holding_time: int,
    half_life: float,
) -> pd.Series:
    """按滚动路径相似性匹配历史片段，并对后续超额收益做指数加权均值后取负号。"""
    _validate_mi(close)
    benchmark_close = _reindex_like(close, benchmark_close)
    rw = int(rw)
    history_window = int(history_window)
    threshold = float(threshold)
    holding_time = int(holding_time)
    half_life = float(half_life)
    out = pd.Series(np.nan, index=close.index, dtype=float)

    close_df = pd.concat([close.rename("close"), benchmark_close.rename("bench")], axis=1)
    for _, g in close_df.groupby(level=1, sort=False):
        p = g["close"].to_numpy(dtype=np.float64)
        b = g["bench"].to_numpy(dtype=np.float64)
        stock_ret = _calc_ret_1d_arr(p)
        bench_ret = _calc_ret_1d_arr(b)
        excess_ret = stock_ret - bench_ret
        samples, ages = _similar_path_samples_1d(p, excess_ret, rw, history_window, threshold, holding_time)
        vals = np.full(len(g), np.nan, dtype=np.float64)
        for i, arr in enumerate(samples):
            if arr.size == 0:
                continue
            order = np.argsort(ages[i])
            ranked = arr[order]
            weights = _exp_decay_weights_ranked(len(ranked), half_life)
            vals[i] = -float(np.sum(weights * ranked))
        out.loc[g.index] = vals
    return out.reindex(close.index)


def similar_path_low_volatility(
    close: pd.Series,
    rw: int,
    history_window: int,
    threshold: float,
    holding_time: int,
    min_matches: int = 5,
    use_exponential_weight: int = 0,
    half_life: float = 20.0,
) -> pd.Series:
    """按路径相似性匹配历史片段，对后续累计收益样本波动率取倒数。"""
    _validate_mi(close)
    rw = int(rw)
    history_window = int(history_window)
    threshold = float(threshold)
    holding_time = int(holding_time)
    min_matches = int(min_matches)
    use_exponential_weight = int(use_exponential_weight)
    half_life = float(half_life)
    out = pd.Series(np.nan, index=close.index, dtype=float)

    for _, s in close.groupby(level=1, sort=False):
        base = s.to_numpy(dtype=np.float64)
        ret = _calc_ret_1d_arr(base)
        samples, ages = _similar_path_samples_1d(base, ret, rw, history_window, threshold, holding_time)
        vals = np.full(len(s), np.nan, dtype=np.float64)
        for i, arr in enumerate(samples):
            if arr.size < min_matches:
                continue
            if use_exponential_weight:
                lam = np.log(2.0) / half_life
                w = np.exp(-lam * ages[i])
                w_sum = float(w.sum())
                if w_sum <= EPS:
                    continue
                mu = float(np.sum(w * arr) / w_sum)
                var = float(np.sum(w * ((arr - mu) ** 2)) / w_sum)
                std_val = float(np.sqrt(max(var, 0.0)))
            else:
                std_val = float(np.std(arr, ddof=0))
            if np.isfinite(std_val) and std_val > EPS:
                vals[i] = 1.0 / std_val
        out.loc[s.index] = vals
    return out.reindex(close.index)


def turnover_survival_loss(
    close: pd.Series,
    ref_price: pd.Series,
    turn: pd.Series,
    window: int,
    turn_scale: float = 1.0,
) -> pd.Series:
    """按换手率诱导的持仓存活权重，对历史参考价高于现价的亏损部分做加权聚合。"""
    _validate_window(window)
    _validate_mi(close)
    ref_price = _reindex_like(close, ref_price)
    turn = _reindex_like(close, turn)
    window = int(window)
    turn_scale = float(turn_scale)

    frame = pd.concat(
        [close.rename("close"), ref_price.rename("ref_price"), turn.rename("turn")],
        axis=1,
    )
    out = pd.Series(np.nan, index=close.index, dtype=float)

    for _, g in frame.groupby(level=1, sort=False):
        c = g["close"].to_numpy(dtype=float)
        p = g["ref_price"].to_numpy(dtype=float)
        v = np.clip(g["turn"].to_numpy(dtype=float) * turn_scale, 0.0, 1.0)
        vals = np.full(len(g), np.nan, dtype=float)

        for t in range(len(g)):
            start = max(0, t - window)
            if t <= start:
                continue
            close_t = c[t]
            if not np.isfinite(close_t) or close_t <= 0.0:
                continue

            hist_p = p[start:t]
            hist_v = v[start:t]
            losses: list[float] = []
            weights: list[float] = []
            m = len(hist_p)
            for j in range(m):
                p_ref = hist_p[j]
                v_ref = hist_v[j]
                if not np.isfinite(p_ref) or not np.isfinite(v_ref):
                    continue
                loss_j = ((close_t - p_ref) / close_t) if close_t < p_ref else 0.0
                survival = 1.0 if j == m - 1 else float(np.prod(1.0 - hist_v[j + 1 :]))
                weights.append(float(v_ref) * survival)
                losses.append(float(loss_j))

            if not weights:
                continue
            w_arr = np.asarray(weights, dtype=float)
            l_arr = np.asarray(losses, dtype=float)
            den = float(w_arr.sum())
            if den <= EPS:
                continue
            vals[t] = float(np.sum((w_arr / den) * l_arr))

        out.loc[g.index] = vals

    return out.reindex(close.index)


def holding_return(
    amount: pd.Series,
    turnover: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """按个股估计历史留存筹码的加权成本，并返回当前价格相对该成本的收益。"""
    _validate_window(window)
    _validate_mi(close)
    amount = _reindex_like(close, amount)
    turnover = _reindex_like(close, turnover)
    window = int(window)

    frame = pd.concat(
        [amount.rename("amount"), turnover.rename("turnover"), close.rename("close")],
        axis=1,
    )
    out = pd.Series(np.nan, index=close.index, dtype=float)

    for _, g in frame.groupby(level=1, sort=False):
        amt = g["amount"].droplevel(1).astype(float)
        turn = g["turnover"].droplevel(1).astype(float)
        cls = g["close"].droplevel(1).astype(float)

        survival = (1.0 - turn).clip(lower=EPS, upper=1.0)
        base_weight = amt * np.exp(-np.log(survival).cumsum())

        hist_den = base_weight.rolling(window=window, min_periods=window).sum().shift(1)
        hist_num = (cls * base_weight).rolling(window=window, min_periods=window).sum().shift(1)
        ref_cost = hist_num / hist_den.replace(0.0, np.nan)
        vals = (cls / ref_cost.replace(0.0, np.nan)) - 1.0

        vals.index = g.index
        out.loc[g.index] = vals.to_numpy(dtype=float)

    return out.reindex(close.index)


def _prospect_w_plus(p: float, gamma: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    num = p**gamma
    den = (p**gamma + (1.0 - p) ** gamma) ** (1.0 / gamma)
    return float(num / den)


def _prospect_w_minus(p: float, delta: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    num = p**delta
    den = (p**delta + (1.0 - p) ** delta) ** (1.0 / delta)
    return float(num / den)


def prospect_pos_component(ret_1d: pd.Series, window: int, alpha: float = 0.88, gamma: float = 0.61) -> pd.Series:
    """按个股滚动计算前景理论的正收益价值分量。"""
    _validate_window(window)
    _validate_mi(ret_1d)
    alpha = float(alpha)
    gamma = float(gamma)

    def _calc(arr: np.ndarray) -> float:
        x = np.asarray(arr, dtype=float)
        if np.any(~np.isfinite(x)):
            return np.nan
        n_total = len(x)
        pos = np.sort(x[x > 0.0])
        n_pos = len(pos)
        if n_pos == 0:
            return 0.0
        out = 0.0
        for j, r in enumerate(pos):
            p_hi = (n_pos - j) / n_total
            p_lo = (n_pos - j - 1) / n_total
            weight = _prospect_w_plus(p_hi, gamma) - _prospect_w_plus(p_lo, gamma)
            out += (float(r) ** alpha) * weight
        return float(out)

    out = (
        ret_1d.groupby(level=1, group_keys=False)
        .rolling(int(window), min_periods=int(window))
        .apply(_calc, raw=True)
        .droplevel(0)
    )
    return out.reindex(ret_1d.index)


def prospect_neg_component(
    ret_1d: pd.Series,
    window: int,
    alpha: float = 0.88,
    lam: float = 2.25,
    delta: float = 0.69,
) -> pd.Series:
    """按个股滚动计算前景理论的负收益价值分量。"""
    _validate_window(window)
    _validate_mi(ret_1d)
    alpha = float(alpha)
    lam = float(lam)
    delta = float(delta)

    def _calc(arr: np.ndarray) -> float:
        x = np.asarray(arr, dtype=float)
        if np.any(~np.isfinite(x)):
            return np.nan
        n_total = len(x)
        neg = np.sort(x[x < 0.0])
        n_neg = len(neg)
        if n_neg == 0:
            return 0.0
        out = 0.0
        for j, r in enumerate(neg):
            p_hi = (j + 1) / n_total
            p_lo = j / n_total
            weight = _prospect_w_minus(p_hi, delta) - _prospect_w_minus(p_lo, delta)
            out += (-lam * ((-float(r)) ** alpha)) * weight
        return float(out)

    out = (
        ret_1d.groupby(level=1, group_keys=False)
        .rolling(int(window), min_periods=int(window))
        .apply(_calc, raw=True)
        .droplevel(0)
    )
    return out.reindex(ret_1d.index)


def sma(x: pd.Series, window: int, m: int = 1) -> pd.Series:
    """按通达信风格参数计算平滑移动平均。"""
    _validate_window(window)
    _validate_mi(x)
    if int(m) <= 0 or int(m) > int(window):
        raise ValueError("sma requires 0 < m <= window")

    alpha = float(m) / float(window)

    def _one_inst(s: pd.Series) -> pd.Series:
        base = s.droplevel(1)
        out = base.ewm(alpha=alpha, adjust=False, min_periods=1).mean()
        out.index = s.index
        return out

    out = x.groupby(level=1, group_keys=False).apply(_one_inst)
    return out.reindex(x.index)


def wma(x: pd.Series, window: int) -> pd.Series:
    """按个股计算线性加权移动平均。"""
    _validate_window(window)
    _validate_mi(x)

    def _weighted_mean(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        weights = np.arange(1, len(arr) + 1, dtype=float)
        weights /= weights.sum()
        return float(np.dot(arr.astype(float), weights))

    out = x.groupby(level=1, group_keys=False).rolling(window).apply(_weighted_mean, raw=True).droplevel(0)
    return out.reindex(x.index)


def macd(x: pd.Series, fast_window: int, slow_window: int) -> pd.Series:
    """快慢 EMA 之差。"""
    _validate_window(fast_window)
    _validate_window(slow_window)
    if int(fast_window) >= int(slow_window):
        raise ValueError("macd requires fast_window < slow_window")
    return ema(x, fast_window) - ema(x, slow_window)


def slope(x: pd.Series, window: int) -> pd.Series:
    """按个股计算滚动线性趋势斜率。"""
    _validate_window(window)
    _validate_mi(x)
    x_axis = np.arange(window, dtype=float)
    x_centered = x_axis - x_axis.mean()
    denom = float(np.dot(x_centered, x_centered)) + EPS

    def _slope(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        y = arr.astype(float)
        y_centered = y - y.mean()
        return float(np.dot(x_centered, y_centered) / denom)

    out = x.groupby(level=1, group_keys=False).rolling(window).apply(_slope, raw=True).droplevel(0)
    return out.reindex(x.index)


def _trend_reg(x: pd.Series, window: int, mode: str) -> pd.Series:
    _validate_window(window)
    _validate_mi(x)
    arr, dates, instruments = _to_wide_2d(x)
    time_axis = np.arange(arr.shape[0], dtype=float)[:, None]
    x_arr = np.broadcast_to(time_axis, arr.shape)
    vals = _rolling_weighted_regression_2d(arr, x_arr, window, np.ones(window, dtype=float), mode=mode)
    return _from_wide_2d(vals, dates, instruments, x.index, list(x.index.names[:2]))


def rsquare(x: pd.Series, window: int) -> pd.Series:
    """按个股计算滚动趋势回归 R 平方。"""
    return _trend_reg(x, window, mode="rsq")


def resi(x: pd.Series, window: int) -> pd.Series:
    """按个股计算滚动趋势回归残差。"""
    return _trend_reg(x, window, mode="residual")


def group_neutral(x: pd.Series, group_map: pd.Series) -> pd.Series:
    """按日对组内均值做中性化。"""
    _validate_mi(x)
    gm = group_map.reindex(x.index)
    df = pd.DataFrame({"x": x, "g": gm})

    def _one_day(d: pd.DataFrame) -> pd.Series:
        if d["g"].isna().all():
            # 当日若没有分组标签，退化为全市场去均值。
            return d["x"] - d["x"].mean()
        grp_mean = d.groupby("g")["x"].transform("mean")
        return d["x"] - grp_mean

    out = df.groupby(level=0, group_keys=False).apply(_one_day)
    out.index = x.index
    return out


def if_then_else(cond: pd.Series, a: pd.Series | float, b: pd.Series | float) -> pd.Series:
    """按条件在两个输入之间逐元素选择。"""
    _validate_mi(cond)
    if isinstance(a, pd.Series):
        sa = _reindex_like(cond, a)
    else:
        sa = pd.Series(float(a), index=cond.index)
    if isinstance(b, pd.Series):
        sb = _reindex_like(cond, b)
    else:
        sb = pd.Series(float(b), index=cond.index)
    return sa.where(cond.astype(bool), sb)


def _rolling_reg(y: pd.Series, x: pd.Series, window: int, half_life: int | None, mode: str) -> pd.Series:
    _validate_window(window)
    _validate_mi(y)
    x = _reindex_like(y, x)
    w = _exp_weights(window, half_life)
    y_arr, dates, instruments = _to_wide_2d(y)
    x_arr, _, _ = _to_wide_2d(x)
    vals = _rolling_weighted_regression_2d(y_arr, x_arr, window, w, mode=mode)
    return _from_wide_2d(vals, dates, instruments, y.index, list(y.index.names[:2]))


def regression_slope(y: pd.Series, x: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """滚动回归斜率，可选半衰期加权。"""
    return _rolling_reg(y, x, window, half_life, mode="slope")


def regression_rsq(y: pd.Series, x: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """滚动回归 R 平方，可选半衰期加权。"""
    return _rolling_reg(y, x, window, half_life, mode="rsq")


def regression_residual(y: pd.Series, x: pd.Series, window: int, half_life: int | None = None) -> pd.Series:
    """滚动回归残差，可选半衰期加权。"""
    return _rolling_reg(y, x, window, half_life, mode="residual")


OPERATOR_REGISTRY_PRO = {
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_var": ts_var,
    "ts_ir": ts_ir,
    "ts_skew": ts_skew,
    "ts_kurt": ts_kurt,
    "ts_max": ts_max,
    "ts_min": ts_min,
    "ts_max_diff": ts_max_diff,
    "ts_min_diff": ts_min_diff,
    "ts_min_max_diff": ts_min_max_diff,
    "ts_sum": ts_sum,
    "ts_count": ts_count,
    "ts_rank": ts_rank,
    "ts_quantile": ts_quantile,
    "ts_med": ts_med,
    "ts_mad": ts_mad,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "ts_cov": ts_cov,
    "ts_corr": ts_corr,
    "rank": rank,
    "zscore": zscore,
    "cs_mean": cs_mean,
    "cs_sum": cs_sum,
    "cs_std": cs_std,
    "cs_skew": cs_skew,
    "cs_zscore": cs_zscore,
    "cs_multi_reg_resid": cs_multi_reg_resid,
    "csad_daily": csad_daily,
    "cs_corr_mean": cs_corr_mean,
    "cssd_daily": cssd_daily,
    "rolling_herding_beta": rolling_herding_beta,
    "rolling_peer_csad_ratio": rolling_peer_csad_ratio,
    "rolling_regression_beta_sum": rolling_regression_beta_sum,
    "ts_sorted_mean_spread": ts_sorted_mean_spread,
    "rolling_cs_spearman_mean": rolling_cs_spearman_mean,
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "log": log,
    "slog1p": slog1p,
    "inv": inv,
    "exp": exp,
    "sqrt": sqrt,
    "power": power,
    "abs": abs_op,
    "sign": sign,
    "not": not_op,
    "greater": greater,
    "less": less,
    "gt": gt,
    "ge": ge,
    "lt": lt,
    "le": le,
    "eq": eq,
    "ne": ne,
    "and": and_op,
    "or": or_op,
    "delay": delay,
    "ref": ref,
    "delta": delta,
    "ts_pct_change": ts_pct_change,
    "ema": ema,
    "ts_linear_decay_mean": ts_linear_decay_mean,
    "ts_exp_weighted_mean_lagged": ts_exp_weighted_mean_lagged,
    "ts_turnover_ref_price": ts_turnover_ref_price,
    "resiliency": resiliency,
    "similar_path_reversal": similar_path_reversal,
    "similar_path_low_volatility": similar_path_low_volatility,
    "turnover_survival_loss": turnover_survival_loss,
    "holding_return": holding_return,
    "prospect_pos_component": prospect_pos_component,
    "prospect_neg_component": prospect_neg_component,
    "sma": sma,
    "wma": wma,
    "macd": macd,
    "slope": slope,
    "rsquare": rsquare,
    "resi": resi,
    "group_mean": group_mean,
    "group_sum": group_sum,
    "group_rank": group_rank,
    "group_bucket": group_bucket,
    "group_combine": group_combine,
    "neighbor_mean": neighbor_mean,
    "cs_reg_resid": cs_reg_resid,
    "group_day_sum": group_day_sum,
    "group_day_bar_sum": group_day_bar_sum,
    "group_day_mean": group_day_mean,
    "group_day_last": group_day_last,
    "group_day_bucket_entropy": group_day_bucket_entropy,
    "group_day_kurt": group_day_kurt,
    "group_day_skew": group_day_skew,
    "group_day_std": group_day_std,
    "group_day_count": group_day_count,
    "group_day_tripower_iv": group_day_tripower_iv,
    "group_day_bipower_iv": group_day_bipower_iv,
    "group_day_vwap": group_day_vwap,
    "rolling_score_top_volume_vwap_ratio": rolling_score_top_volume_vwap_ratio,
    "group_day_top_prod": group_day_top_prod,
    "group_day_top_mean": group_day_top_mean,
    "group_day_cvar": group_day_cvar,
    "group_day_sorted_skew": group_day_sorted_skew,
    "group_day_weighted_skew": group_day_weighted_skew,
    "day_trend_ratio": day_trend_ratio,
    "intraday_maxdrawdown": intraday_maxdrawdown,
    "upper_shadow": upper_shadow,
    "same_bar_rolling_zscore": same_bar_rolling_zscore,
    "same_bar_rolling_std": same_bar_rolling_std,
    "same_bar_rolling_mean": same_bar_rolling_mean,
    "intraday_peak_signal": intraday_peak_signal,
    "intraday_ridge_signal": intraday_ridge_signal,
    "intraday_broadcast": intraday_broadcast,
    "month_broadcast": month_broadcast,
    "intraday_downsample_last": intraday_downsample_last,
    "intraday_downsample_sum": intraday_downsample_sum,
    "group_day_delay": group_day_delay,
    "group_day_log_return": group_day_log_return,
    "intraday_jump_strength": intraday_jump_strength,
    "intraday_jump_count": intraday_jump_count,
    "intraday_jump_return": intraday_jump_return,
    "student_t_cdf": student_t_cdf,
    "ts_sorted_weighted_spread": ts_sorted_weighted_spread,
    "group_day_corr": group_day_corr,
    "group_month_sum": group_month_sum,
    "group_month_mean": group_month_mean,
    "group_month_regression_coef": group_month_regression_coef,
    "group_month_last": group_month_last,
    "group_ex_self_weighted_mean": group_ex_self_weighted_mean,
    "group_neutral": group_neutral,
    "if_then_else": if_then_else,
    "regression_slope": regression_slope,
    "regression_rsq": regression_rsq,
    "regression_residual": regression_residual,
}
