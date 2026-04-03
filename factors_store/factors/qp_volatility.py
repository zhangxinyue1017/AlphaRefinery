from __future__ import annotations

"""QuantsPlaybook volatility factors.

Current migration focus:
- 振幅因子的隐藏结构

Source notebook references:
- QuantsPlaybook/B-因子构建类/剔除跨期截面相关性的纯真波动率因子/py/波动率选股因子_特质波动率.ipynb
- QuantsPlaybook/B-因子构建类/振幅因子的隐藏结构/notebook/振幅因子的隐藏结构.ipynb

This first pass prioritizes the amplitude notebook because it is directly
portable with daily OHLC data. The more elaborate idiosyncratic-volatility
construction from the other notebook is left for a later migration round.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..registry import FactorRegistry

QP_VOL_REQUIRED_FIELDS: tuple[str, ...] = ("high", "low", "close")
QP_VOL_TURNOVER_FIELDS: tuple[str, ...] = ("high", "low", "close", "turnover")
QP_VOL_V2_SOURCE = "quants_playbook_volatility_v2"
VOL_WINDOW = 20
REGISTER_WINDOWS: tuple[int, ...] = (5, 20, 60, 120)
LAMBDA_HIGH = 0.8
LAMBDA_LOW = 0.2
EPS = 1e-12


def _prepare_hlc(data: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_VOL_REQUIRED_FIELDS)
    high_df = data["high"].unstack(level="instrument").sort_index()
    low_df = data["low"].unstack(level="instrument").sort_index()
    close_df = data["close"].unstack(level="instrument").sort_index()
    return high_df, low_df, close_df


def _prepare_turnover_proxy(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=QP_VOL_TURNOVER_FIELDS)
    turnover = data.get("turnover_rate")
    if turnover is None:
        turnover = data["turnover"] / 100.0
    return turnover.unstack(level="instrument").sort_index()


def _rolling_valid_mean(masked_values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    numer = np.where(valid_mask, masked_values, 0.0).sum(axis=2)
    denom = valid_mask.sum(axis=2)
    out = np.full(numer.shape, np.nan, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out


def _rolling_conditional_amplitude(
    amplitude_df: pd.DataFrame,
    close_df: pd.DataFrame,
    *,
    window: int,
    lamb: float,
    side: str,
) -> pd.DataFrame:
    if side not in {"high", "low"}:
        raise ValueError("side must be 'high' or 'low'")
    if window <= 0:
        raise ValueError("window must be positive")

    amplitude_np = amplitude_df.to_numpy(dtype=float, copy=False)
    close_np = close_df.to_numpy(dtype=float, copy=False)
    out = pd.DataFrame(np.nan, index=amplitude_df.index, columns=amplitude_df.columns, dtype=float)
    if len(amplitude_df) < window:
        return out

    amp_view = sliding_window_view(amplitude_np, window_shape=window, axis=0)
    close_view = sliding_window_view(close_np, window_shape=window, axis=0)

    order = np.argsort(close_view, axis=2)
    ranks = np.argsort(order, axis=2) + 1
    pct_rank = ranks / float(window)

    if side == "high":
        bucket_mask = pct_rank >= lamb
    else:
        bucket_mask = pct_rank <= lamb

    valid = bucket_mask & ~np.isnan(amp_view) & ~np.isnan(close_view)
    tail = _rolling_valid_mean(amp_view, valid)
    out.iloc[window - 1 :] = tail
    return out


def _daily_return_vol_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    _, _, close_df = _prepare_hlc(data)
    return close_df.pct_change().rolling(window).std()


def _amplitude_mean_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    high_df, low_df, _ = _prepare_hlc(data)
    amplitude = high_df / (low_df + EPS) - 1.0
    return amplitude.rolling(window).mean()


def _amplitude_high_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    high_df, low_df, close_df = _prepare_hlc(data)
    amplitude = high_df / (low_df + EPS) - 1.0
    return _rolling_conditional_amplitude(
        amplitude,
        close_df,
        window=window,
        lamb=LAMBDA_HIGH,
        side="high",
    )


def _amplitude_low_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    high_df, low_df, close_df = _prepare_hlc(data)
    amplitude = high_df / (low_df + EPS) - 1.0
    return _rolling_conditional_amplitude(
        amplitude,
        close_df,
        window=window,
        lamb=LAMBDA_LOW,
        side="low",
    )


def _amplitude_spread_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    return _amplitude_high_frame(data, window=window) - _amplitude_low_frame(data, window=window)


def _turnover_mean_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    turnover_df = _prepare_turnover_proxy(data)
    return turnover_df.shift(1).rolling(window).mean()


def _cross_sectional_residual_frame(
    target_df: pd.DataFrame,
    control_df: pd.DataFrame,
    *,
    min_obs: int = 20,
) -> pd.DataFrame:
    target_np = target_df.to_numpy(dtype=float, copy=False)
    control_np = control_df.reindex_like(target_df).to_numpy(dtype=float, copy=False)
    out = np.full(target_np.shape, np.nan, dtype=float)

    for row_idx in range(target_np.shape[0]):
        y = target_np[row_idx]
        x = control_np[row_idx]
        valid = np.isfinite(y) & np.isfinite(x)
        if int(valid.sum()) < int(min_obs):
            continue

        xv = x[valid]
        yv = y[valid]
        x_mean = float(xv.mean())
        y_mean = float(yv.mean())
        x_centered = xv - x_mean
        var_x = float(np.dot(x_centered, x_centered))
        if var_x <= EPS:
            out[row_idx, valid] = yv - y_mean
            continue
        beta = float(np.dot(x_centered, yv - y_mean) / var_x)
        alpha = y_mean - beta * x_mean
        out[row_idx, valid] = yv - (alpha + beta * xv)

    return pd.DataFrame(out, index=target_df.index, columns=target_df.columns)


def _realized_vol_de_turn_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    realized_vol = _daily_return_vol_frame(data, window=window)
    turnover_mean = _turnover_mean_frame(data, window=window)
    return _cross_sectional_residual_frame(realized_vol, turnover_mean)


def _amplitude_mean_de_turn_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    amplitude_mean = _amplitude_mean_frame(data, window=window)
    turnover_mean = _turnover_mean_frame(data, window=window)
    return _cross_sectional_residual_frame(amplitude_mean, turnover_mean)


def _amplitude_spread_de_turn_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW) -> pd.DataFrame:
    amplitude_spread = _amplitude_spread_frame(data, window=window)
    turnover_mean = _turnover_mean_frame(data, window=window)
    return _cross_sectional_residual_frame(amplitude_spread, turnover_mean)


def _pure_vol_lag6_resid_frame(data: dict[str, pd.Series], *, window: int = VOL_WINDOW, lag_window: int = 6) -> pd.DataFrame:
    de_turn = _realized_vol_de_turn_frame(data, window=window)
    lagged_mean = de_turn.shift(1).rolling(lag_window).mean()
    return _cross_sectional_residual_frame(de_turn, lagged_mean)


def _pure_amp_spread_lag6_resid_frame(
    data: dict[str, pd.Series],
    *,
    window: int = VOL_WINDOW,
    lag_window: int = 6,
) -> pd.DataFrame:
    de_turn = _amplitude_spread_de_turn_frame(data, window=window)
    lagged_mean = de_turn.shift(1).rolling(lag_window).mean()
    return _cross_sectional_residual_frame(de_turn, lagged_mean)


def _series_from_frame(
    frame: pd.DataFrame,
    *,
    factor_name: str,
) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def qp_volatility_realized_vol_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日实现波动率: 用日收益率标准差刻画短期价格波动强弱。"""
    return _series_from_frame(_daily_return_vol_frame(data, window=20), factor_name="qp_volatility.realized_vol_20")


def qp_volatility_amplitude_mean_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日平均振幅: 近20日 `high / low - 1` 的均值，衡量日内波动区间的常态水平。"""
    return _series_from_frame(_amplitude_mean_frame(data, window=20), factor_name="qp_volatility.amplitude_mean_20")


def qp_volatility_amplitude_high_20(data: dict[str, pd.Series]) -> pd.Series:
    """高价振幅: 仅在近20日收盘位于时间分位前20%的日子上，平均这些交易日的振幅。"""
    return _series_from_frame(_amplitude_high_frame(data, window=20), factor_name="qp_volatility.amplitude_high_20")


def qp_volatility_amplitude_low_20(data: dict[str, pd.Series]) -> pd.Series:
    """低价振幅: 仅在近20日收盘位于时间分位后20%的日子上，平均这些交易日的振幅。"""
    return _series_from_frame(_amplitude_low_frame(data, window=20), factor_name="qp_volatility.amplitude_low_20")


def qp_volatility_amplitude_spread_20(data: dict[str, pd.Series]) -> pd.Series:
    """理想振幅: 高价振幅减去低价振幅，捕捉强势区间与弱势区间的振幅结构差。"""
    return _series_from_frame(
        _amplitude_spread_frame(data, window=20),
        factor_name="qp_volatility.amplitude_spread_20",
    )


def qp_volatility_turnover_mean_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日换手均值: 作为波动率残差化的控制量。"""
    return _series_from_frame(_turnover_mean_frame(data, window=20), factor_name="qp_volatility.turnover_mean_20")


def qp_volatility_realized_vol_de_turn20(data: dict[str, pd.Series]) -> pd.Series:
    """去换手实现波动率: 对 20 日实现波动率做日度截面换手残差。"""
    return _series_from_frame(
        _realized_vol_de_turn_frame(data, window=20),
        factor_name="qp_volatility.realized_vol_de_turn20",
    )


def qp_volatility_amplitude_mean_de_turn20(data: dict[str, pd.Series]) -> pd.Series:
    """去换手平均振幅: 对 20 日平均振幅做日度截面换手残差。"""
    return _series_from_frame(
        _amplitude_mean_de_turn_frame(data, window=20),
        factor_name="qp_volatility.amplitude_mean_de_turn20",
    )


def qp_volatility_amplitude_spread_de_turn20(data: dict[str, pd.Series]) -> pd.Series:
    """去换手振幅结构: 对 20 日高低价振幅差做日度截面换手残差。"""
    return _series_from_frame(
        _amplitude_spread_de_turn_frame(data, window=20),
        factor_name="qp_volatility.amplitude_spread_de_turn20",
    )


def qp_volatility_pure_vol_lag6_resid(data: dict[str, pd.Series]) -> pd.Series:
    """纯真波动率近似: 先去换手，再去除自身过去 6 期截面相关成分。"""
    return _series_from_frame(
        _pure_vol_lag6_resid_frame(data, window=20, lag_window=6),
        factor_name="qp_volatility.pure_vol_lag6_resid",
    )


def qp_volatility_pure_amp_spread_lag6_resid(data: dict[str, pd.Series]) -> pd.Series:
    """纯真振幅结构近似: 先去换手，再去除自身过去 6 期截面相关成分。"""
    return _series_from_frame(
        _pure_amp_spread_lag6_resid_frame(data, window=20, lag_window=6),
        factor_name="qp_volatility.pure_amp_spread_lag6_resid",
    )


def _make_realized_vol_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _daily_return_vol_frame(data, window=window),
            factor_name=f"qp_volatility.realized_vol_{window}",
        )

    return _factor


def _make_amplitude_mean_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _amplitude_mean_frame(data, window=window),
            factor_name=f"qp_volatility.amplitude_mean_{window}",
        )

    return _factor


def _make_amplitude_high_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _amplitude_high_frame(data, window=window),
            factor_name=f"qp_volatility.amplitude_high_{window}",
        )

    return _factor


def _make_amplitude_low_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _amplitude_low_frame(data, window=window),
            factor_name=f"qp_volatility.amplitude_low_{window}",
        )

    return _factor


def _make_amplitude_spread_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _amplitude_spread_frame(data, window=window),
            factor_name=f"qp_volatility.amplitude_spread_{window}",
        )

    return _factor


def register_qp_volatility(registry: FactorRegistry) -> int:
    factor_specs: list[tuple[str, object, str]] = []
    for window in REGISTER_WINDOWS:
        factor_specs.extend(
            (
                (
                    f"qp_volatility.realized_vol_{window}",
                    qp_volatility_realized_vol_20 if window == 20 else _make_realized_vol_factor(window),
                    f"Portable {window}-day realized volatility from daily returns.",
                ),
                (
                    f"qp_volatility.amplitude_mean_{window}",
                    qp_volatility_amplitude_mean_20 if window == 20 else _make_amplitude_mean_factor(window),
                    f"{window}-day average amplitude high/low - 1.",
                ),
                (
                    f"qp_volatility.amplitude_high_{window}",
                    qp_volatility_amplitude_high_20 if window == 20 else _make_amplitude_high_factor(window),
                    f"Average amplitude on high-close days within the trailing {window}-day window.",
                ),
                (
                    f"qp_volatility.amplitude_low_{window}",
                    qp_volatility_amplitude_low_20 if window == 20 else _make_amplitude_low_factor(window),
                    f"Average amplitude on low-close days within the trailing {window}-day window.",
                ),
                (
                    f"qp_volatility.amplitude_spread_{window}",
                    qp_volatility_amplitude_spread_20 if window == 20 else _make_amplitude_spread_factor(window),
                    f"Difference between high-close amplitude and low-close amplitude over {window} days.",
                ),
            )
        )

    for name, func, notes in factor_specs:
        registry.register(
            name,
            func,
            source="quants_playbook_volatility",
            required_fields=QP_VOL_REQUIRED_FIELDS,
            notes=notes,
        )

    v2_specs: tuple[tuple[str, object, tuple[str, ...], str], ...] = (
        (
            "qp_volatility.turnover_mean_20",
            qp_volatility_turnover_mean_20,
            QP_VOL_TURNOVER_FIELDS,
            "20-day mean turnover proxy used as the main residualization control.",
        ),
        (
            "qp_volatility.realized_vol_de_turn20",
            qp_volatility_realized_vol_de_turn20,
            QP_VOL_TURNOVER_FIELDS,
            "20-day realized volatility with same-day cross-sectional turnover residualization.",
        ),
        (
            "qp_volatility.amplitude_mean_de_turn20",
            qp_volatility_amplitude_mean_de_turn20,
            QP_VOL_TURNOVER_FIELDS,
            "20-day amplitude mean with same-day cross-sectional turnover residualization.",
        ),
        (
            "qp_volatility.amplitude_spread_de_turn20",
            qp_volatility_amplitude_spread_de_turn20,
            QP_VOL_TURNOVER_FIELDS,
            "20-day amplitude spread with same-day cross-sectional turnover residualization.",
        ),
        (
            "qp_volatility.pure_vol_lag6_resid",
            qp_volatility_pure_vol_lag6_resid,
            QP_VOL_TURNOVER_FIELDS,
            "Portable pure-vol proxy: realized_vol_de_turn20 residualized against its 6-day lagged mean.",
        ),
        (
            "qp_volatility.pure_amp_spread_lag6_resid",
            qp_volatility_pure_amp_spread_lag6_resid,
            QP_VOL_TURNOVER_FIELDS,
            "Portable pure-amplitude proxy: amplitude_spread_de_turn20 residualized against its 6-day lagged mean.",
        ),
    )

    for name, func, required_fields, notes in v2_specs:
        registry.register(
            name,
            func,
            source=QP_VOL_V2_SOURCE,
            required_fields=required_fields,
            notes=notes,
        )
    return len(factor_specs) + len(v2_specs)


def qp_volatility_source_info() -> dict[str, object]:
    return {
        "source": "quants_playbook_volatility",
        "status": "partially_migrated",
        "implemented_factors": (
            *(f"qp_volatility.realized_vol_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_volatility.amplitude_mean_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_volatility.amplitude_high_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_volatility.amplitude_low_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_volatility.amplitude_spread_{window}" for window in REGISTER_WINDOWS),
        ),
        "registered_windows": REGISTER_WINDOWS,
        "notes": (
            "Current implementation focuses on the amplitude-structure notebook. "
            "The cross-sectional idiosyncratic-volatility regressions are reserved for a later pass."
        ),
    }


def qp_volatility_v2_source_info() -> dict[str, object]:
    return {
        "source": QP_VOL_V2_SOURCE,
        "status": "second_pass_portable",
        "implemented_factors": (
            "qp_volatility.turnover_mean_20",
            "qp_volatility.realized_vol_de_turn20",
            "qp_volatility.amplitude_mean_de_turn20",
            "qp_volatility.amplitude_spread_de_turn20",
            "qp_volatility.pure_vol_lag6_resid",
            "qp_volatility.pure_amp_spread_lag6_resid",
        ),
        "notes": (
            "Second-pass volatility migration that approximates pure-volatility ideas via "
            "cross-sectional turnover residualization plus lag residualization, without external FF-style data."
        ),
    }
