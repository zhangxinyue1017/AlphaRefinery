from __future__ import annotations

"""QuantsPlaybook momentum factors.

Current migration focus:
- 高质量动量因子选股

Source notebook references:
- QuantsPlaybook/B-因子构建类/高质量动量因子选股/高质量动量选股.ipynb
- QuantsPlaybook/B-因子构建类/A股市场中如何构造动量因子？/notebook/A股市场中如何构造动量因子.ipynb

This first pass only migrates the directly portable daily-close / turnover
logic from the high-quality momentum notebook. The second notebook contains
more parameter-sweep style research, so we leave that for a later round.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..operators import cs_zscore
from ..registry import FactorRegistry

QP_MOMENTUM_CLOSE_FIELDS: tuple[str, ...] = ("close",)
QP_MOMENTUM_TURNOVER_FIELDS: tuple[str, ...] = ("close", "turnover")
QP_MOMENTUM_AMPLITUDE_FIELDS: tuple[str, ...] = ("close", "high", "low")
QP_MOMENTUM_V2_SOURCE = "quants_playbook_momentum_v2"

LOOKBACK_RET = 60
LOOKBACK_PATH = 20
REGISTER_WINDOWS: tuple[int, ...] = (5, 20, 60, 120)
AMPLITUDE_SLICE_WINDOWS: tuple[int, ...] = (60, 120)
VOL_PENALTY = 3000.0
AMPLITUDE_BUCKET_Q = 0.30
EPS = 1e-12


def _prepare_close(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=QP_MOMENTUM_CLOSE_FIELDS)
    return data["close"].unstack(level="instrument").sort_index()


def _prepare_hlc(data: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_MOMENTUM_AMPLITUDE_FIELDS)
    close_df = data["close"].unstack(level="instrument").sort_index()
    high_df = data["high"].unstack(level="instrument").sort_index()
    low_df = data["low"].unstack(level="instrument").sort_index()
    return high_df, low_df, close_df


def _prepare_turnover_proxy(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=QP_MOMENTUM_TURNOVER_FIELDS)
    turnover = data.get("turnover_rate")
    if turnover is None:
        # BaoStock turnover is percentage-like; the scale does not affect rank-based usage.
        turnover = data["turnover"] / 100.0
    return turnover.unstack(level="instrument").sort_index()


def _basic_momentum_frame(data: dict[str, pd.Series], *, window: int = LOOKBACK_RET) -> pd.DataFrame:
    close_df = _prepare_close(data)
    prev_daily_ret = close_df.pct_change(fill_method=None).shift(1)
    ret_ex_today = close_df.shift(1) / close_df.shift(window + 1) - 1.0
    ret_std = prev_daily_ret.rolling(window).std()
    return ret_ex_today - VOL_PENALTY * ret_std.pow(2)


def _turnover_stability_frame(data: dict[str, pd.Series], *, window: int = LOOKBACK_RET) -> pd.DataFrame:
    turnover_df = _prepare_turnover_proxy(data)
    turnover_std = turnover_df.shift(1).rolling(window).std()
    return -turnover_std


def _anti_spike_frame(data: dict[str, pd.Series], *, window: int = LOOKBACK_PATH) -> pd.DataFrame:
    close_df = _prepare_close(data)
    prev_daily_ret = close_df.pct_change(fill_method=None).shift(1)
    max_ret = prev_daily_ret.rolling(window).max()
    return -max_ret


def _amplitude_sliced_return_frame(
    data: dict[str, pd.Series],
    *,
    window: int,
    side: str,
    bucket_q: float = AMPLITUDE_BUCKET_Q,
) -> pd.DataFrame:
    if side not in {"low", "high"}:
        raise ValueError("side must be 'low' or 'high'")

    high_df, low_df, close_df = _prepare_hlc(data)
    amplitude_df = (high_df / (low_df + EPS) - 1.0).shift(1)
    prev_daily_ret = close_df.pct_change(fill_method=None).shift(1)

    out = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)
    if len(close_df) < window:
        return out

    amp_view = sliding_window_view(amplitude_df.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    ret_view = sliding_window_view(prev_daily_ret.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    valid = ~np.isnan(amp_view) & ~np.isnan(ret_view)
    full_valid = valid.all(axis=2)

    order = np.argsort(amp_view, axis=2, kind="mergesort")
    ranks = np.argsort(order, axis=2, kind="mergesort") + 1
    pct_rank = ranks / float(window)
    if side == "low":
        bucket_mask = pct_rank <= float(bucket_q)
    else:
        bucket_mask = pct_rank >= float(1.0 - bucket_q)

    tail = np.where(bucket_mask & valid, ret_view, 0.0).sum(axis=2)
    tail[~full_valid] = np.nan
    out.iloc[window - 1 :] = tail
    return out


def _amplitude_spread_momentum_frame(
    data: dict[str, pd.Series],
    *,
    window: int,
    bucket_q: float = AMPLITUDE_BUCKET_Q,
) -> pd.DataFrame:
    low_ret = _amplitude_sliced_return_frame(data, window=window, side="low", bucket_q=bucket_q)
    high_ret = _amplitude_sliced_return_frame(data, window=window, side="high", bucket_q=bucket_q)
    return low_ret - high_ret


def _quality_momentum_frame(data: dict[str, pd.Series], *, window: int = LOOKBACK_RET) -> pd.DataFrame:
    basic = wide_frame_to_series(
        _basic_momentum_frame(data, window=window),
        name=f"qp_momentum.basic_{window}",
    )
    stable_turnover = wide_frame_to_series(
        _turnover_stability_frame(data, window=window),
        name=f"qp_momentum.turnover_stability_{window}",
    )
    anti_spike = wide_frame_to_series(
        _anti_spike_frame(data, window=window),
        name=f"qp_momentum.anti_spike_{window}",
    )
    combo = (
        cs_zscore(basic)
        + cs_zscore(stable_turnover)
        + cs_zscore(anti_spike)
    ) / 3.0
    return combo.unstack(level="instrument").sort_index()


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def qp_momentum_basic_60(data: dict[str, pd.Series]) -> pd.Series:
    """60日基础动量: 用前60日累计收益减去 `3000 * 波动率^2`，更偏好稳健上涨而非高波动上涨。"""
    return _series_from_frame(_basic_momentum_frame(data, window=60), factor_name="qp_momentum.basic_60")


def qp_momentum_turnover_stability_60(data: dict[str, pd.Series]) -> pd.Series:
    """换手稳定性: 取过去60日换手率标准差的相反数，值越大表示成交更平稳、不那么拥挤。"""
    return _series_from_frame(
        _turnover_stability_frame(data, window=60),
        factor_name="qp_momentum.turnover_stability_60",
    )


def qp_momentum_anti_spike_20(data: dict[str, pd.Series]) -> pd.Series:
    """反脉冲收益: 取过去20日单日最大涨幅的相反数，惩罚短期过度冲高的股票。"""
    return _series_from_frame(_anti_spike_frame(data, window=20), factor_name="qp_momentum.anti_spike_20")


def qp_momentum_high_quality_60(data: dict[str, pd.Series]) -> pd.Series:
    """高质量动量: 等权融合基础动量、换手稳定性、反脉冲收益三个子信号。"""
    return _series_from_frame(
        _quality_momentum_frame(data, window=60),
        factor_name="qp_momentum.high_quality_60",
    )


def _make_basic_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _basic_momentum_frame(data, window=window),
            factor_name=f"qp_momentum.basic_{window}",
        )

    return _factor


def _make_turnover_stability_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _turnover_stability_frame(data, window=window),
            factor_name=f"qp_momentum.turnover_stability_{window}",
        )

    return _factor


def _make_anti_spike_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _anti_spike_frame(data, window=window),
            factor_name=f"qp_momentum.anti_spike_{window}",
        )

    return _factor


def _make_high_quality_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _quality_momentum_frame(data, window=window),
            factor_name=f"qp_momentum.high_quality_{window}",
        )

    return _factor


def _make_amp_low_ret_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _amplitude_sliced_return_frame(data, window=window, side="low"),
            factor_name=f"qp_momentum.amp_low_ret_{window}_q30",
        )

    return _factor


def _make_amp_high_ret_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _amplitude_sliced_return_frame(data, window=window, side="high"),
            factor_name=f"qp_momentum.amp_high_ret_{window}_q30",
        )

    return _factor


def _make_amp_spread_ret_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _amplitude_spread_momentum_frame(data, window=window),
            factor_name=f"qp_momentum.amp_spread_ret_{window}_q30",
        )

    return _factor


def register_qp_momentum(registry: FactorRegistry) -> int:
    factor_specs: list[tuple[str, object, tuple[str, ...], str]] = []
    for window in REGISTER_WINDOWS:
        factor_specs.extend(
            (
                (
                    f"qp_momentum.basic_{window}",
                    qp_momentum_basic_60 if window == 60 else _make_basic_factor(window),
                    QP_MOMENTUM_CLOSE_FIELDS,
                    f"Portable {window}-day momentum: lagged return minus volatility penalty.",
                ),
                (
                    f"qp_momentum.turnover_stability_{window}",
                    qp_momentum_turnover_stability_60 if window == 60 else _make_turnover_stability_factor(window),
                    QP_MOMENTUM_TURNOVER_FIELDS,
                    f"Negative {window}-day turnover-rate std; higher values mean more stable participation.",
                ),
                (
                    f"qp_momentum.anti_spike_{window}",
                    qp_momentum_anti_spike_20 if window == 20 else _make_anti_spike_factor(window),
                    QP_MOMENTUM_CLOSE_FIELDS,
                    f"Negative of the prior-{window}-day max daily return; suppresses recent one-day spikes.",
                ),
                (
                    f"qp_momentum.high_quality_{window}",
                    qp_momentum_high_quality_60 if window == 60 else _make_high_quality_factor(window),
                    QP_MOMENTUM_TURNOVER_FIELDS,
                    f"Equal-weight composite of base momentum, turnover stability, and anti-spike signals over {window} days.",
                ),
            )
        )

    for name, func, required_fields, notes in factor_specs:
        registry.register(
            name,
            func,
            source="quants_playbook_momentum",
            required_fields=required_fields,
            notes=notes,
        )

    v2_specs: list[tuple[str, object, tuple[str, ...], str]] = []
    for window in AMPLITUDE_SLICE_WINDOWS:
        v2_specs.extend(
            (
                (
                    f"qp_momentum.amp_low_ret_{window}_q30",
                    _make_amp_low_ret_factor(window),
                    QP_MOMENTUM_AMPLITUDE_FIELDS,
                    f"{window}-day sum of lagged returns on the lowest-amplitude 30% of days in the window.",
                ),
                (
                    f"qp_momentum.amp_high_ret_{window}_q30",
                    _make_amp_high_ret_factor(window),
                    QP_MOMENTUM_AMPLITUDE_FIELDS,
                    f"{window}-day sum of lagged returns on the highest-amplitude 30% of days in the window.",
                ),
                (
                    f"qp_momentum.amp_spread_ret_{window}_q30",
                    _make_amp_spread_ret_factor(window),
                    QP_MOMENTUM_AMPLITUDE_FIELDS,
                    f"{window}-day low-amplitude return sum minus high-amplitude return sum.",
                ),
            )
        )

    for name, func, required_fields, notes in v2_specs:
        registry.register(
            name,
            func,
            source=QP_MOMENTUM_V2_SOURCE,
            required_fields=required_fields,
            notes=notes,
        )
    return len(factor_specs) + len(v2_specs)


def qp_momentum_source_info() -> dict[str, object]:
    return {
        "source": "quants_playbook_momentum",
        "status": "partially_migrated",
        "implemented_factors": (
            *(f"qp_momentum.basic_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_momentum.turnover_stability_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_momentum.anti_spike_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_momentum.high_quality_{window}" for window in REGISTER_WINDOWS),
        ),
        "registered_windows": REGISTER_WINDOWS,
        "notes": (
            "Current implementation is based on the high-quality momentum notebook only. "
            "The broader parameterized RetN_momentum research notebook is reserved for a later migration round."
        ),
    }


def qp_momentum_v2_source_info() -> dict[str, object]:
    return {
        "source": QP_MOMENTUM_V2_SOURCE,
        "status": "second_pass_portable",
        "implemented_factors": (
            *(f"qp_momentum.amp_low_ret_{window}_q30" for window in AMPLITUDE_SLICE_WINDOWS),
            *(f"qp_momentum.amp_high_ret_{window}_q30" for window in AMPLITUDE_SLICE_WINDOWS),
            *(f"qp_momentum.amp_spread_ret_{window}_q30" for window in AMPLITUDE_SLICE_WINDOWS),
        ),
        "notes": (
            "Second-pass momentum migration based on amplitude-sliced momentum. "
            "This keeps the portable daily OHLC version and registers the additions under an isolated v2 source."
        ),
    }
