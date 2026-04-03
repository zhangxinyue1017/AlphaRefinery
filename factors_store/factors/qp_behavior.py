from __future__ import annotations

"""QuantsPlaybook behavior factors.

Current migration focus:
- 处置效应因子

Source notebook reference:
- QuantsPlaybook/B-因子构建类/处置效应因子/py/资本利得突出量CGO与风险偏好_重置.ipynb

This first pass migrates the core daily CGO/reference-price construction.
The later cross-factor regressions and portfolio tests in the notebook are
left for a future round.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..registry import FactorRegistry

QP_BEHAVIOR_REQUIRED_FIELDS: tuple[str, ...] = ("close", "amount", "volume", "turnover", "vwap")
REGISTER_WINDOWS: tuple[int, ...] = (20, 60, 100, 120)
EPS = 1e-12


def _prepare_behavior_inputs(
    data: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_BEHAVIOR_REQUIRED_FIELDS)
    close_df = data["close"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    turnover = data.get("turnover_rate")
    if turnover is None:
        turnover = data["turnover"] / 100.0
    turnover_df = turnover.unstack(level="instrument").sort_index()
    return close_df, amount_df, volume_df, vwap_df, turnover_df


def _rolling_turnover_weights(turnover_view: np.ndarray) -> np.ndarray:
    turn = np.array(turnover_view, dtype=float, copy=True)
    if turn.shape[2] == 0:
        return turn
    turn[..., 0] = 0.0
    survival = 1.0 - np.roll(turn, -1, axis=2)
    weights = np.cumprod(survival[..., ::-1], axis=2)[..., ::-1] * turn
    weights[~np.isfinite(turnover_view)] = np.nan
    return weights


def _rolling_reference_price_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    close_df, amount_df, volume_df, vwap_df, turnover_df = _prepare_behavior_inputs(data)
    avg_price_raw = amount_df / volume_df.replace(0.0, np.nan)
    # BaoStock panel prices are on the adjusted close scale, while amount / volume
    # stays on raw trade-price scale. We rescale average trade price by close/vwap
    # so the reference price lives on the same adjusted-price axis as close.
    scale = close_df / vwap_df.replace(0.0, np.nan)
    avg_price_df = avg_price_raw * scale

    out = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)
    if len(close_df) < window:
        return out

    avg_view = sliding_window_view(avg_price_df.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)
    turn_view = sliding_window_view(turnover_df.to_numpy(dtype=float, copy=False), window_shape=window, axis=0)

    weights = _rolling_turnover_weights(turn_view)
    weight_sum = np.nansum(weights, axis=2)
    ref_price_tail = np.full(weight_sum.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(avg_view).all(axis=2)
        & np.isfinite(turn_view).all(axis=2)
        & (weight_sum > EPS)
    )
    numer = np.nansum(weights * avg_view, axis=2)
    np.divide(numer, weight_sum, out=ref_price_tail, where=valid)

    out.iloc[window - 1 :] = ref_price_tail
    return out


def _rolling_cgo_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    close_df, _, _, _, _ = _prepare_behavior_inputs(data)
    ref_price_df = _rolling_reference_price_frame(data, window=window)
    return close_df / ref_price_df.replace(0.0, np.nan) - 1.0


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def qp_behavior_reference_price_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日参考价格: 用换手率衰减权重对日均成交价加权，近似持仓者的成本中心。"""
    return _series_from_frame(
        _rolling_reference_price_frame(data, window=20),
        factor_name="qp_behavior.reference_price_20",
    )


def qp_behavior_cgo_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日CGO: 当前收盘价相对参考价格的偏离，衡量资本利得突出量。"""
    return _series_from_frame(
        _rolling_cgo_frame(data, window=20),
        factor_name="qp_behavior.cgo_20",
    )


def qp_behavior_cgo_100(data: dict[str, pd.Series]) -> pd.Series:
    """100日CGO: 更长记忆的资本利得突出量，更接近原始 notebook 里的经典窗口。"""
    return _series_from_frame(
        _rolling_cgo_frame(data, window=100),
        factor_name="qp_behavior.cgo_100",
    )


def _make_reference_price_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _rolling_reference_price_frame(data, window=window),
            factor_name=f"qp_behavior.reference_price_{window}",
        )

    return _factor


def _make_cgo_factor(window: int):
    def _factor(data: dict[str, pd.Series]) -> pd.Series:
        return _series_from_frame(
            _rolling_cgo_frame(data, window=window),
            factor_name=f"qp_behavior.cgo_{window}",
        )

    return _factor


def register_qp_behavior(registry: FactorRegistry) -> int:
    factor_specs: list[tuple[str, object, str]] = []
    for window in REGISTER_WINDOWS:
        factor_specs.extend(
            (
                (
                    f"qp_behavior.reference_price_{window}",
                    qp_behavior_reference_price_20 if window == 20 else _make_reference_price_factor(window),
                    f"Turnover-decay reference price over {window} days.",
                ),
                (
                    f"qp_behavior.cgo_{window}",
                    qp_behavior_cgo_20 if window == 20 else (qp_behavior_cgo_100 if window == 100 else _make_cgo_factor(window)),
                    f"Capital gain overhang over {window} days: close / reference_price - 1.",
                ),
            )
        )

    for name, func, notes in factor_specs:
        registry.register(
            name,
            func,
            source="quants_playbook_behavior",
            required_fields=QP_BEHAVIOR_REQUIRED_FIELDS,
            notes=notes,
        )
    return len(factor_specs)


def qp_behavior_source_info() -> dict[str, object]:
    return {
        "source": "quants_playbook_behavior",
        "status": "partially_migrated",
        "implemented_factors": (
            *(f"qp_behavior.reference_price_{window}" for window in REGISTER_WINDOWS),
            *(f"qp_behavior.cgo_{window}" for window in REGISTER_WINDOWS),
        ),
        "registered_windows": REGISTER_WINDOWS,
        "notes": (
            "Current implementation focuses on the core CGO/reference-price logic. "
            "The later risk-preference regressions and CGO-conditioned cross-factor tests are not migrated yet."
        ),
    }
