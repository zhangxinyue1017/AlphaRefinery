from __future__ import annotations

"""QuantsPlaybook price-volume pressure factors.

Current migration focus:
- 基于量价关系度量股票的买卖压力

Source references:
- QuantsPlaybook/B-因子构建类/基于量价关系度量股票的买卖压力/py/CalFunc.py

This first pass keeps the directly portable daily price-volume core:
- rolling volume-weighted price vs simple mean price
- APB-style average-price-bias proxy
- low-price / high-price volume concentration within a trailing window

The original notebook contains additional trading-calendar and portfolio
construction details. Those are intentionally left out here so the factors
stay portable on the plain daily panel contract.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..operators import wide_sma as Mean
from ..registry import FactorRegistry

QP_PRESSURE_PRICE_FIELDS: tuple[str, ...] = ("close", "volume")
QP_PRESSURE_APB_FIELDS: tuple[str, ...] = ("close", "volume", "vwap")
QP_PRESSURE_SOURCE = "quants_playbook_pressure_v1"
PRESSURE_WINDOW = 20
LOW_BUCKET_Q = 0.30
HIGH_BUCKET_Q = 0.70
EPS = 1e-12


def _prepare_price_volume(data: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_PRESSURE_PRICE_FIELDS)
    close_df = data["close"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    return close_df, volume_df


def _prepare_apb_inputs(data: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_PRESSURE_APB_FIELDS)
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    return vwap_df, volume_df


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)


def _rolling_weighted_mean(value_df: pd.DataFrame, weight_df: pd.DataFrame, *, window: int) -> pd.DataFrame:
    value_np = value_df.to_numpy(dtype=float, copy=False)
    weight_np = weight_df.reindex_like(value_df).to_numpy(dtype=float, copy=False)
    out = _empty_like(value_df)
    if window <= 0:
        raise ValueError("window must be positive")
    if len(value_df) < window:
        return out

    value_view = sliding_window_view(value_np, window_shape=window, axis=0)
    weight_view = sliding_window_view(weight_np, window_shape=window, axis=0)
    valid = ~np.isnan(value_view) & ~np.isnan(weight_view)
    full_valid = valid.all(axis=2)
    weight_sum = np.where(valid, weight_view, 0.0).sum(axis=2)
    numer = np.where(valid, value_view * weight_view, 0.0).sum(axis=2)

    tail = np.full(weight_sum.shape, np.nan, dtype=float)
    np.divide(numer, weight_sum, out=tail, where=full_valid & (weight_sum > EPS))
    out.iloc[window - 1 :] = tail
    return out


def _rolling_bucket_price_volume_stats(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    *,
    window: int,
    low_q: float = LOW_BUCKET_Q,
    high_q: float = HIGH_BUCKET_Q,
) -> dict[str, pd.DataFrame]:
    close_np = close_df.to_numpy(dtype=float, copy=False)
    volume_np = volume_df.reindex_like(close_df).to_numpy(dtype=float, copy=False)
    template = _empty_like(close_df)
    outputs = {
        "low_share": template.copy(),
        "high_share": template.copy(),
        "overall_vwap": template.copy(),
        "low_vwap": template.copy(),
        "high_vwap": template.copy(),
    }
    if window <= 0:
        raise ValueError("window must be positive")
    if len(close_df) < window:
        return outputs

    close_view = sliding_window_view(close_np, window_shape=window, axis=0)
    volume_view = sliding_window_view(volume_np, window_shape=window, axis=0)
    valid = ~np.isnan(close_view) & ~np.isnan(volume_view)
    full_valid = valid.all(axis=2)

    order = np.argsort(close_view, axis=2, kind="mergesort")
    ranks = np.argsort(order, axis=2, kind="mergesort") + 1
    pct_rank = ranks / float(window)
    low_mask = pct_rank <= float(low_q)
    high_mask = pct_rank >= float(high_q)

    total_volume = np.where(valid, volume_view, 0.0).sum(axis=2)
    total_price_volume = np.where(valid, close_view * volume_view, 0.0).sum(axis=2)

    low_volume = np.where(low_mask & valid, volume_view, 0.0).sum(axis=2)
    high_volume = np.where(high_mask & valid, volume_view, 0.0).sum(axis=2)
    low_price_volume = np.where(low_mask & valid, close_view * volume_view, 0.0).sum(axis=2)
    high_price_volume = np.where(high_mask & valid, close_view * volume_view, 0.0).sum(axis=2)

    low_share = np.full(total_volume.shape, np.nan, dtype=float)
    high_share = np.full(total_volume.shape, np.nan, dtype=float)
    overall_vwap = np.full(total_volume.shape, np.nan, dtype=float)
    low_vwap = np.full(total_volume.shape, np.nan, dtype=float)
    high_vwap = np.full(total_volume.shape, np.nan, dtype=float)

    np.divide(low_volume, total_volume, out=low_share, where=full_valid & (total_volume > EPS))
    np.divide(high_volume, total_volume, out=high_share, where=full_valid & (total_volume > EPS))
    np.divide(total_price_volume, total_volume, out=overall_vwap, where=full_valid & (total_volume > EPS))
    np.divide(low_price_volume, low_volume, out=low_vwap, where=full_valid & (low_volume > EPS))
    np.divide(high_price_volume, high_volume, out=high_vwap, where=full_valid & (high_volume > EPS))

    outputs["low_share"].iloc[window - 1 :] = low_share
    outputs["high_share"].iloc[window - 1 :] = high_share
    outputs["overall_vwap"].iloc[window - 1 :] = overall_vwap
    outputs["low_vwap"].iloc[window - 1 :] = low_vwap
    outputs["high_vwap"].iloc[window - 1 :] = high_vwap
    return outputs


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def _vwap_minus_mean_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    close_df, volume_df = _prepare_price_volume(data)
    rolling_vwap = _rolling_weighted_mean(close_df, volume_df, window=window)
    rolling_mean = Mean(close_df, window)
    return rolling_vwap - rolling_mean


def _apb_frame(data: dict[str, pd.Series], *, window: int) -> pd.DataFrame:
    vwap_df, volume_df = _prepare_apb_inputs(data)
    rolling_mean_vwap = Mean(vwap_df, window)
    rolling_weighted_vwap = _rolling_weighted_mean(vwap_df, volume_df, window=window)
    return rolling_mean_vwap / rolling_weighted_vwap.replace(0.0, np.nan) - 1.0


def _pressure_component_frames(data: dict[str, pd.Series], *, window: int) -> dict[str, pd.DataFrame]:
    close_df, volume_df = _prepare_price_volume(data)
    stats = _rolling_bucket_price_volume_stats(close_df, volume_df, window=window)
    overall_vwap = stats["overall_vwap"].replace(0.0, np.nan)
    buy_pressure = stats["low_share"] * (overall_vwap - stats["low_vwap"]) / overall_vwap.abs().clip(lower=EPS)
    sell_pressure = stats["high_share"] * (stats["high_vwap"] - overall_vwap) / overall_vwap.abs().clip(lower=EPS)
    return {
        "low_price_volume_share": stats["low_share"],
        "high_price_volume_share": stats["high_share"],
        "low_price_volume_bias": stats["low_share"] - stats["high_share"],
        "buy_pressure": buy_pressure,
        "sell_pressure": sell_pressure,
        "net_pressure": buy_pressure - sell_pressure,
    }


def qp_pressure_vwap_minus_mean_5(data: dict[str, pd.Series]) -> pd.Series:
    """5日量价均衡差: 5日成交量加权价格减简单均价，衡量放量交易更偏向高位还是低位。"""
    return _series_from_frame(
        _vwap_minus_mean_frame(data, window=5),
        factor_name="qp_pressure.vwap_minus_mean_5",
    )


def qp_pressure_vwap_minus_mean_30(data: dict[str, pd.Series]) -> pd.Series:
    """30日量价均衡差: 更长窗口下的量价重心相对均价偏离。"""
    return _series_from_frame(
        _vwap_minus_mean_frame(data, window=30),
        factor_name="qp_pressure.vwap_minus_mean_30",
    )


def qp_pressure_apb_5(data: dict[str, pd.Series]) -> pd.Series:
    """5日 APB 代理: 日度 VWAP 的简单均值相对成交量加权均值的偏离。"""
    return _series_from_frame(_apb_frame(data, window=5), factor_name="qp_pressure.apb_5")


def qp_pressure_apb_30(data: dict[str, pd.Series]) -> pd.Series:
    """30日 APB 代理: 较长窗口下的平均价格偏差。"""
    return _series_from_frame(_apb_frame(data, window=30), factor_name="qp_pressure.apb_30")


def qp_pressure_low_price_volume_share_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日低价量占比: 近20日价格分位较低交易日承载的成交量占比。"""
    frames = _pressure_component_frames(data, window=PRESSURE_WINDOW)
    return _series_from_frame(
        frames["low_price_volume_share"],
        factor_name="qp_pressure.low_price_volume_share_20",
    )


def qp_pressure_high_price_volume_share_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日高价量占比: 近20日价格分位较高交易日承载的成交量占比。"""
    frames = _pressure_component_frames(data, window=PRESSURE_WINDOW)
    return _series_from_frame(
        frames["high_price_volume_share"],
        factor_name="qp_pressure.high_price_volume_share_20",
    )


def qp_pressure_low_price_volume_bias_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日低高价量差: 低价量占比减高价量占比，正值偏向低位承接。"""
    frames = _pressure_component_frames(data, window=PRESSURE_WINDOW)
    return _series_from_frame(
        frames["low_price_volume_bias"],
        factor_name="qp_pressure.low_price_volume_bias_20",
    )


def qp_pressure_buy_pressure_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日买压代理: 低价日成交量占比乘以其相对整体 VWAP 的折价幅度。"""
    frames = _pressure_component_frames(data, window=PRESSURE_WINDOW)
    return _series_from_frame(
        frames["buy_pressure"],
        factor_name="qp_pressure.buy_pressure_20",
    )


def qp_pressure_sell_pressure_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日卖压代理: 高价日成交量占比乘以其相对整体 VWAP 的溢价幅度。"""
    frames = _pressure_component_frames(data, window=PRESSURE_WINDOW)
    return _series_from_frame(
        frames["sell_pressure"],
        factor_name="qp_pressure.sell_pressure_20",
    )


def qp_pressure_net_pressure_20(data: dict[str, pd.Series]) -> pd.Series:
    """20日净压力: 买压代理减卖压代理，正值表示更像低位吸筹、负值表示更像高位派发。"""
    frames = _pressure_component_frames(data, window=PRESSURE_WINDOW)
    return _series_from_frame(
        frames["net_pressure"],
        factor_name="qp_pressure.net_pressure_20",
    )


def register_qp_pressure(registry: FactorRegistry) -> int:
    factor_specs: tuple[tuple[str, object, tuple[str, ...], str], ...] = (
        (
            "qp_pressure.vwap_minus_mean_5",
            qp_pressure_vwap_minus_mean_5,
            QP_PRESSURE_PRICE_FIELDS,
            "5-day rolling volume-weighted close minus simple mean close.",
        ),
        (
            "qp_pressure.vwap_minus_mean_30",
            qp_pressure_vwap_minus_mean_30,
            QP_PRESSURE_PRICE_FIELDS,
            "30-day rolling volume-weighted close minus simple mean close.",
        ),
        (
            "qp_pressure.apb_5",
            qp_pressure_apb_5,
            QP_PRESSURE_APB_FIELDS,
            "5-day APB-style proxy: mean(vwap) / volume-weighted mean(vwap) - 1.",
        ),
        (
            "qp_pressure.apb_30",
            qp_pressure_apb_30,
            QP_PRESSURE_APB_FIELDS,
            "30-day APB-style proxy: mean(vwap) / volume-weighted mean(vwap) - 1.",
        ),
        (
            "qp_pressure.low_price_volume_share_20",
            qp_pressure_low_price_volume_share_20,
            QP_PRESSURE_PRICE_FIELDS,
            "20-day share of volume traded on lower-price days within the trailing window.",
        ),
        (
            "qp_pressure.high_price_volume_share_20",
            qp_pressure_high_price_volume_share_20,
            QP_PRESSURE_PRICE_FIELDS,
            "20-day share of volume traded on higher-price days within the trailing window.",
        ),
        (
            "qp_pressure.low_price_volume_bias_20",
            qp_pressure_low_price_volume_bias_20,
            QP_PRESSURE_PRICE_FIELDS,
            "20-day low-price volume share minus high-price volume share.",
        ),
        (
            "qp_pressure.buy_pressure_20",
            qp_pressure_buy_pressure_20,
            QP_PRESSURE_PRICE_FIELDS,
            "20-day buy-pressure proxy from lower-price volume concentration and discount to overall VWAP.",
        ),
        (
            "qp_pressure.sell_pressure_20",
            qp_pressure_sell_pressure_20,
            QP_PRESSURE_PRICE_FIELDS,
            "20-day sell-pressure proxy from higher-price volume concentration and premium to overall VWAP.",
        ),
        (
            "qp_pressure.net_pressure_20",
            qp_pressure_net_pressure_20,
            QP_PRESSURE_PRICE_FIELDS,
            "20-day buy-pressure proxy minus sell-pressure proxy.",
        ),
    )

    for name, func, required_fields, notes in factor_specs:
        registry.register(
            name,
            func,
            source=QP_PRESSURE_SOURCE,
            required_fields=required_fields,
            notes=notes,
        )
    return len(factor_specs)


def qp_pressure_source_info() -> dict[str, object]:
    return {
        "source": QP_PRESSURE_SOURCE,
        "status": "first_pass_portable",
        "implemented_factors": (
            "qp_pressure.vwap_minus_mean_5",
            "qp_pressure.vwap_minus_mean_30",
            "qp_pressure.apb_5",
            "qp_pressure.apb_30",
            "qp_pressure.low_price_volume_share_20",
            "qp_pressure.high_price_volume_share_20",
            "qp_pressure.low_price_volume_bias_20",
            "qp_pressure.buy_pressure_20",
            "qp_pressure.sell_pressure_20",
            "qp_pressure.net_pressure_20",
        ),
        "notes": (
            "Portable first pass of the QuantsPlaybook price-volume pressure theme. "
            "The original notebook's paused/trading-day handling and downstream portfolio construction are not migrated yet."
        ),
    }
