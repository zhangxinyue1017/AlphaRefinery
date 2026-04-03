from __future__ import annotations

"""QuantsPlaybook chip-distribution factors.

Current migration focus:
- 筹码因子

Source notebook references:
- QuantsPlaybook/B-因子构建类/筹码因子/README.md
- QuantsPlaybook/B-因子构建类/筹码因子/scr/cyq.py
- QuantsPlaybook/B-因子构建类/筹码因子/scr/distribution_of_chips.py
- QuantsPlaybook/B-因子构建类/筹码因子/scr/turnover_coefficient_ops.py

This first pass includes:
- turnover-decay chip moments: ARC / VRC / SRC / KRC
- classic triangular-distribution chip factors: CYQK_C / ASR / CKDW / PRP
"""

import numpy as np
import pandas as pd

from ..contract import validate_data
from ..data import wide_frame_to_series
from ..registry import FactorRegistry

QP_CHIP_MOMENT_FIELDS: tuple[str, ...] = ("close", "turnover")
QP_CHIP_CLASSIC_FIELDS: tuple[str, ...] = ("close", "high", "low", "volume", "turnover")
TURN_MOMENT_WINDOWS: tuple[int, ...] = (60,)
CLASSIC_WINDOWS: tuple[int, ...] = (80,)
EPS = 1e-12


def _prepare_chip_moment_inputs(data: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_CHIP_MOMENT_FIELDS)
    close_df = data["close"].unstack(level="instrument").sort_index()
    turnover = data.get("turnover_rate")
    if turnover is None:
        turnover = data["turnover"] / 100.0
    turnover_df = turnover.unstack(level="instrument").sort_index()
    return close_df, turnover_df


def _prepare_chip_classic_inputs(
    data: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_data(data, required_fields=QP_CHIP_CLASSIC_FIELDS)
    close_df = data["close"].unstack(level="instrument").sort_index()
    high_df = data["high"].unstack(level="instrument").sort_index()
    low_df = data["low"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    turnover = data.get("turnover_rate")
    if turnover is None:
        turnover = data["turnover"] / 100.0
    turnover_df = turnover.unstack(level="instrument").sort_index()
    return close_df, high_df, low_df, volume_df, turnover_df


def _calc_adj_turnover(turnover_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(turnover_arr, dtype=float).flatten().copy()
    rolled = np.roll(arr, -1)
    rolled[-1] = 0.0
    return np.flip(np.flip(1 - rolled).cumprod()) * arr


def _calc_norm_turnover(turnover_arr: np.ndarray) -> np.ndarray:
    adj = _calc_adj_turnover(turnover_arr)
    denom = adj.sum()
    if not np.isfinite(denom) or denom <= EPS:
        return np.full_like(adj, np.nan, dtype=float)
    return adj / denom


def _calc_rc(close_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(close_arr, dtype=float).flatten()
    return 1.0 - arr / arr[-1]


def _rolling_turn_moment_frame(data: dict[str, pd.Series], *, window: int, kind: str) -> pd.DataFrame:
    close_df, turnover_df = _prepare_chip_moment_inputs(data)
    out = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)
    if len(close_df) < window:
        return out

    for col in close_df.columns:
        close = close_df[col].to_numpy(dtype=float, copy=False)
        turnover = turnover_df[col].to_numpy(dtype=float, copy=False)
        values = np.full(len(close), np.nan, dtype=float)
        for end in range(window - 1, len(close)):
            c = close[end - window + 1 : end + 1]
            t = turnover[end - window + 1 : end + 1]
            if np.isnan(c).any() or np.isnan(t).any():
                continue
            weight = _calc_norm_turnover(t)
            if np.isnan(weight).all():
                continue
            rc = _calc_rc(c)
            arc = np.sum(weight * rc)
            vrc = window / (window - 1) * np.sum(weight * np.square(rc - arc))
            if kind == "ARC":
                values[end] = arc
            elif kind == "VRC":
                values[end] = vrc
            elif kind == "SRC":
                if vrc > EPS:
                    values[end] = window / (window - 1) * np.sum(weight * np.power(rc - arc, 3)) / np.power(vrc, 1.5)
            elif kind == "KRC":
                if vrc > EPS:
                    values[end] = window / (window - 1) * np.sum(weight * np.power(rc - arc, 4)) / np.power(vrc, 2)
            else:
                raise ValueError("kind must be ARC/VRC/SRC/KRC")
        out[col] = values
    return out


def _triang_pdf_on_grid(close: float, high: float, low: float, vol: float, xs: np.ndarray) -> np.ndarray:
    pdf = np.zeros_like(xs, dtype=float)
    scale = high - low
    if not np.isfinite(scale) or scale <= EPS or vol <= 0:
        return pdf
    c = (close - low) / scale
    c = min(max(c, 0.0), 1.0)
    peak = low + scale * c
    left = (xs >= low) & (xs <= peak)
    right = (xs > peak) & (xs <= high)
    if c > EPS:
        pdf[left] = 2 * (xs[left] - low) / (c * scale * scale)
    if (1 - c) > EPS:
        pdf[right] = 2 * (high - xs[right]) / ((1 - c) * scale * scale)
    total = pdf.sum()
    if total <= EPS:
        return np.zeros_like(xs, dtype=float)
    return pdf / total * vol


def _chip_metrics_from_window(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    turnover: np.ndarray,
    *,
    step_ratio: float = 0.002,
) -> dict[str, float]:
    if np.isnan(close).any() or np.isnan(high).any() or np.isnan(low).any() or np.isnan(vol).any() or np.isnan(turnover).any():
        return {}
    max_p = np.nanmax(high)
    min_p = np.nanmin(low)
    if not np.isfinite(max_p) or not np.isfinite(min_p) or max_p <= min_p:
        return {}
    step = max((max_p - min_p) * step_ratio, EPS)
    xs = np.arange(min_p, max_p + step, step)
    if len(xs) < 2:
        return {}
    cum_vol = np.zeros(len(xs), dtype=float)
    for c, h, l, v, t in zip(close, high, low, vol, turnover):
        decay = float(t)
        if not np.isfinite(decay) or decay < 0:
            return {}
        curpdf = _triang_pdf_on_grid(c, h, l, v, xs)
        cum_vol = cum_vol * (1 - decay) + curpdf * decay
    total = cum_vol.sum()
    if total <= EPS:
        return {}
    share = cum_vol / total
    cumsum = np.cumsum(share)
    last_close = float(close[-1])

    winner = share[xs <= last_close].sum()
    lower = 0.9 * last_close
    upper = 1.1 * last_close
    asr = share[xs <= upper].sum() - share[xs <= lower].sum()

    winner_half_idx = np.searchsorted(cumsum, 0.5, side="right") - 1
    winner_half_idx = min(max(winner_half_idx, 0), len(xs) - 1)
    cost50 = float(xs[winner_half_idx])

    peak_low = float(xs[np.argmin(cum_vol)])
    peak_high = float(xs[np.argmax(cum_vol)])
    ckdw = np.nan
    if abs(peak_high - peak_low) > EPS:
        ckdw = (cost50 - peak_low) / (peak_high - peak_low)
    prp = np.nan
    if abs(cost50) > EPS:
        prp = last_close / cost50 - 1.0

    return {
        "CYQK_C": winner,
        "ASR": asr,
        "CKDW": ckdw,
        "PRP": prp,
    }


def _rolling_chip_classic_frame(data: dict[str, pd.Series], *, window: int, kind: str) -> pd.DataFrame:
    close_df, high_df, low_df, volume_df, turnover_df = _prepare_chip_classic_inputs(data)
    out = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns, dtype=float)
    if len(close_df) < window:
        return out
    for col in close_df.columns:
        close = close_df[col].to_numpy(dtype=float, copy=False)
        high = high_df[col].to_numpy(dtype=float, copy=False)
        low = low_df[col].to_numpy(dtype=float, copy=False)
        vol = volume_df[col].to_numpy(dtype=float, copy=False)
        turn = turnover_df[col].to_numpy(dtype=float, copy=False)
        values = np.full(len(close), np.nan, dtype=float)
        for end in range(window - 1, len(close)):
            metrics = _chip_metrics_from_window(
                close[end - window + 1 : end + 1],
                high[end - window + 1 : end + 1],
                low[end - window + 1 : end + 1],
                vol[end - window + 1 : end + 1],
                turn[end - window + 1 : end + 1],
            )
            if metrics:
                values[end] = metrics.get(kind, np.nan)
        out[col] = values
    return out


def _series_from_frame(frame: pd.DataFrame, *, factor_name: str) -> pd.Series:
    return wide_frame_to_series(frame, name=factor_name)


def qp_chip_arc_60(data: dict[str, pd.Series]) -> pd.Series:
    """ARC: 换手率半衰期加权下的平均相对收益，衡量筹码整体浮盈/浮亏状态。"""
    return _series_from_frame(_rolling_turn_moment_frame(data, window=60, kind="ARC"), factor_name="qp_chip.arc_60")


def qp_chip_vrc_60(data: dict[str, pd.Series]) -> pd.Series:
    """VRC: 换手率半衰期加权下的收益方差，衡量筹码盈亏分散程度。"""
    return _series_from_frame(_rolling_turn_moment_frame(data, window=60, kind="VRC"), factor_name="qp_chip.vrc_60")


def qp_chip_src_60(data: dict[str, pd.Series]) -> pd.Series:
    """SRC: 换手率半衰期加权收益分布偏度，衡量筹码盈利/亏损是否向一侧偏斜。"""
    return _series_from_frame(_rolling_turn_moment_frame(data, window=60, kind="SRC"), factor_name="qp_chip.src_60")


def qp_chip_krc_60(data: dict[str, pd.Series]) -> pd.Series:
    """KRC: 换手率半衰期加权收益分布峰度，衡量筹码盈亏分化是否极端。"""
    return _series_from_frame(_rolling_turn_moment_frame(data, window=60, kind="KRC"), factor_name="qp_chip.krc_60")


def qp_chip_cyqk_c_t_80(data: dict[str, pd.Series]) -> pd.Series:
    """盈利占比: 三角分布筹码模型下，当前价格以下筹码的占比。"""
    return _series_from_frame(_rolling_chip_classic_frame(data, window=80, kind="CYQK_C"), factor_name="qp_chip.cyqk_c_t_80")


def qp_chip_asr_t_80(data: dict[str, pd.Series]) -> pd.Series:
    """活动筹码: 当前价上下 10% 区间内的筹码占比，刻画当前价附近的筹码密集程度。"""
    return _series_from_frame(_rolling_chip_classic_frame(data, window=80, kind="ASR"), factor_name="qp_chip.asr_t_80")


def qp_chip_ckdw_t_80(data: dict[str, pd.Series]) -> pd.Series:
    """成本重心: 平均成本在筹码分布峰谷之间的位置，衡量筹码是否低位或高位密集。"""
    return _series_from_frame(_rolling_chip_classic_frame(data, window=80, kind="CKDW"), factor_name="qp_chip.ckdw_t_80")


def qp_chip_prp_t_80(data: dict[str, pd.Series]) -> pd.Series:
    """价格相对位置: 当前价格相对平均成本的偏离，衡量当前股价处于筹码成本曲线的高低位置。"""
    return _series_from_frame(_rolling_chip_classic_frame(data, window=80, kind="PRP"), factor_name="qp_chip.prp_t_80")


def register_qp_chip(registry: FactorRegistry) -> int:
    factor_specs = (
        ("qp_chip.arc_60", qp_chip_arc_60, QP_CHIP_MOMENT_FIELDS, "Turnover-decay chip mean return ARC over 60 days."),
        ("qp_chip.vrc_60", qp_chip_vrc_60, QP_CHIP_MOMENT_FIELDS, "Turnover-decay chip variance VRC over 60 days."),
        ("qp_chip.src_60", qp_chip_src_60, QP_CHIP_MOMENT_FIELDS, "Turnover-decay chip skew SRC over 60 days."),
        ("qp_chip.krc_60", qp_chip_krc_60, QP_CHIP_MOMENT_FIELDS, "Turnover-decay chip kurtosis KRC over 60 days."),
        ("qp_chip.cyqk_c_t_80", qp_chip_cyqk_c_t_80, QP_CHIP_CLASSIC_FIELDS, "Triangular-distribution profitable-chip ratio over 80 days."),
        ("qp_chip.asr_t_80", qp_chip_asr_t_80, QP_CHIP_CLASSIC_FIELDS, "Triangular-distribution active-chip ratio over 80 days."),
        ("qp_chip.ckdw_t_80", qp_chip_ckdw_t_80, QP_CHIP_CLASSIC_FIELDS, "Triangular-distribution chip-center metric over 80 days."),
        ("qp_chip.prp_t_80", qp_chip_prp_t_80, QP_CHIP_CLASSIC_FIELDS, "Triangular-distribution price relative position over 80 days."),
    )
    for name, func, req, notes in factor_specs:
        registry.register(
            name,
            func,
            source="quants_playbook_chip",
            required_fields=req,
            notes=notes,
        )
    return len(factor_specs)


def qp_chip_source_info() -> dict[str, object]:
    return {
        "source": "quants_playbook_chip",
        "status": "partially_migrated",
        "implemented_factors": (
            "qp_chip.arc_60",
            "qp_chip.vrc_60",
            "qp_chip.src_60",
            "qp_chip.krc_60",
            "qp_chip.cyqk_c_t_80",
            "qp_chip.asr_t_80",
            "qp_chip.ckdw_t_80",
            "qp_chip.prp_t_80",
        ),
        "notes": "First-pass migration of turnover-decay chip moments and classic triangular chip-distribution factors.",
    }
