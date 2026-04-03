from __future__ import annotations

"""LLM refined candidates for the qp_volatility.amplitude_spread_5 family."""

import numpy as np
import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, decay_linear, frame_to_series, rank, sma, ts_rank, turnover_frame

PARENT_FACTOR = "qp_volatility.amplitude_spread_5"
FAMILY_KEY = "amplitude_structure_family"
SEED_FAMILY = "amplitude_structure"
SUMMARY_GLOB = "llm_refined_amplitude_structure_family_summary_*.csv"

EPS = 1e-12


def _field_frame(data: dict[str, pd.Series], field: str) -> pd.DataFrame:
    validate_data(data, required_fields=(field,))
    return data[field].unstack(level="instrument").sort_index()


def _amplitude_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    high_df = _field_frame(data, "high")
    low_df = _field_frame(data, "low")
    return high_df.div(low_df + EPS) - 1.0


def _state_mask(anchor_df: pd.DataFrame, *, window: int, high_state: bool) -> pd.DataFrame:
    anchor_rank = ts_rank(anchor_df, int(window)).div(float(window))
    if high_state:
        return (anchor_rank >= 0.8).astype(float)
    return (anchor_rank <= 0.2).astype(float)


def _conditional_amplitude(
    data: dict[str, pd.Series],
    *,
    anchor_field: str,
    window: int,
    high_state: bool,
) -> pd.DataFrame:
    amplitude = _amplitude_frame(data)
    anchor_df = _field_frame(data, anchor_field)
    mask = _state_mask(anchor_df, window=int(window), high_state=high_state)
    numer = (amplitude * mask).rolling(int(window)).sum()
    denom = mask.rolling(int(window)).sum().replace(0.0, np.nan)
    return numer.div(denom)


def _amplitude_spread(data: dict[str, pd.Series], *, anchor_field: str = "close", window: int = 5) -> pd.DataFrame:
    high_amp = _conditional_amplitude(data, anchor_field=anchor_field, window=int(window), high_state=True)
    low_amp = _conditional_amplitude(data, anchor_field=anchor_field, window=int(window), high_state=False)
    return high_amp - low_amp


def _amplitude_high(data: dict[str, pd.Series], *, anchor_field: str = "close", window: int = 20) -> pd.DataFrame:
    return _conditional_amplitude(data, anchor_field=anchor_field, window=int(window), high_state=True)


def llm_refined_amplitude_spread_smoothed(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: DeepSeek
    keep-drop status: candidate_pool
    note: raw model expression is structurally identical to the negative parent.
    """
    frame = -_amplitude_spread(data, anchor_field="close", window=5)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_spread_smoothed")


def llm_refined_amplitude_spread_vwap_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "vwap"))
    frame = -_amplitude_spread(data, anchor_field="vwap", window=20)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_spread_vwap_20")


def llm_refined_amplitude_spread_amount_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "amount"))
    amount_df = _field_frame(data, "amount")
    frame = -_amplitude_spread(data, anchor_field="close", window=5) * (1.0 + rank(amount_df))
    return frame_to_series(frame, factor_name="llm_refined.amplitude_spread_amount_confirm")


def llm_refined_amplitude_spread_decay(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: Qwen
    keep-drop status: candidate_pool
    """
    frame = -decay_linear(_amplitude_spread(data, anchor_field="close", window=5), 5)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_spread_decay")


def llm_refined_amplitude_spread_vol_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "turnover"))
    turnover_df = turnover_frame(data)
    frame = -_amplitude_spread(data, anchor_field="close", window=5).div(sma(turnover_df, 5).replace(0.0, np.nan))
    return frame_to_series(frame, factor_name="llm_refined.amplitude_spread_vol_norm")


def llm_refined_amplitude_spread_vwap_5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "vwap"))
    frame = -_amplitude_spread(data, anchor_field="vwap", window=5)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_spread_vwap_5")


def llm_refined_decayed_amplitude_spread(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: GPT
    keep-drop status: candidate_pool
    """
    frame = -decay_linear(_amplitude_spread(data, anchor_field="close", window=5), 5)
    return frame_to_series(frame, factor_name="llm_refined.decayed_amplitude_spread")


def llm_refined_amplitude_state_ratio(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close"))
    high_amp = _conditional_amplitude(data, anchor_field="close", window=5, high_state=True)
    low_amp = _conditional_amplitude(data, anchor_field="close", window=5, high_state=False)
    frame = -high_amp.div(low_amp + EPS)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_state_ratio")


def llm_refined_amplitude_spread_volume_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "volume"))
    volume_df = _field_frame(data, "volume")
    rel_volume = volume_df.div(sma(volume_df, 20).replace(0.0, np.nan))
    frame = -_amplitude_spread(data, anchor_field="close", window=5) * rel_volume
    return frame_to_series(frame, factor_name="llm_refined.amplitude_spread_volume_confirm")


def llm_refined_decay_amp_spread_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: Gemini
    keep-drop status: candidate_pool
    """
    frame = decay_linear(-_amplitude_spread(data, anchor_field="close", window=10), 10)
    return frame_to_series(frame, factor_name="llm_refined.decay_amp_spread_10")


def llm_refined_turnover_state_amp_spread(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "turnover"))
    turnover_df = turnover_frame(data)
    amplitude = _amplitude_frame(data)
    high_mask = _state_mask(turnover_df, window=5, high_state=True)
    low_mask = _state_mask(turnover_df, window=5, high_state=False)
    high_amp = (amplitude * high_mask).rolling(5).sum().div(high_mask.rolling(5).sum().replace(0.0, np.nan))
    low_amp = (amplitude * low_mask).rolling(5).sum().div(low_mask.rolling(5).sum().replace(0.0, np.nan))
    frame = -(high_amp - low_amp)
    return frame_to_series(frame, factor_name="llm_refined.turnover_state_amp_spread")


def llm_refined_volume_confirmed_amp_spread(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_spread_5
    round: manual_refine_20260325
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "amount"))
    amount_df = _field_frame(data, "amount")
    rel_amount = amount_df.div(sma(amount_df, 20).replace(0.0, np.nan))
    frame = -_amplitude_spread(data, anchor_field="close", window=5) * rel_amount
    return frame_to_series(frame, factor_name="llm_refined.volume_confirmed_amp_spread")


def llm_refined_amplitude_high_smoothed(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    frame = -pd.DataFrame.ewm(_amplitude_high(data, anchor_field="close", window=20), span=5, min_periods=5, adjust=False).mean()
    return frame_to_series(frame, factor_name="llm_refined.amplitude_high_smoothed")


def llm_refined_amplitude_high_vwap_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: DeepSeek/Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "vwap"))
    frame = -_amplitude_high(data, anchor_field="vwap", window=20)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_high_vwap_20")


def llm_refined_amplitude_high_amount_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "amount"))
    amount_df = _field_frame(data, "amount")
    frame = -_amplitude_high(data, anchor_field="close", window=20) * (1.0 + rank(amount_df))
    return frame_to_series(frame, factor_name="llm_refined.amplitude_high_amount_confirm")


def llm_refined_amplitude_high_decay(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: Qwen
    keep-drop status: candidate_pool
    """
    frame = -decay_linear(_amplitude_high(data, anchor_field="close", window=20), 5)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_high_decay")


def llm_refined_amplitude_high_liq_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "turnover"))
    turnover_df = turnover_frame(data)
    frame = -_amplitude_high(data, anchor_field="close", window=20).div(sma(turnover_df, 20).replace(0.0, np.nan))
    return frame_to_series(frame, factor_name="llm_refined.amplitude_high_liq_norm")


def llm_refined_decay_amp_high_20_stable(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: Gemini
    keep-drop status: candidate_pool
    """
    frame = decay_linear(-_amplitude_high(data, anchor_field="close", window=20), 10)
    return frame_to_series(frame, factor_name="llm_refined.decay_amp_high_20_stable")


def llm_refined_turnover_state_amp_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "turnover"))
    frame = -_amplitude_high(data, anchor_field="turnover", window=20)
    return frame_to_series(frame, factor_name="llm_refined.turnover_state_amp_20")


def llm_refined_crowded_amp_high_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "turnover"))
    turnover_df = turnover_frame(data)
    frame = -_amplitude_high(data, anchor_field="close", window=20) * ts_rank(turnover_df, 20)
    return frame_to_series(frame, factor_name="llm_refined.crowded_amp_high_20")


def llm_refined_decayed_amplitude_high(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: GPT
    keep-drop status: candidate_pool
    """
    frame = -decay_linear(_amplitude_high(data, anchor_field="close", window=20), 10)
    return frame_to_series(frame, factor_name="llm_refined.decayed_amplitude_high")


def llm_refined_amplitude_high_low_ratio(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close"))
    high_amp = _amplitude_high(data, anchor_field="close", window=20)
    low_amp = _conditional_amplitude(data, anchor_field="close", window=20, high_state=False)
    frame = -high_amp.div(low_amp + EPS)
    return frame_to_series(frame, factor_name="llm_refined.amplitude_high_low_ratio")


def llm_refined_amplitude_high_volume_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_volatility.amplitude_high_20
    round: manual_refine_20260325
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "close", "volume"))
    volume_df = _field_frame(data, "volume")
    rel_volume = volume_df.div(sma(volume_df, 20).replace(0.0, np.nan))
    frame = -_amplitude_high(data, anchor_field="close", window=20) * rel_volume
    return frame_to_series(frame, factor_name="llm_refined.amplitude_high_volume_confirm")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.amplitude_spread_smoothed",
        func=llm_refined_amplitude_spread_smoothed,
        required_fields=("high", "low", "close"),
        notes="DeepSeek refinement candidate for amplitude_structure family; raw returned expression is the negative parent spread itself.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_spread_vwap_20",
        func=llm_refined_amplitude_spread_vwap_20,
        required_fields=("high", "low", "vwap"),
        notes="DeepSeek refinement candidate for amplitude_structure family: close-state spread replaced with 20-day vwap-state spread.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_spread_amount_confirm",
        func=llm_refined_amplitude_spread_amount_confirm,
        required_fields=("high", "low", "close", "amount"),
        notes="DeepSeek refinement candidate for amplitude_structure family: negative amplitude spread confirmed by cross-sectional amount rank.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_spread_decay",
        func=llm_refined_amplitude_spread_decay,
        required_fields=("high", "low", "close"),
        notes="Qwen refinement candidate for amplitude_structure family: decay_linear on the negative 5-day close-state amplitude spread.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_spread_vol_norm",
        func=llm_refined_amplitude_spread_vol_norm,
        required_fields=("high", "low", "close", "turnover"),
        notes="Qwen refinement candidate for amplitude_structure family: negative amplitude spread normalized by 5-day average turnover.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_spread_vwap_5",
        func=llm_refined_amplitude_spread_vwap_5,
        required_fields=("high", "low", "vwap"),
        notes="Qwen refinement candidate for amplitude_structure family: 5-day vwap-state amplitude spread.",
    ),
    FactorSpec(
        name="llm_refined.decayed_amplitude_spread",
        func=llm_refined_decayed_amplitude_spread,
        required_fields=("high", "low", "close"),
        notes="GPT refinement candidate for amplitude_structure family: conservative decayed negative amplitude spread.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_state_ratio",
        func=llm_refined_amplitude_state_ratio,
        required_fields=("high", "low", "close"),
        notes="GPT refinement candidate for amplitude_structure family: ratio of high-state to low-state amplitude under close ranking.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_spread_volume_confirm",
        func=llm_refined_amplitude_spread_volume_confirm,
        required_fields=("high", "low", "close", "volume"),
        notes="GPT refinement candidate for amplitude_structure family: negative amplitude spread confirmed by 20-day relative volume.",
    ),
    FactorSpec(
        name="llm_refined.decay_amp_spread_10",
        func=llm_refined_decay_amp_spread_10,
        required_fields=("high", "low", "close"),
        notes="Gemini refinement candidate for amplitude_structure family: 10-day negative amplitude spread with decay smoothing.",
    ),
    FactorSpec(
        name="llm_refined.turnover_state_amp_spread",
        func=llm_refined_turnover_state_amp_spread,
        required_fields=("high", "low", "turnover"),
        notes="Gemini refinement candidate for amplitude_structure family: turnover-state amplitude spread.",
    ),
    FactorSpec(
        name="llm_refined.volume_confirmed_amp_spread",
        func=llm_refined_volume_confirmed_amp_spread,
        required_fields=("high", "low", "close", "amount"),
        notes="Gemini refinement candidate for amplitude_structure family: negative amplitude spread confirmed by 20-day relative amount.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_high_smoothed",
        func=llm_refined_amplitude_high_smoothed,
        required_fields=("high", "low", "close"),
        notes="DeepSeek refinement candidate for amplitude_structure family: 5-day EMA smoothing on negative amplitude_high_20.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_high_vwap_20",
        func=llm_refined_amplitude_high_vwap_20,
        required_fields=("high", "low", "vwap"),
        notes="DeepSeek/Qwen refinement candidate for amplitude_structure family: vwap-state negative amplitude_high_20.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_high_amount_confirm",
        func=llm_refined_amplitude_high_amount_confirm,
        required_fields=("high", "low", "close", "amount"),
        notes="DeepSeek refinement candidate for amplitude_structure family: negative amplitude_high_20 confirmed by cross-sectional amount rank.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_high_decay",
        func=llm_refined_amplitude_high_decay,
        required_fields=("high", "low", "close"),
        notes="Qwen refinement candidate for amplitude_structure family: decay_linear on negative amplitude_high_20.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_high_liq_norm",
        func=llm_refined_amplitude_high_liq_norm,
        required_fields=("high", "low", "close", "turnover"),
        notes="Qwen refinement candidate for amplitude_structure family: negative amplitude_high_20 normalized by 20-day average turnover.",
    ),
    FactorSpec(
        name="llm_refined.decay_amp_high_20_stable",
        func=llm_refined_decay_amp_high_20_stable,
        required_fields=("high", "low", "close"),
        notes="Gemini refinement candidate for amplitude_structure family: stable 10-day decay on negative amplitude_high_20.",
    ),
    FactorSpec(
        name="llm_refined.turnover_state_amp_20",
        func=llm_refined_turnover_state_amp_20,
        required_fields=("high", "low", "turnover"),
        notes="Gemini refinement candidate for amplitude_structure family: turnover-state negative amplitude_high_20.",
    ),
    FactorSpec(
        name="llm_refined.crowded_amp_high_20",
        func=llm_refined_crowded_amp_high_20,
        required_fields=("high", "low", "close", "turnover"),
        notes="Gemini refinement candidate for amplitude_structure family: negative amplitude_high_20 confirmed by turnover ts-rank.",
    ),
    FactorSpec(
        name="llm_refined.decayed_amplitude_high",
        func=llm_refined_decayed_amplitude_high,
        required_fields=("high", "low", "close"),
        notes="GPT refinement candidate for amplitude_structure family: conservative decayed negative amplitude_high_20.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_high_low_ratio",
        func=llm_refined_amplitude_high_low_ratio,
        required_fields=("high", "low", "close"),
        notes="GPT refinement candidate for amplitude_structure family: ratio of high-state to low-state amplitude under close ranking, 20-day window.",
    ),
    FactorSpec(
        name="llm_refined.amplitude_high_volume_confirm",
        func=llm_refined_amplitude_high_volume_confirm,
        required_fields=("high", "low", "close", "volume"),
        notes="GPT refinement candidate for amplitude_structure family: negative amplitude_high_20 confirmed by 20-day relative volume.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "llm_refined_amplitude_spread_amount_confirm",
    "llm_refined_amplitude_spread_decay",
    "llm_refined_amplitude_spread_smoothed",
    "llm_refined_amplitude_spread_vol_norm",
    "llm_refined_amplitude_spread_volume_confirm",
    "llm_refined_amplitude_spread_vwap_20",
    "llm_refined_amplitude_spread_vwap_5",
    "llm_refined_amplitude_state_ratio",
    "llm_refined_amplitude_high_amount_confirm",
    "llm_refined_amplitude_high_decay",
    "llm_refined_amplitude_high_liq_norm",
    "llm_refined_amplitude_high_low_ratio",
    "llm_refined_amplitude_high_smoothed",
    "llm_refined_amplitude_high_volume_confirm",
    "llm_refined_amplitude_high_vwap_20",
    "llm_refined_crowded_amp_high_20",
    "llm_refined_decay_amp_high_20_stable",
    "llm_refined_decay_amp_spread_10",
    "llm_refined_decayed_amplitude_high",
    "llm_refined_decayed_amplitude_spread",
    "llm_refined_turnover_state_amp_20",
    "llm_refined_turnover_state_amp_spread",
    "llm_refined_volume_confirmed_amp_spread",
]
