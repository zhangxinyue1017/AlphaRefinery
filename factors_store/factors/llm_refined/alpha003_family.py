from __future__ import annotations

"""LLM refined candidates for the alpha101.alpha003 family."""

import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, correlation, frame_to_series, prepare_core_inputs, rank, sma, turnover_frame

PARENT_FACTOR = "alpha101.alpha003"
FAMILY_KEY = "alpha003_family"
SEED_FAMILY = "open_volume_correlation"
SUMMARY_GLOB = "llm_refined_alpha003_family_summary_*.csv"


def llm_refined_open_amount_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    open_df, amount_df, _ = prepare_core_inputs(data)
    frame = -correlation(rank(open_df), rank(amount_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.open_amount_corr_10")


def llm_refined_open_rel_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    open_df, _, volume_df = prepare_core_inputs(data)
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -correlation(rank(open_df), rank(rel_volume), 10)
    return frame_to_series(frame, factor_name="llm_refined.open_rel_volume_corr_10")


def llm_refined_open_rel_volume_corr_14(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    open_df, _, volume_df = prepare_core_inputs(data)
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -correlation(rank(open_df), rank(rel_volume), 14)
    return frame_to_series(frame, factor_name="llm_refined.open_rel_volume_corr_14")


def llm_refined_open_vwap_gap_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("open", "vwap", "turnover"))
    open_df = data["open"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    turnover_df = turnover_frame(data)
    open_vwap_gap = open_df / vwap_df
    frame = -correlation(rank(open_vwap_gap), rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.open_vwap_gap_turnover_corr_10")


def llm_refined_open_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("open", "turnover"))
    open_df = data["open"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -correlation(rank(open_df), rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.open_turnover_corr_10")


def llm_refined_open_vwap_gap_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("open", "vwap", "volume"))
    open_df = data["open"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    open_vwap_gap = open_df / vwap_df
    frame = -correlation(rank(open_vwap_gap), rank(volume_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.open_vwap_gap_volume_corr_10")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.open_amount_corr_10",
        func=llm_refined_open_amount_corr_10,
        required_fields=("open", "amount", "volume"),
        notes="DeepSeek refinement candidate for alpha101.alpha003 family: corr(rank(open), rank(amount), 10).",
    ),
    FactorSpec(
        name="llm_refined.open_rel_volume_corr_10",
        func=llm_refined_open_rel_volume_corr_10,
        required_fields=("open", "amount", "volume"),
        notes="DeepSeek refinement candidate for alpha101.alpha003 family: corr(rank(open), rank(volume / ts_mean(volume, 20)), 10).",
    ),
    FactorSpec(
        name="llm_refined.open_rel_volume_corr_14",
        func=llm_refined_open_rel_volume_corr_14,
        required_fields=("open", "amount", "volume"),
        notes="DeepSeek refinement candidate for alpha101.alpha003 family: corr(rank(open), rank(volume / ts_mean(volume, 20)), 14).",
    ),
    FactorSpec(
        name="llm_refined.open_vwap_gap_turnover_corr_10",
        func=llm_refined_open_vwap_gap_turnover_corr_10,
        required_fields=("open", "vwap", "turnover"),
        notes="DeepSeek refinement candidate for alpha101.alpha003 family: corr(rank(open / vwap), rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.open_turnover_corr_10",
        func=llm_refined_open_turnover_corr_10,
        required_fields=("open", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha003 family: corr(rank(open), rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.open_vwap_gap_volume_corr_10",
        func=llm_refined_open_vwap_gap_volume_corr_10,
        required_fields=("open", "vwap", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha003 family: corr(rank(open / vwap), rank(volume), 10).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "llm_refined_open_amount_corr_10",
    "llm_refined_open_rel_volume_corr_10",
    "llm_refined_open_rel_volume_corr_14",
    "llm_refined_open_turnover_corr_10",
    "llm_refined_open_vwap_gap_turnover_corr_10",
    "llm_refined_open_vwap_gap_volume_corr_10",
]
