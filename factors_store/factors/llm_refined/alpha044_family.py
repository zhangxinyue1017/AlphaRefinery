from __future__ import annotations

"""LLM refined candidates for the alpha101.alpha044 family."""

import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, correlation, frame_to_series, rank, scale, sma, turnover_frame

PARENT_FACTOR = "alpha101.alpha044"
FAMILY_KEY = "alpha044_family"
SEED_FAMILY = "high_ranked_volume_correlation"
SUMMARY_GLOB = "llm_refined_alpha044_family_summary_*.csv"


def llm_refined_neg_high_rank_amount_corr_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "amount"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -correlation(high_df, rank(amount_df), 15)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_rank_amount_corr_15")


def llm_refined_neg_high_rank_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -correlation(high_df, rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_rank_turnover_corr_10")


def llm_refined_neg_vwap_rank_rel_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "volume"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -correlation(vwap_df, rank(rel_volume), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_vwap_rank_rel_volume_corr_10")


def llm_refined_neg_high_vwap_gap_rank_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -correlation(rank(high_df - vwap_df), rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_vwap_gap_rank_turnover_corr_10")


def llm_refined_neg_high_vwap_rank_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    turnover_df = turnover_frame(data)
    intraday_stretch = high_df / vwap_df
    frame = -correlation(intraday_stretch, rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_vwap_rank_turnover_corr_10")


def llm_refined_neg_high_vwap_gap_rel_volume_corr_14(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "volume"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    high_vwap_gap = rank(high_df) - rank(vwap_df)
    frame = -correlation(high_vwap_gap, rank(rel_volume), 14)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_vwap_gap_rel_volume_corr_14")


def llm_refined_neg_rowmax_high_vwap_rank_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    stronger_price = high_df.where(high_df >= vwap_df, vwap_df)
    frame = -correlation(stronger_price, rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_rowmax_high_vwap_rank_turnover_corr_10")


def llm_refined_neg_high_rank_amount_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "amount"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -correlation(high_df, rank(amount_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_rank_amount_corr_10")


def llm_refined_neg_high_scaled_rel_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha044
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "volume"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    scaled_rel_volume = scale(rel_volume)
    frame = -correlation(high_df, rank(scaled_rel_volume), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_scaled_rel_volume_corr_10")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.neg_high_rank_amount_corr_15",
        func=llm_refined_neg_high_rank_amount_corr_15,
        required_fields=("high", "amount"),
        notes="DeepSeek refinement candidate for alpha101.alpha044 family: -corr(high, rank(amount), 15).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_rank_turnover_corr_10",
        func=llm_refined_neg_high_rank_turnover_corr_10,
        required_fields=("high", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha044 family: -corr(high, rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_vwap_rank_rel_volume_corr_10",
        func=llm_refined_neg_vwap_rank_rel_volume_corr_10,
        required_fields=("vwap", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha044 family: -corr(vwap, rank(volume / mean(volume, 20)), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_vwap_gap_rank_turnover_corr_10",
        func=llm_refined_neg_high_vwap_gap_rank_turnover_corr_10,
        required_fields=("high", "vwap", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha044 family: -corr(rank(high - vwap), rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_vwap_rank_turnover_corr_10",
        func=llm_refined_neg_high_vwap_rank_turnover_corr_10,
        required_fields=("high", "vwap", "turnover"),
        notes="DeepSeek refinement candidate for alpha101.alpha044 family: -corr(high / vwap, rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_vwap_gap_rel_volume_corr_14",
        func=llm_refined_neg_high_vwap_gap_rel_volume_corr_14,
        required_fields=("high", "vwap", "volume"),
        notes="DeepSeek refinement candidate for alpha101.alpha044 family: -corr(rank(high) - rank(vwap), rank(volume / mean(volume, 20)), 14).",
    ),
    FactorSpec(
        name="llm_refined.neg_rowmax_high_vwap_rank_turnover_corr_10",
        func=llm_refined_neg_rowmax_high_vwap_rank_turnover_corr_10,
        required_fields=("high", "vwap", "turnover"),
        notes="GPT refinement candidate for alpha101.alpha044 family: -corr(rowmax(high, vwap), rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_rank_amount_corr_10",
        func=llm_refined_neg_high_rank_amount_corr_10,
        required_fields=("high", "amount"),
        notes="GPT refinement candidate for alpha101.alpha044 family: -corr(high, rank(amount), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_scaled_rel_volume_corr_10",
        func=llm_refined_neg_high_scaled_rel_volume_corr_10,
        required_fields=("high", "volume"),
        notes="GPT refinement candidate for alpha101.alpha044 family: -corr(high, rank(scale(volume / mean(volume, 20))), 10).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "llm_refined_neg_high_rank_amount_corr_10",
    "llm_refined_neg_high_rank_amount_corr_15",
    "llm_refined_neg_high_rank_turnover_corr_10",
    "llm_refined_neg_high_scaled_rel_volume_corr_10",
    "llm_refined_neg_high_vwap_gap_rank_turnover_corr_10",
    "llm_refined_neg_high_vwap_gap_rel_volume_corr_14",
    "llm_refined_neg_high_vwap_rank_turnover_corr_10",
    "llm_refined_neg_rowmax_high_vwap_rank_turnover_corr_10",
    "llm_refined_neg_vwap_rank_rel_volume_corr_10",
]
