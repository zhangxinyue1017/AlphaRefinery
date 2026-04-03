from __future__ import annotations

"""LLM refined candidates for the alpha101.alpha040 family."""

import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, correlation, decay_linear, frame_to_series, rank, sma, stddev, turnover_frame

PARENT_FACTOR = "alpha101.alpha040"
FAMILY_KEY = "alpha040_family"
SEED_FAMILY = "high_volatility_times_volume_correlation"
SUMMARY_GLOB = "llm_refined_alpha040_family_summary_*.csv"


def llm_refined_neg_amplitude_std_high_rank_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "low", "volume"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    low_df = data["low"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    amplitude = high_df - low_df
    frame = -rank(stddev(amplitude, 10)) * correlation(high_df, rank(volume_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_amplitude_std_high_rank_volume_corr_10")


def llm_refined_neg_high_std14_turnover_corr_14(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(stddev(high_df, 14)) * correlation(high_df, turnover_df, 14)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_std14_turnover_corr_14")


def llm_refined_neg_high_std10_vwap_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "volume"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    frame = -rank(stddev(high_df, 10)) * correlation(vwap_df, volume_df, 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_std10_vwap_volume_corr_10")


def llm_refined_neg_high_std10_high_rank_rel_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "volume"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -rank(stddev(high_df, 10)) * correlation(high_df, rank(rel_volume), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_std10_high_rank_rel_volume_corr_10")


def llm_refined_neg_vwap_std_amount_corr_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "amount"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -rank(stddev(vwap_df, 15)) * correlation(vwap_df, amount_df, 15)
    return frame_to_series(frame, factor_name="llm_refined.neg_vwap_std_amount_corr_15")


def llm_refined_neg_decay_high_vwap_turnover_corr_rank_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    turnover_df = turnover_frame(data)
    intraday_stretch = high_df / vwap_df
    frame = -rank(decay_linear(intraday_stretch, 10)) * rank(correlation(high_df, turnover_df, 15))
    return frame_to_series(frame, factor_name="llm_refined.neg_decay_high_vwap_turnover_corr_rank_15")


def llm_refined_neg_rowmax_high_vwap_std_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    stronger_price = high_df.where(high_df >= vwap_df, vwap_df)
    frame = -rank(stddev(stronger_price, 10)) * correlation(rank(stronger_price), rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_rowmax_high_vwap_std_turnover_corr_10")


def llm_refined_neg_high_std_rank_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(stddev(high_df, 10)) * correlation(rank(high_df), rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_high_std_rank_turnover_corr_10")


def llm_refined_neg_vwap_std_rank_high_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha040
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "volume"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    frame = -rank(stddev(vwap_df, 10)) * correlation(rank(high_df), rank(volume_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_vwap_std_rank_high_volume_corr_10")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.neg_amplitude_std_high_rank_volume_corr_10",
        func=llm_refined_neg_amplitude_std_high_rank_volume_corr_10,
        required_fields=("high", "low", "volume"),
        notes="DeepSeek refinement candidate for alpha101.alpha040 family: -rank(std(high - low, 10)) * corr(high, rank(volume), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_std14_turnover_corr_14",
        func=llm_refined_neg_high_std14_turnover_corr_14,
        required_fields=("high", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha040 family: -rank(std(high, 14)) * corr(high, turnover, 14).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_std10_vwap_volume_corr_10",
        func=llm_refined_neg_high_std10_vwap_volume_corr_10,
        required_fields=("high", "vwap", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha040 family: -rank(std(high, 10)) * corr(vwap, volume, 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_std10_high_rank_rel_volume_corr_10",
        func=llm_refined_neg_high_std10_high_rank_rel_volume_corr_10,
        required_fields=("high", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha040 family: -rank(std(high, 10)) * corr(high, rank(volume / mean(volume, 20)), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_vwap_std_amount_corr_15",
        func=llm_refined_neg_vwap_std_amount_corr_15,
        required_fields=("vwap", "amount"),
        notes="DeepSeek refinement candidate for alpha101.alpha040 family: -rank(std(vwap, 15)) * corr(vwap, amount, 15).",
    ),
    FactorSpec(
        name="llm_refined.neg_decay_high_vwap_turnover_corr_rank_15",
        func=llm_refined_neg_decay_high_vwap_turnover_corr_rank_15,
        required_fields=("high", "vwap", "turnover"),
        notes="DeepSeek refinement candidate for alpha101.alpha040 family: -rank(decay_linear(high / vwap, 10)) * rank(corr(high, turnover, 15)).",
    ),
    FactorSpec(
        name="llm_refined.neg_rowmax_high_vwap_std_turnover_corr_10",
        func=llm_refined_neg_rowmax_high_vwap_std_turnover_corr_10,
        required_fields=("high", "vwap", "turnover"),
        notes="GPT refinement candidate for alpha101.alpha040 family: -rank(std(rowmax(high, vwap), 10)) * corr(rank(rowmax(high, vwap)), rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_std_rank_turnover_corr_10",
        func=llm_refined_neg_high_std_rank_turnover_corr_10,
        required_fields=("high", "turnover"),
        notes="GPT refinement candidate for alpha101.alpha040 family: -rank(std(high, 10)) * corr(rank(high), rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_vwap_std_rank_high_volume_corr_10",
        func=llm_refined_neg_vwap_std_rank_high_volume_corr_10,
        required_fields=("high", "vwap", "volume"),
        notes="GPT refinement candidate for alpha101.alpha040 family: -rank(std(vwap, 10)) * corr(rank(high), rank(volume), 10).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "llm_refined_neg_amplitude_std_high_rank_volume_corr_10",
    "llm_refined_neg_decay_high_vwap_turnover_corr_rank_15",
    "llm_refined_neg_high_std10_high_rank_rel_volume_corr_10",
    "llm_refined_neg_high_std10_vwap_volume_corr_10",
    "llm_refined_neg_high_std14_turnover_corr_14",
    "llm_refined_neg_high_std_rank_turnover_corr_10",
    "llm_refined_neg_rowmax_high_vwap_std_turnover_corr_10",
    "llm_refined_neg_vwap_std_amount_corr_15",
    "llm_refined_neg_vwap_std_rank_high_volume_corr_10",
]
