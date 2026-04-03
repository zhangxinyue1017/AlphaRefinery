from __future__ import annotations

"""LLM refined candidates for the alpha101.alpha016 family."""

import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, correlation, covariance, frame_to_series, rank, sma, turnover_frame

PARENT_FACTOR = "alpha101.alpha016"
FAMILY_KEY = "alpha016_family"
SEED_FAMILY = "high_volume_covariance"
SUMMARY_GLOB = "llm_refined_alpha016_family_summary_*.csv"


def llm_refined_neg_vwap_amount_corr_rank_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha016
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "amount"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -rank(correlation(rank(vwap_df), rank(amount_df), 15))
    return frame_to_series(frame, factor_name="llm_refined.neg_vwap_amount_corr_rank_15")


def llm_refined_neg_high_turnover_cov_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha016
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(covariance(rank(high_df), rank(turnover_df), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_high_turnover_cov_rank_10")


def llm_refined_neg_high_rel_volume_corr_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha016
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "volume"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -rank(correlation(rank(high_df), rank(rel_volume), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_high_rel_volume_corr_rank_10")


def llm_refined_neg_high_close_spread_turnover_cov_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha016
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "close", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    close_df = data["close"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(covariance(rank(high_df - close_df), rank(turnover_df), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_high_close_spread_turnover_cov_rank_10")


def llm_refined_neg_high_vwap_turnover_corr_rank_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha016
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    turnover_df = turnover_frame(data)
    intraday_stretch = high_df / vwap_df
    frame = -rank(correlation(rank(intraday_stretch), rank(turnover_df), 15))
    return frame_to_series(frame, factor_name="llm_refined.neg_high_vwap_turnover_corr_rank_15")


def llm_refined_neg_high_turnover_corr_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha016
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(correlation(rank(high_df), rank(turnover_df), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_high_turnover_corr_rank_10")


def llm_refined_neg_rowmax_high_vwap_turnover_corr_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha016
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("high", "vwap", "turnover"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    stronger_price = high_df.where(high_df >= vwap_df, vwap_df)
    frame = -rank(correlation(rank(stronger_price), rank(turnover_df), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_rowmax_high_vwap_turnover_corr_rank_10")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.neg_vwap_amount_corr_rank_15",
        func=llm_refined_neg_vwap_amount_corr_rank_15,
        required_fields=("vwap", "amount"),
        notes="DeepSeek refinement candidate for alpha101.alpha016 family: -rank(corr(rank(vwap), rank(amount), 15)).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_turnover_cov_rank_10",
        func=llm_refined_neg_high_turnover_cov_rank_10,
        required_fields=("high", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha016 family: -rank(cov(rank(high), rank(turnover), 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_rel_volume_corr_rank_10",
        func=llm_refined_neg_high_rel_volume_corr_rank_10,
        required_fields=("high", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha016 family: -rank(corr(rank(high), rank(volume / mean(volume, 20)), 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_close_spread_turnover_cov_rank_10",
        func=llm_refined_neg_high_close_spread_turnover_cov_rank_10,
        required_fields=("high", "close", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha016 family: -rank(cov(rank(high - close), rank(turnover), 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_vwap_turnover_corr_rank_15",
        func=llm_refined_neg_high_vwap_turnover_corr_rank_15,
        required_fields=("high", "vwap", "turnover"),
        notes="DeepSeek refinement candidate for alpha101.alpha016 family: -rank(corr(rank(high / vwap), rank(turnover), 15)).",
    ),
    FactorSpec(
        name="llm_refined.neg_high_turnover_corr_rank_10",
        func=llm_refined_neg_high_turnover_corr_rank_10,
        required_fields=("high", "turnover"),
        notes="GPT refinement candidate for alpha101.alpha016 family: -rank(corr(rank(high), rank(turnover), 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_rowmax_high_vwap_turnover_corr_rank_10",
        func=llm_refined_neg_rowmax_high_vwap_turnover_corr_rank_10,
        required_fields=("high", "vwap", "turnover"),
        notes="GPT refinement candidate for alpha101.alpha016 family: -rank(corr(rank(rowmax(high, vwap)), rank(turnover), 10)).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "llm_refined_neg_high_close_spread_turnover_cov_rank_10",
    "llm_refined_neg_high_rel_volume_corr_rank_10",
    "llm_refined_neg_high_turnover_corr_rank_10",
    "llm_refined_neg_high_turnover_cov_rank_10",
    "llm_refined_neg_high_vwap_turnover_corr_rank_15",
    "llm_refined_neg_rowmax_high_vwap_turnover_corr_rank_10",
    "llm_refined_neg_vwap_amount_corr_rank_15",
]
