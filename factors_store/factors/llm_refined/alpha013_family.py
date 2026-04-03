from __future__ import annotations

"""LLM refined candidates for the alpha101.alpha013 family."""

import pandas as pd

from ...contract import validate_data
from .common import (
    FactorSpec,
    correlation,
    covariance,
    decay_linear,
    frame_to_series,
    rank,
    sma,
    turnover_frame,
)

PARENT_FACTOR = "alpha101.alpha013"
FAMILY_KEY = "alpha013_family"
SEED_FAMILY = "close_volume_covariance"
SUMMARY_GLOB = "llm_refined_alpha013_family_summary_*.csv"


def llm_refined_vwap_amount_corr_rank_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha013
    round: legacy_import
    source model: LLM
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "amount"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -rank(correlation(rank(vwap_df), rank(amount_df), 15))
    return frame_to_series(frame, factor_name="llm_refined.vwap_amount_corr_rank_15")


def llm_refined_neg_decay_close_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha013
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "turnover"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    smooth_close = decay_linear(close_df, 5)
    frame = -rank(correlation(rank(smooth_close), rank(turnover_df), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_decay_close_turnover_corr_10")


def llm_refined_neg_close_turnover_cov_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha013
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "turnover"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(covariance(rank(close_df), rank(turnover_df), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_close_turnover_cov_rank_10")


def llm_refined_neg_vwap_volume_corr_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha013
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "volume"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    frame = -rank(correlation(rank(vwap_df), rank(volume_df), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_vwap_volume_corr_rank_10")


def llm_refined_neg_close_rel_volume_cov_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha013
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "volume"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -rank(covariance(rank(close_df), rank(rel_volume), 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_close_rel_volume_cov_rank_10")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.vwap_amount_corr_rank_15",
        func=llm_refined_vwap_amount_corr_rank_15,
        required_fields=("vwap", "amount"),
        notes="LLM refinement candidate for alpha101.alpha013 family: rank(corr(rank(vwap), rank(amount), 15)).",
    ),
    FactorSpec(
        name="llm_refined.neg_decay_close_turnover_corr_10",
        func=llm_refined_neg_decay_close_turnover_corr_10,
        required_fields=("close", "turnover"),
        notes="GPT refinement candidate for alpha101.alpha013 family: -rank(corr(rank(decay_linear(close, 5)), rank(turnover), 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_close_turnover_cov_rank_10",
        func=llm_refined_neg_close_turnover_cov_rank_10,
        required_fields=("close", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha013 family: -rank(cov(rank(close), rank(turnover), 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_vwap_volume_corr_rank_10",
        func=llm_refined_neg_vwap_volume_corr_rank_10,
        required_fields=("vwap", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha013 family: -rank(corr(rank(vwap), rank(volume), 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_close_rel_volume_cov_rank_10",
        func=llm_refined_neg_close_rel_volume_cov_rank_10,
        required_fields=("close", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha013 family: -rank(cov(rank(close), rank(volume / mean(volume, 20)), 10)).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "llm_refined_neg_close_rel_volume_cov_rank_10",
    "llm_refined_neg_close_turnover_cov_rank_10",
    "llm_refined_neg_decay_close_turnover_corr_10",
    "llm_refined_neg_vwap_volume_corr_rank_10",
    "llm_refined_vwap_amount_corr_rank_15",
]
