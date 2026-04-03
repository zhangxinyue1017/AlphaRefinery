from __future__ import annotations

"""LLM refined candidates for the alpha101.alpha069 family."""

import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, correlation, delta, frame_to_series, rank, scale, sma, stddev, ts_max, ts_rank, turnover_frame

PARENT_FACTOR = "alpha101.alpha069"
FAMILY_KEY = "alpha069_family"
SEED_FAMILY = "industry_neutral_nonlinear_flow"
SUMMARY_GLOB = "llm_refined_alpha069_family_summary_*.csv"


def llm_refined_neg_vwap_ts_rank_rel_volume_corr_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "vwap", "volume"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -rank(ts_rank(vwap_df, 5)) * correlation(close_df, rel_volume, 15)
    return frame_to_series(frame, factor_name="llm_refined.neg_vwap_ts_rank_rel_volume_corr_15")


def llm_refined_neg_tsmax_delta_vwap_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "turnover"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(ts_max(delta(vwap_df, 3), 5)) * ts_rank(correlation(vwap_df, turnover_df, 5), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_tsmax_delta_vwap_turnover_corr_10")


def llm_refined_neg_delta_vwap_rank_rel_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "volume"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    frame = -rank(delta(vwap_df, 3)) * correlation(vwap_df, rank(rel_volume), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_delta_vwap_rank_rel_volume_corr_10")


def llm_refined_neg_tsmax_delta_vwap_amount_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "amount"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -rank(ts_max(delta(vwap_df, 3), 10)) * correlation(vwap_df, amount_df, 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_tsmax_delta_vwap_amount_corr_10")


def llm_refined_neg_close_vwap_amount_ratio_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "vwap", "amount"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    amount_avg20 = sma(amount_df, 20).replace(0.0, pd.NA)
    amount_ratio = amount_df / amount_avg20
    frame = -correlation(rank(close_df - vwap_df), rank(amount_ratio), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_close_vwap_amount_ratio_corr_10")


def llm_refined_neg_vwap_volume_corr_ts_rank_close_std_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "vwap", "volume"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    frame = -ts_rank(correlation(vwap_df, volume_df, 15), 20) * rank(stddev(close_df, 10))
    return frame_to_series(frame, factor_name="llm_refined.neg_vwap_volume_corr_ts_rank_close_std_20")


def llm_refined_neg_delta_vwap_ts_rank_vwap_turnover_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("vwap", "turnover"))
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -rank(ts_rank(delta(vwap_df, 3), 5)) * correlation(rank(vwap_df), rank(turnover_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_delta_vwap_ts_rank_vwap_turnover_corr_10")


def llm_refined_neg_delta_close_ts_rank_close_amount_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "amount"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -rank(ts_rank(delta(close_df, 3), 5)) * correlation(rank(close_df), rank(amount_df), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_delta_close_ts_rank_close_amount_corr_10")


def llm_refined_neg_delta_vwap_ts_rank_close_scaled_rel_volume_corr_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha069
    round: legacy_import
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "vwap", "volume"))
    close_df = data["close"].unstack(level="instrument").sort_index()
    vwap_df = data["vwap"].unstack(level="instrument").sort_index()
    volume_df = data["volume"].unstack(level="instrument").sort_index()
    adv20 = sma(volume_df, 20).replace(0.0, pd.NA)
    rel_volume = volume_df / adv20
    scaled_rel_volume = scale(rel_volume)
    frame = -rank(ts_rank(delta(vwap_df, 3), 5)) * correlation(rank(close_df), rank(scaled_rel_volume), 10)
    return frame_to_series(frame, factor_name="llm_refined.neg_delta_vwap_ts_rank_close_scaled_rel_volume_corr_10")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.neg_vwap_ts_rank_rel_volume_corr_15",
        func=llm_refined_neg_vwap_ts_rank_rel_volume_corr_15,
        required_fields=("close", "vwap", "volume"),
        notes="DeepSeek refinement candidate for alpha101.alpha069 family: -rank(ts_rank(vwap, 5)) * corr(close, volume / mean(volume, 20), 15).",
    ),
    FactorSpec(
        name="llm_refined.neg_tsmax_delta_vwap_turnover_corr_10",
        func=llm_refined_neg_tsmax_delta_vwap_turnover_corr_10,
        required_fields=("vwap", "turnover"),
        notes="Qwen refinement candidate for alpha101.alpha069 family: -rank(ts_max(delta(vwap, 3), 5)) * ts_rank(corr(vwap, turnover, 5), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_delta_vwap_rank_rel_volume_corr_10",
        func=llm_refined_neg_delta_vwap_rank_rel_volume_corr_10,
        required_fields=("vwap", "volume"),
        notes="Qwen refinement candidate for alpha101.alpha069 family: -rank(delta(vwap, 3)) * corr(vwap, rank(volume / mean(volume, 20)), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_tsmax_delta_vwap_amount_corr_10",
        func=llm_refined_neg_tsmax_delta_vwap_amount_corr_10,
        required_fields=("vwap", "amount"),
        notes="Qwen refinement candidate for alpha101.alpha069 family: -rank(ts_max(delta(vwap, 3), 10)) * corr(vwap, amount, 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_close_vwap_amount_ratio_corr_10",
        func=llm_refined_neg_close_vwap_amount_ratio_corr_10,
        required_fields=("close", "vwap", "amount"),
        notes="DeepSeek refinement candidate for alpha101.alpha069 family: -corr(rank(close - vwap), rank(amount / mean(amount, 20)), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_vwap_volume_corr_ts_rank_close_std_20",
        func=llm_refined_neg_vwap_volume_corr_ts_rank_close_std_20,
        required_fields=("close", "vwap", "volume"),
        notes="DeepSeek refinement candidate for alpha101.alpha069 family: -ts_rank(corr(vwap, volume, 15), 20) * rank(std(close, 10)).",
    ),
    FactorSpec(
        name="llm_refined.neg_delta_vwap_ts_rank_vwap_turnover_corr_10",
        func=llm_refined_neg_delta_vwap_ts_rank_vwap_turnover_corr_10,
        required_fields=("vwap", "turnover"),
        notes="GPT refinement candidate for alpha101.alpha069 family: -rank(ts_rank(delta(vwap, 3), 5)) * corr(rank(vwap), rank(turnover), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_delta_close_ts_rank_close_amount_corr_10",
        func=llm_refined_neg_delta_close_ts_rank_close_amount_corr_10,
        required_fields=("close", "amount"),
        notes="GPT refinement candidate for alpha101.alpha069 family: -rank(ts_rank(delta(close, 3), 5)) * corr(rank(close), rank(amount), 10).",
    ),
    FactorSpec(
        name="llm_refined.neg_delta_vwap_ts_rank_close_scaled_rel_volume_corr_10",
        func=llm_refined_neg_delta_vwap_ts_rank_close_scaled_rel_volume_corr_10,
        required_fields=("close", "vwap", "volume"),
        notes="GPT refinement candidate for alpha101.alpha069 family: -rank(ts_rank(delta(vwap, 3), 5)) * corr(rank(close), rank(scale(volume / mean(volume, 20))), 10).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "llm_refined_neg_close_vwap_amount_ratio_corr_10",
    "llm_refined_neg_delta_close_ts_rank_close_amount_corr_10",
    "llm_refined_neg_delta_vwap_rank_rel_volume_corr_10",
    "llm_refined_neg_delta_vwap_ts_rank_close_scaled_rel_volume_corr_10",
    "llm_refined_neg_delta_vwap_ts_rank_vwap_turnover_corr_10",
    "llm_refined_neg_tsmax_delta_vwap_amount_corr_10",
    "llm_refined_neg_tsmax_delta_vwap_turnover_corr_10",
    "llm_refined_neg_vwap_ts_rank_rel_volume_corr_15",
    "llm_refined_neg_vwap_volume_corr_ts_rank_close_std_20",
]
