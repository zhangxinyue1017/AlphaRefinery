from __future__ import annotations

"""LLM refined candidates for the alpha191.alpha132 family."""

import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, frame_to_series, rank, sma, stddev, turnover_frame

PARENT_FACTOR = "alpha191.alpha132"
FAMILY_KEY = "alpha132_family"
SEED_FAMILY = "amount_level"
SUMMARY_GLOB = "llm_refined_alpha132_family_summary_*.csv"


def _ema_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.ewm(span=int(window), min_periods=int(window), adjust=False).mean()


def llm_refined_amt_scaled_by_floatcap(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount", "float_market_cap"))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    float_cap_df = data["float_market_cap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    frame = -sma(amount_df, 20) / float_cap_df
    return frame_to_series(frame, factor_name="llm_refined.amt_scaled_by_floatcap")


def llm_refined_amt_volatility_ratio(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -stddev(amount_df, 20) / sma(amount_df, 20).replace(0.0, pd.NA)
    return frame_to_series(frame, factor_name="llm_refined.amt_volatility_ratio")


def llm_refined_amt_turnover_confirmed(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount", "turnover"))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    turnover_df = turnover_frame(data)
    frame = -sma(amount_df, 20) * (1.0 - rank(turnover_df))
    return frame_to_series(frame, factor_name="llm_refined.amt_turnover_confirmed")


def llm_refined_amt_level_ema20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -_ema_frame(amount_df, 20)
    return frame_to_series(frame, factor_name="llm_refined.amt_level_ema20")


def llm_refined_amt_to_floatcap_rank(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount", "float_market_cap"))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    float_cap_df = data["float_market_cap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    frame = -rank(sma(amount_df, 20) / float_cap_df)
    return frame_to_series(frame, factor_name="llm_refined.amt_to_floatcap_rank")


def llm_refined_amt_level_vol_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -sma(amount_df, 20) / stddev(amount_df, 20).replace(0.0, pd.NA)
    return frame_to_series(frame, factor_name="llm_refined.amt_level_vol_confirm")


def llm_refined_amount_cap_norm_28(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount", "float_market_cap"))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    float_cap_df = data["float_market_cap"].unstack(level="instrument").sort_index().replace(0.0, pd.NA)
    frame = -sma(amount_df, 28) / float_cap_df
    return frame_to_series(frame, factor_name="llm_refined.amount_cap_norm_28")


def llm_refined_amount_shock_5_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -sma(amount_df, 5) / sma(amount_df, 20).replace(0.0, pd.NA)
    return frame_to_series(frame, factor_name="llm_refined.amount_shock_5_20")


def llm_refined_alpha132_vol_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha132
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount", "returns"))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    returns_df = data["returns"].unstack(level="instrument").sort_index()
    frame = -sma(amount_df, 20) * stddev(returns_df, 20)
    return frame_to_series(frame, factor_name="llm_refined.alpha132_vol_confirm")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.amt_scaled_by_floatcap",
        func=llm_refined_amt_scaled_by_floatcap,
        required_fields=("amount", "float_market_cap"),
        notes="DeepSeek manual refinement candidate for alpha191.alpha132 family: -mean(amount, 20) / float_market_cap.",
    ),
    FactorSpec(
        name="llm_refined.amt_volatility_ratio",
        func=llm_refined_amt_volatility_ratio,
        required_fields=("amount",),
        notes="DeepSeek manual refinement candidate for alpha191.alpha132 family: -ts_std(amount, 20) / mean(amount, 20).",
    ),
    FactorSpec(
        name="llm_refined.amt_turnover_confirmed",
        func=llm_refined_amt_turnover_confirmed,
        required_fields=("amount", "turnover"),
        notes="DeepSeek manual refinement candidate for alpha191.alpha132 family: -mean(amount, 20) * (1 - cs_rank(turnover)).",
    ),
    FactorSpec(
        name="llm_refined.amt_level_ema20",
        func=llm_refined_amt_level_ema20,
        required_fields=("amount",),
        notes="GPT manual refinement candidate for alpha191.alpha132 family: neg(ema(amount, 20)).",
    ),
    FactorSpec(
        name="llm_refined.amt_to_floatcap_rank",
        func=llm_refined_amt_to_floatcap_rank,
        required_fields=("amount", "float_market_cap"),
        notes="GPT manual refinement candidate for alpha191.alpha132 family: neg(cs_rank(div(mean(amount, 20), float_market_cap))).",
    ),
    FactorSpec(
        name="llm_refined.amt_level_vol_confirm",
        func=llm_refined_amt_level_vol_confirm,
        required_fields=("amount",),
        notes="GPT manual refinement candidate for alpha191.alpha132 family: neg(div(mean(amount, 20), std(amount, 20))).",
    ),
    FactorSpec(
        name="llm_refined.amount_cap_norm_28",
        func=llm_refined_amount_cap_norm_28,
        required_fields=("amount", "float_market_cap"),
        notes="Gemini manual refinement candidate for alpha191.alpha132 family: div(-mean(amount, 28), float_market_cap).",
    ),
    FactorSpec(
        name="llm_refined.amount_shock_5_20",
        func=llm_refined_amount_shock_5_20,
        required_fields=("amount",),
        notes="Gemini manual refinement candidate for alpha191.alpha132 family: -div(mean(amount, 5), mean(amount, 20)).",
    ),
    FactorSpec(
        name="llm_refined.alpha132_vol_confirm",
        func=llm_refined_alpha132_vol_confirm,
        required_fields=("amount", "returns"),
        notes="Qwen manual refinement candidate for alpha191.alpha132 family: -mul(mean(amount, 20), ts_std(returns, 20)).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_alpha132_vol_confirm",
    "llm_refined_amount_cap_norm_28",
    "llm_refined_amount_shock_5_20",
    "llm_refined_amt_level_ema20",
    "llm_refined_amt_level_vol_confirm",
    "llm_refined_amt_scaled_by_floatcap",
    "llm_refined_amt_to_floatcap_rank",
    "llm_refined_amt_turnover_confirmed",
    "llm_refined_amt_volatility_ratio",
]
