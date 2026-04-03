from __future__ import annotations

"""LLM refined candidates for the alpha191.alpha095 family."""

import pandas as pd

from ...contract import validate_data
from .common import FactorSpec, decay_linear, frame_to_series, rank, sma, stddev, turnover_frame

PARENT_FACTOR = "alpha191.alpha095"
FAMILY_KEY = "alpha095_family"
SEED_FAMILY = "amount_volatility"
SUMMARY_GLOB = "llm_refined_alpha095_family_summary_*.csv"


def _ema_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.ewm(span=int(window), min_periods=int(window), adjust=False).mean()


def llm_refined_amt_vol_normalized(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -stddev(amount_df, 20) / sma(amount_df, 20).replace(0.0, pd.NA)
    return frame_to_series(frame, factor_name="llm_refined.amt_vol_normalized")


def llm_refined_turnover_vol_ratio(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("turnover",))
    turnover_df = turnover_frame(data)
    frame = -stddev(turnover_df, 20) / sma(turnover_df, 20).replace(0.0, pd.NA)
    return frame_to_series(frame, factor_name="llm_refined.turnover_vol_ratio")


def llm_refined_amt_vol_price_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount", "returns"))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    returns_df = data["returns"].unstack(level="instrument").sort_index()
    frame = -stddev(amount_df, 20) * (1.0 + rank(returns_df))
    return frame_to_series(frame, factor_name="llm_refined.amt_vol_price_confirm")


def llm_refined_alpha095_decay_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -decay_linear(stddev(amount_df, 20), 5)
    return frame_to_series(frame, factor_name="llm_refined.alpha095_decay_smooth")


def llm_refined_amt_ret_instability(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount", "returns"))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    returns_df = data["returns"].unstack(level="instrument").sort_index()
    frame = -stddev(amount_df, 20) * stddev(returns_df, 20)
    return frame_to_series(frame, factor_name="llm_refined.amt_ret_instability")


def llm_refined_amount_cv_28(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -stddev(amount_df, 28) / sma(amount_df, 28).replace(0.0, pd.NA)
    return frame_to_series(frame, factor_name="llm_refined.amount_cv_28")


def llm_refined_turnover_vol_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("turnover",))
    turnover_df = turnover_frame(data)
    frame = -stddev(turnover_df, 20)
    return frame_to_series(frame, factor_name="llm_refined.turnover_vol_20")


def llm_refined_smoothed_amt_vol(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha191.alpha095
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("amount",))
    amount_df = data["amount"].unstack(level="instrument").sort_index()
    frame = -stddev(_ema_frame(amount_df, 5), 20)
    return frame_to_series(frame, factor_name="llm_refined.smoothed_amt_vol")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.amt_vol_normalized",
        func=llm_refined_amt_vol_normalized,
        required_fields=("amount",),
        notes="DeepSeek manual refinement candidate for alpha191.alpha095 family: -ts_std(amount, 20) / mean(amount, 20).",
    ),
    FactorSpec(
        name="llm_refined.turnover_vol_ratio",
        func=llm_refined_turnover_vol_ratio,
        required_fields=("turnover",),
        notes="DeepSeek manual refinement candidate for alpha191.alpha095 family: -ts_std(turnover, 20) / mean(turnover, 20).",
    ),
    FactorSpec(
        name="llm_refined.amt_vol_price_confirm",
        func=llm_refined_amt_vol_price_confirm,
        required_fields=("amount", "returns"),
        notes="DeepSeek manual refinement candidate for alpha191.alpha095 family: -ts_std(amount, 20) * (1 + cs_rank(returns)).",
    ),
    FactorSpec(
        name="llm_refined.alpha095_decay_smooth",
        func=llm_refined_alpha095_decay_smooth,
        required_fields=("amount",),
        notes="Qwen manual refinement candidate for alpha191.alpha095 family: -decay_linear(std(amount, 20), 5).",
    ),
    FactorSpec(
        name="llm_refined.amt_ret_instability",
        func=llm_refined_amt_ret_instability,
        required_fields=("amount", "returns"),
        notes="GPT manual refinement candidate for alpha191.alpha095 family: -mul(std(amount, 20), std(returns, 20)).",
    ),
    FactorSpec(
        name="llm_refined.amount_cv_28",
        func=llm_refined_amount_cv_28,
        required_fields=("amount",),
        notes="Gemini manual refinement candidate for alpha191.alpha095 family: -div(std(amount, 28), mean(amount, 28)).",
    ),
    FactorSpec(
        name="llm_refined.turnover_vol_20",
        func=llm_refined_turnover_vol_20,
        required_fields=("turnover",),
        notes="Gemini manual refinement candidate for alpha191.alpha095 family: -std(turnover, 20).",
    ),
    FactorSpec(
        name="llm_refined.smoothed_amt_vol",
        func=llm_refined_smoothed_amt_vol,
        required_fields=("amount",),
        notes="GPT manual refinement candidate for alpha191.alpha095 family: -std(ema(amount, 5), 20).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_alpha095_decay_smooth",
    "llm_refined_amt_ret_instability",
    "llm_refined_amt_vol_normalized",
    "llm_refined_amt_vol_price_confirm",
    "llm_refined_amount_cv_28",
    "llm_refined_smoothed_amt_vol",
    "llm_refined_turnover_vol_20",
    "llm_refined_turnover_vol_ratio",
]
