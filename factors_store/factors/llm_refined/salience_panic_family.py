from __future__ import annotations

"""LLM refined candidates for the qp_salience.std_score_60 family."""

import numpy as np
import pandas as pd

from ...contract import validate_data
from ..qp_salience import _calc_sigma, _std_score_frame, _terrified_score_frame
from .common import FactorSpec, decay_linear, frame_to_series, rank, sma, stddev, ts_rank, turnover_frame

PARENT_FACTOR = "qp_salience.std_score_60"
FAMILY_KEY = "salience_panic_family"
SEED_FAMILY = "salience_panic_score"
SUMMARY_GLOB = "llm_refined_salience_panic_family_summary_*.csv"


def _returns_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=("close",))
    close_df = data["close"].unstack(level="instrument").sort_index()
    return close_df.pct_change(fill_method=None)


def _amount_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=("amount",))
    return data["amount"].unstack(level="instrument").sort_index()


def _pre_close_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    validate_data(data, required_fields=("pre_close",))
    return data["pre_close"].unstack(level="instrument").sort_index()


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _salience_sigma_weighted_returns(data: dict[str, pd.Series]) -> pd.DataFrame:
    ret_df = _returns_frame(data)
    sigma = _calc_sigma(ret_df)
    return sigma * ret_df


def _terrified_mix_frame(data: dict[str, pd.Series]) -> pd.DataFrame:
    weighted = _salience_sigma_weighted_returns(data)
    return 0.5 * (sma(weighted, 60) + stddev(weighted, 60))


def _ema_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    return df.ewm(span=int(window), min_periods=int(window), adjust=False).mean()


def llm_refined_panic_smoothed_ema(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    frame = -_ema_frame(stddev(ret_df, 60), 20)
    return frame_to_series(frame, factor_name="llm_refined.panic_smoothed_ema")


def llm_refined_panic_amount_vol_interact(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    amount_df = _amount_frame(data)
    frame = -stddev(ret_df, 60) * (stddev(amount_df, 20) / sma(amount_df, 20).replace(0.0, np.nan))
    return frame_to_series(frame, factor_name="llm_refined.panic_amount_vol_interact")


def llm_refined_panic_reversal_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    frame = -stddev(ret_df, 60) * (1.0 + rank(ret_df))
    return frame_to_series(frame, factor_name="llm_refined.panic_reversal_confirm")


def llm_refined_smoothed_terrified_score_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "turnover"))
    frame = -decay_linear(_terrified_score_frame(data, window=60), 20)
    return frame_to_series(frame, factor_name="llm_refined.smoothed_terrified_score_20")


def llm_refined_panic_turnover_norm_60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "turnover"))
    turnover_df = turnover_frame(data)
    frame = -_std_score_frame(data, window=60) / sma(turnover_df, 60).replace(0.0, np.nan)
    return frame_to_series(frame, factor_name="llm_refined.panic_turnover_norm_60")


def llm_refined_panic_amplitude_interact(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    validate_data(data, required_fields=("close", "turnover", "high", "low", "pre_close"))
    high_df = data["high"].unstack(level="instrument").sort_index()
    low_df = data["low"].unstack(level="instrument").sort_index()
    pre_close_df = _pre_close_frame(data).replace(0.0, np.nan)
    amplitude = (high_df - low_df) / pre_close_df
    frame = -_std_score_frame(data, window=60) * amplitude
    return frame_to_series(frame, factor_name="llm_refined.panic_amplitude_interact")


def llm_refined_decayed_salience_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    weighted = _salience_sigma_weighted_returns(data)
    frame = -decay_linear(stddev(weighted, 60), 10)
    return frame_to_series(frame, factor_name="llm_refined.decayed_salience_std")


def llm_refined_relative_salience_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    weighted = _salience_sigma_weighted_returns(data)
    frame = -stddev(weighted, 60) / (sma(ret_df.abs(), 60) + 0.1)
    return frame_to_series(frame, factor_name="llm_refined.relative_salience_std")


def llm_refined_salience_ema_panic_60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: llm_refine_20260327_r1
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    weighted = _salience_sigma_weighted_returns(data)
    frame = -_ema_frame(stddev(weighted, 60), 10)
    return frame_to_series(frame, factor_name="llm_refined.salience_ema_panic_60")


def llm_refined_salience_amount_normalized_std_60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: llm_refine_20260327_r1
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    weighted = _salience_sigma_weighted_returns(data)
    amount_df = _amount_frame(data)
    rel_amount = amount_df.div(sma(amount_df, 20).replace(0.0, np.nan))
    frame = -stddev(weighted.div(rel_amount + 0.1), 60)
    return frame_to_series(frame, factor_name="llm_refined.salience_amount_normalized_std_60")


def llm_refined_panic_reversal_confirm_tsrank(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    weighted = _salience_sigma_weighted_returns(data)
    frame = -stddev(weighted, 60) * ts_rank(ret_df.abs(), 20)
    return frame_to_series(frame, factor_name="llm_refined.panic_reversal_confirm_tsrank")


def llm_refined_salience_panic_decay(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    frame = -decay_linear(stddev(_cs_zscore(ret_df), 60), 5)
    return frame_to_series(frame, factor_name="llm_refined.salience_panic_decay")


def llm_refined_salience_panic_vol_neut(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    frame = -stddev(_cs_zscore(ret_df), 60) / stddev(ret_df, 60).replace(0.0, np.nan)
    return frame_to_series(frame, factor_name="llm_refined.salience_panic_vol_neut")


def llm_refined_salience_panic_liq_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.std_score_60
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    turnover_df = turnover_frame(data)
    frame = -stddev(_cs_zscore(ret_df), 60) * turnover_df
    return frame_to_series(frame, factor_name="llm_refined.salience_panic_liq_confirm")


def llm_refined_terrified_smoothed(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    frame = -_ema_frame(_terrified_mix_frame(data), 20)
    return frame_to_series(frame, factor_name="llm_refined.terrified_smoothed")


def llm_refined_terrified_amount_vol(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    amount_df = _amount_frame(data)
    frame = -_terrified_mix_frame(data) * (stddev(amount_df, 20) / sma(amount_df, 20).replace(0.0, np.nan))
    return frame_to_series(frame, factor_name="llm_refined.terrified_amount_vol")


def llm_refined_terrified_reversal_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: DeepSeek
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    frame = -_terrified_mix_frame(data) * (1.0 + rank(ret_df))
    return frame_to_series(frame, factor_name="llm_refined.terrified_reversal_confirm")


def llm_refined_decay_terrified_score_60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    frame = decay_linear(-_terrified_mix_frame(data), 20)
    return frame_to_series(frame, factor_name="llm_refined.decay_terrified_score_60")


def llm_refined_ts_rank_terrified_score_60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    frame = ts_rank(-_terrified_mix_frame(data), 60)
    return frame_to_series(frame, factor_name="llm_refined.ts_rank_terrified_score_60")


def llm_refined_norm_terrified_score_60_adv20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: Gemini
    keep-drop status: candidate_pool
    """
    turnover_df = turnover_frame(data)
    frame = -_terrified_mix_frame(data) / sma(turnover_df, 20).replace(0.0, np.nan)
    return frame_to_series(frame, factor_name="llm_refined.norm_terrified_score_60_adv20")


def llm_refined_decayed_terrified_mix(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    frame = -decay_linear(_terrified_mix_frame(data), 10)
    return frame_to_series(frame, factor_name="llm_refined.decayed_terrified_mix")


def llm_refined_relative_terrified_score(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    frame = -(_terrified_mix_frame(data) * 2.0) / (sma(ret_df.abs(), 60) + 0.1)
    return frame_to_series(frame, factor_name="llm_refined.relative_terrified_score")


def llm_refined_terrified_amount_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: GPT
    keep-drop status: candidate_pool
    """
    amount_df = _amount_frame(data)
    frame = -_terrified_mix_frame(data) * (amount_df / sma(amount_df, 20).replace(0.0, np.nan))
    return frame_to_series(frame, factor_name="llm_refined.terrified_amount_confirm")


def llm_refined_salience_terrified_decay(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    zret = _cs_zscore(ret_df)
    frame = -decay_linear(sma(zret, 60) + stddev(zret, 60), 5)
    return frame_to_series(frame, factor_name="llm_refined.salience_terrified_decay")


def llm_refined_salience_terrified_vol_neut(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    zret = _cs_zscore(ret_df)
    frame = -(sma(zret, 60) + stddev(zret, 60)) / stddev(ret_df, 60).replace(0.0, np.nan)
    return frame_to_series(frame, factor_name="llm_refined.salience_terrified_vol_neut")


def llm_refined_salience_terrified_liq_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_salience.terrified_score_60
    round: manual_refine_20260324
    source model: Qwen
    keep-drop status: candidate_pool
    """
    ret_df = _returns_frame(data)
    zret = _cs_zscore(ret_df)
    turnover_df = turnover_frame(data)
    frame = -(sma(zret, 60) + stddev(zret, 60)) * turnover_df
    return frame_to_series(frame, factor_name="llm_refined.salience_terrified_liq_confirm")


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.panic_smoothed_ema",
        func=llm_refined_panic_smoothed_ema,
        required_fields=("close",),
        notes="DeepSeek manual refinement candidate for qp_salience.std_score_60 family: -ema(ts_std(returns, 60), 20).",
    ),
    FactorSpec(
        name="llm_refined.panic_amount_vol_interact",
        func=llm_refined_panic_amount_vol_interact,
        required_fields=("close", "amount"),
        notes="DeepSeek manual refinement candidate for qp_salience.std_score_60 family: -ts_std(returns, 60) * (ts_std(amount, 20) / mean(amount, 20)).",
    ),
    FactorSpec(
        name="llm_refined.panic_reversal_confirm",
        func=llm_refined_panic_reversal_confirm,
        required_fields=("close",),
        notes="DeepSeek manual refinement candidate for qp_salience.std_score_60 family: -ts_std(returns, 60) * (1 + cs_rank(returns)).",
    ),
    FactorSpec(
        name="llm_refined.smoothed_terrified_score_20",
        func=llm_refined_smoothed_terrified_score_20,
        required_fields=("close", "turnover"),
        notes="Gemini manual refinement candidate for qp_salience.std_score_60 family: -decay_linear(qp_salience.terrified_score_60, 20).",
    ),
    FactorSpec(
        name="llm_refined.panic_turnover_norm_60",
        func=llm_refined_panic_turnover_norm_60,
        required_fields=("close", "turnover"),
        notes="Gemini manual refinement candidate for qp_salience.std_score_60 family: -div(qp_salience.std_score_60, mean(turnover, 60)).",
    ),
    FactorSpec(
        name="llm_refined.panic_amplitude_interact",
        func=llm_refined_panic_amplitude_interact,
        required_fields=("close", "turnover", "high", "low", "pre_close"),
        notes="Gemini manual refinement candidate for qp_salience.std_score_60 family: -mul(qp_salience.std_score_60, div(sub(high, low), pre_close)).",
    ),
    FactorSpec(
        name="llm_refined.decayed_salience_std",
        func=llm_refined_decayed_salience_std,
        required_fields=("close",),
        notes="GPT manual refinement candidate for qp_salience.std_score_60 family: -decay_linear(std(sigma * returns, 60), 10).",
    ),
    FactorSpec(
        name="llm_refined.relative_salience_std",
        func=llm_refined_relative_salience_std,
        required_fields=("close",),
        notes="GPT manual refinement candidate for qp_salience.std_score_60 family: -div(std(sigma * returns, 60), mean(abs(returns), 60) + 0.1).",
    ),
    FactorSpec(
        name="llm_refined.salience_ema_panic_60",
        func=llm_refined_salience_ema_panic_60,
        required_fields=("close",),
        notes="GPT-5.4 auto-refine research_winner for qp_salience.std_score_60 family: negative 10-day EMA of 60-day salience-weighted return std.",
    ),
    FactorSpec(
        name="llm_refined.salience_amount_normalized_std_60",
        func=llm_refined_salience_amount_normalized_std_60,
        required_fields=("close", "amount"),
        notes="GPT-5.4 auto-refine research_keep for qp_salience.std_score_60 family: negative 60-day std of salience-weighted returns normalized by relative amount.",
    ),
    FactorSpec(
        name="llm_refined.panic_reversal_confirm_tsrank",
        func=llm_refined_panic_reversal_confirm_tsrank,
        required_fields=("close",),
        notes="GPT manual refinement candidate for qp_salience.std_score_60 family: -mul(std(sigma * returns, 60), ts_rank(abs(returns), 20)).",
    ),
    FactorSpec(
        name="llm_refined.salience_panic_decay",
        func=llm_refined_salience_panic_decay,
        required_fields=("close",),
        notes="Qwen manual refinement candidate for qp_salience.std_score_60 family: -decay_linear(std(cs_zscore(returns), 60), 5).",
    ),
    FactorSpec(
        name="llm_refined.salience_panic_vol_neut",
        func=llm_refined_salience_panic_vol_neut,
        required_fields=("close",),
        notes="Qwen manual refinement candidate for qp_salience.std_score_60 family: -div(std(cs_zscore(returns), 60), std(returns, 60)).",
    ),
    FactorSpec(
        name="llm_refined.salience_panic_liq_confirm",
        func=llm_refined_salience_panic_liq_confirm,
        required_fields=("close", "turnover"),
        notes="Qwen manual refinement candidate for qp_salience.std_score_60 family: -mul(std(cs_zscore(returns), 60), turnover).",
    ),
    FactorSpec(
        name="llm_refined.terrified_smoothed",
        func=llm_refined_terrified_smoothed,
        required_fields=("close",),
        notes="DeepSeek manual refinement candidate for qp_salience.terrified_score_60 family: -ema(0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60)), 20).",
    ),
    FactorSpec(
        name="llm_refined.terrified_amount_vol",
        func=llm_refined_terrified_amount_vol,
        required_fields=("close", "amount"),
        notes="DeepSeek manual refinement candidate for qp_salience.terrified_score_60 family: -0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60)) * (ts_std(amount, 20) / mean(amount, 20)).",
    ),
    FactorSpec(
        name="llm_refined.terrified_reversal_confirm",
        func=llm_refined_terrified_reversal_confirm,
        required_fields=("close",),
        notes="DeepSeek manual refinement candidate for qp_salience.terrified_score_60 family: -0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60)) * (1 + cs_rank(returns)).",
    ),
    FactorSpec(
        name="llm_refined.decay_terrified_score_60",
        func=llm_refined_decay_terrified_score_60,
        required_fields=("close",),
        notes="Gemini manual refinement candidate for qp_salience.terrified_score_60 family: decay_linear(-(0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60))), 20).",
    ),
    FactorSpec(
        name="llm_refined.ts_rank_terrified_score_60",
        func=llm_refined_ts_rank_terrified_score_60,
        required_fields=("close",),
        notes="Gemini manual refinement candidate for qp_salience.terrified_score_60 family: ts_rank(-(0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60))), 60).",
    ),
    FactorSpec(
        name="llm_refined.norm_terrified_score_60_adv20",
        func=llm_refined_norm_terrified_score_60_adv20,
        required_fields=("close", "turnover"),
        notes="Gemini manual refinement candidate for qp_salience.terrified_score_60 family: div(-(0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60))), mean(turnover, 20)).",
    ),
    FactorSpec(
        name="llm_refined.decayed_terrified_mix",
        func=llm_refined_decayed_terrified_mix,
        required_fields=("close",),
        notes="GPT manual refinement candidate for qp_salience.terrified_score_60 family: -decay_linear(0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60)), 10).",
    ),
    FactorSpec(
        name="llm_refined.relative_terrified_score",
        func=llm_refined_relative_terrified_score,
        required_fields=("close",),
        notes="GPT manual refinement candidate for qp_salience.terrified_score_60 family: -div(add(mean(sigma * returns, 60), std(sigma * returns, 60)), add(mean(abs(returns), 60), 0.1)).",
    ),
    FactorSpec(
        name="llm_refined.terrified_amount_confirm",
        func=llm_refined_terrified_amount_confirm,
        required_fields=("close", "amount"),
        notes="GPT manual refinement candidate for qp_salience.terrified_score_60 family: -mul(0.5 * (mean(sigma * returns, 60) + std(sigma * returns, 60)), rel_amount(20)).",
    ),
    FactorSpec(
        name="llm_refined.salience_terrified_decay",
        func=llm_refined_salience_terrified_decay,
        required_fields=("close",),
        notes="Qwen manual refinement candidate for qp_salience.terrified_score_60 family: -decay_linear(add(ts_mean(cs_zscore(returns), 60), ts_std(cs_zscore(returns), 60)), 5).",
    ),
    FactorSpec(
        name="llm_refined.salience_terrified_vol_neut",
        func=llm_refined_salience_terrified_vol_neut,
        required_fields=("close",),
        notes="Qwen manual refinement candidate for qp_salience.terrified_score_60 family: -div(add(ts_mean(cs_zscore(returns), 60), ts_std(cs_zscore(returns), 60)), ts_std(returns, 60)).",
    ),
    FactorSpec(
        name="llm_refined.salience_terrified_liq_confirm",
        func=llm_refined_salience_terrified_liq_confirm,
        required_fields=("close", "turnover"),
        notes="Qwen manual refinement candidate for qp_salience.terrified_score_60 family: -mul(add(ts_mean(cs_zscore(returns), 60), ts_std(cs_zscore(returns), 60)), turnover).",
    ),
)

__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_decayed_salience_std",
    "llm_refined_panic_amplitude_interact",
    "llm_refined_panic_amount_vol_interact",
    "llm_refined_panic_reversal_confirm",
    "llm_refined_panic_reversal_confirm_tsrank",
    "llm_refined_panic_smoothed_ema",
    "llm_refined_panic_turnover_norm_60",
    "llm_refined_decay_terrified_score_60",
    "llm_refined_decayed_terrified_mix",
    "llm_refined_relative_salience_std",
    "llm_refined_relative_terrified_score",
    "llm_refined_salience_amount_normalized_std_60",
    "llm_refined_salience_ema_panic_60",
    "llm_refined_salience_panic_decay",
    "llm_refined_salience_panic_liq_confirm",
    "llm_refined_salience_panic_vol_neut",
    "llm_refined_salience_terrified_decay",
    "llm_refined_salience_terrified_liq_confirm",
    "llm_refined_salience_terrified_vol_neut",
    "llm_refined_smoothed_terrified_score_20",
    "llm_refined_terrified_amount_confirm",
    "llm_refined_terrified_amount_vol",
    "llm_refined_terrified_reversal_confirm",
    "llm_refined_terrified_smoothed",
    "llm_refined_ts_rank_terrified_score_60",
    "llm_refined_norm_terrified_score_60_adv20",
]
