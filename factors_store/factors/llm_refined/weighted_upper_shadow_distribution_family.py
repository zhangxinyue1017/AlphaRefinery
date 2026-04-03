from __future__ import annotations

"""LLM refined candidates for the factor365.weighted_upper_shadow_frequency_40_hl10 family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "factor365.weighted_upper_shadow_frequency_40_hl10"
FAMILY_KEY = "weighted_upper_shadow_distribution_family"
SEED_FAMILY = "weighted_upper_shadow_distribution"
SUMMARY_GLOB = "llm_refined_weighted_upper_shadow_distribution_family_summary_*.csv"


def llm_refined_volume_weighted_shadow_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_22
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(volume_weighted_mean(where((high - close) / add(pre_close, 1e-12) > 0.01, (high - close) / add(pre_close, 1e-12), 0), volume, 10))",
        factor_name="llm_refined.volume_weighted_shadow_10",
    )


def llm_refined_shadow_turnover_weighted(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_23
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, close), add(pre_close, 1e-12)) > 0.01, mul(div(sub(high, close), add(pre_close, 1e-12)), turnover), 0), 10))",
        factor_name="llm_refined.shadow_turnover_weighted",
    )


def llm_refined_shadow_turnover_filtered(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_23
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(mul(gt(div(sub(high, close), add(pre_close, 1e-12)), 0.01), gt(turnover, cs_mean(turnover))), div(sub(high, close), add(pre_close, 1e-12)), 0), 10))",
        factor_name="llm_refined.shadow_turnover_filtered",
    )


def llm_refined_shadow_rank_threshold(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_23
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(gt(cs_rank(div(sub(high, close), add(pre_close, 1e-12))), 0.8), div(sub(high, close), add(pre_close, 1e-12)), 0), 10))",
        factor_name="llm_refined.shadow_rank_threshold",
    )


def llm_refined_amt_weighted_shadow_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_25
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where((high - close) / add(pre_close, 1e-12) > 0.01, (high - close) / add(pre_close, 1e-12) * amount, 0), 10))",
        factor_name="llm_refined.amt_weighted_shadow_10",
    )


def llm_refined_vol_amt_shadow_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_25
    source model: kimi-k2
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where((high - close) / add(ts_std(close, 20), 1e-12) > 0.5, (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.vol_amt_shadow_15",
    )


def llm_refined_amt_shadow_turnover_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_29
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(volume_weighted_mean(where(gt(div(sub(high, close), add(pre_close, 1e-12)), 0.01), mul(div(sub(high, close), add(pre_close, 1e-12)), add(turnover, 1e-12)), 0), amount, 10))",
        factor_name="llm_refined.amt_shadow_turnover_confirm",
    )


def llm_refined_upper_body_reject_amt_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.amt_weighted_shadow_10
    round: llm_refine_round_36
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 10))",
        factor_name="llm_refined.upper_body_reject_amt_10",
    )


def llm_refined_shadow_amt_ema_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.upper_body_reject_amt_10
    round: llm_refine_round_60
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 15))",
        factor_name="llm_refined.shadow_amt_ema_15",
    )


def llm_refined_shadow_length_weighted_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.amt_weighted_shadow_10
    round: llm_refine_round_36
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where((high - close) / add(pre_close, 1e-12) > 0.01, (high - close) / add(pre_close, 1e-12) * amount * log(1 + (high - close)), 0), 10))",
        factor_name="llm_refined.shadow_length_weighted_10",
    )


def llm_refined_wm_half_life_shadow_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.amt_weighted_shadow_10
    round: llm_refine_round_36
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(weighted_mean(where((high - close) / add(pre_close, 1e-12) > 0.01, (high - close) / add(pre_close, 1e-12) * amount, 0), 10, half_life=10))",
        factor_name="llm_refined.wm_half_life_shadow_10",
    )


def llm_refined_turnover_conf_shadow_ema10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.amt_weighted_shadow_10
    round: llm_refine_round_36
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(gt(div(sub(high, close), add(pre_close, 1e-12)), 0.01), mul(div(sub(high, close), add(pre_close, 1e-12)), turnover), 0), 10))",
        factor_name="llm_refined.turnover_conf_shadow_ema10",
    )


def llm_refined_amt_turn_confirm_shadow_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.amt_weighted_shadow_10
    round: llm_refine_round_36
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where((high - close) / add(pre_close, 1e-12) > 0.01, (high - close) / add(pre_close, 1e-12) * amount * turnover, 0), 10))",
        factor_name="llm_refined.amt_turn_confirm_shadow_10",
    )


def llm_refined_shadow_amt_turnover_confirm_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.vol_amt_shadow_15
    round: llm_refine_round_37
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, close), add(ts_std(close, 20), 1e-12)) > 0.5, mul(mul(div(sub(high, close), add(pre_close, 1e-12)), amount), turnover), 0), 15))",
        factor_name="llm_refined.shadow_amt_turnover_confirm_15",
    )


def llm_refined_relative_shadow_length_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.vol_amt_shadow_15
    round: llm_refine_round_37
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where((high - close) / add(high - low, 1e-12) > 0.3, (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.relative_shadow_length_15",
    )


def llm_refined_shadow_range_norm_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.vol_amt_shadow_15
    round: llm_refine_round_37
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where((high - close) / add(sub(high, low), 1e-12) > 0.3, (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.shadow_range_norm_15",
    )


def llm_refined_turnover_shadow_confirm_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.vol_amt_shadow_15
    round: llm_refine_round_37
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(gt(div(sub(high, close), add(ts_std(close, 20), 1e-12)), 0.8), mul(div(sub(high, close), add(pre_close, 1e-12)), turnover), 0), 20))",
        factor_name="llm_refined.turnover_shadow_confirm_20",
    )


def llm_refined_amt_turn_confirm_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.vol_amt_shadow_15
    round: llm_refine_round_37
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(((high - close) / add(ts_std(close, 20), 1e-12) > 0.5) & (turnover > ts_mean(turnover, 20)), (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.amt_turn_confirm_15",
    )


def llm_refined_shadow_turn_confirm_ema(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.upper_body_reject_amt_10
    round: llm_refine_round_46
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, mul(mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), div(turnover, add(ts_mean(turnover, 20), 1e-12))), 0), 10))",
        factor_name="llm_refined.shadow_turn_confirm_ema",
    )


def llm_refined_vol_scaled_shadow_amt_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.upper_body_reject_amt_10
    round: llm_refine_round_47
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, rowmax(open, close)), add(mul(add(pre_close, 1e-12), ts_std(returns, 20)), 1e-12)) > 0.01, mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 15))",
        factor_name="llm_refined.vol_scaled_shadow_amt_15",
    )


def llm_refined_shadow_vol_scaled_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.upper_body_reject_amt_10
    round: llm_refine_round_59
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, rowmax(open, close)), add(ts_std(close, 15), 1e-12)) > 0.01, mul(div(sub(high, rowmax(open, close)), add(ts_std(close, 15), 1e-12)), amount), 0), 10))",
        factor_name="llm_refined.shadow_vol_scaled_15",
    )


def llm_refined_shadow_decay_amt_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llm_refined.upper_body_reject_amt_10
    round: llm_refine_round_48
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(decay_linear(where(gt(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), 0.01), mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 20))",
        factor_name="llm_refined.shadow_decay_amt_20",
    )


def llm_refined_decay_shadow_turn_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llmgen.turnover_confirmed_shadow_15
    round: llm_refine_round_50
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(decay_linear(where(((high - close) / add(ts_std(close, 20), 1e-12) > 0.5) & (turnover > ts_mean(turnover, 10)), div(high - close, add(pre_close, 1e-12)) * amount, 0), 15))",
        factor_name="llm_refined.decay_shadow_turn_15",
    )


def llm_refined_shadow_pos_confirm_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llmgen.turnover_confirmed_shadow_15
    round: llm_refine_round_51
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(((high - close) / add(ts_std(close, 20), 1e-12) > 0.5) & (turnover > ts_mean(turnover, 10)) & (close > ts_mean(close, 20)), (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.shadow_pos_confirm_15",
    )


def llm_refined_shadow_amt_relturn_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llmgen.amt_turn_confirm_15
    round: llm_refine_round_52
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(((high - close) / add(ts_std(close, 20), 1e-12) > 0.5) & (turnover / add(ts_mean(turnover, 20), 1e-12) > 1.2), (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.shadow_amt_relturn_15",
    )


def llm_refined_llmgen_shadow_vol_rel_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llmgen.amt_turn_confirm_15
    round: llm_refine_round_53
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(((high - close) / add(ts_std(close, 20), 1e-12) > div(ts_std(close, 20), add(pre_close, 1e-12))) & (rel_amount(20) > 1), (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.llmgen_shadow_vol_rel_15",
    )


def llm_refined_shadow_wm_half_life_40(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llmgen.amt_turn_confirm_15
    round: llm_refine_round_54
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(weighted_mean(where(((high - close) / add(ts_std(close, 20), 1e-12) > 0.5) & (turnover > ts_mean(turnover, 20)), (high - close) / add(pre_close, 1e-12) * amount, 0), 40, half_life=10))",
        factor_name="llm_refined.shadow_wm_half_life_40",
    )


def llm_refined_llmgen_shadow_std40_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: llmgen.turnover_confirmed_shadow_15
    round: llm_refine_round_60
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(((high - close) / add(ts_std(close, 40), 1e-12) > 0.5) & (turnover > ts_mean(turnover, 10)), (high - close) / add(pre_close, 1e-12) * amount, 0), 15))",
        factor_name="llm_refined.llmgen_shadow_std40_15",
    )


def llm_refined_shadow_turnover_conf_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_73
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(and(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, turnover > ts_mean(turnover, 20)), mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 15))",
        factor_name="llm_refined.shadow_turnover_conf_15",
    )


def llm_refined_shadow_rel_amount_confirm_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_74
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(and(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, rel_amount(20) > 1.2), mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 15))",
        factor_name="llm_refined.shadow_rel_amount_confirm_15",
    )


def llm_refined_shadow_amt_ema_15_th0015(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_75
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.015, mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 15))",
        factor_name="llm_refined.shadow_amt_ema_15_th0015",
    )


def llm_refined_shadow_count_amt_ema_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_73
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(mul(where(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, 1, 0), amount), 15))",
        factor_name="llm_refined.shadow_count_amt_ema_15",
    )


def llm_refined_llmgen_shadow_turnover_conf_15(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_74
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ema(where(and(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, gt(turnover, ts_mean(turnover, 20))), mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 15))",
        factor_name="llm_refined.llmgen_shadow_turnover_conf_15",
    )


def llm_refined_shadow_wma_smooth_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.weighted_upper_shadow_frequency_40_hl10
    round: llm_refine_round_75
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(wma(where(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)) > 0.01, mul(div(sub(high, rowmax(open, close)), add(pre_close, 1e-12)), amount), 0), 20))",
        factor_name="llm_refined.shadow_wma_smooth_20",
    )


FACTOR_SPECS = (
    FactorSpec(
        name="llm_refined.volume_weighted_shadow_10",
        func=llm_refined_volume_weighted_shadow_10,
        required_fields=("high", "close", "pre_close", "volume"),
        notes="Round1 strong winner; base continuous upper-shadow intensity weighted by volume.",
    ),
    FactorSpec(
        name="llm_refined.shadow_turnover_weighted",
        func=llm_refined_shadow_turnover_weighted,
        required_fields=("high", "close", "pre_close", "turnover"),
        notes="Turnover-weighted shadow intensity variant kept for motif coverage.",
    ),
    FactorSpec(
        name="llm_refined.shadow_turnover_filtered",
        func=llm_refined_shadow_turnover_filtered,
        required_fields=("high", "close", "pre_close", "turnover"),
        notes="Low-turnover winner variant using turnover filter; good focused continuation branch.",
    ),
    FactorSpec(
        name="llm_refined.shadow_rank_threshold",
        func=llm_refined_shadow_rank_threshold,
        required_fields=("high", "close", "pre_close"),
        notes="Cross-sectional rank-threshold branch; useful semantic contrast to weighted variants.",
    ),
    FactorSpec(
        name="llm_refined.amt_weighted_shadow_10",
        func=llm_refined_amt_weighted_shadow_10,
        required_fields=("high", "close", "pre_close", "amount"),
        notes="Current strongest first-wave candidate from the family run.",
    ),
    FactorSpec(
        name="llm_refined.vol_amt_shadow_15",
        func=llm_refined_vol_amt_shadow_15,
        required_fields=("high", "close", "pre_close", "amount"),
        notes="Strong amount-weighted extension with volatility thresholding and longer smoothing.",
    ),
    FactorSpec(
        name="llm_refined.amt_shadow_turnover_confirm",
        func=llm_refined_amt_shadow_turnover_confirm,
        required_fields=("high", "close", "pre_close", "amount", "turnover", "volume"),
        notes="Amount-weighted turnover-confirmed branch preserved as a higher-conviction confirmation variant.",
    ),
    FactorSpec(
        name="llm_refined.upper_body_reject_amt_10",
        func=llm_refined_upper_body_reject_amt_10,
        required_fields=("amount", "close", "high", "open", "pre_close"),
        notes="Focused winner from amt_weighted_shadow_10; clearer upper-body rejection semantics.",
    ),
    FactorSpec(
        name="llm_refined.shadow_amt_ema_15",
        func=llm_refined_shadow_amt_ema_15,
        required_fields=("amount", "close", "high", "open", "pre_close"),
        notes="15-day EMA continuation of the upper-body rejection amount line from the Path Eval v2 quality branch.",
    ),
    FactorSpec(
        name="llm_refined.shadow_length_weighted_10",
        func=llm_refined_shadow_length_weighted_10,
        required_fields=("amount", "close", "high", "pre_close"),
        notes="Adds explicit shadow-length weighting on top of the amount-weighted shadow branch.",
    ),
    FactorSpec(
        name="llm_refined.wm_half_life_shadow_10",
        func=llm_refined_wm_half_life_shadow_10,
        required_fields=("amount", "close", "high", "pre_close"),
        notes="Weighted-mean half-life smoothing variant of the amount-weighted shadow line.",
    ),
    FactorSpec(
        name="llm_refined.turnover_conf_shadow_ema10",
        func=llm_refined_turnover_conf_shadow_ema10,
        required_fields=("close", "high", "pre_close", "turnover"),
        notes="Turnover-confirmed EMA shadow branch with lighter confirmation structure.",
    ),
    FactorSpec(
        name="llm_refined.amt_turn_confirm_shadow_10",
        func=llm_refined_amt_turn_confirm_shadow_10,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="Amount-times-turnover confirmed shadow signal from the 10-day main branch.",
    ),
    FactorSpec(
        name="llm_refined.shadow_amt_turnover_confirm_15",
        func=llm_refined_shadow_amt_turnover_confirm_15,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="15-day branch combining amount weighting with turnover confirmation.",
    ),
    FactorSpec(
        name="llm_refined.relative_shadow_length_15",
        func=llm_refined_relative_shadow_length_15,
        required_fields=("amount", "close", "high", "low", "pre_close"),
        notes="Relative shadow-length normalization branch from vol_amt_shadow_15.",
    ),
    FactorSpec(
        name="llm_refined.shadow_range_norm_15",
        func=llm_refined_shadow_range_norm_15,
        required_fields=("amount", "close", "high", "low", "pre_close"),
        notes="Range-normalized 15-day amount-weighted shadow variant.",
    ),
    FactorSpec(
        name="llm_refined.turnover_shadow_confirm_20",
        func=llm_refined_turnover_shadow_confirm_20,
        required_fields=("close", "high", "pre_close", "turnover"),
        notes="Longer-window turnover-confirmed shadow branch using a stricter normalized trigger.",
    ),
    FactorSpec(
        name="llm_refined.amt_turn_confirm_15",
        func=llm_refined_amt_turn_confirm_15,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="15-day amount-plus-turnover confirmed shadow branch with dual condition gating.",
    ),
    FactorSpec(
        name="llm_refined.shadow_turn_confirm_ema",
        func=llm_refined_shadow_turn_confirm_ema,
        required_fields=("amount", "close", "high", "open", "pre_close", "turnover"),
        notes="Upper-body rejection branch augmented with relative-turnover confirmation and EMA smoothing.",
    ),
    FactorSpec(
        name="llm_refined.vol_scaled_shadow_amt_15",
        func=llm_refined_vol_scaled_shadow_amt_15,
        required_fields=("amount", "close", "high", "open", "pre_close", "returns"),
        notes="Volatility-scaled upper-body rejection variant with strong low-turnover profile.",
    ),
    FactorSpec(
        name="llm_refined.shadow_vol_scaled_15",
        func=llm_refined_shadow_vol_scaled_15,
        required_fields=("amount", "close", "high", "open"),
        notes="Std-normalized upper-body rejection branch from Path Eval v2 with strong excess and Sharpe.",
    ),
    FactorSpec(
        name="llm_refined.shadow_decay_amt_20",
        func=llm_refined_shadow_decay_amt_20,
        required_fields=("amount", "close", "high", "open", "pre_close"),
        notes="Decay-linear amount-weighted rejection branch extending the quality line to a 20-day memory.",
    ),
    FactorSpec(
        name="llm_refined.decay_shadow_turn_15",
        func=llm_refined_decay_shadow_turn_15,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="Turnover-confirmation branch rewritten with decay-linear smoothing.",
    ),
    FactorSpec(
        name="llm_refined.shadow_pos_confirm_15",
        func=llm_refined_shadow_pos_confirm_15,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="Position-confirmed turnover shadow winner; current strongest formalized confirmation variant.",
    ),
    FactorSpec(
        name="llm_refined.shadow_amt_relturn_15",
        func=llm_refined_shadow_amt_relturn_15,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="Relative-turnover confirmed amount shadow branch with strong risk-adjusted performance.",
    ),
    FactorSpec(
        name="llm_refined.llmgen_shadow_vol_rel_15",
        func=llm_refined_llmgen_shadow_vol_rel_15,
        required_fields=("amount", "close", "high", "pre_close"),
        notes="Relative-amount confirmed shadow branch; current best-node representative from the dual-parent scheduler.",
    ),
    FactorSpec(
        name="llm_refined.shadow_wm_half_life_40",
        func=llm_refined_shadow_wm_half_life_40,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="Longer-horizon weighted-mean half-life continuation of the turnover-confirmed amount branch.",
    ),
    FactorSpec(
        name="llm_refined.llmgen_shadow_std40_15",
        func=llm_refined_llmgen_shadow_std40_15,
        required_fields=("amount", "close", "high", "pre_close", "turnover"),
        notes="Std40-normalized turnover-confirmed shadow branch; strongest new winner from the Path Eval v2 expandability line.",
    ),
    FactorSpec(
        name="llm_refined.shadow_turnover_conf_15",
        func=llm_refined_shadow_turnover_conf_15,
        required_fields=("amount", "close", "high", "open", "pre_close", "turnover"),
        notes="Auto-promoted pending candidate for weighted_upper_shadow_distribution; run_id=run::weighted_upper_shadow_distribution::r73::0de6210ee502; round_id=73; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.10233833423147573, NetAnn=3.4313677848628252, Turnover=0.16198388687213008.",
    ),
    FactorSpec(
        name="llm_refined.shadow_rel_amount_confirm_15",
        func=llm_refined_shadow_rel_amount_confirm_15,
        required_fields=("amount", "close", "high", "open", "pre_close"),
        notes="Auto-promoted pending candidate for weighted_upper_shadow_distribution; run_id=run::weighted_upper_shadow_distribution::r74::28f640b8f664; round_id=74; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.10123962377061183, NetAnn=3.860856176764008, Turnover=0.1510145673405704.",
    ),
    FactorSpec(
        name="llm_refined.shadow_amt_ema_15_th0015",
        func=llm_refined_shadow_amt_ema_15_th0015,
        required_fields=("amount", "close", "high", "open", "pre_close"),
        notes="Auto-promoted pending candidate for weighted_upper_shadow_distribution; run_id=run::weighted_upper_shadow_distribution::r75::a3ef8a31d45e; round_id=75; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.10194587412178703, NetAnn=3.1818906843755004, Turnover=0.13741409967868176.",
    ),
    FactorSpec(
        name="llm_refined.shadow_count_amt_ema_15",
        func=llm_refined_shadow_count_amt_ema_15,
        required_fields=("amount", "close", "high", "open", "pre_close"),
        notes="Auto-promoted pending candidate for weighted_upper_shadow_distribution; run_id=run::weighted_upper_shadow_distribution::r73::e94b6c8f1a2c; round_id=73; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.10103079950189439, NetAnn=3.5224564956497746, Turnover=0.15479334731676234.",
    ),
    FactorSpec(
        name="llm_refined.llmgen_shadow_turnover_conf_15",
        func=llm_refined_llmgen_shadow_turnover_conf_15,
        required_fields=("amount", "close", "high", "open", "pre_close", "turnover"),
        notes="Auto-promoted pending candidate for weighted_upper_shadow_distribution; run_id=run::weighted_upper_shadow_distribution::r74::d4a6011d1dd2; round_id=74; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.10233833423147573, NetAnn=3.4313677848628252, Turnover=0.16198388687213008.",
    ),
    FactorSpec(
        name="llm_refined.shadow_wma_smooth_20",
        func=llm_refined_shadow_wma_smooth_20,
        required_fields=("amount", "close", "high", "open", "pre_close"),
        notes="Auto-promoted pending candidate for weighted_upper_shadow_distribution; run_id=run::weighted_upper_shadow_distribution::r75::904e63009c21; round_id=75; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.10199799341271538, NetAnn=3.305682497107961, Turnover=0.14068769270979184.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_volume_weighted_shadow_10",
    "llm_refined_shadow_turnover_weighted",
    "llm_refined_shadow_turnover_filtered",
    "llm_refined_shadow_rank_threshold",
    "llm_refined_amt_weighted_shadow_10",
    "llm_refined_vol_amt_shadow_15",
    "llm_refined_amt_shadow_turnover_confirm",
    "llm_refined_upper_body_reject_amt_10",
    "llm_refined_shadow_amt_ema_15",
    "llm_refined_shadow_length_weighted_10",
    "llm_refined_wm_half_life_shadow_10",
    "llm_refined_turnover_conf_shadow_ema10",
    "llm_refined_amt_turn_confirm_shadow_10",
    "llm_refined_shadow_amt_turnover_confirm_15",
    "llm_refined_relative_shadow_length_15",
    "llm_refined_shadow_range_norm_15",
    "llm_refined_turnover_shadow_confirm_20",
    "llm_refined_amt_turn_confirm_15",
    "llm_refined_shadow_turn_confirm_ema",
    "llm_refined_vol_scaled_shadow_amt_15",
    "llm_refined_shadow_vol_scaled_15",
    "llm_refined_shadow_decay_amt_20",
    "llm_refined_decay_shadow_turn_15",
    "llm_refined_shadow_pos_confirm_15",
    "llm_refined_shadow_amt_relturn_15",
    "llm_refined_llmgen_shadow_vol_rel_15",
    "llm_refined_shadow_wm_half_life_40",
    "llm_refined_llmgen_shadow_std40_15",
    "llm_refined_shadow_turnover_conf_15",
    "llm_refined_shadow_rel_amount_confirm_15",
    "llm_refined_shadow_amt_ema_15_th0015",
    "llm_refined_shadow_count_amt_ema_15",
    "llm_refined_llmgen_shadow_turnover_conf_15",
    "llm_refined_shadow_wma_smooth_20",
]
