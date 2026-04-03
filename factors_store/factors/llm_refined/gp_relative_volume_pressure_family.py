from __future__ import annotations

"""Auto-promoted LLM refined candidates for the gp_mined.volume_mean30_over_volume family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "gp_mined.volume_mean30_over_volume"
FAMILY_KEY = "gp_relative_volume_pressure_family"
SEED_FAMILY = "gp_relative_volume_pressure"
SUMMARY_GLOB = "llm_refined_gp_relative_volume_pressure_family_summary_*.csv"


def llm_refined_volume_mean30_over_ema_volume10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_1
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(volume, 30), add(ema(volume, 10), 1e-12))",
        factor_name="llm_refined.volume_mean30_over_ema_volume10",
    )



def llm_refined_amount_vol_ratio_with_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(mul(amount, add(ts_std(amount, 30), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_vol_ratio_with_std",
    )


def llm_refined_gp_volume_mean40_over_volume(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_2
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(volume, 40), add(volume, 1e-12))",
        factor_name="llm_refined.gp_volume_mean40_over_volume",
    )


def llm_refined_ema_vol30_over_vol(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ema(volume, 30), add(volume, 1e-12))",
        factor_name="llm_refined.ema_vol30_over_vol",
    )


def llm_refined_turnover_mean30_over_turnover_std_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_8
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(turnover, 30), add(mul(turnover, add(ts_std(turnover, 30), 1e-12)), 1e-12))",
        factor_name="llm_refined.turnover_mean30_over_turnover_std_norm",
    )


def llm_refined_amount_vol_ratio_smooth_denom(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_7
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(mul(ts_mean(amount, 5), add(ts_std(amount, 30), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_vol_ratio_smooth_denom",
    )


def llm_refined_amount_mean30_over_amount_std_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_38
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 28), add(mul(amount, add(ts_std(amount, 28), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_mean30_over_amount_std_norm",
    )


def llm_refined_turnover_mean30_over_rel_volume_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_38
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(turnover, 30), add(mul(rel_volume(20), add(ts_std(turnover, 30), 1e-12)), 1e-12))",
        factor_name="llm_refined.turnover_mean30_over_rel_volume_std",
    )


def llm_refined_amount_calm_std_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_41
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(ts_std(amount, 30), 1e-12))",
        factor_name="llm_refined.amount_calm_std_norm",
    )


def llm_refined_amount_mean60_rel_vol_ratio(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_41
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 60), add(mul(amount, add(ts_std(amount, 60), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_mean60_rel_vol_ratio",
    )


def llm_refined_amount_mean20_over_amount_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_39
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 20), add(ts_std(amount, 20), 1e-12))",
        factor_name="llm_refined.amount_mean20_over_amount_std",
    )


def llm_refined_amt_mean28_over_ranked_std_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_44
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 28), add(mul(amount, add(ts_rank(ts_std(amount, 28), 60), 1e-12)), 1e-12))",
        factor_name="llm_refined.amt_mean28_over_ranked_std_norm",
    )


def llm_refined_amount_mean28_over_ema_amount_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_42
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 28), add(mul(ema(amount, 5), add(ts_std(amount, 28), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_mean28_over_ema_amount_std",
    )


def llm_refined_amt_mean30_over_relamt_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_46
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(mul(rel_amount(20), add(ts_std(amount, 30), 1e-12)), 1e-12))",
        factor_name="llm_refined.amt_mean30_over_relamt_std",
    )


def llm_refined_gp_amount_mean30_over_std60_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_48
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(mul(amount, add(ts_std(amount, 60), 1e-12)), 1e-12))",
        factor_name="llm_refined.gp_amount_mean30_over_std60_norm",
    )


def llm_refined_amount_mean30_over_amount_std_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_43
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(ts_std(amount, 30), mul(amount, 0.1)))",
        factor_name="llm_refined.amount_mean30_over_amount_std_smooth",
    )


def llm_refined_amount_mean30_over_volatility_weighted_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_47
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(mul(ts_std(amount, 30), ts_std(returns, 30)), 1e-12))",
        factor_name="llm_refined.amount_mean30_over_volatility_weighted_std",
    )


def llm_refined_amount_shrink_share_60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_45
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_sum(greater(sub(delay(amount, 1), amount), 0), 60), add(ts_sum(abs(delta(amount, 1)), 60), 1e-12))",
        factor_name="llm_refined.amount_shrink_share_60",
    )


def llm_refined_amt_mean30_over_ema5_rkstd30(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_52
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), add(mul(ema(amount, 5), add(rank(ts_std(amount, 30)), 1e-12)), 1e-12))",
        factor_name="llm_refined.amt_mean30_over_ema5_rkstd30",
    )


def llm_refined_relamt30_over_relamt5_std30(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_50
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(rel_amount(30), 30), add(mul(ema(rel_amount(30), 5), add(ts_std(amount, 30), 1e-12)), 1e-12))",
        factor_name="llm_refined.relamt30_over_relamt5_std30",
    )


def llm_refined_amt_mean30_volfilter_std20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_51
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 30), mul(amount, add(ts_std(volume, 20), 1e-12)))",
        factor_name="llm_refined.amt_mean30_volfilter_std20",
    )


def llm_refined_amount_mean28_over_vol_adj_ema(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_55
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 28), add(mul(ema(amount, 5), add(ts_std(amount, 20), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_mean28_over_vol_adj_ema",
    )


def llm_refined_amount_mean28_over_rel_amount5_std(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_54
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 28), add(mul(ema(rel_amount(5), 5), add(ts_std(amount, 28), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_mean28_over_rel_amount5_std",
    )


def llm_refined_amount_mean60_over_ema_std_amount(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_57
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12))",
        factor_name="llm_refined.amount_mean60_over_ema_std_amount",
    )


def llm_refined_amt_mean60_over_ema5_std60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_56
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 60), add(mul(ema(amount, 5), add(ts_std(amount, 60), 1e-12)), 1e-12))",
        factor_name="llm_refined.amt_mean60_over_ema5_std60",
    )


def llm_refined_amount_pressure_with_turnover_level(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_58
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)), div(ts_mean(turnover, 20), add(turnover, 1e-12)))",
        factor_name="llm_refined.amount_pressure_with_turnover_level",
    )


def llm_refined_cs_rank_amt_pressure(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_60
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="cs_rank(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)))",
        factor_name="llm_refined.cs_rank_amt_pressure",
    )


def llm_refined_amount_mean60_over_decayema_amount(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_61
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(amount, 60), add(decay_linear(amount, 20), 1e-12))",
        factor_name="llm_refined.amount_mean60_over_decayema_amount",
    )


def llm_refined_amount_mean60_over_ema10_std60_down_share20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_59
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)), div(ts_sum(greater(sub(delay(amount, 1), amount), 0), 20), add(ts_sum(abs(sub(amount, delay(amount, 1))), 20), 1e-12)))",
        factor_name="llm_refined.amount_mean60_over_ema10_std60_down_share20",
    )


def llm_refined_amount_pressure_smooth_turnover(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_65
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)), div(ts_mean(turnover, 20), add(ts_mean(turnover, 5), 1e-12)))",
        factor_name="llm_refined.amount_pressure_smooth_turnover",
    )


def llm_refined_amount_pressure_rel_turnover(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_62
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)), div(ts_mean(turnover, 60), add(turnover, 1e-12)))",
        factor_name="llm_refined.amount_pressure_rel_turnover",
    )


def llm_refined_amt_press_turnover_ema_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_64
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)), div(ema(turnover, 20), add(turnover, 1e-12)))",
        factor_name="llm_refined.amt_press_turnover_ema_smooth",
    )


def llm_refined_opt_shrink_share_blend(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_63
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 40), add(mul(ema(amount, 15), add(ts_std(amount, 40), 1e-12)), 1e-12)), div(ts_sum(greater(sub(delay(turnover, 1), turnover), 0), 20), add(ts_sum(abs(sub(turnover, delay(turnover, 1))), 20), 1e-12)))",
        factor_name="llm_refined.opt_shrink_share_blend",
    )


def llm_refined_amount_pressure_ema_turnover(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_66
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)), div(ts_mean(turnover, 60), add(ema(turnover, 5), 1e-12)))",
        factor_name="llm_refined.amount_pressure_ema_turnover",
    )


def llm_refined_amount_pressure_decay(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_69
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(decay_linear(amount, 10), 1e-12)), div(ts_mean(turnover, 60), add(turnover, 1e-12)))",
        factor_name="llm_refined.amount_pressure_decay",
    )


def llm_refined_pressure_size_neutral(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_67
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(mul(div(ts_mean(amount, 60), add(mul(ema(amount, 10), add(ts_std(amount, 60), 1e-12)), 1e-12)), div(ts_mean(turnover, 60), add(turnover, 1e-12))), div(1, cap))",
        factor_name="llm_refined.pressure_size_neutral",
    )


def llm_refined_amount_turnover_trend_pressure(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_77
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ts_mean(amount, 10), 1e-12)), div(ts_mean(turnover, 20), add(ts_mean(turnover, 5), 1e-12)))",
        factor_name="llm_refined.amount_turnover_trend_pressure",
    )


def llm_refined_amount_pressure_with_vol_filter(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_74
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(mul(div(ts_mean(amount, 60), add(decay_linear(amount, 10), 1e-12)), div(ts_mean(turnover, 60), add(turnover, 1e-12))), div(ts_mean(ts_std(returns, 20), 60), add(ts_std(returns, 20), 1e-12)))",
        factor_name="llm_refined.amount_pressure_with_vol_filter",
    )


def llm_refined_press_with_vol_filter(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_75
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(decay_linear(amount, 10), 1e-12)), div(ts_mean(turnover, 60), add(where(gt(ts_std(turnover, 20), 0.1), turnover, ts_mean(turnover, 20)), 1e-12)))",
        factor_name="llm_refined.press_with_vol_filter",
    )


def llm_refined_amt_press_vol_adj(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_75
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(decay_linear(amount, 10), 1e-12)), div(ts_mean(turnover, 60), add(mul(turnover, ts_std(returns, 20)), 1e-12)))",
        factor_name="llm_refined.amt_press_vol_adj",
    )


def llm_refined_amount_pressure_shrink_share(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_74
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(decay_linear(amount, 10), 1e-12)), div(ts_sum(rowmax(sub(delay(turnover, 1), turnover), 0), 20), add(ts_sum(abs(sub(turnover, delay(turnover, 1))), 20), 1e-12)))",
        factor_name="llm_refined.amount_pressure_shrink_share",
    )


def llm_refined_amt_press_turnover_120(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_76
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(decay_linear(amount, 10), 1e-12)), div(ts_mean(turnover, 120), add(turnover, 1e-12)))",
        factor_name="llm_refined.amt_press_turnover_120",
    )


def llm_refined_amt_press_peak_rel(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_76
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ts_max(amount, 10), 1e-12)), div(ts_mean(turnover, 60), add(turnover, 1e-12)))",
        factor_name="llm_refined.amt_press_peak_rel",
    )


def llm_refined_amt_press_vol_filter(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_77
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(amount, 1e-12)), div(ts_std(amount, 60), add(ts_std(amount, 20), 1e-12)))",
        factor_name="llm_refined.amt_press_vol_filter",
    )


def llm_refined_amt_press_turn_mean10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_84
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ts_mean(amount, 10), 1e-12)), div(ts_mean(turnover, 120), add(turnover, 1e-12)))",
        factor_name="llm_refined.amt_press_turn_mean10",
    )


def llm_refined_amt_press_turn_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_82
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ema(amount, 10), 1e-12)), div(ts_mean(turnover, 120), add(ema(turnover, 10), 1e-12)))",
        factor_name="llm_refined.amt_press_turn_smooth",
    )


def llm_refined_amt_press_std_rel(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_81
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ts_std(amount, 60), 1e-12)), div(ts_mean(turnover, 60), add(turnover, 1e-12)))",
        factor_name="llm_refined.amt_press_std_rel",
    )


def llm_refined_amt_press_peak_rel_relamt(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_78
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ts_max(amount, 10), 1e-12)), div(1, add(rel_amount(20), 1e-12)))",
        factor_name="llm_refined.amt_press_peak_rel_relamt",
    )


def llm_refined_amt_press_ema_denom_stable(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_85
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ema(amount, 10), 1e-12)), div(ts_mean(turnover, 120), add(ema(turnover, 20), 1e-12)))",
        factor_name="llm_refined.amt_press_ema_denom_stable",
    )


def llm_refined_amt_press_ema_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_83
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ema(amount, 28), add(ema(amount, 10), 1e-12)), div(ema(turnover, 60), add(turnover, 1e-12)))",
        factor_name="llm_refined.amt_press_ema_smooth",
    )


def llm_refined_amt_press_std_norm_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_80
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(div(ts_mean(amount, 60), add(ts_std(amount, 10), 1e-12)), div(ts_mean(turnover, 60), add(turnover, 1e-12)))",
        factor_name="llm_refined.amt_press_std_norm_10",
    )


def llm_refined_amt_press_ema_turn_stable(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_110
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_std(amount, 20), 1e-12))), div(ts_mean(turnover, 120), add(ema(turnover, 40), 1e-12)))",
        factor_name="llm_refined.amt_press_ema_turn_stable",
    )


def llm_refined_amt_press_ema_turn_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_112
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_std(amount, 20), 1e-12))), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)))",
        factor_name="llm_refined.amt_press_ema_turn_smooth",
    )


def llm_refined_amt_press_relamt_vol_blend(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_113
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_std(amount, 20), 1e-12))), div(ts_mean(div(volume, add(ts_mean(volume, 60), 1e-12)), 20), add(ts_std(div(volume, add(ts_mean(volume, 60), 1e-12)), 20), 1e-12)))",
        factor_name="llm_refined.amt_press_relamt_vol_blend",
    )


def llm_refined_amt_press_amt_smooth_turn_ratio(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_114
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ema(amount, 20), add(ts_std(amount, 20), 1e-12))), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)))",
        factor_name="llm_refined.amt_press_amt_smooth_turn_ratio",
    )


def llm_refined_amt_press_volatility_filter(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_115
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_std(amount, 20), 1e-12))), where(lt(ts_std(returns, 20), ts_mean(ts_std(returns, 20), 60)), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)), 1))",
        factor_name="llm_refined.amt_press_volatility_filter",
    )


def llm_refined_amt_press_ema_turn_smooth_std40(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_116
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_std(amount, 40), 1e-12))), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)))",
        factor_name="llm_refined.amt_press_ema_turn_smooth_std40",
    )


def llm_refined_amt_press_ema_turn_smooth_std60(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_128
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_std(amount, 60), 1e-12))), div(ema(turnover, 120), add(ema(turnover, 28), 1e-12)))",
        factor_name="llm_refined.amt_press_ema_turn_smooth_std60",
    )


def llm_refined_amt_press_relamt_turnstd(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_127
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_mean(amount, 20), 1e-12))), div(ema(turnover, 120), add(ts_std(turnover, 20), 1e-12)))",
        factor_name="llm_refined.amt_press_relamt_turnstd",
    )


def llm_refined_amt_press_relamt_turn_slowfast(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_127
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_std(amount, 40), 1e-12))), div(ema(turnover, 180), add(ema(turnover, 28), 1e-12)))",
        factor_name="llm_refined.amt_press_relamt_turn_slowfast",
    )


def llm_refined_amt_press_ema_turn_mad_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_129
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(cs_rank(div(ts_mean(amount, 60), add(ts_mad(amount, 40), 1e-12))), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)))",
        factor_name="llm_refined.amt_press_ema_turn_mad_smooth",
    )


def llm_refined_vol_press_tsrank_turn_ema(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.volume_mean30_over_volume
    round: llm_refine_round_129
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(ts_rank(div(ts_mean(volume, 60), add(ts_std(volume, 40), 1e-12)), 250), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)))",
        factor_name="llm_refined.vol_press_tsrank_turn_ema",
    )


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.volume_mean30_over_ema_volume10",
        func=llm_refined_volume_mean30_over_ema_volume10,
        required_fields=("volume",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r1::156eec14023a; round_id=1; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.03421402932795017, NetAnn=0.4825189808151813, Turnover=0.2560076806705744.",
    ),
    FactorSpec(
        name="llm_refined.amount_vol_ratio_with_std",
        func=llm_refined_amount_vol_ratio_with_std,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r4::b4e83ab3bf33; round_id=4; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.1046988747567939, NetAnn=7.721055391628651, Turnover=0.32670272743202244.",
    ),
    FactorSpec(
        name="llm_refined.gp_volume_mean40_over_volume",
        func=llm_refined_gp_volume_mean40_over_volume,
        required_fields=("volume",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r2::2f06840fa25c; round_id=2; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.04116886666947773, NetAnn=0.5380980118282268, Turnover=0.7848041350104276.",
    ),
    FactorSpec(
        name="llm_refined.ema_vol30_over_vol",
        func=llm_refined_ema_vol30_over_vol,
        required_fields=("volume",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r3::73801ab850cb; round_id=3; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.03924417999772626, NetAnn=0.5578636438668423, Turnover=0.927981142775747.",
    ),
    FactorSpec(
        name="llm_refined.turnover_mean30_over_turnover_std_norm",
        func=llm_refined_turnover_mean30_over_turnover_std_norm,
        required_fields=("turnover",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r8::57ec85ffaf91; round_id=8; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07750259228254869, NetAnn=1.3375135741802961, Turnover=0.3754065082762334.",
    ),
    FactorSpec(
        name="llm_refined.amount_vol_ratio_smooth_denom",
        func=llm_refined_amount_vol_ratio_smooth_denom,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r7::1e8c2dc26325; round_id=7; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.10152290282412545, NetAnn=7.849228924824194, Turnover=0.11972996925621399.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean30_over_amount_std_norm",
        func=llm_refined_amount_mean30_over_amount_std_norm,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r38::255fb4c33e25; round_id=38; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.10870812683757958, NetAnn=6.171023429677653, Turnover=0.3252511600320905.",
    ),
    FactorSpec(
        name="llm_refined.turnover_mean30_over_rel_volume_std",
        func=llm_refined_turnover_mean30_over_rel_volume_std,
        required_fields=("turnover", "volume"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r38::d94c6926018b; round_id=38; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.06058262741424183, NetAnn=1.5239175975449024, Turnover=0.7456974907976894.",
    ),
    FactorSpec(
        name="llm_refined.amount_calm_std_norm",
        func=llm_refined_amount_calm_std_norm,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r41::6a5c213b6188; round_id=41; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.05308065723239442, NetAnn=1.5949688286598174, Turnover=0.14995501230880168.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean60_rel_vol_ratio",
        func=llm_refined_amount_mean60_rel_vol_ratio,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r41::a4f2bd345b32; round_id=41; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.10851935244705667, NetAnn=6.273657902611091, Turnover=0.32576581808063765.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean20_over_amount_std",
        func=llm_refined_amount_mean20_over_amount_std,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r39::2637584a147c; round_id=39; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.058013215809085396, NetAnn=2.137245801380474, Turnover=0.2131215290377678.",
    ),
    FactorSpec(
        name="llm_refined.amt_mean28_over_ranked_std_norm",
        func=llm_refined_amt_mean28_over_ranked_std_norm,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r44::1a7275349539; round_id=44; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.06698475860927276, NetAnn=2.8020175627689032, Turnover=0.44938486520055393.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean28_over_ema_amount_std",
        func=llm_refined_amount_mean28_over_ema_amount_std,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r42::dff63c7505c6; round_id=42; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.10504076177328407, NetAnn=6.114276972313688, Turnover=0.13334730485857582.",
    ),
    FactorSpec(
        name="llm_refined.amt_mean30_over_relamt_std",
        func=llm_refined_amt_mean30_over_relamt_std,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r46::6af8604977e0; round_id=46; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.06792807773183311, NetAnn=2.0442537942228878, Turnover=0.7276606089018505.",
    ),
    FactorSpec(
        name="llm_refined.gp_amount_mean30_over_std60_norm",
        func=llm_refined_gp_amount_mean30_over_std60_norm,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r48::6a50a9c13fcd; round_id=48; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.10261368348344353, NetAnn=5.450764973842249, Turnover=0.3315703434183497.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean30_over_amount_std_smooth",
        func=llm_refined_amount_mean30_over_amount_std_smooth,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r43::3855d8f2bdc8; round_id=43; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.0625137283152609, NetAnn=1.995578264726967, Turnover=0.27765566227134875.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean30_over_volatility_weighted_std",
        func=llm_refined_amount_mean30_over_volatility_weighted_std,
        required_fields=("amount", "returns"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r47::f4f2dde2e993; round_id=47; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07243842534875718, NetAnn=1.1471201034400038, Turnover=0.11552728667642037.",
    ),
    FactorSpec(
        name="llm_refined.amount_shrink_share_60",
        func=llm_refined_amount_shrink_share_60,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r45::10a87f2801e1; round_id=45; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07195737952105602, NetAnn=3.101203804231435, Turnover=0.6639487761904898.",
    ),
    FactorSpec(
        name="llm_refined.amt_mean30_over_ema5_rkstd30",
        func=llm_refined_amt_mean30_over_ema5_rkstd30,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r52::6e71a27f2e66; round_id=52; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.1083074698354466, NetAnn=6.208575718833015, Turnover=0.17724498526549182.",
    ),
    FactorSpec(
        name="llm_refined.relamt30_over_relamt5_std30",
        func=llm_refined_relamt30_over_relamt5_std30,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r50::e5cefc598bcc; round_id=50; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.10244998287407124, NetAnn=5.985831450800247, Turnover=0.13735104620546396.",
    ),
    FactorSpec(
        name="llm_refined.amt_mean30_volfilter_std20",
        func=llm_refined_amt_mean30_volfilter_std20,
        required_fields=("amount", "volume"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r51::c676b060ac4d; round_id=51; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07285549866885224, NetAnn=3.1842924626187976, Turnover=0.2893772882428513.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean28_over_vol_adj_ema",
        func=llm_refined_amount_mean28_over_vol_adj_ema,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r55::c6f3da7ede13; round_id=55; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.1062952148410306, NetAnn=6.438102646032664, Turnover=0.1468596344501918.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean28_over_rel_amount5_std",
        func=llm_refined_amount_mean28_over_rel_amount5_std,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r54::268bce6d14e1; round_id=54; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.06217404481664983, NetAnn=1.980078019777908, Turnover=0.34320586553848204.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean60_over_ema_std_amount",
        func=llm_refined_amount_mean60_over_ema_std_amount,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r57::45eb2e7b41d7; round_id=57; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.10006835200917255, NetAnn=5.602781271645858, Turnover=0.08206953449549395.",
    ),
    FactorSpec(
        name="llm_refined.amt_mean60_over_ema5_std60",
        func=llm_refined_amt_mean60_over_ema5_std60,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r56::2c16f1c12351; round_id=56; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.10455850930989144, NetAnn=6.076032160226413, Turnover=0.1252256692643807.",
    ),
    FactorSpec(
        name="llm_refined.amount_pressure_with_turnover_level",
        func=llm_refined_amount_pressure_with_turnover_level,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r58::002398ffe501; round_id=58; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.10924973984563989, NetAnn=6.189571396405084, Turnover=0.34024505498945873.",
    ),
    FactorSpec(
        name="llm_refined.cs_rank_amt_pressure",
        func=llm_refined_cs_rank_amt_pressure,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r60::18c53b2c6bc6; round_id=60; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.10006835200917255, NetAnn=5.602781271645858, Turnover=0.08206953449549395.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean60_over_decayema_amount",
        func=llm_refined_amount_mean60_over_decayema_amount,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r61::5756e7d949ca; round_id=61; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.06576195038793382, NetAnn=3.0902119113126227, Turnover=0.12700361518814446.",
    ),
    FactorSpec(
        name="llm_refined.amount_mean60_over_ema10_std60_down_share20",
        func=llm_refined_amount_mean60_over_ema10_std60_down_share20,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r59::9eaa8c2c86b5; round_id=59; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.1055044967369684, NetAnn=6.380159662812933, Turnover=0.14732164372549011.",
    ),
    FactorSpec(
        name="llm_refined.amount_pressure_smooth_turnover",
        func=llm_refined_amount_pressure_smooth_turnover,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r65::7ca81092cecf; round_id=65; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.10489810404046743, NetAnn=6.116953748976816, Turnover=0.14919775086983345.",
    ),
    FactorSpec(
        name="llm_refined.amount_pressure_rel_turnover",
        func=llm_refined_amount_pressure_rel_turnover,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r62::c1a03c3e2cb8; round_id=62; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.11042702060210169, NetAnn=6.333312493969119, Turnover=0.32285494798589986.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_turnover_ema_smooth",
        func=llm_refined_amt_press_turnover_ema_smooth,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r64::4010b4ba93aa; round_id=64; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.10946999612327274, NetAnn=6.2615918765276675, Turnover=0.3341655181235185.",
    ),
    FactorSpec(
        name="llm_refined.opt_shrink_share_blend",
        func=llm_refined_opt_shrink_share_blend,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r63::fd5c443d60ea; round_id=63; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.10375729970868433, NetAnn=6.34615022821565, Turnover=0.1354946545663289.",
    ),
    FactorSpec(
        name="llm_refined.amount_pressure_ema_turnover",
        func=llm_refined_amount_pressure_ema_turnover,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r66::624b71a0f412; round_id=66; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.10753190909953896, NetAnn=6.302226832246898, Turnover=0.15992961266384514.",
    ),
    FactorSpec(
        name="llm_refined.amount_pressure_decay",
        func=llm_refined_amount_pressure_decay,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r69::4e8134a3f23a; round_id=69; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07285264549973519, NetAnn=2.9271401718660544, Turnover=0.4553042298478954.",
    ),
    FactorSpec(
        name="llm_refined.pressure_size_neutral",
        func=llm_refined_pressure_size_neutral,
        required_fields=("amount", "cap", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r67::fd1760c3d343; round_id=67; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07865076251169376, NetAnn=4.295653598587008, Turnover=0.17640660806687147.",
    ),
    FactorSpec(
        name="llm_refined.amount_turnover_trend_pressure",
        func=llm_refined_amount_turnover_trend_pressure,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r77::52767e16b9c4; round_id=77; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.06489407635842047, NetAnn=2.653293369182279, Turnover=0.24077484922581693.",
    ),
    FactorSpec(
        name="llm_refined.amount_pressure_with_vol_filter",
        func=llm_refined_amount_pressure_with_vol_filter,
        required_fields=("amount", "returns", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r74::648024b61812; round_id=74; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.073106631471532, NetAnn=3.0728978494131125, Turnover=0.39337001910772174.",
    ),
    FactorSpec(
        name="llm_refined.press_with_vol_filter",
        func=llm_refined_press_with_vol_filter,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r75::ca307417bcdd; round_id=75; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.06581263507602193, NetAnn=2.903541208668195, Turnover=0.15248512927869012.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_vol_adj",
        func=llm_refined_amt_press_vol_adj,
        required_fields=("amount", "returns", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r75::8365d42c41ea; round_id=75; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.08767761523101139, NetAnn=2.549153548205643, Turnover=0.4037714948283846.",
    ),
    FactorSpec(
        name="llm_refined.amount_pressure_shrink_share",
        func=llm_refined_amount_pressure_shrink_share,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r74::771043191b95; round_id=74; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.07121078312596925, NetAnn=3.177454896749067, Turnover=0.29324865128558764.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_turnover_120",
        func=llm_refined_amt_press_turnover_120,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r76::9ee87922bc19; round_id=76; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07995976394776746, NetAnn=3.0589485730291033, Turnover=0.44328218913477857.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_peak_rel",
        func=llm_refined_amt_press_peak_rel,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r76::ab157c6c3881; round_id=76; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07500889531926426, NetAnn=3.1148782509811337, Turnover=0.4372107238300755.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_vol_filter",
        func=llm_refined_amt_press_vol_filter,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r77::31cfb5165194; round_id=77; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07207982116647765, NetAnn=3.3643959888870105, Turnover=0.43340183618344397.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_turn_mean10",
        func=llm_refined_amt_press_turn_mean10,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r84::af9a9789767d; round_id=84; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07968438431537124, NetAnn=3.133497112415438, Turnover=0.43587453350888317.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_turn_smooth",
        func=llm_refined_amt_press_turn_smooth,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r82::d003c84bc10f; round_id=82; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.07576167627309757, NetAnn=3.106563399554177, Turnover=0.17982299393189283.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_std_rel",
        func=llm_refined_amt_press_std_rel,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r81::b67054fbd261; round_id=81; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07295548482805005, NetAnn=2.2613874070343765, Turnover=0.6548036042924686.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_peak_rel_relamt",
        func=llm_refined_amt_press_peak_rel_relamt,
        required_fields=("amount",),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r78::be5b0bad11c0; round_id=78; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.07473519367474506, NetAnn=3.100294392148374, Turnover=0.5346290390445138.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_ema_denom_stable",
        func=llm_refined_amt_press_ema_denom_stable,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r85::1b7a5e1f09a8; round_id=85; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07468459949104846, NetAnn=3.0267734840330442, Turnover=0.15507510884003242.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_ema_smooth",
        func=llm_refined_amt_press_ema_smooth,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r83::d660d84c6bfe; round_id=83; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07105583471662602, NetAnn=2.7172768913043157, Turnover=0.6394905778380048.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_std_norm_10",
        func=llm_refined_amt_press_std_norm_10,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r80::eae13028f749; round_id=80; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07843998897205759, NetAnn=3.2615363590902264, Turnover=0.41236619427855825.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_ema_turn_stable",
        func=llm_refined_amt_press_ema_turn_stable,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r110::8f93e775ac40; round_id=110; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.07287696109144035, NetAnn=3.1208517871293395, Turnover=0.12961892056027213.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_ema_turn_smooth",
        func=llm_refined_amt_press_ema_turn_smooth,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r112::4288e56512dd; round_id=112; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07350324208841179, NetAnn=3.5946630472313474, Turnover=0.1312800351826376.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_relamt_vol_blend",
        func=llm_refined_amt_press_relamt_vol_blend,
        required_fields=("amount", "volume"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r113::46cfb96fd2f4; round_id=113; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.06500864599769801, NetAnn=2.5585844155813984, Turnover=0.18021455696691127.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_amt_smooth_turn_ratio",
        func=llm_refined_amt_press_amt_smooth_turn_ratio,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r114::119c37da7d4d; round_id=114; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.06250307098239846, NetAnn=2.5021686847205484, Turnover=0.17404431148160895.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_volatility_filter",
        func=llm_refined_amt_press_volatility_filter,
        required_fields=("amount", "turnover", "returns"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r115::bad80a045b27; round_id=115; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07162556785110707, NetAnn=3.3435100280557792, Turnover=0.1480528371656674.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_ema_turn_smooth_std40",
        func=llm_refined_amt_press_ema_turn_smooth_std40,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r116::7fc817b31a09; round_id=116; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.06703984219989191, NetAnn=2.938853671846892, Turnover=0.10425560625091383.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_ema_turn_smooth_std60",
        func=llm_refined_amt_press_ema_turn_smooth_std60,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r128::440005dc179f; round_id=128; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.0561043708147077, NetAnn=1.9808003427546694, Turnover=0.08573105468901078.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_relamt_turnstd",
        func=llm_refined_amt_press_relamt_turnstd,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r127::2ab9c57a3434; round_id=127; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.06775661689280829, NetAnn=2.9289456811524746, Turnover=0.131986631709471.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_relamt_turn_slowfast",
        func=llm_refined_amt_press_relamt_turn_slowfast,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r127::ba41913b864a; round_id=127; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.06718143081086274, NetAnn=3.025090446524292, Turnover=0.09523258335159927.",
    ),
    FactorSpec(
        name="llm_refined.amt_press_ema_turn_mad_smooth",
        func=llm_refined_amt_press_ema_turn_mad_smooth,
        required_fields=("amount", "turnover"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r129::d31fc64eb3e4; round_id=129; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07003922002304055, NetAnn=3.0104146795572566, Turnover=0.10827466670918749.",
    ),
    FactorSpec(
        name="llm_refined.vol_press_tsrank_turn_ema",
        func=llm_refined_vol_press_tsrank_turn_ema,
        required_fields=("turnover", "volume"),
        notes="Auto-promoted pending candidate for gp_relative_volume_pressure; run_id=run::gp_relative_volume_pressure::r129::aedbca43eb11; round_id=129; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07088607481475478, NetAnn=3.561305780216, Turnover=0.1286900995738562.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_volume_mean30_over_ema_volume10",
    "llm_refined_amount_vol_ratio_with_std",
    "llm_refined_gp_volume_mean40_over_volume",
    "llm_refined_ema_vol30_over_vol",
    "llm_refined_turnover_mean30_over_turnover_std_norm",
    "llm_refined_amount_vol_ratio_smooth_denom",
    "llm_refined_amount_mean30_over_amount_std_norm",
    "llm_refined_turnover_mean30_over_rel_volume_std",
    "llm_refined_amount_calm_std_norm",
    "llm_refined_amount_mean60_rel_vol_ratio",
    "llm_refined_amount_mean20_over_amount_std",
    "llm_refined_amt_mean28_over_ranked_std_norm",
    "llm_refined_amount_mean28_over_ema_amount_std",
    "llm_refined_amt_mean30_over_relamt_std",
    "llm_refined_gp_amount_mean30_over_std60_norm",
    "llm_refined_amount_mean30_over_amount_std_smooth",
    "llm_refined_amount_mean30_over_volatility_weighted_std",
    "llm_refined_amount_shrink_share_60",
    "llm_refined_amt_mean30_over_ema5_rkstd30",
    "llm_refined_relamt30_over_relamt5_std30",
    "llm_refined_amt_mean30_volfilter_std20",
    "llm_refined_amount_mean28_over_vol_adj_ema",
    "llm_refined_amount_mean28_over_rel_amount5_std",
    "llm_refined_amount_mean60_over_ema_std_amount",
    "llm_refined_amt_mean60_over_ema5_std60",
    "llm_refined_amount_pressure_with_turnover_level",
    "llm_refined_cs_rank_amt_pressure",
    "llm_refined_amount_mean60_over_decayema_amount",
    "llm_refined_amount_mean60_over_ema10_std60_down_share20",
    "llm_refined_amount_pressure_smooth_turnover",
    "llm_refined_amount_pressure_rel_turnover",
    "llm_refined_amt_press_turnover_ema_smooth",
    "llm_refined_opt_shrink_share_blend",
    "llm_refined_amount_pressure_ema_turnover",
    "llm_refined_amount_pressure_decay",
    "llm_refined_pressure_size_neutral",
    "llm_refined_amount_turnover_trend_pressure",
    "llm_refined_amount_pressure_with_vol_filter",
    "llm_refined_press_with_vol_filter",
    "llm_refined_amt_press_vol_adj",
    "llm_refined_amount_pressure_shrink_share",
    "llm_refined_amt_press_turnover_120",
    "llm_refined_amt_press_peak_rel",
    "llm_refined_amt_press_vol_filter",
    "llm_refined_amt_press_turn_mean10",
    "llm_refined_amt_press_turn_smooth",
    "llm_refined_amt_press_std_rel",
    "llm_refined_amt_press_peak_rel_relamt",
    "llm_refined_amt_press_ema_denom_stable",
    "llm_refined_amt_press_ema_smooth",
    "llm_refined_amt_press_std_norm_10",
    "llm_refined_amt_press_ema_turn_stable",
    "llm_refined_amt_press_ema_turn_smooth",
    "llm_refined_amt_press_relamt_vol_blend",
    "llm_refined_amt_press_amt_smooth_turn_ratio",
    "llm_refined_amt_press_volatility_filter",
    "llm_refined_amt_press_ema_turn_smooth_std40",
    "llm_refined_amt_press_ema_turn_smooth_std60",
    "llm_refined_amt_press_relamt_turnstd",
    "llm_refined_amt_press_relamt_turn_slowfast",
    "llm_refined_amt_press_ema_turn_mad_smooth",
    "llm_refined_vol_press_tsrank_turn_ema",
]
