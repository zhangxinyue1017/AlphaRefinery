from __future__ import annotations

"""Auto-promoted LLM refined candidates for the factor365.ideal_amplitude_20_q25 family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "factor365.ideal_amplitude_20_q25"
FAMILY_KEY = "ideal_amplitude_structure_family"
SEED_FAMILY = "ideal_amplitude_structure"
SUMMARY_GLOB = "llm_refined_ideal_amplitude_structure_family_summary_*.csv"


def llm_refined_zscore_amplitude_spread_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(zscore(ts_mean(where(close > delay(close, 1), high - low, 0), 20) - ts_mean(where(close < delay(close, 1), high - low, 0), 20)))",
        factor_name="llm_refined.zscore_amplitude_spread_20",
    )


def llm_refined_turnover_weighted_amplitude_contrast_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_mean(where(turnover > ts_mean(turnover, 20), high - low, 0), 20) / add(ts_mean(where(turnover <= ts_mean(turnover, 20), high - low, 0), 20) + 0.0001, 1e-12))",
        factor_name="llm_refined.turnover_weighted_amplitude_contrast_20",
    )


def llm_refined_vwap_state_amplitude_spread_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_mean(where(close > vwap, high - low, 0), 20) - ts_mean(where(close < vwap, high - low, 0), 20))",
        factor_name="llm_refined.vwap_state_amplitude_spread_20",
    )



def llm_refined_ideal_amp_smooth_28(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(sub(ts_mean(where(gt(div(sub(close, low), add(add(sub(high, low), 0.001), 1e-12)), 0.75), div(sub(high, low), add(pre_close, 1e-12)), 0), 28), ts_mean(where(lt(div(sub(close, low), add(add(sub(high, low), 0.001), 1e-12)), 0.25), div(sub(high, low), add(pre_close, 1e-12)), 0), 28)))",
        factor_name="llm_refined.ideal_amp_smooth_28",
    )


def llm_refined_factor_opt_vwap_state(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_2
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_sorted_mean_spread(high - low, vwap, 20, 0.25))",
        factor_name="llm_refined.factor_opt_vwap_state",
    )


def llm_refined_factor_opt_long_window(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_2
    source model: deepseek-v3.1
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_sorted_mean_spread(high - low, close, 40, 0.25))",
        factor_name="llm_refined.factor_opt_long_window",
    )


def llm_refined_ideal_amp_open_state(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_8
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(sub(ts_mean(where(gt(close, open), sub(high, low), 0), 20), ts_mean(where(lt(close, open), sub(high, low), 0), 20)))",
        factor_name="llm_refined.ideal_amp_open_state",
    )


def llm_refined_ideal_amp_vol_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_8
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(sub(ts_mean(where(and(gt(ts_rank(close, 20), 0.75), gt(turnover, ts_mean(turnover, 20))), sub(high, low), 0), 20), ts_mean(where(lt(ts_rank(close, 20), 0.25), sub(high, low), 0), 20)))",
        factor_name="llm_refined.ideal_amp_vol_confirm",
    )


def llm_refined_vwap_state_turn_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_16
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ts_sorted_mean_spread(high - low, vwap, 20, 0.25), ts_rank(turnover, 20)))",
        factor_name="llm_refined.vwap_state_turn_confirm",
    )


def llm_refined_amp_smooth_vwap_state(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_18
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_sorted_mean_spread(ts_mean(high - low, 3), vwap, 20, 0.25))",
        factor_name="llm_refined.amp_smooth_vwap_state",
    )


def llm_refined_volume_confirm_amplitude_q25(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: factor365.ideal_amplitude_20_q25
    round: llm_refine_round_20
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_sorted_mean_spread(high - low, vwap, 20, 0.25) * ts_rank(volume, 10))",
        factor_name="llm_refined.volume_confirm_amplitude_q25",
    )


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.zscore_amplitude_spread_20",
        func=llm_refined_zscore_amplitude_spread_20,
        required_fields=("close", "high", "low"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r4::23755bfe8c76; round_id=4; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.059333187807664986, NetAnn=2.4465146258538613, Turnover=0.3306132412847389.",
    ),
    FactorSpec(
        name="llm_refined.turnover_weighted_amplitude_contrast_20",
        func=llm_refined_turnover_weighted_amplitude_contrast_20,
        required_fields=("high", "low", "turnover"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r4::23755bfe8c76; round_id=4; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.052982609393580765, NetAnn=1.7076025619581943, Turnover=0.2094323137242066.",
    ),
    FactorSpec(
        name="llm_refined.vwap_state_amplitude_spread_20",
        func=llm_refined_vwap_state_amplitude_spread_20,
        required_fields=("close", "high", "low", "vwap"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r4::23755bfe8c76; round_id=4; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.052289228112268285, NetAnn=1.3183395512007192, Turnover=0.04629407429545586.",
    ),
    FactorSpec(
        name="llm_refined.ideal_amp_smooth_28",
        func=llm_refined_ideal_amp_smooth_28,
        required_fields=("close", "high", "low", "pre_close"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r3::882cfd04b65e; round_id=3; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.031194012471620295, NetAnn=1.9446467751364462, Turnover=0.2825534324563134.",
    ),
    FactorSpec(
        name="llm_refined.factor_opt_vwap_state",
        func=llm_refined_factor_opt_vwap_state,
        required_fields=("high", "low", "vwap"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r2::1b82a87cc571; round_id=2; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07343121634806356, NetAnn=3.0191583093728696, Turnover=0.27999768431133054.",
    ),
    FactorSpec(
        name="llm_refined.factor_opt_long_window",
        func=llm_refined_factor_opt_long_window,
        required_fields=("close", "high", "low"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r2::1b82a87cc571; round_id=2; source_model=deepseek-v3.1; decision=research_keep. Selection summary: RankIC=0.07269369849280292, NetAnn=2.4174814133905143, Turnover=0.14927192574825032.",
    ),
    FactorSpec(
        name="llm_refined.ideal_amp_open_state",
        func=llm_refined_ideal_amp_open_state,
        required_fields=("close", "high", "low", "open"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r8::06eee5be19db; round_id=8; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.06311234946329754, NetAnn=2.4533963213387846, Turnover=0.32090476765007386.",
    ),
    FactorSpec(
        name="llm_refined.ideal_amp_vol_confirm",
        func=llm_refined_ideal_amp_vol_confirm,
        required_fields=("close", "high", "low", "turnover"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r8::06eee5be19db; round_id=8; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.07107194413889929, NetAnn=1.8796695959075862, Turnover=0.111546479975907.",
    ),
    FactorSpec(
        name="llm_refined.vwap_state_turn_confirm",
        func=llm_refined_vwap_state_turn_confirm,
        required_fields=("high", "low", "turnover", "vwap"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r16::75a175a68ebd; round_id=16; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.07824911287489943, NetAnn=3.4633833725137615, Turnover=0.4613789235063234.",
    ),
    FactorSpec(
        name="llm_refined.amp_smooth_vwap_state",
        func=llm_refined_amp_smooth_vwap_state,
        required_fields=("high", "low", "vwap"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r18::5f0d45242100; round_id=18; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07336801831874065, NetAnn=3.240857780312912, Turnover=0.22178576723962987.",
    ),
    FactorSpec(
        name="llm_refined.volume_confirm_amplitude_q25",
        func=llm_refined_volume_confirm_amplitude_q25,
        required_fields=("high", "low", "volume", "vwap"),
        notes="Auto-promoted pending candidate for ideal_amplitude_structure; run_id=run::ideal_amplitude_structure::r20::081f9e53aa57; round_id=20; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.07690934078136068, NetAnn=3.2330643756374693, Turnover=0.49416871618732977.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_zscore_amplitude_spread_20",
    "llm_refined_turnover_weighted_amplitude_contrast_20",
    "llm_refined_vwap_state_amplitude_spread_20",
    "llm_refined_ideal_amp_smooth_28",
    "llm_refined_factor_opt_vwap_state",
    "llm_refined_factor_opt_long_window",
    "llm_refined_ideal_amp_open_state",
    "llm_refined_ideal_amp_vol_confirm",
    "llm_refined_vwap_state_turn_confirm",
    "llm_refined_amp_smooth_vwap_state",
    "llm_refined_volume_confirm_amplitude_q25",
]
