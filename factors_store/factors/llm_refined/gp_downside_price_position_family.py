from __future__ import annotations

"""Auto-promoted LLM refined candidates for the gp_mined.low_over_high family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "gp_mined.low_over_high"
FAMILY_KEY = "gp_downside_price_position_family"
SEED_FAMILY = "gp_downside_price_position"
SUMMARY_GLOB = "llm_refined_gp_downside_price_position_family_summary_*.csv"


def llm_refined_low_over_high_smooth5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.low_over_high
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="ts_mean(div(low, high), 5)",
        factor_name="llm_refined.low_over_high_smooth5",
    )



def llm_refined_ema_low_over_high(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.low_over_high
    round: llm_refine_round_1
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ema(low, 5), high)",
        factor_name="llm_refined.ema_low_over_high",
    )


def llm_refined_low_over_high_decay5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.low_over_high
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="decay_linear(div(low, high), 5)",
        factor_name="llm_refined.low_over_high_decay5",
    )


def llm_refined_llmgen_low_over_high_volnorm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.low_over_high
    round: llm_refine_round_6
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(div(low, high), 5), add(ts_std(div(low, high), 5), 1e-12))",
        factor_name="llm_refined.llmgen_low_over_high_volnorm",
    )


def llm_refined_llmgen_low_high_decay10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.low_over_high
    round: llm_refine_round_7
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="decay_linear(div(low, high), 10)",
        factor_name="llm_refined.llmgen_low_high_decay10",
    )


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.low_over_high_smooth5",
        func=llm_refined_low_over_high_smooth5,
        required_fields=("high", "low"),
        notes="Auto-promoted pending candidate for gp_downside_price_position; run_id=run::gp_downside_price_position::r4::dab4905ac4b5; round_id=4; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.08897250208695356, NetAnn=1.404900362980599, Turnover=0.2882321601717922.",
    ),
    FactorSpec(
        name="llm_refined.ema_low_over_high",
        func=llm_refined_ema_low_over_high,
        required_fields=("high", "low"),
        notes="Auto-promoted pending candidate for gp_downside_price_position; run_id=run::gp_downside_price_position::r1::dab578422fb8; round_id=1; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.067398800877641, NetAnn=0.9382972996394583, Turnover=0.7502685520196881.",
    ),
    FactorSpec(
        name="llm_refined.low_over_high_decay5",
        func=llm_refined_low_over_high_decay5,
        required_fields=("high", "low"),
        notes="Auto-promoted pending candidate for gp_downside_price_position; run_id=run::gp_downside_price_position::r3::972e42d0b129; round_id=3; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.09042527924030702, NetAnn=1.463988204825439, Turnover=0.3546539074992222.",
    ),
    FactorSpec(
        name="llm_refined.llmgen_low_over_high_volnorm",
        func=llm_refined_llmgen_low_over_high_volnorm,
        required_fields=("high", "low"),
        notes="Auto-promoted pending candidate for gp_downside_price_position; run_id=run::gp_downside_price_position::r6::d8b64f999868; round_id=6; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07128764810260983, NetAnn=1.1084439471882135, Turnover=0.5114795632133493.",
    ),
    FactorSpec(
        name="llm_refined.llmgen_low_high_decay10",
        func=llm_refined_llmgen_low_high_decay10,
        required_fields=("high", "low"),
        notes="Auto-promoted pending candidate for gp_downside_price_position; run_id=run::gp_downside_price_position::r7::6141d6d2436c; round_id=7; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.09089312978431476, NetAnn=1.396039427046619, Turnover=0.2064785892383976.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_low_over_high_smooth5",
    "llm_refined_ema_low_over_high",
    "llm_refined_low_over_high_decay5",
    "llm_refined_llmgen_low_over_high_volnorm",
    "llm_refined_llmgen_low_high_decay10",
]
