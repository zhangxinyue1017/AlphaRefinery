from __future__ import annotations

"""Auto-promoted LLM refined candidates for the qp_pressure.net_pressure_20 family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "qp_pressure.net_pressure_20"
FAMILY_KEY = "qp_low_price_accumulation_pressure_family"
SEED_FAMILY = "qp_low_price_accumulation_pressure"
SUMMARY_GLOB = "llm_refined_qp_low_price_accumulation_pressure_family_summary_*.csv"


def llm_refined_net_pressure_lowprice_turnover_confirm_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_24
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.net_pressure_lowprice_turnover_confirm_20",
    )



def llm_refined_low_price_vol_gain_filter_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_28
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), mul(volume, gt(pct_chg, 0)), 0), 20), add(ts_sum(volume, 20), 1e-12))",
        factor_name="llm_refined.low_price_vol_gain_filter_20",
    )


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.net_pressure_lowprice_turnover_confirm_20",
        func=llm_refined_net_pressure_lowprice_turnover_confirm_20,
        required_fields=("close", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r24::2ed0ad23da98; round_id=24; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07197727701296824, NetAnn=3.5087856935665203, Turnover=0.14367904303651544.",
    ),
    FactorSpec(
        name="llm_refined.low_price_vol_gain_filter_20",
        func=llm_refined_low_price_vol_gain_filter_20,
        required_fields=("close", "pct_chg", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r28::7a725ebeb204; round_id=28; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.03840983248001412, NetAnn=4.02347353260608, Turnover=0.24451071537510335.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_net_pressure_lowprice_turnover_confirm_20",
    "llm_refined_low_price_vol_gain_filter_20",
]
