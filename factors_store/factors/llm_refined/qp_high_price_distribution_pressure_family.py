from __future__ import annotations

"""Auto-promoted LLM refined candidates for the qp_pressure.high_price_volume_share_20 family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "qp_pressure.high_price_volume_share_20"
FAMILY_KEY = "qp_high_price_distribution_pressure_family"
SEED_FAMILY = "qp_high_price_distribution_pressure"
SUMMARY_GLOB = "llm_refined_qp_high_price_distribution_pressure_family_summary_*.csv"


def llm_refined_qp_pressure_hp_vwap_vol_share(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.high_price_volume_share_20
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(sumif(volume, 20, vwap > ts_quantile(vwap, 20, 0.7)) / add(ts_sum(volume, 20), 1e-12))",
        factor_name="llm_refined.qp_pressure_hp_vwap_vol_share",
    )



def llm_refined_qp_pressure_hp_vol_share_smooth_5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.high_price_volume_share_20
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(decay_linear(sumif(volume, 20, close > ts_quantile(close, 20, 0.7)) / add(ts_sum(volume, 20), 1e-12), 5))",
        factor_name="llm_refined.qp_pressure_hp_vol_share_smooth_5",
    )


def llm_refined_qp_pressure_hp_vol_share_40(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.high_price_volume_share_20
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(sumif(volume, 40, close > ts_quantile(close, 40, 0.7)) / add(ts_sum(volume, 40), 1e-12))",
        factor_name="llm_refined.qp_pressure_hp_vol_share_40",
    )


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.qp_pressure_hp_vwap_vol_share",
        func=llm_refined_qp_pressure_hp_vwap_vol_share,
        required_fields=("volume", "vwap"),
        notes="Auto-promoted pending candidate for qp_high_price_distribution_pressure; run_id=run::qp_high_price_distribution_pressure::r3::9907fce2be44; round_id=3; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.054673031079691764, NetAnn=0.7094104898897613, Turnover=0.1814519016754772.",
    ),
    FactorSpec(
        name="llm_refined.qp_pressure_hp_vol_share_smooth_5",
        func=llm_refined_qp_pressure_hp_vol_share_smooth_5,
        required_fields=("close", "volume"),
        notes="Auto-promoted pending candidate for qp_high_price_distribution_pressure; run_id=run::qp_high_price_distribution_pressure::r3::9907fce2be44; round_id=3; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0514278295005472, NetAnn=0.8195467765169122, Turnover=0.1424250612928291.",
    ),
    FactorSpec(
        name="llm_refined.qp_pressure_hp_vol_share_40",
        func=llm_refined_qp_pressure_hp_vol_share_40,
        required_fields=("close", "volume"),
        notes="Auto-promoted pending candidate for qp_high_price_distribution_pressure; run_id=run::qp_high_price_distribution_pressure::r3::9907fce2be44; round_id=3; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0499646097479208, NetAnn=0.846301726526582, Turnover=0.0951746179314401.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_qp_pressure_hp_vwap_vol_share",
    "llm_refined_qp_pressure_hp_vol_share_smooth_5",
    "llm_refined_qp_pressure_hp_vol_share_40",
]
