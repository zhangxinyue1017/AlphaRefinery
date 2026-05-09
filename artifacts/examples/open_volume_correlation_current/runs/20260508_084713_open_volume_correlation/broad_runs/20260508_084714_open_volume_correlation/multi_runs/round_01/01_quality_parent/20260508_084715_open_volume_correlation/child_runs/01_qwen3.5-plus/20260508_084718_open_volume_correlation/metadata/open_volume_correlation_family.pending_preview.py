from __future__ import annotations

"""Auto-promoted LLM refined candidates for the alpha101.alpha003 family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "alpha101.alpha003"
FAMILY_KEY = "open_volume_correlation_family"
SEED_FAMILY = "open_volume_correlation"
SUMMARY_GLOB = "llm_refined_open_volume_correlation_family_summary_*.csv"


def llm_refined_open_vol_cov_vol_adj(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_7
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-div(ts_cov(open, volume, 10), add(ts_mean(volume, 20), 1e-12))",
        factor_name="llm_refined.open_vol_cov_vol_adj",
    )



def llm_refined_turnover_adjusted_corr(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_5
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-ts_corr(open, volume, 10) * div(ema(turnover, 20), add(ema(turnover, 120), 1e-12))",
        factor_name="llm_refined.turnover_adjusted_corr",
    )


def llm_refined_open_amt_corr_ema(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_8
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-ema(ts_corr(cs_rank(open), cs_rank(amount), 10), 5)",
        factor_name="llm_refined.open_amt_corr_ema",
    )


def llm_refined_open_vol_corr_amt_conf(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_6
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-ts_corr(cs_rank(open), cs_rank(volume), 10) * cs_rank(amount)",
        factor_name="llm_refined.open_vol_corr_amt_conf",
    )


def llm_refined_open_vol_corr_resid_size(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_9
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="cs_reg_resid(neg(ts_corr(cs_rank(open), cs_rank(volume), 10)), size, 20)",
        factor_name="llm_refined.open_vol_corr_resid_size",
    )


def llm_refined_donor_transfer_open_volume(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_10
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-cs_rank(ts_cov(cs_rank(ema(open, 10)), cs_rank(ema(volume, 20)), 5))",
        factor_name="llm_refined.donor_transfer_open_volume",
    )


def llm_refined_open_vol_ema_cov_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_11
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-cs_rank(ts_cov(cs_rank(ema(open, 5)), cs_rank(ema(volume, 5)), 10))",
        factor_name="llm_refined.open_vol_ema_cov_10",
    )


def llm_refined_open_volume_corr_turnover_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_18
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-mul(ts_corr(cs_rank(open), cs_rank(volume), 10), ema(cs_rank(turnover), 5))",
        factor_name="llm_refined.open_volume_corr_turnover_confirm",
    )


def llm_refined_open_vol_corr_20d(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_16
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_corr(open, volume, 20))",
        factor_name="llm_refined.open_vol_corr_20d",
    )


def llm_refined_zscore_open_turnover_corr(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_19
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(ts_corr(zscore(open), zscore(turnover), 10))",
        factor_name="llm_refined.zscore_open_turnover_corr",
    )


def llm_refined_open_vol_corr_liquidity_filter(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_23
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-where(gt(rel_volume(20), 1), ts_corr(cs_rank(open), cs_rank(volume), 10), 0)",
        factor_name="llm_refined.open_vol_corr_liquidity_filter",
    )


def llm_refined_open_vol_cov_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_6
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="-ts_cov(cs_rank(open), cs_rank(volume), 10)",
        factor_name="llm_refined.open_vol_cov_rank_10",
    )


def llm_refined_high_vol_corr_rank_10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_6
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="-ts_corr(cs_rank(high), cs_rank(volume), 10)",
        factor_name="llm_refined.high_vol_corr_rank_10",
    )


def llm_refined_open_volume_corr_rsq_gate(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_18
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="-mul(ts_corr(cs_rank(open), cs_rank(volume), 10), ema(regression_rsq(open, volume, 20), 5))",
        factor_name="llm_refined.open_volume_corr_rsq_gate",
    )


def llm_refined_open_volume_corr_resid_adj(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_13
    source model: claude-sonnet-4-6
    keep-drop status: research_keep_exploratory
    """
    return evaluate_expression_factor(
        data,
        expression="-ts_corr(cs_rank(open), cs_rank(volume), 10) - 0.5 * cs_reg_resid(cs_rank(open), cs_rank(volume), 20)",
        factor_name="llm_refined.open_volume_corr_resid_adj",
    )


def llm_refined_open_vol_corr_ema_gate(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_29
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(neg(ema(ts_corr(cs_rank(open), cs_rank(volume), 10), 5)), rowmin(div(rel_volume(20), 2), 1))",
        factor_name="llm_refined.open_vol_corr_ema_gate",
    )


def llm_refined_open_vol_corr_win20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_31
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="-where(gt(rel_volume(20), 1), ts_corr(cs_rank(open), cs_rank(volume), 20), 0)",
        factor_name="llm_refined.open_vol_corr_win20",
    )


def llm_refined_open_vol_corr_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_33
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="-sma(where(gt(rel_volume(20), 1), ts_corr(cs_rank(open), cs_rank(volume), 10), 0), 5, 1)",
        factor_name="llm_refined.open_vol_corr_smooth",
    )


def llm_refined_open_vol_corr_decay(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_31
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="decay_linear(-where(gt(rel_volume(20), 1), ts_corr(cs_rank(open), cs_rank(volume), 10), 0), 5)",
        factor_name="llm_refined.open_vol_corr_decay",
    )


def llm_refined_open_vol_corr_decay_gate(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_29
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(neg(decay_linear(ts_corr(cs_rank(open), cs_rank(volume), 10), 5)), rowmin(div(rel_volume(20), 2), 1))",
        factor_name="llm_refined.open_vol_corr_decay_gate",
    )


def llm_refined_open_vol_corr_amount_rank(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_29
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(decay_linear(ts_corr(cs_rank(open), cs_rank(amount), 10), 5))",
        factor_name="llm_refined.open_vol_corr_amount_rank",
    )


def llm_refined_r2_turnover_weight(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_37
    source model: kimi-k2
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(ema(add(ts_cov(returns, open, 5), regression_rsq(returns, market_return, 60)), 20), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)))",
        factor_name="llm_refined.r2_turnover_weight",
    )


def llm_refined_corr_resid_turnover_gate(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_39
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ema(add(cs_reg_resid(ts_corr(cs_rank(open), cs_rank(turnover), 10), size, 20), regression_rsq(returns, market_return, 60)), 20), div(ema(turnover, 120), add(ema(turnover, 40), 1e-12))))",
        factor_name="llm_refined.corr_resid_turnover_gate",
    )


def llm_refined_turnover_ratio_smoothed_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_39
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ema(add(ts_corr(cs_rank(open), cs_rank(turnover), 10), regression_rsq(returns, market_return, 60)), 20), ema(div(ema(turnover, 120), add(ema(turnover, 20), 1e-12)), 20)))",
        factor_name="llm_refined.turnover_ratio_smoothed_confirm",
    )


def llm_refined_open_liquidity_corr_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_41
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ema(add(ts_corr(returns, open, 5), regression_rsq(returns, market_return, 60)), 20), div(ema(turnover, 120), add(ema(turnover, 20), 1e-12))))",
        factor_name="llm_refined.open_liquidity_corr_norm",
    )


def llm_refined_open_vol_amount_corr(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_42
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 5), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 20), 1e-12))))",
        factor_name="llm_refined.open_vol_amount_corr",
    )


def llm_refined_open_vol_rank_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_42
    source model: kimi-k2
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 5), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 40), div(ema(cs_rank(turnover), 120), add(ema(cs_rank(turnover), 20), 1e-12))))",
        factor_name="llm_refined.open_vol_rank_smooth",
    )


def llm_refined_open_vol_decay_rsq(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_42
    source model: kimi-k2
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(decay_linear(add(ts_corr(cs_rank(returns), cs_rank(open), 5), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 40), div(ema(cs_rank(turnover), 120), add(ema(cs_rank(turnover), 20), 1e-12))))",
        factor_name="llm_refined.open_vol_decay_rsq",
    )


def llm_refined_decorrelated_turnover_open_r2(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_43
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ema(ts_cov(returns, open, 5), 20), div(ema(sub(turnover, volume), 120), add(ema(turnover, 20), 1e-12))))",
        factor_name="llm_refined.decorrelated_turnover_open_r2",
    )


def llm_refined_open_vol_corr_stable(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: alpha101.alpha003
    round: llm_refine_round_45
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 10), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 40), 1e-12))))",
        factor_name="llm_refined.open_vol_corr_stable",
    )


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.open_vol_cov_vol_adj",
        func=llm_refined_open_vol_cov_vol_adj,
        required_fields=("open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r7::fa1a4c8e452c; round_id=7; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.06246882274034917, NetAnn=3.264265960118525, Turnover=0.4217100957161621.",
    ),
    FactorSpec(
        name="llm_refined.turnover_adjusted_corr",
        func=llm_refined_turnover_adjusted_corr,
        required_fields=("open", "turnover", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r5::7763a9f1279c; round_id=5; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.05103599090696367, NetAnn=2.1197211182136373, Turnover=0.4741037878105079.",
    ),
    FactorSpec(
        name="llm_refined.open_amt_corr_ema",
        func=llm_refined_open_amt_corr_ema,
        required_fields=("amount", "open"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r8::fa89ad1d3128; round_id=8; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.037601701768010115, NetAnn=1.071084366835605, Turnover=0.30055657643117034.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_amt_conf",
        func=llm_refined_open_vol_corr_amt_conf,
        required_fields=("amount", "open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r6::5608d4435206; round_id=6; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.059118701161265864, NetAnn=2.109043181909456, Turnover=0.49307922134942633.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_resid_size",
        func=llm_refined_open_vol_corr_resid_size,
        required_fields=("open", "size", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r9::c306656244a2; round_id=9; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.03336330874207299, NetAnn=0.9625278274543927, Turnover=0.5149145054017756.",
    ),
    FactorSpec(
        name="llm_refined.donor_transfer_open_volume",
        func=llm_refined_donor_transfer_open_volume,
        required_fields=("open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r10::a85596b06e2f; round_id=10; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.03503506100693046, NetAnn=1.5049163546816873, Turnover=0.5123034991051776.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_ema_cov_10",
        func=llm_refined_open_vol_ema_cov_10,
        required_fields=("open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r11::5dcd65ee45bd; round_id=11; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.04018928649167463, NetAnn=1.5267524521681706, Turnover=0.35254134105594326.",
    ),
    FactorSpec(
        name="llm_refined.open_volume_corr_turnover_confirm",
        func=llm_refined_open_volume_corr_turnover_confirm,
        required_fields=("open", "turnover", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r18::434a7b4ecc7e; round_id=18; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.049971103500368395, NetAnn=1.8031968294379834, Turnover=0.44387703158267966.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_20d",
        func=llm_refined_open_vol_corr_20d,
        required_fields=("open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r16::cf4dff187918; round_id=16; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.047752105138980504, NetAnn=2.3024415941635548, Turnover=0.28682516915320627.",
    ),
    FactorSpec(
        name="llm_refined.zscore_open_turnover_corr",
        func=llm_refined_zscore_open_turnover_corr,
        required_fields=("open", "turnover"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r19::54246c8ffd5e; round_id=19; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.037528541047792845, NetAnn=1.214758536481365, Turnover=0.5109386050859896.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_liquidity_filter",
        func=llm_refined_open_vol_corr_liquidity_filter,
        required_fields=("open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r23::74c1deaabaac; round_id=23; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.05210630056476849, NetAnn=14.964130894426134, Turnover=0.9977434724295804.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_cov_rank_10",
        func=llm_refined_open_vol_cov_rank_10,
        required_fields=("open", "volume"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r6::5608d4435206; round_id=6; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0446, NetAnn=1.58, Turnover=0.424, RankICIR=0.6442.",
    ),
    FactorSpec(
        name="llm_refined.high_vol_corr_rank_10",
        func=llm_refined_high_vol_corr_rank_10,
        required_fields=("high", "volume"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r6::5608d4435206; round_id=6; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0462, NetAnn=0.82, Turnover=0.460, RankICIR=0.6066.",
    ),
    FactorSpec(
        name="llm_refined.open_volume_corr_rsq_gate",
        func=llm_refined_open_volume_corr_rsq_gate,
        required_fields=("open", "volume"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r18::434a7b4ecc7e; round_id=18; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.0410, NetAnn=1.48, Turnover=0.433, RankICIR=0.5743.",
    ),
    FactorSpec(
        name="llm_refined.open_volume_corr_resid_adj",
        func=llm_refined_open_volume_corr_resid_adj,
        required_fields=("open", "volume"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r13::5dcd65ee45bd; round_id=13; source_model=claude-sonnet-4-6; decision=research_keep_exploratory. Selection summary: RankIC=0.0453, NetAnn=1.64, Turnover=0.489, RankICIR=0.5755.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_ema_gate",
        func=llm_refined_open_vol_corr_ema_gate,
        required_fields=("open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r29::89821e6aa540; round_id=29; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.049354818403542215, NetAnn=1.7441254943384665, Turnover=0.4319502247216192.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_win20",
        func=llm_refined_open_vol_corr_win20,
        required_fields=("open", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r31::4d1f6bc1ddfc; round_id=31; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.05648030665754672, NetAnn=10.308828413701258, Turnover=0.8054294983082361.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_smooth",
        func=llm_refined_open_vol_corr_smooth,
        required_fields=("open", "volume"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r33::d83bc9abd285; round_id=33; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.0564, NetAnn=1.85, Turnover=0.261, RankICIR=0.6668.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_decay",
        func=llm_refined_open_vol_corr_decay,
        required_fields=("open", "volume"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r31::4d1f6bc1ddfc; round_id=31; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0520, NetAnn=1.60, Turnover=0.432, RankICIR=0.7000.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_decay_gate",
        func=llm_refined_open_vol_corr_decay_gate,
        required_fields=("open", "volume"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r29::89821e6aa540; round_id=29; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=0.0473, NetAnn=1.66, Turnover=0.447, RankICIR=0.7225.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_amount_rank",
        func=llm_refined_open_vol_corr_amount_rank,
        required_fields=("amount", "open"),
        notes="Manually promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r29::89821e6aa540; round_id=29; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=0.0361, NetAnn=1.01, Turnover=0.326, RankICIR=0.5004.",
    ),
    FactorSpec(
        name="llm_refined.r2_turnover_weight",
        func=llm_refined_r2_turnover_weight,
        required_fields=("market_return", "open", "returns", "turnover"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r37::38d6c23a4ba4; round_id=37; source_model=kimi-k2; decision=research_keep. Selection summary: RankIC=0.06977438168765637, NetAnn=2.637592328881568, Turnover=0.06850167133492328.",
    ),
    FactorSpec(
        name="llm_refined.corr_resid_turnover_gate",
        func=llm_refined_corr_resid_turnover_gate,
        required_fields=("market_return", "open", "returns", "size", "turnover"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r39::d807c28f688e; round_id=39; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=-0.0027175738933902897, NetAnn=-0.08799103529793528, Turnover=0.1483163345619921.",
    ),
    FactorSpec(
        name="llm_refined.turnover_ratio_smoothed_confirm",
        func=llm_refined_turnover_ratio_smoothed_confirm,
        required_fields=("market_return", "open", "returns", "turnover"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r39::d807c28f688e; round_id=39; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=-0.009809184993177095, NetAnn=-0.14050554654864145, Turnover=0.1374981329186164.",
    ),
    FactorSpec(
        name="llm_refined.open_liquidity_corr_norm",
        func=llm_refined_open_liquidity_corr_norm,
        required_fields=("market_return", "open", "returns", "turnover"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r41::7bae95abe7d8; round_id=41; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0008731278643962313, NetAnn=-0.026823619670270804, Turnover=0.21443279674903087.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_amount_corr",
        func=llm_refined_open_vol_amount_corr,
        required_fields=("amount", "market_return", "open", "returns"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r42::17d3b3e6fcb3; round_id=42; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.0487854655358091, NetAnn=1.3202159200357717, Turnover=0.25530171719492967.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_rank_smooth",
        func=llm_refined_open_vol_rank_smooth,
        required_fields=("market_return", "open", "returns", "turnover"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r42::17d3b3e6fcb3; round_id=42; source_model=kimi-k2; decision=research_keep. Selection summary: RankIC=0.038535858329931796, NetAnn=0.6488215319308108, Turnover=0.18104120899765022.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_decay_rsq",
        func=llm_refined_open_vol_decay_rsq,
        required_fields=("market_return", "open", "returns", "turnover"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r42::17d3b3e6fcb3; round_id=42; source_model=kimi-k2; decision=research_keep. Selection summary: RankIC=0.03842354567818526, NetAnn=0.6406163075789431, Turnover=0.17382783169350458.",
    ),
    FactorSpec(
        name="llm_refined.decorrelated_turnover_open_r2",
        func=llm_refined_decorrelated_turnover_open_r2,
        required_fields=("open", "returns", "turnover", "volume"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r43::dacaebb30939; round_id=43; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.034595410291391335, NetAnn=0.727589612133158, Turnover=0.10868001248022163.",
    ),
    FactorSpec(
        name="llm_refined.open_vol_corr_stable",
        func=llm_refined_open_vol_corr_stable,
        required_fields=("amount", "market_return", "open", "returns"),
        notes="Auto-promoted pending candidate for open_volume_correlation; run_id=run::open_volume_correlation::r45::6d157bc298f9; round_id=45; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.04200080449751649, NetAnn=1.2253288993118345, Turnover=0.1883450778955242.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_open_vol_cov_vol_adj",
    "llm_refined_turnover_adjusted_corr",
    "llm_refined_open_amt_corr_ema",
    "llm_refined_open_vol_corr_amt_conf",
    "llm_refined_open_vol_corr_resid_size",
    "llm_refined_donor_transfer_open_volume",
    "llm_refined_open_vol_ema_cov_10",
    "llm_refined_open_volume_corr_turnover_confirm",
    "llm_refined_open_vol_corr_20d",
    "llm_refined_zscore_open_turnover_corr",
    "llm_refined_open_vol_corr_liquidity_filter",
    "llm_refined_open_vol_cov_rank_10",
    "llm_refined_high_vol_corr_rank_10",
    "llm_refined_open_volume_corr_rsq_gate",
    "llm_refined_open_volume_corr_resid_adj",
    "llm_refined_open_vol_corr_ema_gate",
    "llm_refined_open_vol_corr_win20",
    "llm_refined_open_vol_corr_smooth",
    "llm_refined_open_vol_corr_decay",
    "llm_refined_open_vol_corr_decay_gate",
    "llm_refined_open_vol_corr_amount_rank",
    "llm_refined_r2_turnover_weight",
    "llm_refined_corr_resid_turnover_gate",
    "llm_refined_turnover_ratio_smoothed_confirm",
    "llm_refined_open_liquidity_corr_norm",
    "llm_refined_open_vol_amount_corr",
    "llm_refined_open_vol_rank_smooth",
    "llm_refined_open_vol_decay_rsq",
    "llm_refined_decorrelated_turnover_open_r2",
    "llm_refined_open_vol_corr_stable",
]
