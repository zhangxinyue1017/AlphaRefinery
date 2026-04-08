from __future__ import annotations

"""Auto-promoted LLM refined candidates for the gp_mined.close_mean60_over_close family."""

import pandas as pd

from .common import FactorSpec, evaluate_expression_factor

PARENT_FACTOR = "gp_mined.close_mean60_over_close"
FAMILY_KEY = "gp_historical_anchor_ratio_family"
SEED_FAMILY = "gp_historical_anchor_ratio"
SUMMARY_GLOB = "llm_refined_gp_historical_anchor_ratio_family_summary_*.csv"


def llm_refined_quantile_anchor(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_7
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_quantile(close, 60, 0.2), add(close, 1e-12))",
        factor_name="llm_refined.quantile_anchor",
    )



def llm_refined_mean60_over_close_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(sma(ts_mean(close, 60), 5), add(close, 1e-12))",
        factor_name="llm_refined.mean60_over_close_smooth",
    )


def llm_refined_close_mean40(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_5
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 40), add(close, 1e-12))",
        factor_name="llm_refined.close_mean40",
    )


def llm_refined_close_mean40_zscore(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_5
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="zscore(div(ts_mean(close, 40), add(close, 1e-12)))",
        factor_name="llm_refined.close_mean40_zscore",
    )


def llm_refined_price_decay_linear(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_2
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_linear_decay_mean(close, 60), add(close, 1e-12))",
        factor_name="llm_refined.price_decay_linear",
    )


def llm_refined_gp_mined_close_mean60_over_max20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 60), add(ts_max(close, 20), 1e-12))",
        factor_name="llm_refined.gp_mined_close_mean60_over_max20",
    )


def llm_refined_gp_mined_close_mean60_over_decay5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 60), add(decay_linear(close, 5), 1e-12))",
        factor_name="llm_refined.gp_mined_close_mean60_over_decay5",
    )


def llm_refined_mean60_over_close_voladj(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_1
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(div(ts_mean(close, 60), add(close, 1e-12)), add(add(1, ts_std(returns, 20)), 1e-12))",
        factor_name="llm_refined.mean60_over_close_voladj",
    )


def llm_refined_mean60_over_ema10_close(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_1
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 60), add(ema(close, 10), 1e-12))",
        factor_name="llm_refined.mean60_over_ema10_close",
    )


def llm_refined_close_mean60_over_mean5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 60), add(ts_mean(close, 5), 1e-12))",
        factor_name="llm_refined.close_mean60_over_mean5",
    )


def llm_refined_mean60_sub_close(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="sub(ts_mean(close, 60), close)",
        factor_name="llm_refined.mean60_sub_close",
    )


def llm_refined_mean60_over_close_vol_norm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(div(ts_mean(close, 60), add(close, 1e-12)), add(ts_std(close, 20), 1e-12))",
        factor_name="llm_refined.mean60_over_close_vol_norm",
    )


def llm_refined_close_ema60_over_close(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_1
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ema(close, 60), add(close, 1e-12))",
        factor_name="llm_refined.close_ema60_over_close",
    )


def llm_refined_close_mean60_over_close_volstable(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_1
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(div(ts_mean(close, 60), add(close, 1e-12)), add(ts_std(returns, 20), 1e-12))",
        factor_name="llm_refined.close_mean60_over_close_volstable",
    )


def llm_refined_gp_mined_close_mean60_amountnorm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_2
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(div(ts_mean(close, 60), add(close, 1e-12)), add(ts_std(amount, 20), 1e-12))",
        factor_name="llm_refined.gp_mined_close_mean60_amountnorm",
    )


def llm_refined_gp_log_anchor_diff(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_8
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="sub(log(ts_mean(close, 60)), log(close))",
        factor_name="llm_refined.gp_log_anchor_diff",
    )


def llm_refined_anchor_distance_zscore(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_2
    source model: deepseek-v3.1
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="zscore(div(ts_mean(close, 60), add(close, 1e-12)))",
        factor_name="llm_refined.anchor_distance_zscore",
    )


def llm_refined_scaled_mean60_over_close(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="scale(div(ts_mean(close, 60), add(close, 1e-12)))",
        factor_name="llm_refined.scaled_mean60_over_close",
    )


def llm_refined_mean60_over_close_adj_turnover(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 60), add(close, 1e-12)) * (1 - ts_mean(turnover, 20))",
        factor_name="llm_refined.mean60_over_close_adj_turnover",
    )


def llm_refined_ema40_close_ratio(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_5
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ema(close, 40), add(close, 1e-12))",
        factor_name="llm_refined.ema40_close_ratio",
    )


def llm_refined_gp_optimized_smooth_mean60_over_close(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_3
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="ts_mean(div(ts_mean(close, 60), add(close, 1e-12)), 3)",
        factor_name="llm_refined.gp_optimized_smooth_mean60_over_close",
    )


def llm_refined_close_mean60_ema_smooth(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_4
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="div(ema(ts_mean(close, 60), 5), add(close, 1e-12))",
        factor_name="llm_refined.close_mean60_ema_smooth",
    )


def llm_refined_close_mean60_over_turn_price20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_11
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 60), add(ts_turnover_ref_price(close, turnover, 20), 1e-12))",
        factor_name="llm_refined.close_mean60_over_turn_price20",
    )


def llm_refined_close_mean60_vwap(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_12
    source model: deepseek-v3.1
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(vwap, 60), add(close, 1e-12))",
        factor_name="llm_refined.close_mean60_vwap",
    )


def llm_refined_gp_mean60_over_close_decay5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_13
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="decay_linear(div(ts_mean(close, 60), add(close, 1e-12)), 5)",
        factor_name="llm_refined.gp_mean60_over_close_decay5",
    )


def llm_refined_mean40_over_ema10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: gp_mined.close_mean60_over_close
    round: llm_refine_round_14
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="div(ts_mean(close, 40), add(ema(close, 10), 1e-12))",
        factor_name="llm_refined.mean40_over_ema10",
    )


FACTOR_SPECS: tuple[FactorSpec, ...] = (
    FactorSpec(
        name="llm_refined.quantile_anchor",
        func=llm_refined_quantile_anchor,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r7::5c91a4671d33; round_id=7; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.0818310950571949, NetAnn=2.762501513520002, Turnover=0.26059122998010154.",
    ),
    FactorSpec(
        name="llm_refined.mean60_over_close_smooth",
        func=llm_refined_mean60_over_close_smooth,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r4::1218ea0eb0a5; round_id=4; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.0763855023219811, NetAnn=3.298676936003396, Turnover=0.23250593571469477.",
    ),
    FactorSpec(
        name="llm_refined.close_mean40",
        func=llm_refined_close_mean40,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r5::16fe48d6d8d3; round_id=5; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.07420245131350155, NetAnn=3.2341486487138624, Turnover=0.3007723476390672.",
    ),
    FactorSpec(
        name="llm_refined.close_mean40_zscore",
        func=llm_refined_close_mean40_zscore,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r5::cf57ec750838; round_id=5; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.07420245131350155, NetAnn=3.2341486487138624, Turnover=0.3007723476390672.",
    ),
    FactorSpec(
        name="llm_refined.price_decay_linear",
        func=llm_refined_price_decay_linear,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r2::941f14ec38a3; round_id=2; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07570035389147224, NetAnn=3.2305449515529867, Turnover=0.31563945802034155.",
    ),
    FactorSpec(
        name="llm_refined.gp_mined_close_mean60_over_max20",
        func=llm_refined_gp_mined_close_mean60_over_max20,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r3::55692267a338; round_id=3; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.0723884964825112, NetAnn=2.42338445669475, Turnover=0.11793202545479747.",
    ),
    FactorSpec(
        name="llm_refined.gp_mined_close_mean60_over_decay5",
        func=llm_refined_gp_mined_close_mean60_over_decay5,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r3::55692267a338; round_id=3; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.07213011848239062, NetAnn=3.3615332846383428, Turnover=0.13005823199643773.",
    ),
    FactorSpec(
        name="llm_refined.mean60_over_close_voladj",
        func=llm_refined_mean60_over_close_voladj,
        required_fields=("close", "returns"),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r1::244ff46fcb81; round_id=1; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.08138474998203878, NetAnn=3.5415605879297054, Turnover=0.2419129723747912.",
    ),
    FactorSpec(
        name="llm_refined.mean60_over_ema10_close",
        func=llm_refined_mean60_over_ema10_close,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r1::244ff46fcb81; round_id=1; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=0.06690333793673396, NetAnn=3.0662349607103083, Turnover=0.09209138726833176.",
    ),
    FactorSpec(
        name="llm_refined.close_mean60_over_mean5",
        func=llm_refined_close_mean60_over_mean5,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r3::50c5ba9ca573; round_id=3; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.0688744019520206, NetAnn=3.1318453210409043, Turnover=0.12066303081436747.",
    ),
    FactorSpec(
        name="llm_refined.mean60_sub_close",
        func=llm_refined_mean60_sub_close,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r3::50c5ba9ca573; round_id=3; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.06489181631338542, NetAnn=3.0789915816509694, Turnover=0.22365886446589148.",
    ),
    FactorSpec(
        name="llm_refined.mean60_over_close_vol_norm",
        func=llm_refined_mean60_over_close_vol_norm,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r3::50c5ba9ca573; round_id=3; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0694343713142492, NetAnn=1.791620877983259, Turnover=0.08033163119822997.",
    ),
    FactorSpec(
        name="llm_refined.close_ema60_over_close",
        func=llm_refined_close_ema60_over_close,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r1::01fd016fc939; round_id=1; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.07849340707087805, NetAnn=3.6558973576611047, Turnover=0.29447337020033676.",
    ),
    FactorSpec(
        name="llm_refined.close_mean60_over_close_volstable",
        func=llm_refined_close_mean60_over_close_volstable,
        required_fields=("close", "returns"),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r1::01fd016fc939; round_id=1; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=0.08441763221099344, NetAnn=1.4628197067947362, Turnover=0.1405554969192592.",
    ),
    FactorSpec(
        name="llm_refined.gp_mined_close_mean60_amountnorm",
        func=llm_refined_gp_mined_close_mean60_amountnorm,
        required_fields=("amount", "close"),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r2::d21f63548c9a; round_id=2; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.09912375595978373, NetAnn=5.495274925241898, Turnover=0.08659251283812677.",
    ),
    FactorSpec(
        name="llm_refined.gp_log_anchor_diff",
        func=llm_refined_gp_log_anchor_diff,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r8::10aab4c236e6; round_id=8; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.07666891161459244, NetAnn=3.4496731733864587, Turnover=0.2508876026895289.",
    ),
    FactorSpec(
        name="llm_refined.anchor_distance_zscore",
        func=llm_refined_anchor_distance_zscore,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r2::66f0f16da3cc; round_id=2; source_model=deepseek-v3.1; decision=research_winner. Selection summary: RankIC=0.07666891099742072, NetAnn=3.4496731733864587, Turnover=0.2508876026895289.",
    ),
    FactorSpec(
        name="llm_refined.scaled_mean60_over_close",
        func=llm_refined_scaled_mean60_over_close,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r4::9412cb5248a8; round_id=4; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07590943226122712, NetAnn=5.091702698983418, Turnover=0.20833493405384593.",
    ),
    FactorSpec(
        name="llm_refined.mean60_over_close_adj_turnover",
        func=llm_refined_mean60_over_close_adj_turnover,
        required_fields=("close", "turnover"),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r4::9412cb5248a8; round_id=4; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.0846844152751022, NetAnn=3.7931812020892055, Turnover=0.22234481245360785.",
    ),
    FactorSpec(
        name="llm_refined.ema40_close_ratio",
        func=llm_refined_ema40_close_ratio,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r5::80274f25f163; round_id=5; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.0766976216907497, NetAnn=3.3481916008711208, Turnover=0.3500368126981543.",
    ),
    FactorSpec(
        name="llm_refined.gp_optimized_smooth_mean60_over_close",
        func=llm_refined_gp_optimized_smooth_mean60_over_close,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r3::a05135546292; round_id=3; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.07276889845374801, NetAnn=3.4641505817094878, Turnover=0.1498191305209542.",
    ),
    FactorSpec(
        name="llm_refined.close_mean60_ema_smooth",
        func=llm_refined_close_mean60_ema_smooth,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r4::d111494e883d; round_id=4; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.07645663876888713, NetAnn=3.2787762679278174, Turnover=0.24093545981675601.",
    ),
    FactorSpec(
        name="llm_refined.close_mean60_over_turn_price20",
        func=llm_refined_close_mean60_over_turn_price20,
        required_fields=("close", "turnover"),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r11::cbedaf2960dc; round_id=11; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=0.06382263500679847, NetAnn=2.8652948677097116, NetExcess=0.15121552601190902, Turnover=0.0840155104085715.",
    ),
    FactorSpec(
        name="llm_refined.close_mean60_vwap",
        func=llm_refined_close_mean60_vwap,
        required_fields=("close", "vwap"),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r12::2e99b1698234; round_id=12; source_model=deepseek-v3.1; decision=research_keep. Selection summary: RankIC=0.011878846914610967, NetAnn=0.6606790119721944, NetExcess=-0.4837678722196578, Turnover=0.042682954220500814. Promotion context: new_family_broad fallback keep for branch preservation.",
    ),
    FactorSpec(
        name="llm_refined.gp_mean60_over_close_decay5",
        func=llm_refined_gp_mean60_over_close_decay5,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r13::6b581850802c; round_id=13; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.07185358543243982, NetAnn=3.347209585196296, NetExcess=0.35125630379374173, Turnover=0.12893140916406778.",
    ),
    FactorSpec(
        name="llm_refined.mean40_over_ema10",
        func=llm_refined_mean40_over_ema10,
        required_fields=("close",),
        notes="Auto-promoted pending candidate for gp_historical_anchor_ratio; run_id=run::gp_historical_anchor_ratio::r14::085a6e2a0439; round_id=14; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.0669031029692361, NetAnn=2.985786765719221, NetExcess=0.2188511383497307, Turnover=0.11805456446558318. Promotion context: new_family_broad fallback keep for branch preservation.",
    ),
)


__all__ = [
    "FACTOR_SPECS",
    "FAMILY_KEY",
    "PARENT_FACTOR",
    "SEED_FAMILY",
    "SUMMARY_GLOB",
    "llm_refined_quantile_anchor",
    "llm_refined_mean60_over_close_smooth",
    "llm_refined_close_mean40",
    "llm_refined_close_mean40_zscore",
    "llm_refined_price_decay_linear",
    "llm_refined_gp_mined_close_mean60_over_max20",
    "llm_refined_gp_mined_close_mean60_over_decay5",
    "llm_refined_mean60_over_close_voladj",
    "llm_refined_mean60_over_ema10_close",
    "llm_refined_close_mean60_over_mean5",
    "llm_refined_mean60_sub_close",
    "llm_refined_mean60_over_close_vol_norm",
    "llm_refined_close_ema60_over_close",
    "llm_refined_close_mean60_over_close_volstable",
    "llm_refined_gp_mined_close_mean60_amountnorm",
    "llm_refined_gp_log_anchor_diff",
    "llm_refined_anchor_distance_zscore",
    "llm_refined_scaled_mean60_over_close",
    "llm_refined_mean60_over_close_adj_turnover",
    "llm_refined_ema40_close_ratio",
    "llm_refined_gp_optimized_smooth_mean60_over_close",
    "llm_refined_close_mean60_ema_smooth",
    "llm_refined_close_mean60_over_turn_price20",
    "llm_refined_close_mean60_vwap",
    "llm_refined_gp_mean60_over_close_decay5",
    "llm_refined_mean40_over_ema10",
]
