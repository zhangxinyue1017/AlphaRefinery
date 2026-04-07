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


def llm_refined_vwap_anchor_net_pressure_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_31
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_mean(vwap, 5)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_mean(vwap, 5)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.vwap_anchor_net_pressure_20",
    )


def llm_refined_low_price_range_pressure_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(low, ts_quantile(low, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(high, ts_quantile(high, 20, 0.7)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_price_range_pressure_20",
    )


def llm_refined_low_price_amt_pressure_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_price_amt_pressure_20",
    )


def llm_refined_low_price_mean_anchor_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_mean(close, 20)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_mean(close, 20)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_price_mean_anchor_20",
    )


def llm_refined_net_pressure_lowprice_relvol_turnover_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(div(turnover, add(ts_mean(turnover, 20), 1e-12)), 5))",
        factor_name="llm_refined.net_pressure_lowprice_relvol_turnover_confirm",
    )


def llm_refined_net_pressure_lowprice_turnover_vwap_confirm(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(vwap, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(vwap, 20, 0.7)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.net_pressure_lowprice_turnover_vwap_confirm",
    )


def llm_refined_low_vwap_net_amt_ema5(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(vwap, ts_quantile(vwap, 20, 0.3)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12)), div(ts_sum(if_then_else(ge(vwap, ts_quantile(vwap, 20, 0.7)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12))), ema(turnover, 5))",
        factor_name="llm_refined.low_vwap_net_amt_ema5",
    )


def llm_refined_net_pressure_lowprice_turnover_confirm_vwap(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(vwap, ts_quantile(vwap, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(vwap, ts_quantile(vwap, 20, 0.7)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.net_pressure_lowprice_turnover_confirm_vwap",
    )


def llm_refined_net_pressure_lowprice_turnover_smooth20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), amount, 0), 20), add(ts_sum(amount, 20), 1e-12))), ts_mean(turnover, 10))",
        factor_name="llm_refined.net_pressure_lowprice_turnover_smooth20",
    )


def llm_refined_lowprice_net_pressure_relvol_adj_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: claude-sonnet-4-6
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(div(volume, add(ts_mean(volume, 20), 1e-12)), 5))",
        factor_name="llm_refined.lowprice_net_pressure_relvol_adj_20",
    )


def llm_refined_low_tail_shadow_net_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_0
    source model: kimi-k2
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), mul(volume, sub(high, close)), 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), mul(volume, sub(close, low)), 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_tail_shadow_net_20",
    )


def llm_refined_low_price_pressure_smooth_40(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_33
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 40, 0.3)), volume, 0), 40), add(ts_sum(volume, 40), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 40, 0.7)), volume, 0), 40), add(ts_sum(volume, 40), 1e-12))), ts_mean(turnover, 10))",
        factor_name="llm_refined.low_price_pressure_smooth_40",
    )


def llm_refined_low_price_rel_amt_confirm_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_33
    source model: qwen3.5-plus
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))), div(ts_mean(amount, 10), add(ts_mean(amount, 20), 1e-12)))",
        factor_name="llm_refined.low_price_rel_amt_confirm_20",
    )


def llm_refined_low_tail_share_bias_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_46
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), mul(volume, div(sub(high, close), add(high, 1e-12))), 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), mul(volume, div(sub(close, low), add(close, 1e-12))), 0), 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_tail_share_bias_20",
    )


def llm_refined_low_tail_percent_vol_share_10_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_54
    source model: claude-sonnet-4-6
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), volume, 0), 10), add(ts_sum(volume, 10), 1e-12)), div(ts_sum(volume, 20), add(ts_sum(volume, 20), 1e-12))), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_tail_percent_vol_share_10_20",
    )


def llm_refined_low_tail_share_confirm_pos_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_53
    source model: qwen3.5-plus
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="sub(div(ts_sum(if_then_else(and(le(close, ts_quantile(close, 20, 0.3)), gt(close, open)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12)), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_tail_share_confirm_pos_20",
    )


def llm_refined_low_tail_net_shadow_share_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_51
    source model: gpt-5.4
    keep-drop status: research_winner
    """
    return evaluate_expression_factor(
        data,
        expression="sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), mul(volume, div(sub(high, close), add(high, 1e-12))), 0), 20), add(ts_sum(volume, 20), 1e-12)), div(ts_sum(if_then_else(ge(close, ts_quantile(close, 20, 0.7)), mul(volume, div(sub(close, low), add(close, 1e-12))), 0), 20), add(ts_sum(volume, 20), 1e-12)))",
        factor_name="llm_refined.low_tail_net_shadow_share_20",
    )


def llm_refined_low_tail_amount_bias_20(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_51
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), mul(amount, div(sub(high, close), add(high, 1e-12))), mul(amount, div(sub(close, low), add(close, 1e-12)))), 20), add(ts_sum(amount, 20), 1e-12)), ts_mean(turnover, 5)), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_tail_amount_bias_20",
    )


def llm_refined_low_tail_share_bias_smooth10(data: dict[str, pd.Series]) -> pd.Series:
    """parent factor: qp_pressure.net_pressure_20
    round: llm_refine_round_51
    source model: gpt-5.4
    keep-drop status: research_keep
    """
    return evaluate_expression_factor(
        data,
        expression="mul(sub(div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), mul(volume, div(sub(high, close), add(high, 1e-12))), mul(volume, div(sub(close, low), add(close, 1e-12)))), 20), add(ts_sum(volume, 20), 1e-12)), ts_mean(turnover, 10)), ts_mean(turnover, 5))",
        factor_name="llm_refined.low_tail_share_bias_smooth10",
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
    FactorSpec(
        name="llm_refined.vwap_anchor_net_pressure_20",
        func=llm_refined_vwap_anchor_net_pressure_20,
        required_fields=("close", "turnover", "volume", "vwap"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r31::b24fdc12fd32; round_id=31; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.06850736736800961, NetAnn=1.44891504348087, Turnover=0.13784212402143955.",
    ),
    FactorSpec(
        name="llm_refined.low_price_range_pressure_20",
        func=llm_refined_low_price_range_pressure_20,
        required_fields=("high", "low", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r33::f840b0bd28d2; round_id=0; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.070759294022393, NetAnn=3.582259953053269, Turnover=0.1451469013104364.",
    ),
    FactorSpec(
        name="llm_refined.low_price_amt_pressure_20",
        func=llm_refined_low_price_amt_pressure_20,
        required_fields=("amount", "close", "turnover"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r33::f840b0bd28d2; round_id=0; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.0726992698461867, NetAnn=3.462122363038701, Turnover=0.1439534314802692.",
    ),
    FactorSpec(
        name="llm_refined.low_price_mean_anchor_20",
        func=llm_refined_low_price_mean_anchor_20,
        required_fields=("amount", "close", "turnover"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r33::f840b0bd28d2; round_id=0; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.071580585419955, NetAnn=3.412353179001383, Turnover=0.1452232717040895.",
    ),
    FactorSpec(
        name="llm_refined.net_pressure_lowprice_relvol_turnover_confirm",
        func=llm_refined_net_pressure_lowprice_relvol_turnover_confirm,
        required_fields=("close", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r34::0bd329f6b9a2; round_id=0; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.0595037017920056, NetAnn=2.504297845021931, Turnover=0.1897644278120738.",
    ),
    FactorSpec(
        name="llm_refined.net_pressure_lowprice_turnover_vwap_confirm",
        func=llm_refined_net_pressure_lowprice_turnover_vwap_confirm,
        required_fields=("close", "turnover", "volume", "vwap"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r34::0bd329f6b9a2; round_id=0; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.0698895068371533, NetAnn=1.7285466308251758, Turnover=0.1244717521371594.",
    ),
    FactorSpec(
        name="llm_refined.low_vwap_net_amt_ema5",
        func=llm_refined_low_vwap_net_amt_ema5,
        required_fields=("amount", "turnover", "vwap"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r35::a70b10f0b791; round_id=0; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.0718273373406394, NetAnn=3.461905345052717, Turnover=0.1469533503504391.",
    ),
    FactorSpec(
        name="llm_refined.net_pressure_lowprice_turnover_confirm_vwap",
        func=llm_refined_net_pressure_lowprice_turnover_confirm_vwap,
        required_fields=("turnover", "volume", "vwap"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r39::b424580801a3; round_id=0; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.0703614294719753, NetAnn=3.4752957026455897, Turnover=0.1431185317218902.",
    ),
    FactorSpec(
        name="llm_refined.net_pressure_lowprice_turnover_smooth20",
        func=llm_refined_net_pressure_lowprice_turnover_smooth20,
        required_fields=("amount", "close", "turnover"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r39::b424580801a3; round_id=0; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.0713732483178242, NetAnn=3.421231973590265, Turnover=0.1265329181650595.",
    ),
    FactorSpec(
        name="llm_refined.lowprice_net_pressure_relvol_adj_20",
        func=llm_refined_lowprice_net_pressure_relvol_adj_20,
        required_fields=("close", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r39::b424580801a3; round_id=0; source_model=claude-sonnet-4-6; decision=research_keep. Selection summary: RankIC=0.0595432598241227, NetAnn=2.517204882316307, Turnover=0.189702112315389.",
    ),
    FactorSpec(
        name="llm_refined.low_tail_shadow_net_20",
        func=llm_refined_low_tail_shadow_net_20,
        required_fields=("close", "high", "low", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r40::f52c300db899; round_id=0; source_model=kimi-k2; decision=research_winner. Selection summary: RankIC=0.0792930148773068, NetAnn=4.566061395405938, Turnover=0.1390177581551766.",
    ),
    FactorSpec(
        name="llm_refined.low_price_pressure_smooth_40",
        func=llm_refined_low_price_pressure_smooth_40,
        required_fields=("close", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r33::f840b0bd28d2; round_id=33; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.06260556981995218, NetAnn=2.812845081260121, Turnover=0.08049264390218332.",
    ),
    FactorSpec(
        name="llm_refined.low_price_rel_amt_confirm_20",
        func=llm_refined_low_price_rel_amt_confirm_20,
        required_fields=("amount", "close", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r33::f840b0bd28d2; round_id=33; source_model=qwen3.5-plus; decision=research_keep. Selection summary: RankIC=0.05865151180507591, NetAnn=2.3825099895347983, Turnover=0.16225508180244042.",
    ),
    FactorSpec(
        name="llm_refined.low_tail_share_bias_20",
        func=llm_refined_low_tail_share_bias_20,
        required_fields=("close", "high", "low", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r46::4b567928f91f; round_id=46; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.08183160493605482, NetAnn=4.338235478041858, Turnover=0.14952149826772915.",
    ),
    FactorSpec(
        name="llm_refined.low_tail_percent_vol_share_10_20",
        func=llm_refined_low_tail_percent_vol_share_10_20,
        required_fields=("close", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r54::f4d2025a2e38; round_id=54; source_model=claude-sonnet-4-6; decision=research_winner. Selection summary: RankIC=0.08144370011672797, NetAnn=1.187815911927545, Turnover=0.1867161149099953.",
    ),
    FactorSpec(
        name="llm_refined.low_tail_share_confirm_pos_20",
        func=llm_refined_low_tail_share_confirm_pos_20,
        required_fields=("close", "open", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r53::abe303dd3be4; round_id=53; source_model=qwen3.5-plus; decision=research_winner. Selection summary: RankIC=0.06684063112444746, NetAnn=2.5686098201014422, Turnover=0.19102061365203818.",
    ),
    FactorSpec(
        name="llm_refined.low_tail_net_shadow_share_20",
        func=llm_refined_low_tail_net_shadow_share_20,
        required_fields=("close", "high", "low", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r51::c368c2884a51; round_id=51; source_model=gpt-5.4; decision=research_winner. Selection summary: RankIC=0.07852143054981309, NetAnn=4.194036014501205, Turnover=0.15271747067238586.",
    ),
    FactorSpec(
        name="llm_refined.low_tail_amount_bias_20",
        func=llm_refined_low_tail_amount_bias_20,
        required_fields=("amount", "close", "high", "low", "turnover"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r51::c368c2884a51; round_id=51; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=0.050020160415280776, NetAnn=1.7084722774118846, Turnover=0.20634626079225554.",
    ),
    FactorSpec(
        name="llm_refined.low_tail_share_bias_smooth10",
        func=llm_refined_low_tail_share_bias_smooth10,
        required_fields=("close", "high", "low", "turnover", "volume"),
        notes="Auto-promoted pending candidate for qp_low_price_accumulation_pressure; run_id=run::qp_low_price_accumulation_pressure::r51::c368c2884a51; round_id=51; source_model=gpt-5.4; decision=research_keep. Selection summary: RankIC=0.045225313189996494, NetAnn=1.5911509252262195, Turnover=0.17768257221799796.",
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
    "llm_refined_vwap_anchor_net_pressure_20",
    "llm_refined_low_price_range_pressure_20",
    "llm_refined_low_price_amt_pressure_20",
    "llm_refined_low_price_mean_anchor_20",
    "llm_refined_net_pressure_lowprice_relvol_turnover_confirm",
    "llm_refined_net_pressure_lowprice_turnover_vwap_confirm",
    "llm_refined_low_vwap_net_amt_ema5",
    "llm_refined_net_pressure_lowprice_turnover_confirm_vwap",
    "llm_refined_net_pressure_lowprice_turnover_smooth20",
    "llm_refined_lowprice_net_pressure_relvol_adj_20",
    "llm_refined_low_tail_shadow_net_20",
    "llm_refined_low_price_pressure_smooth_40",
    "llm_refined_low_price_rel_amt_confirm_20",
    "llm_refined_low_tail_share_bias_20",
    "llm_refined_low_tail_percent_vol_share_10_20",
    "llm_refined_low_tail_share_confirm_pos_20",
    "llm_refined_low_tail_net_shadow_share_20",
    "llm_refined_low_tail_amount_bias_20",
    "llm_refined_low_tail_share_bias_smooth10",
]
