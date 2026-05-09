# LLM Refine Evaluation Report

- family: `open_volume_correlation`
- canonical_seed: `alpha101.alpha003`
- panel_path: `/root/dmd/BaoStock/panel.parquet`
- benchmark_path: `/root/dmd/BaoStock/Index/sh.000001.csv`
- start: `2023-01-03`
- end: `2026-03-09`
- horizon: `5`
- n_groups: `5`
- cost_bps: `10.0`
- rows_after_filter: `3485810`
- instruments_after_filter: `5248`

## Results

| Factor | Role | Model | WinnerScore | RankICIR | NRankICIR | Net Sharpe | NSharpe | StyleCorr | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `alpha101.alpha003` | canonical_seed | Baseline | NA | 0.5193 | 0.2232 | 4.2477 | 1.6485 | 0.0942 | canonical_seed |
| `llmgen.open_vol_amount_corr` | parent | Baseline | NA | 0.6686 | 0.2048 | 4.5023 | 0.2779 | 0.0715 | parent |
| `alpha101.alpha006` | peer | Baseline | NA | 0.4412 | 0.2798 | 4.0373 | 2.7588 | 0.1123 | peer |
| `llmgen.open_vol_corr_stable` | candidate | qwen3.5-plus | 0.9200 | 0.6163 | 0.1928 | 4.6216 | 1.0952 | 0.0770 | research_keep |
| `llmgen.open_amt_cov_smooth` | candidate | qwen3.5-plus | 0.0800 | 0.4152 | -0.0488 | 2.4776 | 1.2963 | 0.0207 | research_keep_exploratory |

## Parent Reference

- expression: `neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 5), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 20), 1e-12))))`
- RankIC: `0.0488`
- RankICIR: `0.6686`
- Neutral RankICIR: `0.2048`
- Net Ann Return: `1.3202`
- Net Excess Ann Return: `-0.3493`
- Neutral Net Sharpe: `0.2779`
- Mean Turnover: `0.2553`

## Research Gate Notes

### llmgen.open_vol_corr_stable
- model: qwen3.5-plus
- evaluation_stage: selection
- decision_scope: formal_decision
- expression: `neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 10), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 40), 1e-12))))`
- research_decision: research_keep
- reason: broad gate: beats parent on NetSharpe; full_metrics=4/4
- neutral RankICIR: 0.1928
- neutral NetSharpe: 1.0952
- avg_abs_style_corr: 0.0770
- nearest_decorrelation_target: 
- corr_to_nearest_decorrelation_target: NA
- avg_abs_decorrelation_target_corr: NA
- decorrelation_gate_action: pass
- decorrelation_gate_reason: no decorrelation diagnostics available
- decorrelation_keep_allowed: True
- decorrelation_winner_allowed: True
- decorrelation_reference_allowed: False
- top_style_exposure: STOM (-0.1818)
- raw_neutral_icir_gap: 0.4235
- raw_neutral_sharpe_gap: 3.5264
- neutral_winner_guard_passed: True
- stage_winner_guard: new_family_broad winner requires material gain on RankICIR / NetAnn / NetExcess

### llmgen.open_amt_cov_smooth
- model: qwen3.5-plus
- evaluation_stage: selection
- decision_scope: formal_decision
- expression: `neg(mul(ema(ts_cov(cs_rank(ema(open, 10)), cs_rank(ema(amount, 20)), 5), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 20), 1e-12))))`
- research_decision: research_keep_exploratory
- reason: exploration rescue (score=0.68): does not beat parent on main metrics but passes quality floor with novel structure
- neutral RankICIR: -0.0488
- neutral NetSharpe: 1.2963
- avg_abs_style_corr: 0.0207
- nearest_decorrelation_target: 
- corr_to_nearest_decorrelation_target: NA
- avg_abs_decorrelation_target_corr: NA
- decorrelation_gate_action: pass
- decorrelation_gate_reason: no decorrelation diagnostics available
- decorrelation_keep_allowed: True
- decorrelation_winner_allowed: True
- decorrelation_reference_allowed: False
- top_style_exposure: DASTD (-0.0574)
- raw_neutral_icir_gap: 0.4641
- raw_neutral_sharpe_gap: 1.1813
- neutral_winner_guard_passed: True
- stage_winner_guard: new_family_broad winner requires NetSharpe improvement
