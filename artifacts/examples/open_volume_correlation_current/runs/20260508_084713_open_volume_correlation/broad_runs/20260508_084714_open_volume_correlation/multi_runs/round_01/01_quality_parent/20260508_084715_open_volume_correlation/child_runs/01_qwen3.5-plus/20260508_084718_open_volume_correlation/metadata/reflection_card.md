# Reflection Card

- family: `open_volume_correlation`
- run_id: `run::open_volume_correlation::r45::6d157bc298f9`
- generated_at: `2026-05-08T09:25:11.037333+00:00`
- summary: open_volume_correlation: winner=(none), keeps=2, failures=3

## Selected Parent
- factor_name: `llmgen.open_vol_amount_corr`
- candidate_id: `cand::open_volume_correlation::r42::2fbe706f075169f6`
- status: `explicit_seed`

## Current Winner
- (none)

## Current Keeps
- `llmgen.open_amt_cov_smooth` [research_keep_exploratory] model=qwen3.5-plus reason=exploration rescue (score=0.68): does not beat parent on main metrics but passes quality floor with novel structure
- `llmgen.open_vol_corr_stable` [research_keep] model=qwen3.5-plus reason=broad gate: beats parent on NetSharpe; full_metrics=4/4

## Current Failures
- `llmgen.open_vol_corr_simple` [drop_light_rerank] model=qwen3.5-plus reason=(none)
- `llmgen.open_vwap_corr_stretch` [drop_light_rerank] model=qwen3.5-plus reason=(none)
- `llmgen.open_turnover_corr_div` [drop_light_rerank] model=qwen3.5-plus reason=(none)

## Recent Winners
- `llmgen.open_vol_amount_corr` [research_winner] model=kimi-k2 reason=baseline row
- `llmgen.open_vol_decay_corr` [research_winner] model=claude-sonnet-4-6 reason=baseline row
- `llmgen.open_vol_corr_win20` [research_winner] model=qwen3.5-plus reason=best research keep candidate by composite winner_score=0.7560 (sharpe=1.00, excess=0.00, ann=1.00, icir=1.00, ic=1.00, turnover=0.20); broad gate: beats parent on RankIC, RankICIR; full_metrics=4/4

## Recent Keeps
- `llmgen.r2_turnover_weight_open_amount_norm` [research_keep] model=claude-sonnet-4-6 reason=broad gate: beats parent on RankIC, RankICIR, NetAnn, NetExcess, NetSharpe; full_metrics=4/4
- `llmgen.decorrelated_turnover_open_r2` [research_keep] model=claude-sonnet-4-6 reason=broad gate: beats parent on RankIC, RankICIR, NetAnn, NetExcess, NetSharpe; full_metrics=4/4
- `llmgen.r2_turnover_weight_smoothed` [research_keep] model=claude-sonnet-4-6 reason=broad gate: beats parent on RankIC, RankICIR, NetAnn, NetExcess, NetSharpe; full_metrics=4/4

## Recent Failures
- `llmgen.r2_turnover_weight_ema_decay` [drop_redundant_parent] model=claude-sonnet-4-6 reason=corr with parent llmgen.r2_turnover_weight = 0.9752 >= 0.950
- `llmgen.r2_turnover_weight_plus_volume_confirm` [drop_redundant_parent] model=claude-sonnet-4-6 reason=corr with parent llmgen.r2_turnover_weight = 0.9577 >= 0.950
- `llmgen.r2_turnover_weight_volatility_adj` [research_drop] model=claude-sonnet-4-6 reason=evaluation failed

## Top Success Tags
- cs_rank: 5
- ts_corr: 4
- amount: 3

## Top Failure Reasons
- corr with parent llmgen.r2_turnover_weight = 0.9752 >= 0.950: 1
- corr with parent llmgen.r2_turnover_weight = 0.9577 >= 0.950: 1
- evaluation failed: 1

## Model Contribution
- qwen3.5-plus: 2

## Suggested Next Focus
- 优先考虑增加平滑、分母稳定化或更低换手的确认项。
