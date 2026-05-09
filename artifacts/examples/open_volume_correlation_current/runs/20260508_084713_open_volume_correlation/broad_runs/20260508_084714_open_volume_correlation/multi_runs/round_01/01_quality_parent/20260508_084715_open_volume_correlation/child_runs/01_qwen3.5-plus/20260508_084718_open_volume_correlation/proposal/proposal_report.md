# LLM Refine Run

- family: `open_volume_correlation`
- canonical_seed: `alpha101.alpha003`
- parent_factor: `llmgen.open_vol_amount_corr`

## Diagnosed Weaknesses
- 同家族冗余度较高
- 流动性暴露显著
- 信号抖动导致换手高
- 尺度敏感性较强

## Refinement Rationale
平滑输入降噪，引入波动过滤，替换流动性维度。

## Candidates
### open_vol_corr_stable
- candidate_role: conservative
- expression: `neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 10), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 40), 1e-12))))`
- explanation: 放宽相关窗口，平滑流动性分母，降低信号抖动。
- rationale: 窗口放宽减少噪声，分母平滑稳定信号。
- expected_improvement: stability
- risk: signal_weakening

### open_amt_cov_smooth
- candidate_role: donor_transfer
- expression: `neg(mul(ema(ts_cov(cs_rank(ema(open, 10)), cs_rank(ema(amount, 20)), 5), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 20), 1e-12))))`
- explanation: 借鉴平滑协方差动机，对开盘与金额做平滑协方差。
- rationale: 平滑输入提升协方差稳定性，借鉴成功动机。
- expected_improvement: rank_icir
- risk: overfitting
