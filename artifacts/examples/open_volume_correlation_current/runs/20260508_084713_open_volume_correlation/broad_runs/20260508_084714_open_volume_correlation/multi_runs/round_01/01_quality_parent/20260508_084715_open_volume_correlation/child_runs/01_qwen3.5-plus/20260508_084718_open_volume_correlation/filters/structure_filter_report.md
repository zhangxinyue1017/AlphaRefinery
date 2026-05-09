# Structure Filter Report

- family: `open_volume_correlation`
- canonical_seed: `alpha101.alpha003`
- kept_count: `2`
- dropped_count: `0`

## Kept Candidates
- `open_vol_corr_stable`: `neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 10), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 40), 1e-12))))`
- `open_amt_cov_smooth`: `neg(mul(ema(ts_cov(cs_rank(ema(open, 10)), cs_rank(ema(amount, 20)), 5), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 20), 1e-12))))`

## Dropped Candidates
- (none)
