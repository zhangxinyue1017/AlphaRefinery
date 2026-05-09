# Light Rerank Report

- family: `open_volume_correlation`
- parent: `llmgen.open_vol_amount_corr`
- parent_expression: `neg(mul(ema(add(ts_corr(cs_rank(returns), cs_rank(open), 5), regression_rsq(cs_rank(returns), cs_rank(market_return), 60)), 20), div(ema(cs_rank(amount), 120), add(ema(cs_rank(amount), 20), 1e-12))))`
- requested_candidates: 6
- final_target: 2
- selected_count: 2
- dropped_count: 4
- role_slots: conservative, donor_transfer, decorrelating, confirmation, simplifying, stretch

## Selected

- `open_vol_corr_stable` | role=`conservative` | score=0.530 | parent_sim=0.900 | peer_sim=0.000 | reason=role_coverage:conservative
- `open_amt_cov_smooth` | role=`donor_transfer` | score=0.745 | parent_sim=0.595 | peer_sim=0.595 | reason=role_coverage:donor_transfer

## Dropped

- `open_turnover_corr_div` | role=`decorrelating` | score=0.315 | parent_sim=0.885 | peer_sim=0.885 | reason=
- `open_vol_corr_volgate` | role=`confirmation` | score=0.545 | parent_sim=0.775 | peer_sim=0.775 | reason=
- `open_vol_corr_simple` | role=`simplifying` | score=0.808 | parent_sim=0.592 | peer_sim=0.592 | reason=
- `open_vwap_corr_stretch` | role=`stretch` | score=0.636 | parent_sim=0.644 | peer_sim=0.758 | reason=
