# LLM Refine Research Funnel

这份评估默认遵循：先看分项指标，再看总分；总分只作辅助，不单独决定好坏。

当前主口径以 `sign-aware uplift` 为准：
- canonical seed 保留原始方向，方便回溯文献/注册来源
- preferred refine seed 代表当前系统默认 refine 方向
- family 总结与 profile 分层总结优先看 sign-aware 指标

admission/library 本版先不作为核心评判标准，避免把未定型的库门槛混入主评估。

## Run Summary

- total_runs: `6`
- total_families: `3`

### Top Runs by `delta_sign_aware_seed_to_winner_net_excess_ann_return`

| Family | Run Kind | Target | Winner | SignAware ΔExcess | SignAware ΔSharpe | SignAware ΔICIR | SignAware Uplift | Formalized |
|---|---|---|---|---:|---:|---:|---:|---:|
| weighted_upper_shadow_distribution | scheduler | default | `llmgen.shadow_wm_half_life_40` | 1.0717 | 5.5555 | 1.0005 | 2.6027 | 0 |
| weighted_upper_shadow_distribution | scheduler | complementarity | `llmgen.shadow_wma_smooth_20` | 0.8817 | 4.5827 | 0.9827 | 2.2042 | 3 |
| weighted_upper_shadow_distribution | scheduler | raw_alpha | `llmgen.shadow_amt_ema_15_th0015` | 0.8039 | 4.5457 | 0.9868 | 2.1459 | 3 |
| qp_high_price_distribution_pressure | scheduler | raw_alpha | `qp_high_price_distribution_pressure_init.qp_pressure_hp_vwap_vol_share` | 0.6924 | 8.7060 | 1.1354 | 2.5625 | 2 |
| gp_relative_volume_pressure | scheduler | complementarity | `llmgen.amt_press_ema_turn_mad_smooth` | 0.3206 | 1.9980 | 0.1806 | 0.8118 | 3 |
| gp_relative_volume_pressure | scheduler | raw_alpha | `llmgen.vol_press_tsrank_turn_ema` | -0.4298 | 3.0634 | 0.2778 | 0.8968 | 2 |

## Family Summary

| Family | Canonical Seed | Preferred Seed | Runs | SignAware Median ΔExcess | KeepRate | WinnerRate | Top3 +ExcessRate | Formalized | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| weighted_upper_shadow_distribution | `factor365.weighted_upper_shadow_frequency_40_hl10` | `factor365.weighted_upper_shadow_frequency_40_hl10` | 3 | 0.8817 | 0.6667 | 0.1605 | 1.0000 | 6 | donor_mode |
| qp_high_price_distribution_pressure | `qp_pressure.high_price_volume_share_20` | `qp_pressure.high_price_volume_share_20` | 1 | 0.6924 | 0.0759 | 0.0127 | 0.0000 | 2 | keep_refining |
| gp_relative_volume_pressure | `gp_mined.volume_mean30_over_volume` | `gp_mined.volume_mean30_over_volume` | 2 | -0.0546 | 0.6548 | 0.2071 | 0.5000 | 5 | keep_refining |

## Family x Profile Summary

| Family | Profile | Runs | SignAware Median ΔExcess | KeepRate | WinnerRate | Top3 +ExcessRate | Formalized | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gp_relative_volume_pressure | complementarity | 1 | 0.3206 | 0.6429 | 0.2143 | 0.6667 | 3 | keep_refining |
| gp_relative_volume_pressure | raw_alpha | 1 | -0.4298 | 0.6667 | 0.2000 | 0.3333 | 2 | keep_refining |
| qp_high_price_distribution_pressure | raw_alpha | 1 | 0.6924 | 0.0759 | 0.0127 | 0.0000 | 2 | keep_refining |
| weighted_upper_shadow_distribution | default | 1 | 1.0717 | 0.7778 | 0.1481 | 1.0000 | 0 | focused_refine_only |
| weighted_upper_shadow_distribution | complementarity | 1 | 0.8817 | 0.6111 | 0.1667 | 1.0000 | 3 | donor_mode |
| weighted_upper_shadow_distribution | raw_alpha | 1 | 0.8039 | 0.6111 | 0.1667 | 1.0000 | 3 | donor_mode |
