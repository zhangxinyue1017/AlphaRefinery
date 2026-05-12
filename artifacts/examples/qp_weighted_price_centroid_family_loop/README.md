# QP Weighted Price Centroid Family Loop Example

This example packages a current `run_refine_family_loop` experiment for the
`qp_weighted_price_centroid` family. It shows how AlphaRefinery turns a public
seed into a stronger, volume-confirmed centroid factor through broad search,
anchor selection, and focused refinement.

## Why this example

- The seed is public and easy to explain:
  `qp_pressure.vwap_minus_mean_30`.
- The family-loop result has a clear measurable uplift from seed to anchor.
- The mechanism remains interpretable: weighted price centroid displacement
  becomes more useful after volume confirmation.
- The example also exposes limits: focused refinement adds only modest uplift,
  turnover rises, and decorrelation targets were not configured.

## Seed

- Family: `qp_weighted_price_centroid`
- Canonical seed: `qp_pressure.vwap_minus_mean_30`
- Seed pool formula:
  `volume_weighted_mean(close, volume, 30) - mean(close, 30)`
- Effective search direction:
  `neg(volume_weighted_mean(close, volume, 30) - mean(close, 30))`

Interpretation: when the volume-weighted price centroid is above the simple
price mean, trading is concentrated at more expensive locations. In this sample,
the useful direction is negative, consistent with fragile high-cost turnover.

## Main Results

| Stage | Factor | ICIR | Sharpe | TO |
|---|---|---:|---:|---:|
| Seed baseline | `qp_pressure.vwap_minus_mean_30` | 0.500 | 3.91 | 0.136 |
| Broad anchor | `llmgen.vwap_mean_volconfirm_20` | 0.732 | 6.06 | 0.216 |
| Focused best | `llmgen.llmgen_vwap_typical_medvol20` | 0.761 | 6.30 | 0.291 |

The main gain is from seed to anchor: ICIR improves from `0.500` to `0.732`.
Focused refinement improves the best anchor further to `0.761`, but the gain is
smaller and comes with higher turnover.

## What To Read

- `tables/seed_anchor_focused_comparison.csv`
  - aligned seed, anchor, and focused-best metrics
- `tables/top_candidates.csv`
  - compact top-candidate table from the run
- `reports/experiment_analysis_report.md`
  - full analysis of broad/focused stages, model contributions, and risks
- `reports/family_loop_summary.md`
  - controller-level stage transition and next-step summary
- `source_artifacts.json`
  - provenance for all copied and generated files

## Workflow Shown

1. Broad search starts from the sign-adjusted centroid seed.
2. The family-loop selects `vwap_mean_volconfirm_20` as the anchor.
3. Focused refinement explores nearby variants around typical price and volume
   normalization.
4. The best focused candidate replaces rolling mean volume with median volume,
   improving robustness to abnormal volume spikes.

## Limits

- This is a research/family-loop example, not an admission or production
  ingestion artifact.
- No provider credentials or raw provider environment are included.
- Decorrelation diagnostics are mostly `unknown` because this run did not set
  explicit decorrelation targets.
- Some top keeps have higher ICIR than the selected winner but also much higher
  turnover, so they need separate validation before reuse.

## Re-run Sketch

Source provider credentials separately, then run:

```bash
python -m factors_store.llm_refine.cli.run_refine_family_loop \
  --family qp_weighted_price_centroid \
  --target-profile raw_alpha \
  --models gpt-5.4,deepseek-v3.1,qwen3.5-plus,claude-sonnet-4-6,kimi-k2 \
  --broad-policy-preset exploratory \
  --focused-policy-preset balanced \
  --n-candidates 8 \
  --focused-n-candidates 6 \
  --broad-max-rounds 3 \
  --focused-max-rounds 3 \
  --max-parallel 5 \
  --no-auto-apply-promotion
```

The exact command used for this packaged run is also recorded in
`config/run_command.sh`, with credentials intentionally omitted.
