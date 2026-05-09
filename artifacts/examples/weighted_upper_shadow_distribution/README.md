# Weighted Upper Shadow Distribution Example

This example is a compact public artifact for the `weighted_upper_shadow_distribution`
family. It shows how AlphaRefinery turns a single seed factor into a maintained
family with multiple interpretable branches.

## Why this example

- The seed is easy to explain: upper-shadow events represent intraday rejection
  or selling pressure.
- The family demonstrates the main AlphaRefinery workflow: family-level search,
  branch preservation, target-conditioned refinement, and promotion-oriented
  review.
- The result is readable without exposing provider secrets, raw prompts, or large
  local runtime outputs.

## Seed

- Family: `weighted_upper_shadow_distribution`
- Canonical seed: `factor365.weighted_upper_shadow_frequency_40_hl10`
- Seed expression:
  `neg(weighted_mean(1((high - max(open, close)) / preclose > 0.01), 40, half_life=10))`
- Search interpretation: start from upper-shadow frequency, then refine toward
  amount weighting, turnover confirmation, and volatility-normalized geometry.

## What to read

- `tables/selected_candidates.csv`
  - compact table of representative candidates and metrics
- `reports/family_report_20260402.md`
  - family-level research summary
- `reports/research_funnel_report.md`
  - cross-family evaluator summary showing this family as the strongest public example
- `source_artifacts.json`
  - provenance for the original internal artifacts used to build this public example

## Main takeaways

The family converged into three useful branches:

- `upper-body rejection x amount`
- `turnover / relative-turnover confirmation`
- `shadow geometry / volatility normalization`

The strongest public narrative is not a single best formula. It is the transition
from a simple upper-shadow event seed into a small portfolio of related, testable
candidate structures.

## Re-run sketch

The original full experiment used provider-backed LLM runs. A minimal rerun shape
is:

```bash
python -m factors_store.llm_refine.cli.run_refine_multi_model \
  --family weighted_upper_shadow_distribution \
  --parent-factor factor365.weighted_upper_shadow_frequency_40_hl10 \
  --target-profile complementarity
```

For a fresh family-level controller run:

```bash
python -m factors_store.llm_refine.cli.run_refine_family_loop \
  --family weighted_upper_shadow_distribution \
  --seed factor365.weighted_upper_shadow_frequency_40_hl10 \
  --target-profile raw_alpha
```

Provider credentials are intentionally not included in this example.
