#!/usr/bin/env bash
set -euo pipefail

# Provider credentials are intentionally omitted. Source your own provider env first.
# Example:
#   source ./llm_refine_provider_env.sh

python -m factors_store.llm_refine.cli.run_refine_family_loop   --family qp_weighted_price_centroid   --target-profile raw_alpha   --models gpt-5.4,deepseek-v3.1,qwen3.5-plus,claude-sonnet-4-6,kimi-k2   --broad-policy-preset exploratory   --focused-policy-preset balanced   --n-candidates 8   --focused-n-candidates 6   --broad-max-rounds 3   --focused-max-rounds 3   --max-parallel 5   --no-auto-apply-promotion   --additional-notes "Preserve the volume-weighted price centroid mechanism, compare against simple mean/close anchors, prefer variants that are not near-duplicates of APB, low-price accumulation pressure, or high-price distribution pressure."
