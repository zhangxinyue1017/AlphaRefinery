# US Live LLM Round

Phase 2C used the existing `run_refine_loop` CLI, `config/us_research_pilot.yaml`, and the `open_volume_correlation` family. No market rows or raw panel were included in the model prompt. Credentials remained in the ignored `api_keys.env` file and were mapped to `LLM_API_KEY` only inside the process.

## Dry-run

- Run directory: `artifacts/us_research_live_2c/dry_run/runs/20261123_215305_open_volume_correlation`
- Provider/model configuration: `openai` / `gpt-5.4`
- Parent: `alpha101.alpha003`
- Parent expression: `-corr(rank(open), rank(volume), 10)`
- Requested candidates: 2
- Parse retries: 0
- Automatic promotion: disabled
- Search: 2023-01-03 through 2023-12-29
- Selection: 2024-01-02 through 2025-12-31
- Final OOS: unconfigured
- US settings: capitalization inference, proxy exposures, and neutralization disabled

The dry-run completed prompt construction, seed resolution, placeholder parsing, evaluation routing, search-state writing, and isolated archive/output creation.

## Live attempt

- Run ID: `run::open_volume_correlation::r1::e21091642f66`
- Run directory: `artifacts/us_research_live_2c/live/runs/20261123_215513_open_volume_correlation`
- Archive: `artifacts/us_research_live_2c/live/archive.db`
- Provider/model: `openai` / `gpt-5.4`
- Provider retries: disabled, including OpenAI SDK retries
- Parse/validation retries: 0
- Actual API request count: 1
- Token usage: unavailable because no completion was produced
- Final run status: `failed`

OpenAI returned HTTP 400 before generation because `gpt-5.4` requires `max_completion_tokens` instead of the legacy `max_tokens` parameter. The hard one-request cap was respected: there was no retry or second request. Consequently, no LLM response, candidate parsing, expression validation, factor evaluation, decision, or final search-state update exists for this live attempt. The isolated archive contains one failed run and no candidate, evaluation, or lineage rows.

The compatibility fix selects `max_completion_tokens` for OpenAI GPT-5 and o-series models while preserving `max_tokens` for existing compatible providers.

## Live GPT-5.6 Sol round

- Run ID: `run::open_volume_correlation::r1::1eb203784197`
- Run directory: `artifacts/us_research_live_2c/live_gpt56/runs/20261123_215842_open_volume_correlation`
- Archive: `artifacts/us_research_live_2c/live_gpt56/archive.db`
- Provider/model: `openai` / `gpt-5.6-sol`
- Actual API request count: 1
- Retries: disabled at provider, SDK, parsing, and validation levels
- Token usage: 3,144 prompt, 3,240 completion, 6,384 total; completion usage includes 2,419 reasoning tokens
- Runtime status: `completed` in 33.771 seconds
- Automatic promotion: disabled; pending manifest entries: 0

The provider response parsed successfully, and expression auto-repair normalized aliases and protected division denominators. Seed-stage role expansion requested six generated candidates even though the CLI final target was two. Light rerank retained these two:

1. `smoothed_gap_volume_corr`: `neg(ts_mean(ts_corr(cs_rank(div(open, add(pre_close, 1e-12))), cs_rank(volume), 10), 3))`
2. `gap_relative_volume_corr`: `neg(ts_corr(cs_rank(div(open, add(pre_close, 1e-12))), cs_rank(div(volume, add(ts_mean(volume, 20), 1e-12))), 10))`

Both passed parser/operator validation but failed factor evaluation in both search and selection with `unknown token: pre_close`. The US panel does not contain that field. Both received the formal decision `research_drop`; no metrics, retained child, or winner was produced. The archive contains one completed run, six generated candidate records, three selection evaluation rows including the parent, and six lineage edges.

The original CLI records `successful_rounds=1` even when the evaluated child list is empty because it registers every completed expansion with `success=True`. The final search state therefore contains only the seed node and no child nodes. This is a bookkeeping inconsistency in the original orchestration and was left unchanged for this phase.

The hard requirement of no more than two model-generated candidates was not met: `--n-candidates 2` controlled the post-rerank target, while automatic seed-stage role expansion asked the model for six. A future strict-cap invocation must avoid seed-stage expansion, for example by using the existing focused-refine stage with an explicit parent. This run is reproducible with the existing CLI arguments below, but repeating it would consume another model request:

```bash
export LLM_PROVIDER_NAME=openai
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY="$CHATGPT_API_KEY"
export LLM_MODEL=gpt-5.6-sol
export LLM_MAX_TOKENS=4000
export LLM_TIMEOUT=180
export LLM_DISABLE_PROVIDER_RETRIES=1

python -m factors_store.llm_refine.cli.run_refine_loop \
  --seed-pool config/us_research_pilot.yaml \
  --family open_volume_correlation \
  --n-candidates 2 \
  --runs-dir artifacts/us_research_live_2c/live_gpt56/runs \
  --archive-db artifacts/us_research_live_2c/live_gpt56/archive.db \
  --name-prefix us_llm_live \
  --round-id 1 \
  --retry-on-parse-fail 0 \
  --no-auto-apply-promotion
```

The ten-stock limitations from Phase 2A/2B remain unchanged: parent portfolio metrics are incomplete because five-quantile grouping fails on tied parent values, candidate groups contain roughly two stocks, and overlapping five-day forward-return portfolio metrics are engineering diagnostics rather than executable returns. This round verifies the live orchestration path and its failure handling, but it does not provide a successfully computed LLM candidate or a research winner.
