# US Live LLM Pipeline Verification (Phase 2E)

Phase 2E used the existing `run_refine_loop` path after the Phase 2D prompt and field-validation changes in commit `bd1b5db`. The input panel was the ignored local Alpaca snapshot with SHA-256 `9985f004a46d67b73d3e20478456d5f0e4d671061f45d1abda771478dbc8839b`.

## Preflight

The print-only preflight used `focused_refine`, explicit parent `alpha101.alpha003`, and a target of two candidates. Assertions against the fully assembled prompt confirmed:

- available fields were exactly `close, high, low, open, returns, volume, vwap`;
- the model was asked for exactly two candidates, with no seed-stage role expansion;
- selection-stage metrics and raw market rows were absent;
- `pre_close` was unavailable and the prompt recommended `delay(close, 1)`;
- the search and selection windows came from `config/us_research_pilot.yaml`.

Preflight artifacts are under `artifacts/us_research_live_2e/preflight`. The live directory did not exist before the request.

## Live execution

- Run ID: `run::open_volume_correlation::r1::1b2246aa5ac6`
- Run directory: `artifacts/us_research_live_2e/live/runs/20261123_234224_open_volume_correlation`
- Archive: `artifacts/us_research_live_2e/live/archive.db`
- Provider/model: `openai` / `gpt-5.6-sol`
- Stage/family: `focused_refine` / `open_volume_correlation`
- Parent: `alpha101.alpha003`, `-corr(rank(open), rank(volume), 10)`
- Actual API requests: 1
- Retries: disabled in the provider/SDK and set to zero for parse/validation failures
- Token usage: 2,249 prompt, 1,751 completion, 4,000 total; completion usage included 1,347 reasoning tokens
- Runtime: 26.146 seconds; status `completed`
- Automatic promotion: disabled; pending promotion entries: 0

The credential was read from the ignored `api_keys.env` file and mapped to `LLM_API_KEY` only in the process. No credentials or raw market data were added to the prompt or artifacts tracked by Git.

The reproducible invocation is:

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
  --stage-mode focused_refine \
  --current-parent-name alpha101.alpha003 \
  --current-parent-expression '-corr(rank(open), rank(volume), 10)' \
  --n-candidates 2 \
  --runs-dir artifacts/us_research_live_2e/live/runs \
  --archive-db artifacts/us_research_live_2e/live/archive.db \
  --name-prefix us_llm_2e \
  --round-id 1 \
  --retry-on-parse-fail 0 \
  --prompt-template-version current_compact \
  --no-auto-apply-promotion
```

Do not rerun this command against the same archive when reproducing the recorded single-request experiment; use a fresh isolated output root.

## Generated candidates and validation

Both generated candidates parsed, passed validation after deterministic alias/safe-division repair, computed successfully, and entered search and selection evaluation.

| Candidate | Model expression | Executed expression | Validation result |
|---|---|---|---|
| `overnight_open_volume_corr` | `neg(corr(rank(div(open, delay(close, 1))), rank(volume), 10))` | `neg(ts_corr(cs_rank(div(open, add(delay(close, 1), 1e-12))), cs_rank(volume), 10))` | passed after `corr`/`rank` normalization and safe denominator insertion |
| `stable_open_volume_corr` | `neg(corr(rank(div(open, mean(delay(close, 1), 3))), rank(volume), 10))` | `neg(ts_corr(cs_rank(div(open, add(ts_mean(delay(close, 1), 3), 1e-12))), cs_rank(volume), 10))` | passed after `corr`/`rank`/`mean` normalization and safe denominator insertion |

## Evaluation and decisions

Search covered 2023-01-03 through 2023-12-29. Selection covered 2024-01-02 through 2025-12-31. The horizon was five days, with ten minimum stocks, five groups, 10 bps costs, no market-cap inference, no proxy exposures, and no neutralization. Final OOS remained unconfigured.

| Stage | Factor | Rank IC | ICIR | Net ann. return | Net excess ann. return | Net Sharpe | Turnover | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Search | Parent | 0.031546 | 0.097566 | NA | NA | NA | NA | parent |
| Search | Overnight | -0.068163 | -0.210929 | -0.880081 | -0.986916 | -4.139854 | 0.594714 | `research_drop` (diagnostic only) |
| Search | Stable | -0.010470 | -0.034301 | -0.686850 | -0.964558 | -1.937808 | 0.537778 | `research_drop` (diagnostic only) |
| Selection | Parent | 0.014178 | 0.041770 | NA | NA | NA | NA | parent |
| Selection | Overnight | 0.054917 | 0.177615 | 2.231382 | -0.318264 | 2.643804 | 0.623326 | `research_keep` |
| Selection | Stable | 0.041542 | 0.133384 | 1.582676 | -0.454290 | 2.150068 | 0.578919 | `research_keep` |

Selection is the configured formal decision stage. Both candidates were kept because they beat the parent on Rank IC and ICIR with all four required candidate metrics present. Neither became a `research_winner`: the winner guard requires ICIR at least 0.30. Their negative net excess returns also remain visible and were not used to weaken or bypass any gate.

The search state recorded one successful expansion, two child nodes, two keeps, zero winners, and `overnight_open_volume_corr` as the current best node. The archive contains one completed run, two candidates, three selection evaluations including the parent, and two lineage edges. This verifies generation, JSON parsing, expression validation, factor computation, both evaluation stages, formal research decisions, search-state update, lineage, report generation, and archive persistence.

## Limitations

- Ten stocks produce coarse cross-sectional ranks and roughly two stocks per portfolio group. These metrics are engineering diagnostics, not publication-quality evidence or an executable trading backtest.
- Parent portfolio metrics remain unavailable because tied/coarse parent values do not form usable five-quantile portfolios. The run emitted the existing all-NA `idxmax`/`idxmin` future warnings without failing.
- No benchmark file, neutralized metrics, style diagnostics, decorrelation target, or final OOS result was configured.
- The successful live round does not validate multi-round scheduling, promotion, or trading, all of which remained disabled or out of scope.

## Offline verification

`PYTHONPATH=. MPLCONFIGDIR=/tmp/alpharefinery-mpl /opt/anaconda3/bin/python -m pytest -q` passed all 20 tests. A preliminary bare `pytest -q` invocation failed during collection because that launcher did not place the repository root on `sys.path`; rerunning through Python with the project root on `PYTHONPATH` resolved the environment issue without a code change. `git diff --check` and the report secret/raw-symbol scan also passed.
