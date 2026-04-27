# llm_refine
> The flagship research engine inside AlphaRefinery for family-level LLM-guided factor refinement

## ✨ What makes `llm_refine` different

- 🧠 **Family-level refinement**, not one-shot formula mutation
- 🧭 **Broad -> Anchor -> Focused** staged search progression
- 🌿 **Dual-parent branch preservation** with path-aware continuation
- 🎯 **Target-conditioned search** beyond raw-alpha-only optimization
- 🧩 **Context-aware decision support** for rerank, anchor selection, and next-step recommendation
- 🪢 **Shared context alignment** across prompting, decision trace, and orchestration
- 🪄 **De-correlation-aware refinement** for lower-redundancy candidate generation

---

## Overview

`llm_refine` is the core research subsystem inside **AlphaRefinery** for **family-level factor search, refinement, evaluation, and continuation**.

It is not designed as a thin prompt wrapper or a one-off expression generator.  
Instead, it organizes factor research as a **structured search process** over a factor family, where each round can contribute to:

- search-space opening,
- parent selection,
- branch continuation,
- evaluation-aware rerank,
- archive and promotion decisions,
- and next-step orchestration.

At a high level, `llm_refine` connects:

- seed family definition,
- parent selection,
- LLM proposal generation,
- parser / repair,
- evaluation / redundancy checks,
- archive / promotion,
- search-state update and continuation.

In practice, it behaves more like:

- a **family-level factor research engine**,
- a **best-first / dual-parent search layer**,
- and a **control surface for iterative research progression**.

For repository-level context, see:

- [AlphaRefinery/README.md](../../README.md)

---

## Why `llm_refine`

Many factor-generation workflows stop at one of these points:

- mutating a single seed,
- sampling multiple expressions from a prompt,
- evaluating a local batch,
- or manually continuing from a top candidate.

`llm_refine` focuses on a different problem:

> **How to turn factor refinement into a staged, repeatable, and controllable family-level research process.**

This means the subsystem is explicitly designed around:

- staged search rather than flat generation,
- continuation policy rather than isolated rounds,
- branch preservation rather than immediate top1 collapse,
- and target-aware refinement rather than raw-alpha-only expansion.

---

## Scope of this README

This document focuses on the `llm_refine` subsystem itself:

1. what problem it solves,
2. how the search process is organized,
3. which entry points fit which research scenarios,
4. how the internal layers are divided,
5. and where to continue reading in code and docs.

---

## Core Ideas

### 1. Family-level search, not isolated formula proposals

`llm_refine` treats refinement as search over a **family-level state**, not as a sequence of disconnected candidate batches.

This is why it supports:

- family loop orchestration,
- staged progression control,
- parent selection beyond immediate top1,
- branch-aware continuation,
- and search-policy tuning.

### 2. Controlled progression, not flat batch generation

The subsystem can explicitly separate:

- **Broad** exploration for motif opening and search-space coverage,
- **Anchor** graduation for parent selection,
- **Focused** continuation for local deepening and confirmation.

This gives the system a more deliberate search progression than plain multi-sample prompting.

### 3. Branch preservation, not premature winner-take-all collapse

Strong families do not always evolve along a single line.  
`llm_refine` therefore preserves branch diversity long enough to let the search process learn from it.

This includes:

- dual-parent continuation,
- path-aware evaluation,
- comparative continuation across rounds,
- and parent selection beyond raw top1 scores.

### 4. Target-conditioned refinement, not raw-alpha-only optimization

The subsystem can refine toward different research objectives, such as:

- `raw_alpha`
- `deployability`
- `complementarity`
- `robustness`

This makes refinement more useful for downstream research goals such as:

- promotion,
- redundancy control,
- factor library complementarity,
- and optional admission-oriented evaluation.

### 5. Context-aware decision support, not scattered local heuristics

Several decision points that were previously separate are being unified into a more shared decision layer.

This currently includes:

- round-level rerank,
- anchor selection,
- family-level next action recommendation,
- optional de-correlation-aware candidate preference.

The goal is not over-automation.  
The goal is to make the refinement loop **more consistent, more traceable, and easier to reason about**.

---

## Current Key Capabilities

- `run_refine_loop`
  - single-round smoke / single-parent refine
- `run_refine_multi_model`
  - focused multi-model round
- `run_refine_multi_model_scheduler`
  - unified search + multi-round scheduler
- `run_refine_family_explore`
  - multi-seed breadth exploration
  - transitional family-level orchestration entry
- `run_refine_family_loop`
  - `Broad -> Anchor Graduation -> Focused` family controller v1
- MMR rerank
- dual-parent round v1
- Path Evaluation v2
- Target-Conditioned Search v1
- Context-aware decision layer v1
  - round-level rerank
  - family-loop anchor selection
  - next action recommendation
- De-correlation refine support
  - target-aware prompt block
  - de-correlation diagnostics
  - rerank hook
- `NA-heavy keep / best_node` tightening

---

## Typical Research Flow

A simplified view of how `llm_refine` is typically used:

```mermaid
graph LR
    A[Seed / Parent] --> B[Prompting]
    B --> C[LLM Proposal]
    C --> D[Parsing / Repair]
    D --> E[Evaluation / Redundancy]
    E --> F[Rerank / Selection]
    F --> G[Archive / Promotion / Continuation]
```

For family-level control, the process can be staged as:

```mermaid
graph LR
    A[Family Start] --> B[Broad]
    B --> C[Anchor Graduation]
    C --> D[Focused]
    D --> E[Branch Continue / Promote / Stop]
```

---

## Before You Run Anything

For any `llm_refine` CLI workflow, the default convention is:

```bash
cd AlphaRefinery
cp -n ./llm_refine_provider_env.example.sh ./llm_refine_provider_env.sh
source ./llm_refine_provider_env.sh
```

### Why this is the default

* `run_refine_*` CLI tools read provider settings from environment variables unless they are explicitly overridden
* if those variables are not loaded, the CLI may fall back to built-in defaults
* those defaults are only intended as local fallbacks, not as the normal research configuration

So the current convention is:

* copy `llm_refine_provider_env.example.sh` into a local `llm_refine_provider_env.sh`
* source that local file
* then run any `run_refine_*`, scheduler, or family-loop command

---

## Directory Layout

```text
factors_store/llm_refine/
├── README.md
├── docs/
├── cli/
├── core/
├── prompting/
├── parsing/
├── evaluation/
├── knowledge/
└── search/
```

### Responsibility by layer

* `config.py`

  * shared default values for `llm_refine`, including common run / provider / path defaults
* `cli/`

  * runnable entry points and scheduler orchestration
* `core/`

  * provider / archive / model / seed loading infrastructure
* `prompting/`

  * prompt construction, export, PromptPlan, prompt block structuring
* `parsing/`

  * parser / validator / repair / expression engine
* `evaluation/`

  * evaluator / redundancy / promotion
* `knowledge/`

  * retrieval / reflection / next experiment planning
* `search/`

  * frontier / policy / engine / objectives / path evaluation
* `docs/`

  * subsystem design, tuning, and usage notes

---

## Most Common Entry Points

| Entry                              | Best for                                                  |
| ---------------------------------- | --------------------------------------------------------- |
| `run_refine_loop`                  | Smoke testing whether a family can run                    |
| `run_refine_multi_model`           | Focused round around an existing parent                   |
| `run_refine_multi_model_scheduler` | Automatically continuing across multiple rounds           |
| `run_refine_family_explore`        | New family breadth exploration when no main line is clear |
| `run_refine_family_loop`           | Broad pass, anchor selection, and focused continuation    |

---

## Mode Layering

`llm_refine` uses several orthogonal knobs to define what a round is trying to do and how it should behave:

* `stage_mode`
* `target_profile`
* `policy_preset`
* `mode`

| Layer           | Answers                                                   | Typical field    |
| --------------- | --------------------------------------------------------- | ---------------- |
| Stage layer     | What stage of the research process this round belongs to  | `stage_mode`     |
| Objective layer | What kind of factor this round prefers to optimize toward | `target_profile` |
| Style layer     | How aggressive or conservative the search should be       | `policy_preset`  |
| Searcher layer  | How candidates and frontier are organized                 | `mode`           |

---

## Additional Notes on the Decision Layer

Beyond the explicit mode layering above, recent iterations have started to consolidate previously scattered selection logic into a shared decision layer.

This currently unifies:

* multi-model round `best_candidate / best_keep / rerank preview`
* family-loop `anchor selection`
* family-loop `next action recommendation`

It reads context such as:

* `stage_mode`
* `target_profile`
* `policy_preset`
* optional `decorrelation_targets`
* neutralized evaluation diagnostics

The goal is not to introduce a large rule engine.
The goal is to let broad / focused / family-loop stages make **more coherent decisions under a shared context view**.

---

## Additional Notes on Shared Context Resolution

Between `DecisionContext` and `PromptPlan`, the subsystem has also added a lighter shared interpretation layer:

* `ContextEvidence`
* `ContextProfile`
* `resolve_context_profile(...)`
* `OrchestrationProfile`
* `resolve_orchestration_profile(...)`

This layer does not replace raw runtime arguments.
Instead, it interprets them into a reusable intermediate view shared by:

* prompt block planning,
* decision trace,
* and family / scheduler orchestration.

Current dimensions include:

* `search_phase`
* `exploration_pressure`
* `redundancy_pressure`
* `prompt_constraint_style`
* `memory_mode`
* `examples_mode`
* `branching_bias`
* `next_action_bias`

These are already recorded into `prompt_trace`, which makes it easier to diagnose:

* why a round was guided or strict,
* why donor / bootstrap blocks were enabled or disabled,
* whether the context looked more like opening, refining, or confirming.

An additional `OrchestrationProfile` has also been added for the scheduler / family-loop layer.

Its role is currently limited and conservative. It mainly serves:

* stage resolve recommendation,
* round strategy trace,
* explicit recording of promotion / parent selection / termination bias,
* orchestration-aware summaries in top-level `summary.json / summary.md`

The long-term goal is:

* first, let orchestration speak the same context language as prompting and decision
* then gradually automate only the most stable and most interpretable orchestration behaviors

---

## Additional Notes on De-correlation Refine

De-correlation is not a globally forced objective.
It is an explicit refinement direction that can be enabled when needed.

Currently implemented pieces include:

* de-correlation target sets in prompting,
* nearest-target / average-target correlation diagnostics in evaluation,
* soft rerank adjustment hooks.

This is most useful when:

* a family already has a strong main line,
* redundancy against the library becomes visible,
* and you want to continue producing candidates that are still good, but less similar.

---

## Additional Notes on PromptPlan and Prompt Block Structuring

Recent work has introduced a lighter structural layer in prompting, while **not changing the final natural-language prompt format shown to the model**.

This is not a hard prompt engine.
It is a relatively conservative `PromptPlan` layer that controls:

* prompt block inclusion,
* budget,
* and style.

It currently covers three main areas:

* `memory`

  * whether to include recent winners / keeps / failures / lineage / reflection, and how many
* `constraints`

  * how to organize anti-patterns, allowed edit types, de-correlation guidance
* `examples`

  * whether family examples, bootstrap frontier, donor motifs are included, and in what amount

The point is not to compress away family semantics.
It is to make the previously scattered block-level if/else logic more explicit and traceable.

Current run artifacts already record:

* `stage_mode`
* `target_profile`
* `policy_preset`
* `context_evidence`
* `context_profile`
* `prompt_plan`

This makes it easier to inspect:

* why memory blocks were enabled,
* why donor motifs were not included,
* how constraints were structured,
* without inferring all of that only from the final prompt text.

### Per-run objective override

In addition to the static `primary_objective` and `secondary_objective` stored in `config/refinement_seed_pool.yaml`, all three main CLI entry points (`run_refine_loop`, `run_refine_multi_model`, `run_refine_multi_model_scheduler`) accept:

* `--primary-objective TEXT`
* `--secondary-objective TEXT`

When provided, these values **override** the seed-pool defaults and are injected directly into the prompt block:

```text
本轮优化优先级：
- 主目标：{primary_objective}
- 次目标：{secondary_objective}
```

This is useful when:

* you want to run the same family with a different emphasis on a given day (e.g., "lower turnover first" vs. "boost ICIR first"),
* you are experimenting with prompt framing without editing the tracked seed pool file,
* or you want scheduler rounds to vary objectives across stages.

Example:

```bash
python -m factors_store.llm_refine.cli.run_refine_multi_model_scheduler \
  --family open_volume_correlation \
  --primary-objective "降低换手，提升因子稳定性" \
  --secondary-objective "在不明显增加换手的前提下提升 RankICIR" \
  --stage-mode new_family_broad \
  --policy-preset exploratory \
  --n-candidates 8
```

---

## 1. `stage_mode`

`stage_mode` is the semantic stage label of a round.
It decides **what this round is doing** inside the overall family research process.

| `stage_mode`       | Meaning                                       | Typical behavior                                                 | Best for                     |
| ------------------ | --------------------------------------------- | ---------------------------------------------------------------- | ---------------------------- |
| `auto`             | Let the system infer the stage from context   | seed-stage may be enabled by context                             | compatibility / general runs |
| `new_family_broad` | First broad round for a new family            | force seed-stage, bootstrap / donor / richer role slots          | first launch of a new family |
| `broad_followup`   | Follow-up rounds within broad search          | still opening search space, but no longer first-round seed-stage | later broad rounds           |
| `focused_refine`   | Local refinement around an existing main line | smaller-step edits, less unnecessary branching                   | once an anchor exists        |
| `confirmation`     | Confirm stronger candidates                   | more stability / continuity oriented                             | before freeze / promotion    |
| `donor_validation` | Validate donor / transfer hypotheses          | focus on whether transferred motifs hold                         | donor experiments            |

In short:

* `stage_mode` answers **what this round is doing**

---

## 2. `target_profile`

`target_profile` is the objective preference of a round.
It decides **what kind of factor this round wants**.

Current options come from [policy.py](./search/policy.py):

| `target_profile`  | Meaning                                                 | Emphasis                                | Best for                             |
| ----------------- | ------------------------------------------------------- | --------------------------------------- | ------------------------------------ |
| `raw_alpha`       | prioritize stronger raw alpha                           | `Ann` / `Excess` / `ICIR`               | broad and early / mid focused search |
| `deployability`   | prioritize more constraint-friendly candidates          | constraint / deployability friendliness | late-stage refinement                |
| `complementarity` | prioritize lower redundancy and library complementarity | library diversification                 | admission-oriented refinement        |
| `robustness`      | prioritize stability across regimes and conditions      | stability / regime robustness           | confirmation / late-stage checks     |

In short:

* `target_profile` answers **what this round wants**

---

## 3. `policy_preset`

`policy_preset` is the search-style package.
It decides **how aggressively or conservatively the round should search**.

Current options also come from [policy.py](./search/policy.py):

| `policy_preset` | Meaning                              | Typical characteristics                                | Best for                      |
| --------------- | ------------------------------------ | ------------------------------------------------------ | ----------------------------- |
| `balanced`      | balanced default configuration       | moderate exploration / quality / redundancy trade-off  | most runs                     |
| `exploratory`   | more willing to open structure space | higher exploration / novelty bonus / larger pool       | broad runs, new families      |
| `conservative`  | emphasize safer local improvement    | stronger turnover / complexity penalty, less branching | focused / confirmation rounds |

In short:

* `policy_preset` answers **how this round searches**

---

## 4. `mode`

`mode` is closer to the searcher organization itself.
It decides **how the search tree or frontier is traversed**.

Typical modes in [policy.py](./search/policy.py) include:

| `mode`                   | Meaning                           | Typical behavior                                                               | Best for                           |
| ------------------------ | --------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------- |
| `multi_model_best_first` | multi-model best-first search     | unify candidates from multiple models and continue the most promising ones     | multi-model / scheduler main path  |
| `family_breadth_first`   | family-level breadth-first search | prefer opening branches and motifs rather than only chasing top1 neighborhoods | broad exploration / family explore |

In short:

* `mode` answers **how the search tree is walked**

---

## Recommended Combinations

| Task                                       | `stage_mode`       | `target_profile`                | `policy_preset`             | Common `mode`                                      |
| ------------------------------------------ | ------------------ | ------------------------------- | --------------------------- | -------------------------------------------------- |
| first broad round for a new family         | `new_family_broad` | `raw_alpha`                     | `exploratory`               | `multi_model_best_first` or `family_breadth_first` |
| later broad rounds                         | `broad_followup`   | `raw_alpha`                     | `exploratory` or `balanced` | `multi_model_best_first`                           |
| focused refinement around an existing line | `focused_refine`   | `raw_alpha`                     | `balanced`                  | `multi_model_best_first`                           |
| de-correlation / admission-oriented refine | `focused_refine`   | `complementarity`               | `balanced`                  | `multi_model_best_first`                           |
| late confirmation / freeze checks          | `confirmation`     | `robustness` or `deployability` | `conservative`              | `multi_model_best_first`                           |

---

## Current Dual-Parent Semantics

When dual-parent is triggered, the system currently keeps two parallel continuation lines:

* `quality_parent`
* `expandability_parent`

Current execution style:

* dual parent within the same round,
* child batches from both parents can launch in parallel,
* children are merged afterward for unified continuation and comparison.

---

## Target-Conditioned Search v1

Currently supported target profiles:

* `raw_alpha`
* `deployability`
* `complementarity`
* `robustness`

Currently implemented aware modules:

* `Constraint-aware`
* `Portfolio-aware`

Interfaces currently reserved:

* `Regime-aware`
* `Transfer-aware`

---

## Where to Read Next

### Want the formal search formulation

* [../../docs/family_search_formulation.md](../../docs/family_search_formulation.md)

### Want to know which entry to use

* [docs/modes.md](./docs/modes.md)

### Want to inspect frontier / MMR / dual-parent

* [docs/search_and_dual_parent.md](./docs/search_and_dual_parent.md)

### Want to tune `SearchPolicy`

* [docs/policy_tuning.md](./docs/policy_tuning.md)

### Want to inspect `research_keep / research_winner / promotion`

* [docs/evaluation_and_promotion.md](./docs/evaluation_and_promotion.md)

### Want to understand `Path Evaluation v2`

* [docs/path_evaluation.md](./docs/path_evaluation.md)

---

## Quick Examples

All examples assume you have already run:

```bash
cd AlphaRefinery
cp -n ./llm_refine_provider_env.example.sh ./llm_refine_provider_env.sh
source ./llm_refine_provider_env.sh
```

### Single-round smoke test

```bash
python -m factors_store.llm_refine.cli.run_refine_loop \
  --family salience_panic_score \
  --n-candidates 3 \
  --auto-parent
```

### Focused multi-model round

```bash
python -m factors_store.llm_refine.cli.run_refine_multi_model \
  --family weighted_upper_shadow_distribution \
  --models gpt-5.4,deepseek-v3.1,qwen3.5-plus \
  --policy-preset balanced \
  --target-profile complementarity \
  --n-candidates 6
```

### Multi-model scheduler with objective override

```bash
python -m factors_store.llm_refine.cli.run_refine_multi_model_scheduler \
  --family open_volume_correlation \
  --models gpt-5.4,deepseek-v3.1,qwen3.5-plus,kimi-k2,claude-sonnet-4-6 \
  --stage-mode new_family_broad \
  --policy-preset exploratory \
  --target-profile raw_alpha \
  --n-candidates 8 \
  --max-rounds 4 \
  --max-parallel 5 \
  --primary-objective "降低换手，提升因子稳定性" \
  --secondary-objective "在不明显增加换手的前提下提升 RankICIR" \
  --auto-apply-promotion
```

---

## Artifact Directories

* single-round:

  * `artifacts/runs/llm_refine_single/`
* multi-model:

  * `artifacts/runs/llm_refine_multi/`
* multi-model scheduler:

  * `artifacts/runs/llm_refine_multi_scheduler/`
* family explore:

  * `artifacts/runs/llm_refine_family_explore/`
* family loop:

  * `artifacts/runs/llm_refine_family_loop/`
