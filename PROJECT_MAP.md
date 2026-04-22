# AlphaRefinery Project Map

This document is the repository navigation guide for **AlphaRefinery**.

It answers one practical question:

> **Where does each public part of the project live, and what is each layer responsible for?**

This file is intentionally different from the root [`README.md`](./README.md):

- [`README.md`](./README.md)
  - explains **what AlphaRefinery is**, its flagship subsystem, and the main workflow
- [`PROJECT_MAP.md`](./PROJECT_MAP.md)
  - explains **how the public repository is organized**

In short:

- **README** = what the project is
- **PROJECT_MAP** = where the public pieces live

> Note: this document describes the **public repository structure**.  
> Local/private factor assets, provider secrets, heavy runtime artifacts, and research-specific outputs may exist outside version control and are intentionally not documented here in detail.

---

## Top-level layout

```text
AlphaRefinery/
├── README.md
├── README_CN.md
├── PROJECT_MAP.md
├── pyproject.toml
├── requirements.txt
├── llm_refine_provider_env.example.sh
├── run_refine.sh
├── config/
├── docs/
├── factors_store/
├── scripts/
└── artifacts/
```

### Top-level files

* [`README.md`](./README.md)

  * repository homepage and English project overview
* [`README_CN.md`](./README_CN.md)

  * Chinese project overview
* [`PROJECT_MAP.md`](./PROJECT_MAP.md)

  * repository navigation guide
* [`requirements.txt`](./requirements.txt)

  * flat dependency list for environments that do not use editable installs
* [`pyproject.toml`](./pyproject.toml)

  * packaging metadata for standalone editable installs with `pip install -e .`
* [`llm_refine_provider_env.example.sh`](./llm_refine_provider_env.example.sh)

  * template for local provider configuration
* [`run_refine.sh`](./run_refine.sh)

  * convenience shell entrypoint for local refinement runs

---

## `config/`: tracked shared configuration

```text
config/
├── factor_manifests/
│   ├── alpha158.yaml
│   └── alpha360.yaml
└── refinement_seed_pool.yaml
```

This directory stores tracked shared configuration.

* [`config/refinement_seed_pool.yaml`](./config/refinement_seed_pool.yaml)

  * seed-family configuration for `llm_refine`
* [`config/factor_manifests/`](./config/factor_manifests/)

  * example manifests for public factor workflows

---

## `docs/`: documentation assets

```text
docs/
├── family_search_formulation.md
└── assets/
    └── alpharefinery_cover.png
```

This directory stores project-level method notes and documentation assets.

* [`docs/family_search_formulation.md`](./docs/family_search_formulation.md)

  * formalizes family-level staged refinement as a sequential search / decision problem
* [`docs/assets/`](./docs/assets/)

  * README and documentation assets

---

## `factors_store/`: core code layer

```text
factors_store/
├── __init__.py
├── __main__.py
├── _bootstrap.py
├── contract.py
├── data.py
├── data_paths.py
├── eval.py
├── operators.py
├── registry.py
├── _vendor/
├── factors/
└── llm_refine/
```

This is the main code layer of AlphaRefinery.

### Foundation modules

* [`contract.py`](./factors_store/contract.py)

  * field-level data contract
* [`data.py`](./factors_store/data.py)

  * data loading and normalization
* [`data_paths.py`](./factors_store/data_paths.py)

  * shared local data-path handling
* [`operators.py`](./factors_store/operators.py)

  * common factor operators
* [`registry.py`](./factors_store/registry.py)

  * central factor registry
* [`eval.py`](./factors_store/eval.py)

  * daily-frequency evaluation entry

---

## `factors_store/_vendor/`: vendored runtime support

This directory contains vendored runtime components used by AlphaRefinery, including operator/runtime support and evaluation helpers.

It is supporting infrastructure rather than the flagship contribution of the project.

---

## `factors_store/factors/`: public factor layer

```text
factors_store/factors/
├── __init__.py
├── alpha101_like.py
├── alpha158_like.py
├── alpha191_like.py
├── alpha360_like.py
├── seed_baselines.py
└── llm_refined/
```

This layer stores the **public factor code** retained in the repository.

### Public factor groups

* `alpha*_like.py`

  * benchmark-style alpha families kept as public examples / baselines
* `seed_baselines.py`

  * baseline / seed factors used for initialization or comparison
* `llm_refined/`

  * public scaffolding for promoted refinement families

### `llm_refined/`

```text
factors_store/factors/llm_refined/
├── __init__.py
└── common.py
```

This directory represents the **formal promotion target** for refinement results.

In the public repository, it keeps only shared scaffolding and helper utilities.
Local/private `*_family.py` results are intentionally excluded from version control.

### Local ignored integrations

Local optional downstream adapters, such as private admission or factor-library bridge code, may exist under the working tree for internal workflows.
They are intentionally ignored by git and are not part of the public/core `llm_refine` repository surface.

---

## `factors_store/llm_refine/`: flagship research engine

```text
factors_store/llm_refine/
├── README.md
├── config.py
├── cli/
├── core/
├── docs/
├── evaluation/
├── knowledge/
├── parsing/
├── prompting/
└── search/
```

This is the **flagship subsystem** of AlphaRefinery.

`llm_refine` is responsible for:

* family-level factor refinement,
* staged search progression,
* branch preservation,
* target-conditioned search,
* evaluation-aware continuation,
* and promotion-oriented research loops.

For subsystem-level usage and design details, see:

* [`factors_store/llm_refine/README.md`](./factors_store/llm_refine/README.md)

### Main subpackages

* `cli/`

  * runnable entrypoints and orchestration
* `core/`

  * providers, models, archive, seed loading
* `prompting/`

  * prompt construction and prompt planning
* `parsing/`

  * parser, validator, repair, expression execution
* `evaluation/`

  * evaluator, redundancy, promotion, research funnel
* `knowledge/`

  * retrieval, reflection, round1/bootstrap support, next-step planning
* `search/`

  * search state, policy, objectives, frontier, decision logic
* `docs/`

  * subsystem-specific design and tuning notes

---

## `scripts/`: local maintenance helpers

```text
scripts/
├── organize_runs_by_family.py
└── refresh_run_indexes.py
```

These scripts support local artifact organization and run-index refresh tasks.
They are not required by the core `llm_refine` loop, but are useful for maintaining long-running research workspaces.

---

## `artifacts/`: runtime output root

```text
artifacts/
└── README.md
```

The repository keeps only the top-level artifact README under version control.
Runtime outputs are ignored by git, including:

* `artifacts/runs/**`
* `artifacts/reports/**`
* `artifacts/logs/**`
* `artifacts/llm_refine_promotions/**`
* optional local downstream integration outputs

This keeps the standalone repository lightweight while preserving a stable default output layout for local research runs.

---

## Public reading path

### If you are new to the repository

1. [README.md](./README.md)
2. [PROJECT_MAP.md](./PROJECT_MAP.md)
3. [docs/family_search_formulation.md](./docs/family_search_formulation.md)
4. [factors_store/llm_refine/README.md](./factors_store/llm_refine/README.md)

### If you want to inspect the flagship subsystem

Start with:

* `factors_store/llm_refine/`
* [`factors_store/llm_refine/README.md`](./factors_store/llm_refine/README.md)
* `factors_store/llm_refine/search/`
* `factors_store/llm_refine/evaluation/`

### If you want to inspect public factor examples

Start with:

* `factors_store/factors/`
* [`factors_store/registry.py`](./factors_store/registry.py)
* [`factors_store/operators.py`](./factors_store/operators.py)

---

## One-line mental model

If you remember only one thing, remember this:

* `factors_store/factors/` = public formal factor examples
* `factors_store/llm_refine/` = flagship family-level refinement engine
* the rest of the repository = supporting infrastructure around that engine
