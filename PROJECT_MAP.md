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
├── requirements.txt
├── llm_refine_provider_env.example.sh
├── config/
├── docs/
└── factors_store/
```

### Top-level files

* [`README.md`](./README.md)

  * repository homepage and English project overview
* [`README_CN.md`](./README_CN.md)

  * Chinese project overview
* [`PROJECT_MAP.md`](./PROJECT_MAP.md)

  * repository navigation guide
* [`requirements.txt`](./requirements.txt)

  * Python dependencies for the public workflows
* [`llm_refine_provider_env.example.sh`](./llm_refine_provider_env.example.sh)

  * template for local provider configuration

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
└── assets/
    └── alpharefinery_cover.png
```

This directory stores documentation assets used by the repository, such as the README cover figure.

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
├── llm_refine/
└── autofactorset_bridge/
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

## `factors_store/autofactorset_bridge/`: optional downstream adapter

This layer provides an optional bridge between formal registry factors and downstream admission workflows.

It is not required for the core `llm_refine` research loop.

Representative scripts include:

* [`evaluate_registry_manifest.py`](./factors_store/autofactorset_bridge/evaluate_registry_manifest.py)
* [`batch_import_from_manifest.py`](./factors_store/autofactorset_bridge/batch_import_from_manifest.py)

---

## Public reading path

### If you are new to the repository

1. [README.md](./README.md)
2. [PROJECT_MAP.md](./PROJECT_MAP.md)
3. [factors_store/llm_refine/README.md](./factors_store/llm_refine/README.md)

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