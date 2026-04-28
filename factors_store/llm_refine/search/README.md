# llm_refine.search Layout

`search/` is the strategy-control layer for family-state refinement. It is split
into four subpackages so search mechanics, candidate decisions, stage routing,
and artifact adapters stay separate.

| Layer | Path | Responsibility |
| --- | --- | --- |
| Core search | `search/core/` | Search nodes, budgets, frontier ranking, search scoring, normalizers, target-profile weights. |
| Candidate decision | `search/decision/` | Candidate features, winner/keep rerank, decorrelation scoring and gates. |
| Stage transition | `search/transition/` | Runtime context, legacy stage advisory, signal extraction, shadow table policy. |
| IO adapters | `search/io/` | Loading evaluated run artifacts into search records. |

## Dependency Direction

Prefer imports from the concrete layer that owns the concept:

```python
from ..search.core.policy import SearchPolicy
from ..search.decision.engine import DecisionEngine
from ..search.transition.stage_transition import resolve_stage_transition_from_state
from ..search.io.run_ingest import load_multi_run_candidate_records
```

The top-level `search/__init__.py` remains a public facade for common imports
such as `SearchEngine`, `SearchPolicy`, and `SearchBudget`.

## Public Facade

The top-level `search/__init__.py` is the only flat public facade. It re-exports
common symbols such as `SearchPolicy`, `SearchEngine`, `SearchBudget`, and
stage-transition dataclasses.

Implementation imports should use the layered paths directly. The old flat
implementation modules have been removed to keep the package boundary explicit.

## Naming Boundary

- `core/` answers: which parent/search node should we explore next?
- `decision/` answers: which evaluated child should be kept or treated as winner?
- `transition/` answers: which stage/action should the family run next?
- `io/` answers: how do completed run artifacts become search records?

## Current Policy Surfaces

| Surface | Owner | Notes |
| --- | --- | --- |
| Search scoring / frontier policy | `core/policy.py`, `core/scoring.py` | Target profiles, frontier scoring, branch value, MMR-related search behavior. |
| Candidate rerank / keep / winner | `decision/engine.py`, `decision/features.py` | Produces rerank previews and round-level best candidate / best keep records. |
| De-correlation policy | `decision/decorrelation_policy.py` | Unified grade, score, rerank adjustment, and complementarity/decorrelation gates. |
| Stage transition policy | `transition/stage_transition.py`, `transition/signals.py`, `transition/table_policy.py` | Legacy transition decision plus explicit signals and shadow table recommendation. |

For the stage signal thresholds and shadow table, see
[`../docs/stage_transition_signals.md`](../docs/stage_transition_signals.md).
