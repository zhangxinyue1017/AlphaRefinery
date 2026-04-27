'''Search package for family-state refinement.

Exports policy, state, engine, scoring, normalization, stage-transition, and run-ingest helpers.
'''

from __future__ import annotations

from .decision_context import DecisionContext, FamilyDecisionState
from .decision_engine import DecisionEngine
from .decision_features import CandidateDecisionFeatures
from .context_resolver import (
    ContextEvidence,
    ContextProfile,
    OrchestrationProfile,
    resolve_context_profile,
    resolve_orchestration_profile,  # deprecated: use resolve_stage_transition / resolve_stage_transition_from_state
)
from .engine import SearchEngine
from .frontier import SearchFrontier
from .normalization import SearchNormalizer, build_search_normalizer
from .policy import SearchPolicy
from .run_ingest import (
    load_candidate_records_from_completed_runs,
    load_multi_run_candidate_records,
    load_single_run_candidate_records,
    resolve_materialized_child_run_dir,
    resolve_materialized_multi_run_dir,
    resolve_materialized_single_run_dir,
)
from .scoring import compute_base_score, compute_frontier_score, winner_improved
from .stage_transition import (
    EvaluationFeedback,
    FamilyState,
    PhasePolicyRule,
    RefinementAction,
    StageTransitionDecision,
    StageTransitionEvidence,
    build_stage_transition_evidence,
    build_stage_transition_shadow,
    get_phase_policy_table,
    resolve_stage_transition,
    resolve_stage_transition_from_state,
)
from .state import SearchBudget, SearchEdge, SearchNode

__all__ = [
    "SearchBudget",
    "CandidateDecisionFeatures",
    "ContextEvidence",
    "ContextProfile",
    "OrchestrationProfile",
    "DecisionContext",
    "DecisionEngine",
    "EvaluationFeedback",
    "FamilyDecisionState",
    "FamilyState",
    "PhasePolicyRule",
    "RefinementAction",
    "SearchEdge",
    "SearchEngine",
    "SearchFrontier",
    "SearchNode",
    "SearchNormalizer",
    "SearchPolicy",
    "StageTransitionDecision",
    "StageTransitionEvidence",
    "build_stage_transition_evidence",
    "build_stage_transition_shadow",
    "get_phase_policy_table",
    "build_search_normalizer",
    "compute_base_score",
    "compute_frontier_score",
    "load_candidate_records_from_completed_runs",
    "load_multi_run_candidate_records",
    "load_single_run_candidate_records",
    "resolve_materialized_child_run_dir",
    "resolve_materialized_multi_run_dir",
    "resolve_materialized_single_run_dir",
    "resolve_context_profile",
    "resolve_orchestration_profile",
    "resolve_stage_transition",
    "resolve_stage_transition_from_state",
    "winner_improved",
]
