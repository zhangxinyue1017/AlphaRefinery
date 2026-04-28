'''Stage-transition evidence, signals, and policy tables.'''

from __future__ import annotations

from .context_resolver import (
    ContextEvidence,
    ContextProfile,
    OrchestrationProfile,
    resolve_context_profile,
    resolve_orchestration_profile,
)
from .signals import SignalExtractor, StageTransitionSignals
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
from .table_policy import (
    ShadowPolicyRule,
    compare_stage_transition_decisions,
    get_shadow_stage_policy_table,
    resolve_shadow_table_policy,
)

__all__ = [
    "ContextEvidence",
    "ContextProfile",
    "EvaluationFeedback",
    "FamilyState",
    "OrchestrationProfile",
    "PhasePolicyRule",
    "RefinementAction",
    "ShadowPolicyRule",
    "SignalExtractor",
    "StageTransitionDecision",
    "StageTransitionEvidence",
    "StageTransitionSignals",
    "build_stage_transition_evidence",
    "build_stage_transition_shadow",
    "compare_stage_transition_decisions",
    "get_phase_policy_table",
    "get_shadow_stage_policy_table",
    "resolve_context_profile",
    "resolve_orchestration_profile",
    "resolve_shadow_table_policy",
    "resolve_stage_transition",
    "resolve_stage_transition_from_state",
]
