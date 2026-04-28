'''Stage-transition advisory rules for family-state search.

Maps runtime evidence into continue, confirmation, complementarity, reopen, terminate, or repair recommendations.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return value == value


@dataclass(frozen=True)
class FamilyState:
    family_id: str
    stage: str = "auto"
    target_profile: str = "raw_alpha"
    parent_set: tuple[dict[str, Any], ...] = ()
    best_node: dict[str, Any] | None = None
    frontier_nodes: tuple[dict[str, Any], ...] = ()
    motif_state: dict[str, Any] = field(default_factory=dict)
    redundancy_state: dict[str, Any] = field(default_factory=dict)
    failure_state: dict[str, Any] = field(default_factory=dict)
    promotion_state: dict[str, Any] = field(default_factory=dict)
    budget_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RefinementAction:
    stage_mode: str = "auto"
    target_profile: str = "raw_alpha"
    policy_preset: str = "balanced"
    parent_selection: str = "best_node"
    decorrelation_targets: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    n_candidates: int = 0
    max_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationFeedback:
    status: str = ""
    search_improved: bool = False
    winner: dict[str, Any] | None = None
    keep: dict[str, Any] | None = None
    best_anchor: dict[str, Any] | None = None
    passed_anchor_count: int = 0
    focused_best_node: dict[str, Any] | None = None
    consecutive_no_improve: int = 0
    children_collected: int = 0
    children_added_to_search: int = 0
    high_corr_count: int = 0
    high_turnover_count: int = 0
    validation_fail_count: int = 0
    budget_exhausted: bool = False
    frontier_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageTransitionEvidence:
    family: str
    current_stage: str = "auto"
    target_profile: str = "raw_alpha"
    policy_preset: str = "balanced"
    last_round_status: str = ""
    last_round_search_improved: bool = False
    last_round_winner: dict[str, Any] | None = None
    last_round_keep: dict[str, Any] | None = None
    best_anchor: dict[str, Any] | None = None
    passed_anchor_count: int = 0
    focused_best_node: dict[str, Any] | None = None
    consecutive_no_improve: int = 0
    children_collected: int = 0
    children_added_to_search: int = 0
    high_corr_count: int = 0
    high_turnover_count: int = 0
    validation_fail_count: int = 0
    budget_exhausted: bool = False
    frontier_exhausted: bool = False
    has_decorrelation_targets: bool = False
    frontier_nodes: tuple[dict[str, Any], ...] = ()
    motif_state: dict[str, Any] = field(default_factory=dict)
    redundancy_state: dict[str, Any] = field(default_factory=dict)
    failure_state: dict[str, Any] = field(default_factory=dict)
    budget_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageTransitionDecision:
    current_stage: str
    next_stage: str
    action: str
    confidence: str
    reason: str
    rationale_tags: tuple[str, ...] = ()
    parent_selection_bias: str = "best_node"
    target_profile_bias: str = "keep_current"
    termination_bias: str = "normal"
    branch_reopen_candidates: tuple[str, ...] = ()
    mode: str = "advisory"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhasePolicyRule:
    rule_id: str
    phase: str
    input_signals: tuple[str, ...]
    output_action: str
    next_stage: str
    parent_selection_bias: str = "best_node"
    target_profile_bias: str = "keep_current"
    termination_bias: str = "normal"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PHASE_POLICY_TABLE: tuple[PhasePolicyRule, ...] = (
    PhasePolicyRule(
        rule_id="failed_round_repair",
        phase="any",
        input_signals=("last_round_status=failed", "validation_fail_count?"),
        output_action="repair_or_retry",
        next_stage="broad_followup_or_current",
        parent_selection_bias="diversify_branch",
        description="A failed run should not advance phase automatically; repair the failure first.",
    ),
    PhasePolicyRule(
        rule_id="empty_flat_stop_or_reopen",
        phase="any",
        input_signals=("no winner", "no keep", "children_collected<=0", "search_improved=false"),
        output_action="reopen_broad_or_freeze",
        next_stage="broad_followup_or_terminate",
        parent_selection_bias="diversify_branch",
        termination_bias="stop_if_not_focused",
        description="A flat round without usable children should either reopen another branch or stop the family.",
    ),
    PhasePolicyRule(
        rule_id="broad_anchor_graduation",
        phase="new_family_broad|broad_followup|family_loop",
        input_signals=("anchor_strength=passed", "passed_anchor_count>0"),
        output_action="graduate_anchor",
        next_stage="focused_refine",
        parent_selection_bias="best_anchor",
        description="A broad candidate that passes the anchor gate becomes the focused parent.",
    ),
    PhasePolicyRule(
        rule_id="broad_strong_winner",
        phase="new_family_broad|broad_followup|family_loop",
        input_signals=("search_improved=true", "winner_quality=strong_or_usable"),
        output_action="continue_focused",
        next_stage="focused_refine",
        parent_selection_bias="best_node",
        description="A broad winner with usable quality should be exploited as the mainline.",
    ),
    PhasePolicyRule(
        rule_id="broad_saturation",
        phase="new_family_broad|broad_followup|family_loop",
        input_signals=("children_collected>=20", "search_improved=false", "no winner", "no keep"),
        output_action="terminate",
        next_stage="terminate",
        termination_bias="stop",
        description="A broad phase that spends material candidate budget without usable output is saturated.",
    ),
    PhasePolicyRule(
        rule_id="focused_complementarity_confirm",
        phase="focused_refine",
        input_signals=("target_profile=complementarity", "winner_quality=usable"),
        output_action="confirmation",
        next_stage="confirmation",
        termination_bias="stop_early_if_flat",
        description="A usable complementarity winner should be confirmed rather than over-mined.",
    ),
    PhasePolicyRule(
        rule_id="focused_high_turnover_switch",
        phase="focused_refine",
        input_signals=("winner_quality=usable_or_strong", "turnover_pressure=true"),
        output_action="switch_to_complementarity",
        next_stage="focused_refine",
        parent_selection_bias="low_corr_parent",
        target_profile_bias="complementarity",
        description="A high-turnover focused winner should move into deployability/complementarity pressure.",
    ),
    PhasePolicyRule(
        rule_id="focused_material_gain_continue",
        phase="focused_refine",
        input_signals=("winner_quality=strong_or_material_gain", "material_gain=true"),
        output_action="continue_focused",
        next_stage="focused_refine",
        parent_selection_bias="best_node",
        description="A focused branch with material incremental value should continue.",
    ),
    PhasePolicyRule(
        rule_id="focused_corr_pressure_reopen",
        phase="focused_refine",
        input_signals=("corr_pressure=true", "decorrelation_targets?"),
        output_action="switch_to_complementarity",
        next_stage="focused_refine",
        parent_selection_bias="low_corr_parent",
        target_profile_bias="complementarity",
        description="A redundant focused branch should reopen a lower-corr branch or target complementarity.",
    ),
    PhasePolicyRule(
        rule_id="focused_turnover_confirmation",
        phase="focused_refine",
        input_signals=("turnover_pressure=true", "no stronger material gain"),
        output_action="confirmation",
        next_stage="confirmation",
        target_profile_bias="deployability",
        termination_bias="stop_early_if_flat",
        description="Turnover pressure without material gain should trigger deployability confirmation.",
    ),
    PhasePolicyRule(
        rule_id="focused_usable_no_gain_confirm",
        phase="focused_refine",
        input_signals=("winner_or_keep=true", "material_gain=false"),
        output_action="confirmation",
        next_stage="confirmation",
        termination_bias="stop_early_if_flat",
        description="A usable candidate without material incremental gain should be confirmed/frozen.",
    ),
    PhasePolicyRule(
        rule_id="confirmation_terminal",
        phase="confirmation|donor_validation",
        input_signals=("confirmation_context=true",),
        output_action="freeze_or_promote",
        next_stage="terminate",
        termination_bias="stop",
        description="Confirmation and donor validation are terminal advisory phases unless manually reopened.",
    ),
)


def get_phase_policy_table() -> tuple[dict[str, Any], ...]:
    """Return the explicit phase-policy table used to audit transition rules."""

    return tuple(rule.to_dict() for rule in PHASE_POLICY_TABLE)


def _is_strong(payload: dict[str, Any]) -> bool:
    icir = _safe_float(payload.get("quick_rank_icir"))
    sharpe = _safe_float(payload.get("net_sharpe"))
    excess = _safe_float(payload.get("net_excess_ann_return"))
    return bool(
        payload
        and (
            (_finite(icir) and icir >= 0.45 and _finite(sharpe) and sharpe >= 3.0)
            or (_finite(excess) and excess > 0.0 and _finite(sharpe) and sharpe >= 2.0)
        )
    )


def _is_usable(payload: dict[str, Any]) -> bool:
    icir = _safe_float(payload.get("quick_rank_icir"))
    sharpe = _safe_float(payload.get("net_sharpe"))
    return bool(payload and _finite(icir) and icir >= 0.45 and _finite(sharpe) and sharpe >= 2.0)


def _is_high_turnover(payload: dict[str, Any], threshold: float = 0.40) -> bool:
    turnover = _safe_float(payload.get("mean_turnover"))
    return bool(_finite(turnover) and turnover >= float(threshold))


def _material_gain(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    if not candidate or not baseline:
        return False
    excess_gain = _safe_float(candidate.get("net_excess_ann_return"), 0.0) - _safe_float(
        baseline.get("net_excess_ann_return"), 0.0
    )
    icir_gain = _safe_float(candidate.get("quick_rank_icir"), 0.0) - _safe_float(
        baseline.get("quick_rank_icir"), 0.0
    )
    sharpe_gain = _safe_float(candidate.get("net_sharpe"), 0.0) - _safe_float(
        baseline.get("net_sharpe"), 0.0
    )
    return excess_gain >= 0.02 or icir_gain >= 0.05 or sharpe_gain >= 0.25


def _factor_name(payload: dict[str, Any]) -> str:
    return str(payload.get("factor_name") or payload.get("candidate_name") or "").strip()


def build_stage_transition_evidence(
    state: FamilyState,
    action: RefinementAction,
    feedback: EvaluationFeedback,
) -> StageTransitionEvidence:
    failure_state = dict(state.failure_state or {})
    budget_state = dict(state.budget_state or {})
    redundancy_state = dict(state.redundancy_state or {})
    motif_state = dict(state.motif_state or {})
    return StageTransitionEvidence(
        family=str(state.family_id),
        current_stage=str(action.stage_mode or state.stage or "auto"),
        target_profile=str(action.target_profile or state.target_profile or "raw_alpha"),
        policy_preset=str(action.policy_preset or "balanced"),
        last_round_status=str(feedback.status or ""),
        last_round_search_improved=bool(feedback.search_improved),
        last_round_winner=dict(feedback.winner or {}),
        last_round_keep=dict(feedback.keep or {}),
        best_anchor=dict(feedback.best_anchor or {}),
        passed_anchor_count=int(feedback.passed_anchor_count or 0),
        focused_best_node=dict(feedback.focused_best_node or {}),
        consecutive_no_improve=int(
            feedback.consecutive_no_improve
            or budget_state.get("consecutive_no_improve")
            or 0
        ),
        children_collected=int(feedback.children_collected or budget_state.get("children_collected") or 0),
        children_added_to_search=int(
            feedback.children_added_to_search or budget_state.get("children_added_to_search") or 0
        ),
        high_corr_count=int(feedback.high_corr_count or failure_state.get("high_corr_count") or 0),
        high_turnover_count=int(
            feedback.high_turnover_count or failure_state.get("high_turnover_count") or 0
        ),
        validation_fail_count=int(
            feedback.validation_fail_count or failure_state.get("validation_fail_count") or 0
        ),
        budget_exhausted=bool(feedback.budget_exhausted or budget_state.get("budget_exhausted")),
        frontier_exhausted=bool(feedback.frontier_exhausted or budget_state.get("frontier_exhausted")),
        has_decorrelation_targets=bool(
            action.decorrelation_targets
            or redundancy_state.get("decorrelation_targets")
            or redundancy_state.get("has_decorrelation_targets")
        ),
        frontier_nodes=tuple(dict(item) for item in (state.frontier_nodes or ()) if isinstance(item, dict)),
        motif_state=motif_state,
        redundancy_state=redundancy_state,
        failure_state=failure_state,
        budget_state=budget_state,
    )


def resolve_stage_transition_from_state(
    state: FamilyState,
    action: RefinementAction,
    feedback: EvaluationFeedback,
) -> StageTransitionDecision:
    return resolve_stage_transition(build_stage_transition_evidence(state, action, feedback))


def build_stage_transition_shadow(
    *,
    legacy_decision: dict[str, Any],
    family_state_decision: StageTransitionDecision | dict[str, Any],
) -> dict[str, Any]:
    legacy = dict(legacy_decision or {})
    family_decision = (
        family_state_decision.to_dict()
        if isinstance(family_state_decision, StageTransitionDecision)
        else dict(family_state_decision or {})
    )
    legacy_stage = str(
        legacy.get("recommended_stage_mode")
        or legacy.get("recommended_next_stage_preset")
        or legacy.get("next_stage")
        or ""
    )
    legacy_action = str(
        legacy.get("round_strategy")
        or legacy.get("recommended_next_step")
        or legacy.get("action")
        or ""
    )
    family_stage = str(family_decision.get("next_stage") or "")
    family_action = str(family_decision.get("action") or "")
    return {
        "mode": "shadow",
        "legacy_next_stage": legacy_stage,
        "legacy_action": legacy_action,
        "legacy_confidence": str(legacy.get("confidence") or ""),
        "legacy_reason": str(legacy.get("recommended_reason") or legacy.get("reason") or ""),
        "family_state_next_stage": family_stage,
        "family_state_action": family_action,
        "family_state_confidence": str(family_decision.get("confidence") or ""),
        "family_state_reason": str(family_decision.get("reason") or ""),
        "stage_agrees": bool(legacy_stage and family_stage and legacy_stage == family_stage),
        "action_agrees": bool(legacy_action and family_action and legacy_action == family_action),
        "legacy_tags": tuple(legacy.get("rationale_tags") or ()),
        "family_state_tags": tuple(family_decision.get("rationale_tags") or ()),
    }


def resolve_stage_transition(evidence: StageTransitionEvidence) -> StageTransitionDecision:
    """Return an advisory stage transition decision without changing execution.

    This function is intentionally deterministic and side-effect free. It
    centralizes the stage-transition vocabulary so scheduler and family-loop
    summaries can be audited with the same semantics before any automatic
    enforcement is introduced.
    """

    stage = str(evidence.current_stage or "auto").strip() or "auto"
    target_profile = str(evidence.target_profile or "raw_alpha").strip() or "raw_alpha"
    status = str(evidence.last_round_status or "").strip().lower()
    winner = dict(evidence.last_round_winner or {})
    keep = dict(evidence.last_round_keep or {})
    anchor = dict(evidence.best_anchor or {})
    focused = dict(evidence.focused_best_node or {})
    tags: list[str] = []
    branch_reopen: list[str] = []

    if status == "failed":
        tags.append("last_round_failed")
        if evidence.validation_fail_count > 0:
            tags.append("validation_failures")
        return StageTransitionDecision(
            current_stage=stage,
            next_stage="broad_followup" if stage in {"auto", "new_family_broad", "broad_followup"} else stage,
            action="repair_or_retry",
            confidence="low",
            reason="last round failed; keep execution manual and inspect failure reasons",
            rationale_tags=tuple(tags),
            parent_selection_bias="diversify_branch",
            termination_bias="normal",
        )

    empty_flat = bool(
        not winner
        and not keep
        and int(evidence.children_collected or 0) <= 0
        and not evidence.last_round_search_improved
    )
    if empty_flat and status in {"", "ok", "partial_success"}:
        tags.append("empty_flat_round")
        if stage == "focused_refine":
            if target_profile == "complementarity":
                tags.append("complementarity_flat")
                return StageTransitionDecision(
                    current_stage=stage,
                    next_stage="focused_refine",
                    action="reopen_broad",
                    confidence="medium",
                    reason="complementarity focused round was flat; reopen another low-corr branch",
                    rationale_tags=tuple(tags),
                    parent_selection_bias="low_corr_parent",
                    target_profile_bias="complementarity",
                    branch_reopen_candidates=tuple(branch_reopen),
                )
            if evidence.has_decorrelation_targets:
                tags.append("decorrelation_flat")
                return StageTransitionDecision(
                    current_stage=stage,
                    next_stage="focused_refine",
                    action="switch_to_complementarity",
                    confidence="medium",
                    reason="focused decorrelation round was flat; switch toward complementarity targets",
                    rationale_tags=tuple(tags),
                    parent_selection_bias="low_corr_parent",
                    target_profile_bias="complementarity",
                )
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="broad_followup",
                action="reopen_broad",
                confidence="medium",
                reason="focused round produced no children or usable candidate; reopen a different branch",
                rationale_tags=tuple(tags),
                parent_selection_bias="diversify_branch",
                branch_reopen_candidates=tuple(branch_reopen),
            )
        return StageTransitionDecision(
            current_stage=stage,
            next_stage="terminate",
            action="freeze_or_switch_family",
            confidence="medium",
            reason="round produced no children, no improvement, and no usable candidate",
            rationale_tags=tuple(tags),
            termination_bias="stop",
        )

    if evidence.frontier_exhausted:
        tags.append("frontier_exhausted")
        return StageTransitionDecision(
            current_stage=stage,
            next_stage="terminate",
            action="freeze_or_switch_family",
            confidence="high",
            reason="search frontier is exhausted",
            rationale_tags=tuple(tags),
            termination_bias="stop",
        )
    if evidence.budget_exhausted:
        tags.append("budget_exhausted")

    if stage in {"auto", "new_family_broad", "broad_followup", "family_loop"}:
        if anchor or evidence.passed_anchor_count > 0:
            tags.append("anchor_available")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="graduate_anchor",
                confidence="high" if anchor else "medium",
                reason="at least one candidate passed anchor graduation evidence",
                rationale_tags=tuple(tags),
                parent_selection_bias="best_anchor",
            )
        if evidence.last_round_search_improved and _is_strong(winner):
            tags.extend(["search_improved", "strong_winner"])
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="exploit_mainline",
                confidence="medium",
                reason="broad search improved and produced a strong winner",
                rationale_tags=tuple(tags),
                parent_selection_bias="best_node",
            )
        if evidence.last_round_search_improved and _is_usable(winner):
            tags.extend(["search_improved", "usable_winner"])
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="exploit_mainline",
                confidence="medium",
                reason="broad search improved and produced a usable winner",
                rationale_tags=tuple(tags),
                parent_selection_bias="best_node",
            )
        if evidence.last_round_search_improved and int(evidence.children_added_to_search or 0) > 0:
            tags.extend(["search_improved", "children_added"])
            if target_profile == "complementarity" and not winner and not keep:
                tags.append("complementarity_without_visible_winner")
                return StageTransitionDecision(
                    current_stage=stage,
                    next_stage="terminate",
                    action="freeze_or_switch_family",
                    confidence="medium",
                    reason="complementarity run spent budget without a visible usable winner",
                    rationale_tags=tuple(tags),
                    termination_bias="stop",
                )
            if evidence.budget_exhausted:
                tags.append("budget_exhausted_without_visible_winner")
                return StageTransitionDecision(
                    current_stage=stage,
                    next_stage="broad_followup",
                    action="continue_broad_search",
                    confidence="low",
                    reason="broad search improved but no visible winner is available; continue broad or inspect artifacts",
                    rationale_tags=tuple(tags),
                    parent_selection_bias="diversify_branch",
                )
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="exploit_mainline",
                confidence="low",
                reason="broad search improved and added candidates, but winner evidence is incomplete",
                rationale_tags=tuple(tags),
                parent_selection_bias="best_node",
            )
        if (
            not evidence.last_round_search_improved
            and int(evidence.children_collected or 0) >= 20
            and not winner
            and not keep
        ):
            tags.append("broad_saturation")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="terminate",
                action="freeze_or_switch_family",
                confidence="medium",
                reason="broad search spent substantial candidate budget without improvement or usable candidates",
                rationale_tags=tuple(tags),
                termination_bias="stop",
            )
        tags.append("continue_broad")
        return StageTransitionDecision(
            current_stage=stage,
            next_stage="broad_followup",
            action="continue_broad_search",
            confidence="medium",
            reason="no anchor-level evidence yet; continue broad search or diversify parent choice",
            rationale_tags=tuple(tags),
            parent_selection_bias="diversify_branch",
        )

    if stage == "focused_refine":
        baseline = anchor or winner
        best_focused = focused or winner
        if best_focused and target_profile == "complementarity" and _is_usable(best_focused):
            tags.append("complementarity_usable_winner")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="confirmation",
                action="confirm_and_freeze",
                confidence="medium",
                reason="complementarity focused round produced a usable winner; confirm/freeze before more mining",
                rationale_tags=tuple(tags),
                termination_bias="stop_early_if_flat",
            )
        if best_focused and _is_high_turnover(best_focused) and target_profile != "complementarity":
            tags.append("high_turnover_winner")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="switch_to_complementarity",
                confidence="medium",
                reason="focused winner is strong but high-turnover; switch toward complementarity/deployability constraints",
                rationale_tags=tuple(tags),
                parent_selection_bias="low_corr_parent",
                target_profile_bias="complementarity",
            )
        if best_focused and _is_strong(best_focused):
            tags.append("focused_strong_winner")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="continue_focused",
                confidence="medium",
                reason="focused round produced a strong winner; continue this mainline before freezing",
                rationale_tags=tuple(tags),
                parent_selection_bias="best_node",
            )
        if best_focused and baseline and _material_gain(best_focused, baseline):
            tags.append("focused_material_gain")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="continue_focused",
                confidence="medium",
                reason="focused best node still has material gain versus baseline",
                rationale_tags=tuple(tags),
                parent_selection_bias="best_node",
            )
        if evidence.high_corr_count > 0 or evidence.has_decorrelation_targets or target_profile == "complementarity":
            tags.append("redundancy_pressure")
            if _factor_name(keep):
                branch_reopen.append(_factor_name(keep))
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="focused_refine",
                action="switch_to_complementarity",
                confidence="medium",
                reason="focused branch is constrained by redundancy pressure; prefer low-corr branch reopening",
                rationale_tags=tuple(tags),
                parent_selection_bias="low_corr_parent",
                target_profile_bias="complementarity",
                branch_reopen_candidates=tuple(branch_reopen),
            )
        if evidence.high_turnover_count > 0:
            tags.append("turnover_pressure")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="confirmation",
                action="deployability_confirmation",
                confidence="medium",
                reason="focused branch needs deployability confirmation rather than more raw-alpha mining",
                rationale_tags=tuple(tags),
                target_profile_bias="deployability",
                termination_bias="stop_early_if_flat",
            )
        if winner or keep:
            tags.append("usable_candidate_without_material_gain")
            return StageTransitionDecision(
                current_stage=stage,
                next_stage="confirmation",
                action="confirm_and_freeze",
                confidence="medium",
                reason="usable candidate exists but focused improvement is not material",
                rationale_tags=tuple(tags),
                termination_bias="stop_early_if_flat",
            )
        tags.append("focused_flat")
        return StageTransitionDecision(
            current_stage=stage,
            next_stage="broad_followup",
            action="reopen_broad",
            confidence="medium",
            reason="focused refinement is flat and produced no usable candidate",
            rationale_tags=tuple(tags),
            parent_selection_bias="diversify_branch",
        )

    if stage in {"confirmation", "donor_validation"}:
        tags.append("confirming_context")
        return StageTransitionDecision(
            current_stage=stage,
            next_stage="terminate",
            action="freeze_or_promote",
            confidence="high",
            reason="confirmation/donor validation is a terminal advisory stage unless operator reopens search",
            rationale_tags=tuple(tags),
            termination_bias="stop",
        )

    tags.append("unknown_stage")
    return StageTransitionDecision(
        current_stage=stage,
        next_stage="broad_followup",
        action="manual_review",
        confidence="low",
        reason="stage is not recognized by the transition resolver",
        rationale_tags=tuple(tags),
        parent_selection_bias="diversify_branch",
    )
