from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return value == value


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
    high_corr_count: int = 0
    high_turnover_count: int = 0
    validation_fail_count: int = 0
    budget_exhausted: bool = False
    frontier_exhausted: bool = False
    has_decorrelation_targets: bool = False

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

    if evidence.budget_exhausted or evidence.frontier_exhausted:
        tags.append("budget_or_frontier_exhausted")
        return StageTransitionDecision(
            current_stage=stage,
            next_stage="terminate",
            action="freeze_or_switch_family",
            confidence="high",
            reason="search budget or frontier is exhausted",
            rationale_tags=tuple(tags),
            termination_bias="stop",
        )

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
                action="decorrelate_or_reopen_branch",
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
