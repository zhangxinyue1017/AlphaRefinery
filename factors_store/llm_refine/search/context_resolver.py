from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ContextEvidence:
    family: str
    stage_mode: str = "auto"
    target_profile: str = "raw_alpha"
    policy_preset: str = "balanced"
    is_seed_stage: bool = False
    has_bootstrap_frontier: bool = False
    has_donor_motifs: bool = False
    has_decorrelation_targets: bool = False
    selected_parent_kind: str = ""
    requested_candidate_count: int = 0
    final_candidate_target: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_runtime(
        cls,
        *,
        family: str,
        stage_mode: str = "auto",
        target_profile: str = "raw_alpha",
        policy_preset: str = "balanced",
        is_seed_stage: bool = False,
        has_bootstrap_frontier: bool = False,
        has_donor_motifs: bool = False,
        has_decorrelation_targets: bool = False,
        selected_parent_kind: str = "",
        requested_candidate_count: int = 0,
        final_candidate_target: int = 0,
    ) -> "ContextEvidence":
        return cls(
            family=str(family or "").strip(),
            stage_mode=str(stage_mode or "auto").strip() or "auto",
            target_profile=str(target_profile or "raw_alpha").strip() or "raw_alpha",
            policy_preset=str(policy_preset or "balanced").strip() or "balanced",
            is_seed_stage=bool(is_seed_stage),
            has_bootstrap_frontier=bool(has_bootstrap_frontier),
            has_donor_motifs=bool(has_donor_motifs),
            has_decorrelation_targets=bool(has_decorrelation_targets),
            selected_parent_kind=str(selected_parent_kind or "").strip(),
            requested_candidate_count=max(int(requested_candidate_count or 0), 0),
            final_candidate_target=max(int(final_candidate_target or 0), 0),
        )


@dataclass(frozen=True)
class ContextProfile:
    search_phase: str = "expanding"
    exploration_pressure: str = "medium"
    redundancy_pressure: str = "low"
    prompt_constraint_style: str = "structured"
    memory_mode: str = "standard"
    examples_mode: str = "family_only"
    branching_bias: str = "allow_one_branch"
    next_action_bias: str = "continue_focused"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OrchestrationProfile:
    recommended_stage_mode: str = "broad_followup"
    round_strategy: str = "open_space"
    promotion_bias: str = "normal"
    parent_selection_bias: str = "best_node"
    termination_bias: str = "normal"
    confidence: str = "medium"
    rationale_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_context_profile(evidence: ContextEvidence) -> ContextProfile:
    stage_mode = str(evidence.stage_mode or "auto").strip() or "auto"
    target_profile = str(evidence.target_profile or "raw_alpha").strip() or "raw_alpha"
    policy_preset = str(evidence.policy_preset or "balanced").strip() or "balanced"

    search_phase = "expanding"
    if stage_mode == "new_family_broad":
        search_phase = "opening"
    elif stage_mode == "broad_followup":
        search_phase = "expanding"
    elif stage_mode == "focused_refine":
        search_phase = "refining"
    elif stage_mode in {"confirmation", "donor_validation"}:
        search_phase = "confirming"

    exploration_pressure = "medium"
    if search_phase == "opening":
        exploration_pressure = "high" if policy_preset == "exploratory" else "medium"
    elif search_phase == "refining":
        exploration_pressure = "low" if policy_preset != "exploratory" else "medium"
    elif search_phase == "confirming":
        exploration_pressure = "low"
    elif policy_preset == "exploratory":
        exploration_pressure = "high"
    elif policy_preset == "conservative":
        exploration_pressure = "low"

    redundancy_pressure = "low"
    if evidence.has_decorrelation_targets or target_profile == "complementarity":
        redundancy_pressure = "high"
    elif evidence.has_donor_motifs or search_phase == "expanding":
        redundancy_pressure = "medium"

    prompt_constraint_style = "structured"
    if search_phase == "opening" and policy_preset == "exploratory":
        prompt_constraint_style = "guided"
    elif search_phase == "confirming" or target_profile in {"deployability", "robustness"}:
        prompt_constraint_style = "strict"

    memory_mode = "standard"
    if search_phase == "opening":
        memory_mode = "light"
    elif search_phase == "confirming":
        memory_mode = "rich"

    examples_mode = "family_only"
    if evidence.is_seed_stage and evidence.has_bootstrap_frontier and evidence.has_donor_motifs:
        examples_mode = "family_plus_bootstrap_and_donor"
    elif evidence.is_seed_stage and evidence.has_bootstrap_frontier:
        examples_mode = "family_plus_bootstrap"
    elif evidence.has_donor_motifs:
        examples_mode = "family_plus_donor"

    branching_bias = "allow_one_branch"
    if search_phase == "opening":
        branching_bias = "encourage_branching"
    elif search_phase == "refining":
        branching_bias = "allow_one_branch" if redundancy_pressure == "high" else "stay_local"
    elif search_phase == "confirming":
        branching_bias = "stay_local"

    next_action_bias = "continue_focused"
    if search_phase == "opening":
        next_action_bias = "reopen_broad"
    elif search_phase == "confirming":
        next_action_bias = "confirmation_ready"

    return ContextProfile(
        search_phase=search_phase,
        exploration_pressure=exploration_pressure,
        redundancy_pressure=redundancy_pressure,
        prompt_constraint_style=prompt_constraint_style,
        memory_mode=memory_mode,
        examples_mode=examples_mode,
        branching_bias=branching_bias,
        next_action_bias=next_action_bias,
    )


def resolve_orchestration_profile(
    *,
    evidence: ContextEvidence,
    context_profile: ContextProfile,
    last_round_status: str = "",
    last_round_search_improved: bool = False,
    last_round_winner: dict[str, Any] | None = None,
    last_round_keep: dict[str, Any] | None = None,
    recommended_stage_mode_hint: str = "",
) -> OrchestrationProfile:
    normalized_stage = str(evidence.stage_mode or "auto").strip() or "auto"
    hint = str(recommended_stage_mode_hint or "").strip()
    winner = dict(last_round_winner or {})
    keep = dict(last_round_keep or {})
    status = str(last_round_status or "").strip().lower()

    recommended_stage_mode = "broad_followup"
    round_strategy = "open_space"
    promotion_bias = "normal"
    parent_selection_bias = "best_node"
    termination_bias = "normal"
    confidence = "medium"
    rationale_tags: list[str] = []

    if hint:
        recommended_stage_mode = hint
        round_strategy = {
            "focused_refine": "exploit_mainline",
            "confirmation": "confirm_and_freeze",
            "donor_validation": "confirm_and_freeze",
            "new_family_broad": "open_space",
            "broad_followup": "open_space",
        }.get(hint, "open_space")
        confidence = "high"
        rationale_tags.append("external_stage_recommendation")
    elif normalized_stage in {"auto", "new_family_broad", "broad_followup"}:
        winner_icir = _safe_float(winner.get("quick_rank_icir"))
        winner_sharpe = _safe_float(winner.get("net_sharpe"))
        winner_excess = _safe_float(winner.get("net_excess_ann_return"))
        strong_winner = bool(
            winner
            and (
                (
                    winner_icir == winner_icir
                    and winner_icir >= 0.35
                    and winner_sharpe == winner_sharpe
                    and winner_sharpe >= 1.2
                )
                or (winner_excess == winner_excess and winner_excess >= 0.0)
            )
        )
        if last_round_search_improved and strong_winner:
            recommended_stage_mode = "focused_refine"
            round_strategy = "exploit_mainline"
            confidence = "medium"
            rationale_tags.extend(["search_improved", "winner_ready_for_focused"])
        else:
            recommended_stage_mode = "broad_followup"
            round_strategy = "open_space"
            parent_selection_bias = "diversify_branch"
            rationale_tags.append("continue_broad")
    elif normalized_stage == "focused_refine":
        recommended_stage_mode = "focused_refine"
        round_strategy = "exploit_mainline"
        termination_bias = "stop_early_if_flat" if not last_round_search_improved else "normal"
        if keep and not winner:
            rationale_tags.append("keep_without_new_winner")
        if not last_round_search_improved:
            rationale_tags.append("focused_flat")
    elif normalized_stage in {"confirmation", "donor_validation"} or context_profile.search_phase == "confirming":
        recommended_stage_mode = "confirmation"
        round_strategy = "confirm_and_freeze"
        promotion_bias = "conservative"
        termination_bias = "stop_early_if_flat"
        confidence = "high"
        rationale_tags.append("confirming_context")

    if status == "failed":
        confidence = "low"
        rationale_tags.append("last_round_failed")

    return OrchestrationProfile(
        recommended_stage_mode=recommended_stage_mode,
        round_strategy=round_strategy,
        promotion_bias=promotion_bias,
        parent_selection_bias=parent_selection_bias,
        termination_bias=termination_bias,
        confidence=confidence,
        rationale_tags=tuple(rationale_tags),
    )
