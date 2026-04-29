'''Round-level execution planning for stage transition decisions.

Stage policy answers the research question: what should the family do next?
The round controller answers the execution question: should the runner launch
another round now, and under which bounded authority?
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..policy_config import DEFAULT_POLICY_CONFIG, RefinePolicyConfig
from .signals import StageTransitionSignals
from .stage_transition import StageTransitionDecision


_ORDERED_VALUES: dict[str, tuple[str, ...]] = {
    "winner_quality": ("none", "weak", "usable", "strong"),
    "corr_pressure": ("low", "medium", "high", "critical"),
    "turnover_pressure": ("low", "medium", "high", "critical"),
}

_NEXT_ROUND_ACTIONS = {
    "continue_focused",
    "reopen_broad",
    "switch_to_complementarity",
}


@dataclass(frozen=True)
class RoundTransitionPlan:
    transition_authority: str
    control_effective: bool
    execute_next_round: bool
    next_stage_mode: str
    next_target_profile: str
    stage_action: str
    stage_next_stage: str
    policy_extension_granted: bool
    policy_extension_count: int
    stop_reason: str
    reason: str
    budget_gate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_round_transition_plan(
    *,
    stage_transition: StageTransitionDecision | dict[str, Any],
    signals: StageTransitionSignals | dict[str, Any],
    saturation_assessment: Any | None = None,
    current_stage: str,
    target_profile: str,
    rounds_completed: int,
    base_max_rounds: int,
    max_total_rounds: int,
    policy_extension_count: int,
    max_policy_extensions: int,
    transition_authority: str = "",
    config: RefinePolicyConfig | None = None,
) -> RoundTransitionPlan:
    cfg = config or DEFAULT_POLICY_CONFIG
    round_cfg = cfg.round_transition
    authority = str(transition_authority or round_cfg.default_authority).strip().lower()
    if authority not in set(round_cfg.allowed_authorities):
        authority = round_cfg.default_authority

    decision = _decision_dict(stage_transition)
    signal_payload = _signal_dict(signals)
    saturation_payload = _payload_dict(saturation_assessment)
    action = str(decision.get("action") or "").strip()
    stage_next = str(decision.get("next_stage") or current_stage or "auto").strip() or "auto"
    saturation_grade = str(saturation_payload.get("grade") or "low").strip().lower() or "low"

    base_limit = max(int(base_max_rounds or 0), 0)
    requested_total_limit = int(max_total_rounds or 0)
    extension_limit = max(int(max_policy_extensions or 0), 0)
    total_limit = max(requested_total_limit, base_limit)
    if total_limit <= 0:
        total_limit = max(base_limit + extension_limit, 1)
    rounds_done = max(int(rounds_completed or 0), 0)
    extension_count_before = max(int(policy_extension_count or 0), 0)
    base_budget_remaining = rounds_done < base_limit
    total_budget_remaining = rounds_done < total_limit
    extension_budget_remaining = extension_count_before < extension_limit
    needs_extension = not base_budget_remaining
    control_effective = authority == "guarded_control"

    budget_gate: dict[str, Any] = {
        "policy_config_version": round_cfg.version,
        "base_max_rounds": base_limit,
        "max_total_rounds": total_limit,
        "rounds_completed": rounds_done,
        "base_budget_remaining": base_budget_remaining,
        "total_budget_remaining": total_budget_remaining,
        "needs_policy_extension": needs_extension,
        "policy_extension_count_before": extension_count_before,
        "max_policy_extensions": extension_limit,
        "extension_budget_remaining": extension_budget_remaining,
        "saturation_grade": saturation_grade,
        "control_effective": control_effective,
    }

    if bool(signal_payload.get("frontier_exhausted")):
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next=stage_next,
            target_profile=target_profile,
            extension_count=extension_count_before,
            stop_reason="frontier_exhausted",
            reason="round controller stopped because frontier_exhausted=true",
            budget_gate=budget_gate,
        )
    if bool(signal_payload.get("budget_exhausted")):
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next=stage_next,
            target_profile=target_profile,
            extension_count=extension_count_before,
            stop_reason="budget_exhausted",
            reason="round controller stopped because budget_exhausted=true",
            budget_gate=budget_gate,
        )
    if action == "terminate":
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next="terminate",
            target_profile=target_profile,
            extension_count=extension_count_before,
            stop_reason="stage_policy_terminate",
            reason="stage policy requested terminate",
            budget_gate=budget_gate,
        )
    if action == "confirmation":
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next="confirmation",
            target_profile=target_profile,
            extension_count=extension_count_before,
            stop_reason="stage_policy_confirmation",
            reason="stage policy requested confirmation; first round-controller version stops and records the handoff",
            budget_gate=budget_gate,
        )
    if action not in _NEXT_ROUND_ACTIONS:
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next=stage_next,
            target_profile=target_profile,
            extension_count=extension_count_before,
            stop_reason="unsupported_stage_action",
            reason=f"round controller does not execute stage action: {action}",
            budget_gate=budget_gate,
        )
    if not total_budget_remaining:
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next=stage_next,
            target_profile=target_profile,
            extension_count=extension_count_before,
            stop_reason="max_total_rounds",
            reason="stage policy requested another round but max_total_rounds was reached",
            budget_gate=budget_gate,
        )

    next_stage, next_target = _next_execution_target(
        action=action,
        stage_next=stage_next,
        current_stage=current_stage,
        target_profile=target_profile,
    )
    if base_budget_remaining:
        return RoundTransitionPlan(
            transition_authority=authority,
            control_effective=control_effective,
            execute_next_round=True,
            next_stage_mode=next_stage,
            next_target_profile=next_target,
            stage_action=action,
            stage_next_stage=stage_next,
            policy_extension_granted=False,
            policy_extension_count=extension_count_before,
            stop_reason="",
            reason="base round budget remains; stage policy action is executable",
            budget_gate=budget_gate,
        )

    extension_ok, extension_reason = _policy_extension_allowed(
        action=action,
        current_stage=current_stage,
        signals=signal_payload,
        saturation_grade=saturation_grade,
        config=cfg,
    )
    budget_gate["extension_safety_passed"] = extension_ok
    budget_gate["extension_safety_reason"] = extension_reason

    if authority != "guarded_control":
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next=next_stage,
            target_profile=next_target,
            extension_count=extension_count_before,
            stop_reason="max_rounds",
            reason=(
                "stage policy requested another round after base max_rounds, "
                f"but transition_authority={authority} cannot grant extensions"
            ),
            budget_gate=budget_gate,
        )
    if not extension_budget_remaining:
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next=next_stage,
            target_profile=next_target,
            extension_count=extension_count_before,
            stop_reason="max_policy_extensions",
            reason="stage policy requested another round but max_policy_extensions was reached",
            budget_gate=budget_gate,
        )
    if not extension_ok:
        return _stop_plan(
            authority=authority,
            control_effective=control_effective,
            action=action,
            stage_next=next_stage,
            target_profile=next_target,
            extension_count=extension_count_before,
            stop_reason="policy_extension_denied",
            reason=extension_reason,
            budget_gate=budget_gate,
        )

    return RoundTransitionPlan(
        transition_authority=authority,
        control_effective=control_effective,
        execute_next_round=True,
        next_stage_mode=next_stage,
        next_target_profile=next_target,
        stage_action=action,
        stage_next_stage=stage_next,
        policy_extension_granted=True,
        policy_extension_count=extension_count_before + 1,
        stop_reason="",
        reason="guarded_control granted one policy extension",
        budget_gate=budget_gate,
    )


def _decision_dict(decision: StageTransitionDecision | dict[str, Any]) -> dict[str, Any]:
    if isinstance(decision, StageTransitionDecision):
        return decision.to_dict()
    return dict(decision or {})


def _signal_dict(signals: StageTransitionSignals | dict[str, Any]) -> dict[str, Any]:
    if isinstance(signals, StageTransitionSignals):
        return signals.to_dict()
    return dict(signals or {})


def _payload_dict(payload: Any | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if hasattr(payload, "to_dict"):
        return dict(payload.to_dict())
    return dict(payload or {})


def _next_execution_target(
    *,
    action: str,
    stage_next: str,
    current_stage: str,
    target_profile: str,
) -> tuple[str, str]:
    if action == "continue_focused":
        return "focused_refine", str(target_profile or "raw_alpha")
    if action == "reopen_broad":
        return "broad_followup", str(target_profile or "raw_alpha")
    if action == "switch_to_complementarity":
        return "focused_refine", "complementarity"
    return str(stage_next or current_stage or "auto"), str(target_profile or "raw_alpha")


def _policy_extension_allowed(
    *,
    action: str,
    current_stage: str,
    signals: dict[str, Any],
    saturation_grade: str,
    config: RefinePolicyConfig,
) -> tuple[bool, str]:
    round_cfg = config.round_transition
    if saturation_grade not in set(round_cfg.extension_safe_saturation_grades):
        return False, f"saturation grade {saturation_grade} is outside extension-safe grades"
    if action == "continue_focused":
        if not _ordered_at_least(
            str(signals.get("winner_quality") or "none"),
            round_cfg.extension_min_winner_quality,
            "winner_quality",
        ):
            return False, "continue_focused extension requires usable-or-better winner_quality"
        broad_like_stage = str(current_stage or "").strip() in {
            "new_family_broad",
            "broad_followup",
            "family_loop",
            "auto",
        }
        if not broad_like_stage and not bool(signals.get("material_gain")):
            return False, "continue_focused extension requires material_gain=true"
        if not _ordered_at_most(
            str(signals.get("turnover_pressure") or "low"),
            round_cfg.extension_max_turnover_pressure,
            "turnover_pressure",
        ):
            return False, "continue_focused extension denied by turnover pressure"
        if not _ordered_at_most(
            str(signals.get("corr_pressure") or "low"),
            round_cfg.extension_max_corr_pressure,
            "corr_pressure",
        ):
            return False, "continue_focused extension denied by correlation pressure"
        return True, "continue_focused extension safety checks passed"
    if action == "reopen_broad":
        return True, "reopen_broad can use a guarded extension when total budget remains"
    if action == "switch_to_complementarity":
        return True, "switch_to_complementarity can use a guarded extension when total budget remains"
    return False, f"unsupported extension action: {action}"


def _ordered_at_least(actual: str, expected: str, field: str) -> bool:
    order = _ORDERED_VALUES.get(field)
    if not order:
        return False
    ranks = {value: idx for idx, value in enumerate(order)}
    return ranks.get(actual, -1) >= ranks.get(expected, 10**6)


def _ordered_at_most(actual: str, expected: str, field: str) -> bool:
    order = _ORDERED_VALUES.get(field)
    if not order:
        return False
    ranks = {value: idx for idx, value in enumerate(order)}
    return ranks.get(actual, 10**6) <= ranks.get(expected, -1)


def _stop_plan(
    *,
    authority: str,
    control_effective: bool,
    action: str,
    stage_next: str,
    target_profile: str,
    extension_count: int,
    stop_reason: str,
    reason: str,
    budget_gate: dict[str, Any],
) -> RoundTransitionPlan:
    return RoundTransitionPlan(
        transition_authority=authority,
        control_effective=control_effective,
        execute_next_round=False,
        next_stage_mode=stage_next,
        next_target_profile=str(target_profile or "raw_alpha"),
        stage_action=action,
        stage_next_stage=stage_next,
        policy_extension_granted=False,
        policy_extension_count=int(extension_count),
        stop_reason=stop_reason,
        reason=reason,
        budget_gate=budget_gate,
    )
