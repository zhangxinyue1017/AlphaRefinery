'''Shadow table policy for stage-transition auditing.

The table policy consumes explicit signals and emits a shadow decision. It does
not replace the legacy resolver; it exists to compare decisions in artifacts.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .signals import StageTransitionSignals
from .stage_transition import StageTransitionDecision, StageTransitionEvidence


_ORDERED_VALUES: dict[str, tuple[str, ...]] = {
    "anchor_strength": ("none", "weak", "passed", "strong"),
    "winner_quality": ("none", "weak", "usable", "strong"),
    "corr_pressure": ("low", "medium", "high", "critical"),
    "turnover_pressure": ("low", "medium", "high", "critical"),
    "frontier_health": ("exhausted", "low", "medium", "high"),
    "model_consensus": ("low", "medium", "high"),
}

_VALID_ACTIONS = {
    "continue_focused",
    "reopen_broad",
    "switch_to_complementarity",
    "confirmation",
    "terminate",
}


@dataclass(frozen=True)
class ShadowPolicyRule:
    rule_id: str
    phase: str
    conditions: tuple[str, ...]
    action: str
    next_stage: str
    specificity: int = 0
    parent_selection_bias: str = "best_node"
    target_profile_bias: str = "keep_current"
    termination_bias: str = "normal"
    confidence: str = "medium"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SHADOW_STAGE_POLICY_TABLE: tuple[ShadowPolicyRule, ...] = (
    ShadowPolicyRule(
        rule_id="frontier_exhausted_terminal",
        phase="any",
        conditions=("frontier_exhausted=true",),
        action="terminate",
        next_stage="terminate",
        specificity=100,
        termination_bias="stop",
        confidence="high",
        reason="frontier_exhausted is a hard terminal signal before empty-flat handling",
    ),
    ShadowPolicyRule(
        rule_id="validation_fail_reopen",
        phase="any",
        conditions=("validation_fail_count>0",),
        action="reopen_broad",
        next_stage="broad_followup",
        specificity=95,
        parent_selection_bias="diversify_branch",
        confidence="low",
        reason="validation failures should reopen/repair instead of advancing phase",
    ),
    ShadowPolicyRule(
        rule_id="focused_turnover_no_gain_switch",
        phase="focused_refine",
        conditions=("turnover_pressure>=high", "material_gain=false"),
        action="switch_to_complementarity",
        next_stage="focused_refine",
        specificity=90,
        parent_selection_bias="low_corr_parent",
        target_profile_bias="complementarity",
        reason="high turnover plus no material gain should target complementarity, not confirmation",
    ),
    ShadowPolicyRule(
        rule_id="focused_turnover_switch",
        phase="focused_refine",
        conditions=("turnover_pressure>=high",),
        action="switch_to_complementarity",
        next_stage="focused_refine",
        specificity=85,
        parent_selection_bias="low_corr_parent",
        target_profile_bias="complementarity",
        reason="turnover pressure outranks strong-winner continuation",
    ),
    ShadowPolicyRule(
        rule_id="focused_corr_critical_switch",
        phase="focused_refine",
        conditions=("corr_pressure>=high",),
        action="switch_to_complementarity",
        next_stage="focused_refine",
        specificity=80,
        parent_selection_bias="low_corr_parent",
        target_profile_bias="complementarity",
        reason="correlation pressure should move the branch toward complementarity",
    ),
    ShadowPolicyRule(
        rule_id="broad_corr_critical_reopen",
        phase="new_family_broad|broad_followup|family_loop|auto",
        conditions=("corr_pressure>=critical",),
        action="reopen_broad",
        next_stage="broad_followup",
        specificity=78,
        parent_selection_bias="diversify_branch",
        reason="critical broad-stage crowding should reopen/diversify before graduation",
    ),
    ShadowPolicyRule(
        rule_id="broad_anchor_continue",
        phase="new_family_broad|broad_followup|family_loop|auto",
        conditions=("anchor_strength>=passed", "turnover_pressure<=medium"),
        action="continue_focused",
        next_stage="focused_refine",
        specificity=70,
        parent_selection_bias="best_anchor",
        confidence="high",
        reason="passed anchor with manageable turnover should move into focused refinement",
    ),
    ShadowPolicyRule(
        rule_id="broad_usable_winner_continue",
        phase="new_family_broad|broad_followup|family_loop|auto",
        conditions=("winner_quality>=usable", "turnover_pressure<=medium"),
        action="continue_focused",
        next_stage="focused_refine",
        specificity=65,
        parent_selection_bias="best_node",
        reason="usable broad winner with manageable turnover should be exploited",
    ),
    ShadowPolicyRule(
        rule_id="focused_complementarity_confirm",
        phase="focused_refine",
        conditions=("target_profile=complementarity", "winner_quality>=usable", "turnover_pressure<=medium"),
        action="confirmation",
        next_stage="confirmation",
        specificity=64,
        termination_bias="stop_early_if_flat",
        reason="usable complementarity result should be confirmed before more mining",
    ),
    ShadowPolicyRule(
        rule_id="focused_material_gain_continue",
        phase="focused_refine",
        conditions=("winner_quality>=usable", "material_gain=true", "turnover_pressure<=medium"),
        action="continue_focused",
        next_stage="focused_refine",
        specificity=60,
        parent_selection_bias="best_node",
        reason="focused result has usable quality and material incremental gain",
    ),
    ShadowPolicyRule(
        rule_id="focused_usable_no_gain_confirm",
        phase="focused_refine",
        conditions=(
            "winner_quality>=usable",
            "material_gain=false",
            "turnover_pressure<=medium",
            "corr_pressure<=medium",
        ),
        action="confirmation",
        next_stage="confirmation",
        specificity=55,
        termination_bias="stop_early_if_flat",
        reason="usable focused result lacks material gain and has no major pressure",
    ),
    ShadowPolicyRule(
        rule_id="no_improve_reopen",
        phase="any",
        conditions=("no_improve_count>=2", "frontier_health>=medium"),
        action="reopen_broad",
        next_stage="broad_followup",
        specificity=50,
        parent_selection_bias="diversify_branch",
        reason="repeated no-improve rounds with live frontier should reopen broad search",
    ),
    ShadowPolicyRule(
        rule_id="no_improve_terminate",
        phase="any",
        conditions=("no_improve_count>=2", "frontier_health<=low"),
        action="terminate",
        next_stage="terminate",
        specificity=48,
        termination_bias="stop",
        reason="repeated no-improve rounds with weak frontier should terminate",
    ),
    ShadowPolicyRule(
        rule_id="empty_or_low_frontier_broad",
        phase="new_family_broad|broad_followup|family_loop|auto",
        conditions=("winner_quality=none", "anchor_strength=none", "frontier_health<=low"),
        action="terminate",
        next_stage="terminate",
        specificity=40,
        termination_bias="stop",
        reason="broad search has no usable signal and low frontier health",
    ),
    ShadowPolicyRule(
        rule_id="broad_default_reopen",
        phase="new_family_broad|broad_followup|family_loop|auto",
        conditions=(),
        action="reopen_broad",
        next_stage="broad_followup",
        specificity=1,
        parent_selection_bias="diversify_branch",
        confidence="low",
        reason="default broad-stage shadow action keeps search open",
    ),
    ShadowPolicyRule(
        rule_id="focused_default_reopen",
        phase="focused_refine",
        conditions=(),
        action="reopen_broad",
        next_stage="broad_followup",
        specificity=1,
        parent_selection_bias="diversify_branch",
        confidence="low",
        reason="default focused-stage shadow action reopens broad search",
    ),
    ShadowPolicyRule(
        rule_id="terminal_phase_terminate",
        phase="confirmation|donor_validation",
        conditions=(),
        action="terminate",
        next_stage="terminate",
        specificity=1,
        termination_bias="stop",
        confidence="high",
        reason="confirmation and donor-validation remain terminal in stage transition",
    ),
)


def get_shadow_stage_policy_table() -> tuple[dict[str, Any], ...]:
    return tuple(rule.to_dict() for rule in SHADOW_STAGE_POLICY_TABLE)


def resolve_shadow_table_policy(
    evidence: StageTransitionEvidence,
    signals: StageTransitionSignals,
) -> StageTransitionDecision:
    stage = str(evidence.current_stage or "auto").strip() or "auto"
    sorted_rules = sorted(
        enumerate(SHADOW_STAGE_POLICY_TABLE),
        key=lambda item: (-int(item[1].specificity), item[0]),
    )
    evaluation_trace: list[dict[str, Any]] = []
    for _, rule in sorted_rules:
        phase_matches = _phase_matches(rule.phase, stage)
        condition_results = [_condition_matches(condition, evidence, signals) for condition in rule.conditions]
        matched = bool(phase_matches and all(result["matched"] for result in condition_results))
        evaluation_trace.append(
            {
                "rule_id": rule.rule_id,
                "phase_matches": phase_matches,
                "conditions": condition_results,
                "matched": matched,
            }
        )
        if matched:
            action = rule.action if rule.action in _VALID_ACTIONS else "reopen_broad"
            tags = (
                "shadow_table",
                f"shadow_rule:{rule.rule_id}",
                f"anchor_strength:{signals.anchor_strength}",
                f"winner_quality:{signals.winner_quality}",
                f"corr_pressure:{signals.corr_pressure}",
                f"turnover_pressure:{signals.turnover_pressure}",
                f"frontier_health:{signals.frontier_health}",
            )
            return StageTransitionDecision(
                current_stage=stage,
                next_stage=rule.next_stage,
                action=action,
                confidence=rule.confidence,
                reason=rule.reason,
                rationale_tags=tags,
                parent_selection_bias=rule.parent_selection_bias,
                target_profile_bias=rule.target_profile_bias,
                termination_bias=rule.termination_bias,
                mode="shadow_table",
            )

    return StageTransitionDecision(
        current_stage=stage,
        next_stage="broad_followup",
        action="reopen_broad",
        confidence="low",
        reason="no shadow table rule matched",
        rationale_tags=("shadow_table", "shadow_rule:none"),
        parent_selection_bias="diversify_branch",
        mode="shadow_table",
    )


def compare_stage_transition_decisions(
    *,
    legacy_decision: StageTransitionDecision | dict[str, Any],
    shadow_decision: StageTransitionDecision | dict[str, Any],
) -> dict[str, Any]:
    legacy = legacy_decision.to_dict() if isinstance(legacy_decision, StageTransitionDecision) else dict(legacy_decision or {})
    shadow = shadow_decision.to_dict() if isinstance(shadow_decision, StageTransitionDecision) else dict(shadow_decision or {})
    return {
        "mode": "legacy_vs_shadow_table",
        "legacy_next_stage": str(legacy.get("next_stage") or ""),
        "legacy_action": str(legacy.get("action") or ""),
        "legacy_reason": str(legacy.get("reason") or ""),
        "shadow_next_stage": str(shadow.get("next_stage") or ""),
        "shadow_action": str(shadow.get("action") or ""),
        "shadow_reason": str(shadow.get("reason") or ""),
        "stage_agrees": bool(legacy.get("next_stage") and legacy.get("next_stage") == shadow.get("next_stage")),
        "action_agrees": bool(legacy.get("action") and legacy.get("action") == shadow.get("action")),
        "legacy_tags": tuple(legacy.get("rationale_tags") or ()),
        "shadow_tags": tuple(shadow.get("rationale_tags") or ()),
    }


def _phase_matches(raw_phase: str, stage: str) -> bool:
    if raw_phase == "any":
        return True
    return stage in {part.strip() for part in raw_phase.split("|") if part.strip()}


def _condition_matches(condition: str, evidence: StageTransitionEvidence, signals: StageTransitionSignals) -> dict[str, Any]:
    text = str(condition or "").strip()
    if not text:
        return {"condition": text, "matched": True, "actual": ""}
    for operator in (" in ", ">=", "<=", "!=", "=", ">", "<"):
        if operator in text:
            left, right = text.split(operator, 1)
            field = left.strip()
            expected = right.strip()
            actual = _lookup_value(field, evidence, signals)
            matched = _compare(actual, operator.strip(), expected, field)
            return {
                "condition": text,
                "field": field,
                "operator": operator.strip(),
                "expected": expected,
                "actual": actual,
                "matched": matched,
            }
    actual = _lookup_value(text, evidence, signals)
    return {"condition": text, "actual": actual, "matched": bool(actual)}


def _lookup_value(field: str, evidence: StageTransitionEvidence, signals: StageTransitionSignals) -> Any:
    if hasattr(signals, field):
        return getattr(signals, field)
    if field == "phase":
        return str(evidence.current_stage or "auto")
    if field == "target_profile":
        return str(evidence.target_profile or "")
    if hasattr(evidence, field):
        return getattr(evidence, field)
    return None


def _compare(actual: Any, operator: str, expected: str, field: str) -> bool:
    if operator == "in":
        expected_values = {
            item.strip()
            for item in expected.strip("{}").split(",")
            if item.strip()
        }
        return str(actual) in expected_values
    if operator in {"=", "!="}:
        matched = _normalize_scalar(actual) == _normalize_scalar(expected)
        return matched if operator == "=" else not matched
    if field in _ORDERED_VALUES:
        return _compare_ordered(str(actual), operator, expected, _ORDERED_VALUES[field])
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    return False


def _normalize_scalar(value: Any) -> Any:
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text == "true"
    return text


def _compare_ordered(actual: str, operator: str, expected: str, order: tuple[str, ...]) -> bool:
    ranks = {value: idx for idx, value in enumerate(order)}
    if actual not in ranks or expected not in ranks:
        return False
    left = ranks[actual]
    right = ranks[expected]
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    return False
