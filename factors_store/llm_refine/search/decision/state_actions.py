'''Compact state/action vocabulary for public scheduler summaries.

The stage-transition table remains the execution-facing policy. This module
maps that detailed policy into a small core vocabulary that is easier to
explain in docs and open-source artifacts.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


FamilyState = Literal["new", "exploring", "refining", "saturated", "held"]
FamilyAction = Literal[
    "start_broad",
    "continue_focused",
    "reopen_broad",
    "switch_objective",
    "confirm",
    "export_donor",
    "import_donor",
    "hold",
    "stop",
]
TransferMode = Literal["none", "export", "import"]

CoreFamilyState = FamilyState
CoreFamilyAction = FamilyAction


_ACTION_MAP: dict[str, FamilyAction] = {
    "continue_focused": "continue_focused",
    "exploit_mainline": "continue_focused",
    "graduate_anchor": "continue_focused",
    "reopen_broad": "reopen_broad",
    "repair_or_retry": "reopen_broad",
    "reopen_broad_or_freeze": "reopen_broad",
    "switch_to_complementarity": "switch_objective",
    "confirmation": "confirm",
    "freeze_or_promote": "hold",
    "freeze_or_switch_family": "hold",
    "terminate": "stop",
}


@dataclass(frozen=True)
class FamilyFlow:
    family_state: FamilyState
    recommended_action: FamilyAction
    action_reason: str
    next_run_hint: dict[str, Any]
    transfer: dict[str, Any]
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CoreFamilyFlow = FamilyFlow


def map_stage_action(action: str) -> FamilyAction:
    """Map detailed stage-policy actions into the compact public vocabulary."""

    text = str(action or "").strip()
    return _ACTION_MAP.get(text, "hold")


def build_core_transfer_summary(
    *,
    has_donor_motifs: bool = False,
    donor_motifs_count: int = 0,
    donor_families: list[str] | tuple[str, ...] | None = None,
    saturation_grade: str = "",
    winner_quality: str = "",
    recommended_action: FamilyAction = "hold",
) -> dict[str, Any]:
    families = [str(item).strip() for item in donor_families or () if str(item).strip()]
    if has_donor_motifs or donor_motifs_count > 0 or families:
        mode: TransferMode = "import"
        action: FamilyAction = "import_donor"
    elif str(saturation_grade or "").strip().lower() in {"high", "critical"} and str(
        winner_quality or ""
    ).strip().lower() in {"usable", "strong"}:
        mode = "export"
        action = "export_donor"
    else:
        mode = "none"
        action = recommended_action

    return {
        "mode": mode,
        "recommended_transfer_action": action,
        "donor_motifs_count": max(int(donor_motifs_count or 0), 0),
        "donor_families": families,
    }


def derive_core_family_state(
    *,
    stage_mode: str,
    stage_action: str,
    round_status: str = "",
    saturation_grade: str = "",
    target_profile: str = "",
    rounds_completed: int = 0,
) -> FamilyState:
    stage = str(stage_mode or "auto").strip() or "auto"
    action = str(stage_action or "").strip()
    status = str(round_status or "").strip().lower()
    grade = str(saturation_grade or "").strip().lower()
    target = str(target_profile or "").strip().lower()

    if int(rounds_completed or 0) <= 0 and stage in {"auto", "new_family_broad"}:
        return "new"
    if action in {"terminate", "freeze_or_promote", "freeze_or_switch_family"}:
        return "held"
    if grade in {"high", "critical"} or action == "switch_to_complementarity" or target == "complementarity":
        return "saturated"
    if status == "failed":
        return "exploring"
    if stage in {"focused_refine", "confirmation", "donor_validation"}:
        return "refining"
    return "exploring"


def build_core_family_flow_summary(
    *,
    family: str,
    stage_mode: str,
    target_profile: str,
    policy_preset: str = "",
    rounds_completed: int = 0,
    round_status: str = "",
    stage_transition: dict[str, Any] | None = None,
    stage_transition_signals: dict[str, Any] | None = None,
    saturation_assessment: dict[str, Any] | None = None,
    round_transition_plan: dict[str, Any] | None = None,
    prompt_trace: dict[str, Any] | None = None,
    donor_families: list[str] | tuple[str, ...] | None = None,
) -> FamilyFlow:
    transition = dict(stage_transition or {})
    signals = dict(stage_transition_signals or {})
    saturation = dict(saturation_assessment or {})
    round_plan = dict(round_transition_plan or {})
    prompt = dict(prompt_trace or {})

    detailed_action = str(transition.get("action") or round_plan.get("stage_action") or "")
    core_action = map_stage_action(detailed_action)
    saturation_grade = str(saturation.get("grade") or "")
    donor_motifs_count = _safe_int(prompt.get("donor_motifs_count"), default=0)
    has_donor_motifs = donor_motifs_count > 0 or bool(
        dict(prompt.get("context_evidence") or {}).get("has_donor_motifs")
    )
    family_state = derive_core_family_state(
        stage_mode=str(transition.get("current_stage") or stage_mode),
        stage_action=detailed_action,
        round_status=round_status,
        saturation_grade=saturation_grade,
        target_profile=target_profile,
        rounds_completed=int(rounds_completed or 0),
    )
    next_run_hint = {
        "stage_mode": str(round_plan.get("next_stage_mode") or transition.get("next_stage") or stage_mode or "auto"),
        "target_profile": str(round_plan.get("next_target_profile") or target_profile or "raw_alpha"),
        "policy_preset": str(policy_preset or prompt.get("policy_preset") or ""),
        "execute_next_round": bool(round_plan.get("execute_next_round", True)),
    }
    transfer = build_core_transfer_summary(
        has_donor_motifs=has_donor_motifs,
        donor_motifs_count=donor_motifs_count,
        donor_families=donor_families,
        saturation_grade=saturation_grade,
        winner_quality=str(signals.get("winner_quality") or ""),
        recommended_action=core_action,
    )
    return FamilyFlow(
        family_state=family_state,
        recommended_action=core_action,
        action_reason=str(transition.get("reason") or round_plan.get("reason") or ""),
        next_run_hint=next_run_hint,
        transfer=transfer,
        source={
            "family": str(family or ""),
            "stage_policy_action": detailed_action,
            "stage_policy_next_stage": str(transition.get("next_stage") or ""),
            "stage_policy_confidence": str(transition.get("confidence") or ""),
            "saturation_grade": saturation_grade,
            "winner_quality": str(signals.get("winner_quality") or ""),
        },
    )


def _safe_int(value: object, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
