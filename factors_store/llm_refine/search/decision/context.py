'''Decision context records for reranking and family-state advice.

Carries stage, target profile, policy preset, decorrelation targets, and admission intent.
'''

from __future__ import annotations

from dataclasses import dataclass

from ..transition.context_resolver import ContextProfile


@dataclass(frozen=True)
class DecisionContext:
    family: str
    stage_mode: str = "auto"
    target_profile: str = "raw_alpha"
    policy_preset: str = "balanced"
    decorrelation_targets: tuple[str, ...] = ()
    neutralized_eval_enabled: bool = True
    admission_intent: bool = False
    context_profile: ContextProfile | None = None

    @property
    def decorrelation_enabled(self) -> bool:
        return bool(self.decorrelation_targets)

    @classmethod
    def from_runtime(
        cls,
        *,
        family: str,
        stage_mode: str = "auto",
        target_profile: str = "raw_alpha",
        policy_preset: str = "balanced",
        decorrelation_targets: list[str] | tuple[str, ...] | None = None,
        neutralized_eval_enabled: bool = True,
        admission_intent: bool = False,
        context_profile: ContextProfile | None = None,
    ) -> "DecisionContext":
        normalized_targets = tuple(str(item).strip() for item in (decorrelation_targets or ()) if str(item).strip())
        return cls(
            family=str(family or "").strip(),
            stage_mode=str(stage_mode or "auto").strip() or "auto",
            target_profile=str(target_profile or "raw_alpha").strip() or "raw_alpha",
            policy_preset=str(policy_preset or "balanced").strip() or "balanced",
            decorrelation_targets=normalized_targets,
            neutralized_eval_enabled=bool(neutralized_eval_enabled),
            admission_intent=bool(admission_intent),
            context_profile=context_profile,
        )


@dataclass(frozen=True)
class FamilyDecisionState:
    family: str
    target_profile: str
    broad_best_node: dict[str, object]
    broad_best_candidate: dict[str, object]
    broad_best_keep: dict[str, object]
    broad_display_node: dict[str, object]
    broad_snapshot: dict[str, object]
    anchor_selection: dict[str, object]
    focused_best_node: dict[str, object]
    focused_best_candidate: dict[str, object]
    focused_best_keep: dict[str, object]
    focused_display_node: dict[str, object]
    broad_stop_reason: str
    focused_stop_reason: str
    broad_returncode: int
    focused_returncode: int

    @classmethod
    def from_family_loop_inputs(
        cls,
        *,
        family: str,
        target_profile: str,
        broad_summary: dict[str, object],
        broad_snapshot: dict[str, object],
        anchor_selection: dict[str, object],
        focused_summary: dict[str, object],
        broad_returncode: int = 0,
        focused_returncode: int = 0,
    ) -> "FamilyDecisionState":
        broad_best = dict(broad_summary.get("best_node") or {})
        broad_best_candidate = dict(
            broad_summary.get("last_round_best_candidate")
            or broad_summary.get("last_round_best_keep")
            or broad_summary.get("last_round_winner")
            or {}
        )
        broad_best_keep = dict(broad_summary.get("last_round_best_keep") or {})
        broad_display = dict(broad_best_candidate or broad_best)
        focused_best = dict(focused_summary.get("best_node") or {})
        focused_best_candidate = dict(
            focused_summary.get("last_round_best_candidate")
            or focused_summary.get("last_round_best_keep")
            or focused_summary.get("last_round_winner")
            or {}
        )
        focused_best_keep = dict(focused_summary.get("last_round_best_keep") or {})
        focused_display = dict(focused_best_candidate or focused_best)
        return cls(
            family=str(family or "").strip(),
            target_profile=str(target_profile or "raw_alpha").strip() or "raw_alpha",
            broad_best_node=broad_best,
            broad_best_candidate=broad_best_candidate,
            broad_best_keep=broad_best_keep,
            broad_display_node=broad_display,
            broad_snapshot=dict(broad_snapshot or {}),
            anchor_selection=dict(anchor_selection or {}),
            focused_best_node=focused_best,
            focused_best_candidate=focused_best_candidate,
            focused_best_keep=focused_best_keep,
            focused_display_node=focused_display,
            broad_stop_reason=str(broad_summary.get("stop_reason", "") or ""),
            focused_stop_reason=str(focused_summary.get("stop_reason", "") or ""),
            broad_returncode=int(broad_returncode),
            focused_returncode=int(focused_returncode),
        )
