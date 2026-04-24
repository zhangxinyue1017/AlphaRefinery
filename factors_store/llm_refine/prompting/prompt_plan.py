'''Prompt planning structures and mode resolver integration.

Chooses memory, constraint, example, and decorrelation guidance blocks for each run context.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..search.context_resolver import ContextEvidence, ContextProfile, resolve_context_profile


@dataclass(frozen=True)
class PromptMemoryPlan:
    include: bool = True
    max_winners: int = 2
    max_keeps: int = 2
    max_failures: int = 3
    include_lineage: bool = True
    include_reflection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptConstraintPlan:
    include: bool = True
    style: str = "structured"
    include_family_constraints: bool = True
    include_axes_guidance: bool = True
    include_allowed_edit_types: bool = True
    include_anti_patterns: bool = True
    include_decorrelation_guidance: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptExamplesPlan:
    include: bool = True
    include_family_formulas: bool = True
    family_formula_limit: int | None = None
    include_bootstrap_frontier: bool = False
    bootstrap_frontier_limit: int | None = None
    include_donor_motifs: bool = False
    donor_motif_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptPlan:
    stage_mode: str = "auto"
    target_profile: str = "raw_alpha"
    policy_preset: str = "balanced"
    memory: PromptMemoryPlan = PromptMemoryPlan()
    constraints: PromptConstraintPlan = PromptConstraintPlan()
    examples: PromptExamplesPlan = PromptExamplesPlan()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_mode": self.stage_mode,
            "target_profile": self.target_profile,
            "policy_preset": self.policy_preset,
            "memory": self.memory.to_dict(),
            "constraints": self.constraints.to_dict(),
            "examples": self.examples.to_dict(),
        }


def build_prompt_plan(
    *,
    stage_mode: str = "auto",
    target_profile: str = "raw_alpha",
    policy_preset: str = "balanced",
    is_seed_stage: bool = False,
    has_donor_motifs: bool = False,
    has_bootstrap_frontier: bool = False,
    has_decorrelation_targets: bool = False,
    context_profile: ContextProfile | None = None,
) -> PromptPlan:
    normalized_stage = str(stage_mode or "auto").strip() or "auto"
    normalized_target = str(target_profile or "raw_alpha").strip() or "raw_alpha"
    normalized_policy = str(policy_preset or "balanced").strip() or "balanced"
    profile = context_profile or resolve_context_profile(
        ContextEvidence.from_runtime(
            family="",
            stage_mode=normalized_stage,
            target_profile=normalized_target,
            policy_preset=normalized_policy,
            is_seed_stage=is_seed_stage,
            has_donor_motifs=has_donor_motifs,
            has_bootstrap_frontier=has_bootstrap_frontier,
            has_decorrelation_targets=has_decorrelation_targets,
        )
    )

    memory = PromptMemoryPlan()
    if profile.memory_mode == "light":
        memory = PromptMemoryPlan(
            include=True,
            max_winners=1,
            max_keeps=1,
            max_failures=2,
            include_lineage=True,
            include_reflection=True,
        )
    elif profile.memory_mode == "rich":
        memory = PromptMemoryPlan(
            include=True,
            max_winners=3,
            max_keeps=2,
            max_failures=4,
            include_lineage=True,
            include_reflection=True,
        )

    constraints = PromptConstraintPlan(
        include=True,
        style=profile.prompt_constraint_style,
        include_family_constraints=True,
        include_axes_guidance=True,
        include_allowed_edit_types=True,
        include_anti_patterns=True,
        include_decorrelation_guidance=bool(has_decorrelation_targets),
    )

    include_bootstrap_frontier = profile.examples_mode in {
        "family_plus_bootstrap",
        "family_plus_bootstrap_and_donor",
    }
    include_donor_motifs = profile.examples_mode in {
        "family_plus_donor",
        "family_plus_bootstrap_and_donor",
    }
    examples = PromptExamplesPlan(
        include=True,
        include_family_formulas=True,
        family_formula_limit=None,
        include_bootstrap_frontier=bool(include_bootstrap_frontier and is_seed_stage and has_bootstrap_frontier),
        bootstrap_frontier_limit=None,
        include_donor_motifs=bool(include_donor_motifs and has_donor_motifs),
        donor_motif_limit=None,
    )

    return PromptPlan(
        stage_mode=normalized_stage,
        target_profile=normalized_target,
        policy_preset=normalized_policy,
        memory=memory,
        constraints=constraints,
        examples=examples,
    )
