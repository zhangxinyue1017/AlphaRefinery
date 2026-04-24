'''Dataclasses and typed records for refinement inputs and outputs.

Defines seed families, proposals, candidates, validation records, and run metadata.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)


@dataclass(frozen=True)
class EvaluationWindow:
    start: str = ""
    end: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationProtocol:
    search: EvaluationWindow
    selection: EvaluationWindow
    final_oos: EvaluationWindow | None = None
    prompt_history_stage: str = "search"
    keep_decision_stage: str = "selection"
    promote_stage: str = "selection"

    def to_dict(self) -> dict[str, Any]:
        return {
            "search": self.search.to_dict(),
            "selection": self.selection.to_dict(),
            "final_oos": self.final_oos.to_dict() if self.final_oos is not None else None,
            "prompt_history_stage": self.prompt_history_stage,
            "keep_decision_stage": self.keep_decision_stage,
            "promote_stage": self.promote_stage,
        }


@dataclass(frozen=True)
class SeedFamily:
    family: str
    priority: str
    canonical_seed: str
    aliases: tuple[str, ...]
    direction: str
    implementation_paths: tuple[str, ...]
    formulas: dict[str, str]
    interpretation: str
    likely_weaknesses: tuple[str, ...]
    refinement_axes: tuple[str, ...]
    preferred_refine_seed: str = ""
    formula_directions: dict[str, str] = field(default_factory=dict)
    primary_objective: str = ""
    secondary_objective: str = ""
    hard_constraints: tuple[str, ...] = ()
    candidate_roles: tuple[str, ...] = ()
    anti_patterns: tuple[str, ...] = ()
    allowed_edit_types: tuple[str, ...] = ()
    relation_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SeedPool:
    version: int
    created_at: str
    project: str
    purpose: str
    evaluation_defaults: dict[str, Any]
    evaluation_protocol: EvaluationProtocol | None
    refinement_principles: tuple[str, ...]
    families: tuple[SeedFamily, ...]
    llm_refinement_template: dict[str, Any]

    def get_family(self, family: str) -> SeedFamily:
        for item in self.families:
            if item.family == family:
                return item
        raise KeyError(f"unknown family: {family}")


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str = DEFAULT_PROVIDER_NAME
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: float = DEFAULT_TIMEOUT


@dataclass(frozen=True)
class RefinementCandidate:
    name: str
    expression: str
    explanation: str
    candidate_role: str = ""
    rationale: str = ""
    expected_improvement: str = ""
    risk: str = ""
    source_model: str = ""
    source_provider: str = ""
    parent_factor: str = ""
    family: str = ""
    candidate_id: str = ""
    round_id: int = 1
    parent_candidate_id: str = ""
    status: str = "proposed"
    validation_warnings: tuple[str, ...] = ()

    def to_library_item(self, name_prefix: str) -> dict[str, Any]:
        return {
            "name": f"{name_prefix}.{self.name}",
            "expr": self.expression,
            "candidate_role": self.candidate_role,
            "explanation": self.explanation,
            "rationale": self.rationale,
            "expected_improvement": self.expected_improvement,
            "risk": self.risk,
            "source_model": self.source_model,
            "source_provider": self.source_provider,
            "parent_factor": self.parent_factor,
            "family": self.family,
            "candidate_id": self.candidate_id,
            "round_id": self.round_id,
            "parent_candidate_id": self.parent_candidate_id,
            "status": self.status,
            "validation_warnings": list(self.validation_warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMProposal:
    parent_factor: str
    diagnosed_weaknesses: tuple[str, ...]
    refinement_rationale: str
    candidates: tuple[RefinementCandidate, ...]
    expected_behavior_change: str = ""
    risk_notes: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload
