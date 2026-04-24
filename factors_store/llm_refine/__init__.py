'''LLM-guided factor refinement package.

Contains prompt generation, parsing, evaluation, search, archive, and orchestration components.
'''

from __future__ import annotations

from .core.archive import create_run_dir, write_run_artifacts
from .evaluation.evaluator import evaluate_refinement_run, register_proposal_candidates
from .parsing.expression_engine import ExpressionEvaluationError, WideExpressionEngine, guess_required_fields
from .core.models import (
    LLMProposal,
    LLMProviderConfig,
    PromptBundle,
    RefinementCandidate,
    SeedFamily,
    SeedPool,
)
from .prompting.prompt_builder import build_refinement_prompt
from .core.providers import OpenAICompatProvider
from .core.seed_loader import load_seed_pool
from .search import SearchBudget, SearchEdge, SearchEngine, SearchFrontier, SearchNode, SearchPolicy

__all__ = [
    "LLMProposal",
    "LLMProviderConfig",
    "PromptBundle",
    "RefinementCandidate",
    "SearchBudget",
    "SearchEdge",
    "SearchEngine",
    "SearchFrontier",
    "SearchNode",
    "SearchPolicy",
    "SeedFamily",
    "SeedPool",
    "ExpressionEvaluationError",
    "WideExpressionEngine",
    "guess_required_fields",
    "OpenAICompatProvider",
    "build_refinement_prompt",
    "create_run_dir",
    "evaluate_refinement_run",
    "load_seed_pool",
    "register_proposal_candidates",
    "write_run_artifacts",
]
