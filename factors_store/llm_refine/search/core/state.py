'''Dataclasses for search nodes, budgets, actions, and feedback.

Stores family frontier state, node metrics, parent-child relationships, and evaluation outcomes.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SearchNode:
    node_id: str
    family: str
    factor_name: str
    expression: str
    candidate_id: str = ""
    parent_candidate_id: str = ""
    root_candidate_id: str = ""
    branch_key: str = ""
    motif_signature: str = ""
    source_run_id: str = ""
    source_model: str = ""
    source_provider: str = ""
    round_id: int = 0
    depth: int = 0
    node_kind: str = "candidate"
    status: str = ""
    quick_rank_ic_mean: float | None = None
    quick_rank_icir: float | None = None
    net_ann_return: float | None = None
    net_excess_ann_return: float | None = None
    net_sharpe: float | None = None
    mean_turnover: float | None = None
    metrics_completeness: float = 1.0
    missing_core_metrics_count: int = 0
    eligible_for_best_node: bool = True
    evaluated_at: str = ""
    created_at: str = ""
    operator_count: int = 0
    token_count: int = 0
    expression_depth: int = 0
    complexity: float = 0.0
    performance_score: float = 0.0
    quality_score: float = 0.0
    retrieval_score: float = 0.0
    novelty_score: float = 0.0
    exploration_score: float = 0.0
    underexplored_bonus_score: float = 0.0
    complexity_penalty_score: float = 0.0
    depth_penalty_score: float = 0.0
    turnover_penalty_score: float = 0.0
    branch_penalty_score: float = 0.0
    redundancy_penalty_score: float = 0.0
    status_bonus_score: float = 0.0
    base_score: float = 0.0
    frontier_score: float = 0.0
    visits: int = 0
    expansions: int = 0
    successful_expansions: int = 0
    children_total: int = 0
    children_kept: int = 0
    children_winners: int = 0
    child_best_gain: float = 0.0
    child_avg_gain: float = 0.0
    child_top3_mean_gain: float = 0.0
    child_median_gain: float = 0.0
    child_positive_gain_rate: float = 0.0
    child_gain_std: float = 0.0
    child_model_support: int = 0
    child_model_support_rate: float = 0.0
    child_cross_model_convergence: float = 0.0
    child_mutation_diversity: int = 0
    child_positive_excess_rate: float = 0.0
    child_low_turnover_rate: float = 0.0
    child_full_metrics_rate: float = 0.0
    child_admission_friendly_rate: float = 0.0
    child_high_quality_novel_count: int = 0
    child_new_motif_success_rate: float = 0.0
    last_success_round: int = 0
    rounds_since_last_success: int = 0
    expandability_score: float = 0.0
    expandability_confidence: float = 0.0
    effective_expandability_score: float = 0.0
    branch_value_score: float = 0.0
    constraint_score: float = 0.0
    portfolio_score: float = 0.0
    regime_score: float = 0.0
    transfer_score: float = 0.0
    target_conditioned_score: float = 0.0
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_name"] = self.factor_name
        return payload


@dataclass(frozen=True)
class SearchEdge:
    parent_node_id: str
    child_node_id: str
    edge_type: str = "refine"
    source_run_id: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchBudget:
    max_rounds: int = 2
    family_budget: int = 4
    branch_budget: int = 2
    max_frontier_size: int = 8
    max_depth: int = 4
    stop_if_no_improve: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
