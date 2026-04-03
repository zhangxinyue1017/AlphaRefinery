from __future__ import annotations

from collections import Counter
from typing import Iterable

from .normalization import SearchNormalizer
from .policy import SearchPolicy
from .scoring import compute_frontier_score, pairwise_similarity
from .state import SearchBudget, SearchNode


class SearchFrontier:
    def __init__(self) -> None:
        self._nodes: dict[str, SearchNode] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def values(self) -> Iterable[SearchNode]:
        return self._nodes.values()

    def contains(self, node_id: str) -> bool:
        return node_id in self._nodes

    def add_or_update(self, node: SearchNode) -> None:
        self._nodes[node.node_id] = node

    def remove(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)

    def get(self, node_id: str) -> SearchNode | None:
        return self._nodes.get(node_id)

    def _cap_ok(
        self,
        *,
        node: SearchNode,
        branch_counter: Counter[str],
        motif_counter: Counter[str],
        branch_cap: int,
        motif_cap: int,
    ) -> bool:
        branch_key = node.branch_key or node.root_candidate_id or node.candidate_id or node.node_id
        motif_key = node.motif_signature or "literal"
        if branch_cap > 0 and branch_counter[branch_key] >= branch_cap:
            return False
        if motif_cap > 0 and motif_counter[motif_key] >= motif_cap:
            return False
        return True

    def _mmr_rerank(
        self,
        *,
        candidates: list[SearchNode],
        policy: SearchPolicy,
        max_frontier_size: int,
        branch_cap: int,
        motif_cap: int,
    ) -> list[SearchNode]:
        if not candidates:
            return []

        relevant = candidates[: max(int(policy.mmr_candidate_pool_size), max_frontier_size, 1)]
        raw_scores = [float(node.frontier_score) for node in relevant]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_span = max(max_score - min_score, 1e-12)

        branch_counter: Counter[str] = Counter()
        motif_counter: Counter[str] = Counter()
        selected: list[SearchNode] = []
        remaining = list(relevant)

        while remaining and len(selected) < max_frontier_size:
            best_idx: int | None = None
            best_mmr = float("-inf")
            best_similarity = 0.0
            for idx, node in enumerate(remaining):
                if not self._cap_ok(
                    node=node,
                    branch_counter=branch_counter,
                    motif_counter=motif_counter,
                    branch_cap=branch_cap,
                    motif_cap=motif_cap,
                ):
                    continue

                normalized_relevance = (float(node.frontier_score) - min_score) / score_span
                max_similarity = 0.0
                if selected:
                    max_similarity = max(pairwise_similarity(node, chosen, policy) for chosen in selected)
                mmr_score = policy.mmr_lambda * normalized_relevance - (1.0 - policy.mmr_lambda) * max_similarity
                if mmr_score > best_mmr:
                    best_idx = idx
                    best_mmr = mmr_score
                    best_similarity = max_similarity

            if best_idx is None:
                break

            chosen = remaining.pop(best_idx)
            chosen.score_breakdown = {
                **dict(chosen.score_breakdown),
                "pre_mmr_frontier_score": float(chosen.frontier_score),
                "mmr_max_similarity": float(best_similarity),
                "mmr_lambda": float(policy.mmr_lambda),
                "mmr_score": float(best_mmr),
            }
            chosen.frontier_score = float(best_mmr)
            selected.append(chosen)
            branch_key = chosen.branch_key or chosen.root_candidate_id or chosen.candidate_id or chosen.node_id
            motif_key = chosen.motif_signature or "literal"
            branch_counter[branch_key] += 1
            motif_counter[motif_key] += 1

        return selected

    def ranked(
        self,
        *,
        policy: SearchPolicy,
        budget: SearchBudget,
        total_visits: int,
        seen_expression_count: int,
        branch_stats: dict[str, dict[str, int]] | None = None,
        motif_counts: dict[str, int] | None = None,
        normalizer: SearchNormalizer | None = None,
        reference_nodes: list[SearchNode] | None = None,
    ) -> list[SearchNode]:
        eligible: list[SearchNode] = []
        branch_stats = branch_stats or {}
        motif_counts = motif_counts or {}
        reference_nodes = list(reference_nodes or self._nodes.values())
        for node in self._nodes.values():
            if node.depth >= int(budget.max_depth):
                continue
            if node.expansions >= int(budget.branch_budget):
                continue
            branch_usage = branch_stats.get(node.branch_key or node.root_candidate_id or node.candidate_id or node.node_id, {})
            node.frontier_score = compute_frontier_score(
                node,
                policy=policy,
                total_visits=total_visits,
                seen_expression_count=seen_expression_count,
                branch_usage_count=int(branch_usage.get("expansions", 0) + branch_usage.get("visits", 0)),
                motif_usage_count=int(motif_counts.get(node.motif_signature or "literal", 0)),
                branch_budget=int(budget.branch_budget),
                normalizer=normalizer,
                reference_nodes=reference_nodes,
            )
            eligible.append(node)
        eligible.sort(
            key=lambda item: (
                item.frontier_score,
                item.base_score,
                -item.branch_penalty_score,
                -item.depth,
                -item.complexity,
            ),
            reverse=True,
        )

        if not policy.frontier_rerank:
            return eligible[: max(int(budget.max_frontier_size), 1)]

        branch_counter: Counter[str] = Counter()
        motif_counter: Counter[str] = Counter()
        reranked: list[SearchNode] = []
        max_frontier_size = max(int(budget.max_frontier_size), 1)
        branch_cap = max(int(policy.branch_frontier_cap), 0)
        motif_cap = max(int(policy.motif_frontier_cap), 0)

        if policy.mmr_rerank:
            mmr_ranked = self._mmr_rerank(
                candidates=eligible,
                policy=policy,
                max_frontier_size=max_frontier_size,
                branch_cap=branch_cap,
                motif_cap=motif_cap,
            )
            if mmr_ranked:
                return mmr_ranked

        for node in eligible:
            if not self._cap_ok(
                node=node,
                branch_counter=branch_counter,
                motif_counter=motif_counter,
                branch_cap=branch_cap,
                motif_cap=motif_cap,
            ):
                continue
            reranked.append(node)
            branch_key = node.branch_key or node.root_candidate_id or node.candidate_id or node.node_id
            motif_key = node.motif_signature or "literal"
            branch_counter[branch_key] += 1
            motif_counter[motif_key] += 1
            if len(reranked) >= max_frontier_size:
                break

        if reranked:
            return reranked
        return eligible[:max_frontier_size]

    def snapshot(
        self,
        *,
        policy: SearchPolicy,
        budget: SearchBudget,
        total_visits: int,
        seen_expression_count: int,
        branch_stats: dict[str, dict[str, int]] | None = None,
        motif_counts: dict[str, int] | None = None,
        normalizer: SearchNormalizer | None = None,
        reference_nodes: list[SearchNode] | None = None,
    ) -> list[dict[str, object]]:
        return [
            {
                "node_id": node.node_id,
                "factor_name": node.factor_name,
                "candidate_id": node.candidate_id,
                "root_candidate_id": node.root_candidate_id,
                "branch_key": node.branch_key,
                "motif_signature": node.motif_signature,
                "depth": node.depth,
                "status": node.status,
                "base_score": node.base_score,
                "quality_score": node.quality_score,
                "retrieval_score": node.retrieval_score,
                "frontier_score": node.frontier_score,
                "performance_score": node.performance_score,
                "novelty_score": node.novelty_score,
                "exploration_score": node.exploration_score,
                "underexplored_bonus_score": node.underexplored_bonus_score,
                "branch_penalty_score": node.branch_penalty_score,
                "redundancy_penalty_score": node.redundancy_penalty_score,
                "complexity_penalty_score": node.complexity_penalty_score,
                "depth_penalty_score": node.depth_penalty_score,
                "turnover_penalty_score": node.turnover_penalty_score,
                "visits": node.visits,
                "expansions": node.expansions,
                "children_total": node.children_total,
                "children_kept": node.children_kept,
                "children_winners": node.children_winners,
                "child_best_gain": node.child_best_gain,
                "child_avg_gain": node.child_avg_gain,
                "child_top3_mean_gain": node.child_top3_mean_gain,
                "child_median_gain": node.child_median_gain,
                "child_positive_gain_rate": node.child_positive_gain_rate,
                "child_gain_std": node.child_gain_std,
                "child_model_support": node.child_model_support,
                "child_model_support_rate": node.child_model_support_rate,
                "child_cross_model_convergence": node.child_cross_model_convergence,
                "child_mutation_diversity": node.child_mutation_diversity,
                "child_positive_excess_rate": node.child_positive_excess_rate,
                "child_low_turnover_rate": node.child_low_turnover_rate,
                "child_full_metrics_rate": node.child_full_metrics_rate,
                "child_admission_friendly_rate": node.child_admission_friendly_rate,
                "child_high_quality_novel_count": node.child_high_quality_novel_count,
                "child_new_motif_success_rate": node.child_new_motif_success_rate,
                "last_success_round": node.last_success_round,
                "rounds_since_last_success": node.rounds_since_last_success,
                "expandability_score": node.expandability_score,
                "expandability_confidence": node.expandability_confidence,
                "effective_expandability_score": node.effective_expandability_score,
                "branch_value_score": node.branch_value_score,
                "constraint_score": node.constraint_score,
                "portfolio_score": node.portfolio_score,
                "regime_score": node.regime_score,
                "transfer_score": node.transfer_score,
                "target_conditioned_score": node.target_conditioned_score,
                "mutation_class": str(node.metadata.get("mutation_class", "") or ""),
                "operator_skeleton": str(node.metadata.get("operator_skeleton", "") or ""),
                "economic_family_tags": list(node.metadata.get("economic_family_tags") or ()),
                "score_breakdown": dict(node.score_breakdown),
            }
            for node in self.ranked(
                policy=policy,
                budget=budget,
                total_visits=total_visits,
                seen_expression_count=seen_expression_count,
                branch_stats=branch_stats,
                motif_counts=motif_counts,
                normalizer=normalizer,
                reference_nodes=reference_nodes,
            )
        ]
