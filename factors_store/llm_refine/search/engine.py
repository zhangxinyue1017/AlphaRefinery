'''Best-first search engine for refinement frontier management.

Selects parents, records expansions, tracks branch value, and updates searchable family state.
'''

from __future__ import annotations

from collections import Counter, defaultdict
import statistics
from typing import Any

from ..core.archive import make_seed_candidate_id, utc_now_iso
from .frontier import SearchFrontier
from .normalization import SearchNormalizer
from .policy import SearchPolicy
from .scoring import (
    compute_base_score,
    compute_branch_value_score,
    compute_expandability_score,
    compute_parent_child_gain,
    expression_motif_signature,
    pairwise_similarity,
    winner_improved,
)
from .state import SearchBudget, SearchEdge, SearchNode


class SearchEngine:
    def __init__(
        self,
        *,
        family: str,
        budget: SearchBudget,
        policy: SearchPolicy,
        normalizer: SearchNormalizer | None = None,
    ) -> None:
        self.family = family
        self.budget = budget
        self.policy = policy
        self.normalizer = normalizer
        self.frontier = SearchFrontier()
        self.nodes: dict[str, SearchNode] = {}
        self.edges: list[SearchEdge] = []
        self.best_node_id: str = ""
        self.total_visits: int = 0
        self.attempts_used: int = 0
        self.successful_rounds: int = 0
        self.consecutive_no_improve: int = 0
        self.event_log: list[dict[str, Any]] = []
        self._seen_expressions: set[str] = set()

    def _ranked_frontier(self) -> list[SearchNode]:
        return self.frontier.ranked(
            policy=self.policy,
            budget=self.budget,
            total_visits=self.total_visits,
            seen_expression_count=len(self._seen_expressions),
            branch_stats=self._branch_stats(),
            motif_counts=self._motif_counts(),
            normalizer=self.normalizer,
            reference_nodes=list(self.nodes.values()),
        )

    def _best_node_sort_key(self, node: SearchNode) -> tuple[float, float]:
        return (1.0 if bool(getattr(node, "eligible_for_best_node", True)) else 0.0, float(node.base_score))

    def _selection_reason_tags(
        self,
        *,
        selected: SearchNode,
        runner_up: SearchNode | None,
    ) -> list[str]:
        breakdown = dict(getattr(selected, "score_breakdown", {}) or {})
        tags: list[str] = []
        if float(breakdown.get("expandability_bonus", 0.0) or 0.0) > 0.05:
            tags.append("expandability_bonus")
        if float(breakdown.get("branch_value_bonus", 0.0) or 0.0) > 0.05:
            tags.append("branch_value_bonus")
        if float(breakdown.get("target_conditioned_score", 0.0) or 0.0) > 0.05:
            tags.append(f"target_fit:{self.policy.target_profile}")
        if float(breakdown.get("constraint_score", 0.0) or 0.0) > 0.7:
            tags.append("constraint_fit")
        if float(breakdown.get("portfolio_score", 0.0) or 0.0) > 0.7:
            tags.append("portfolio_fit")
        if float(breakdown.get("novelty_score", 0.0) or 0.0) > 0.05:
            tags.append("novelty_support")
        if float(breakdown.get("underexplored_bonus", 0.0) or 0.0) > 0.05:
            tags.append("underexplored_bonus")
        if float(breakdown.get("redundancy_penalty", 0.0) or 0.0) < 0.08:
            tags.append("low_redundancy")
        if bool(breakdown.get("eligible_for_best_node", True)):
            tags.append("complete_metrics")
        if runner_up is not None:
            runner = dict(getattr(runner_up, "score_breakdown", {}) or {})
            if float(breakdown.get("quality_score", 0.0) or 0.0) > float(runner.get("quality_score", 0.0) or 0.0):
                tags.append("quality_win")
            if float(breakdown.get("retrieval_score", 0.0) or 0.0) > float(runner.get("retrieval_score", 0.0) or 0.0):
                tags.append("retrieval_win")
        return list(dict.fromkeys(tags))

    def _secondary_parent_reason_tags(
        self,
        *,
        primary: SearchNode,
        secondary: SearchNode,
        similarity: float,
    ) -> list[str]:
        tags = ["dual_parent_expandability"]
        primary_branch_signal = max(float(primary.branch_value_score), float(primary.effective_expandability_score))
        secondary_branch_signal = max(float(secondary.branch_value_score), float(secondary.effective_expandability_score))
        if secondary_branch_signal > primary_branch_signal:
            tags.append("branch_value_lead")
        if float(secondary.target_conditioned_score) > float(primary.target_conditioned_score):
            tags.append(f"target_fit:{self.policy.target_profile}")
        if float(similarity) < float(self.policy.dual_parent_similarity_threshold):
            tags.append("diversified_branch")
        if str(secondary.branch_key or "") != str(primary.branch_key or ""):
            tags.append("branch_complement")
        if str(secondary.motif_signature or "") != str(primary.motif_signature or ""):
            tags.append("motif_complement")
        return list(dict.fromkeys(tags))

    def _selection_pool_payload(self, ranked: list[SearchNode]) -> list[dict[str, Any]]:
        return [
            {
                "node_id": item.node_id,
                "factor_name": item.factor_name,
                "branch_key": item.branch_key,
                "motif_signature": item.motif_signature,
                "frontier_score": item.frontier_score,
                "base_score": item.base_score,
                "branch_value_score": item.branch_value_score,
                "effective_expandability_score": item.effective_expandability_score,
                "constraint_score": item.constraint_score,
                "portfolio_score": item.portfolio_score,
                "target_conditioned_score": item.target_conditioned_score,
                "score_breakdown": dict(item.score_breakdown),
            }
            for item in ranked[: max(int(self.policy.selection_pool_size), 1)]
        ]

    def _select_secondary_parent(
        self,
        *,
        primary: SearchNode,
        candidates: list[SearchNode],
    ) -> tuple[SearchNode | None, float | None]:
        if not bool(self.policy.dual_parent_enabled):
            return None, None
        if int(self.policy.dual_parent_max_parents) < 2:
            return None, None

        best_candidate: SearchNode | None = None
        best_similarity: float | None = None
        best_key: tuple[float, float, float] | None = None
        complementarity_mode = str(self.policy.target_profile or "").strip().lower() == "complementarity"
        target_bonus_weight = 0.45 if complementarity_mode else 0.25
        portfolio_bonus_weight = 0.25 if complementarity_mode else 0.0
        primary_frontier = float(primary.frontier_score)
        primary_expandability = (
            max(
                float(primary.branch_value_score),
                float(primary.effective_expandability_score),
            )
            + target_bonus_weight * float(primary.target_conditioned_score)
            + portfolio_bonus_weight * float(primary.portfolio_score)
        )
        min_expandability = max(
            primary_expandability + float(self.policy.dual_parent_min_expandability_advantage),
            0.03 if complementarity_mode else 0.05,
        )

        for candidate in candidates:
            if candidate.node_id == primary.node_id:
                continue
            delta = primary_frontier - float(candidate.frontier_score)
            if delta > float(self.policy.dual_parent_delta_threshold):
                continue
            similarity = float(pairwise_similarity(primary, candidate, self.policy))
            if similarity >= float(self.policy.dual_parent_similarity_threshold):
                continue
            if str(candidate.branch_key or "") == str(primary.branch_key or ""):
                continue
            candidate_branch_signal = (
                max(
                    float(candidate.branch_value_score),
                    float(candidate.effective_expandability_score),
                )
                + target_bonus_weight * float(candidate.target_conditioned_score)
                + portfolio_bonus_weight * float(candidate.portfolio_score)
            )
            if candidate_branch_signal < min_expandability:
                continue
            if complementarity_mode:
                key = (
                    candidate_branch_signal,
                    float(candidate.portfolio_score),
                    1.0 - similarity,
                    float(candidate.frontier_score),
                    float(candidate.base_score),
                )
            else:
                key = (
                    candidate_branch_signal,
                    float(candidate.frontier_score),
                    float(candidate.base_score),
                )
            if best_key is None or key > best_key:
                best_candidate = candidate
                best_similarity = similarity
                best_key = key
        return best_candidate, best_similarity

    def _register_node(self, node: SearchNode, *, allow_frontier: bool = True) -> SearchNode:
        candidate_id = node.candidate_id or node.node_id
        node.root_candidate_id = node.root_candidate_id or candidate_id
        node.branch_key = node.branch_key or node.root_candidate_id
        node.motif_signature = node.motif_signature or expression_motif_signature(node.expression)
        node.base_score = compute_base_score(node, self.policy, normalizer=self.normalizer)
        self.nodes[node.node_id] = node
        if node.expression:
            self._seen_expressions.add(node.expression.strip())
        if allow_frontier:
            self.frontier.add_or_update(node)
        best = self.best_node
        if best is None or self._best_node_sort_key(node) > self._best_node_sort_key(best):
            self.best_node_id = node.node_id
        return node

    @property
    def best_node(self) -> SearchNode | None:
        if not self.best_node_id:
            return None
        return self.nodes.get(self.best_node_id)

    def register_seed(
        self,
        *,
        factor_name: str,
        expression: str,
        candidate_id: str = "",
        node_kind: str = "seed",
        status: str = "seed",
        source_run_id: str = "",
        source_model: str = "",
        source_provider: str = "",
        round_id: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> SearchNode:
        payload = metrics or {}
        node = SearchNode(
            node_id=candidate_id or make_seed_candidate_id(factor_name),
            family=self.family,
            factor_name=factor_name,
            expression=expression,
            candidate_id=candidate_id or make_seed_candidate_id(factor_name),
            root_candidate_id=candidate_id or make_seed_candidate_id(factor_name),
            branch_key=candidate_id or make_seed_candidate_id(factor_name),
            source_run_id=source_run_id,
            source_model=source_model,
            source_provider=source_provider,
            round_id=int(round_id),
            node_kind=node_kind,
            status=status,
            quick_rank_ic_mean=payload.get("quick_rank_ic_mean"),
            quick_rank_icir=payload.get("quick_rank_icir"),
            net_ann_return=payload.get("net_ann_return"),
            net_excess_ann_return=payload.get("net_excess_ann_return"),
            net_sharpe=payload.get("net_sharpe"),
            mean_turnover=payload.get("mean_turnover"),
            evaluated_at=str(payload.get("evaluated_at", "")),
            created_at=utc_now_iso(),
        )
        return self._register_node(node)

    def _branch_stats(self) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"nodes": 0, "visits": 0, "expansions": 0, "successful_expansions": 0}
        )
        for node in self.nodes.values():
            branch_key = node.branch_key or node.root_candidate_id or node.candidate_id or node.node_id
            row = stats[branch_key]
            row["nodes"] += 1
            row["visits"] += int(node.visits)
            row["expansions"] += int(node.expansions)
            row["successful_expansions"] += int(node.successful_expansions)
        return dict(stats)

    def _motif_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for node in self.nodes.values():
            counts[node.motif_signature or "literal"] += 1
        return dict(counts)

    def record_attempt(self, *, parent_node_id: str, note: str = "") -> None:
        self.attempts_used += 1
        self.event_log.append(
            {
                "event": "attempt",
                "at": utc_now_iso(),
                "parent_node_id": parent_node_id,
                "note": note,
            }
        )

    def select_next_parents(self) -> list[dict[str, Any]]:
        ranked = self._ranked_frontier()
        if not ranked:
            return []
        primary = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        primary.visits += 1
        selected_parents: list[dict[str, Any]] = [
            {
                "node": primary,
                "role": "quality_parent",
                "reason_tags": self._selection_reason_tags(selected=primary, runner_up=runner_up),
                "similarity_to_primary": None,
            }
        ]
        secondary, secondary_similarity = self._select_secondary_parent(
            primary=primary,
            candidates=ranked[1 : max(int(self.policy.selection_pool_size), 2)],
        )
        if secondary is not None:
            secondary.visits += 1
            selected_parents.append(
                {
                    "node": secondary,
                    "role": "expandability_parent",
                    "reason_tags": self._secondary_parent_reason_tags(
                        primary=primary,
                        secondary=secondary,
                        similarity=float(secondary_similarity or 0.0),
                    ),
                    "similarity_to_primary": float(secondary_similarity or 0.0),
                }
            )
        self.total_visits += len(selected_parents)
        selection_pool = self._selection_pool_payload(ranked)
        self.event_log.append(
            {
                "event": "select",
                "at": utc_now_iso(),
                "node_id": primary.node_id,
                "factor_name": primary.factor_name,
                "branch_key": primary.branch_key,
                "motif_signature": primary.motif_signature,
                "frontier_score": primary.frontier_score,
                "base_score": primary.base_score,
                "selected_parent_score_breakdown": dict(primary.score_breakdown),
                "selected_parent_reason_tags": selected_parents[0]["reason_tags"],
                "runner_up": (
                    {
                        "node_id": runner_up.node_id,
                        "factor_name": runner_up.factor_name,
                        "frontier_score": runner_up.frontier_score,
                        "base_score": runner_up.base_score,
                        "score_breakdown": dict(runner_up.score_breakdown),
                    }
                    if runner_up is not None
                    else None
                ),
                "selected_vs_runnerup_delta": (
                    float(primary.frontier_score) - float(runner_up.frontier_score)
                    if runner_up is not None
                    else None
                ),
                "score_breakdown": dict(primary.score_breakdown),
                "dual_parent_triggered": len(selected_parents) > 1,
                "selected_parents": [
                    {
                        "node_id": entry["node"].node_id,
                        "factor_name": entry["node"].factor_name,
                        "expression": entry["node"].expression,
                        "candidate_id": entry["node"].candidate_id,
                        "branch_key": entry["node"].branch_key,
                        "motif_signature": entry["node"].motif_signature,
                        "frontier_score": entry["node"].frontier_score,
                        "base_score": entry["node"].base_score,
                        "score_breakdown": dict(entry["node"].score_breakdown),
                        "role": str(entry["role"]),
                        "reason_tags": list(entry["reason_tags"]),
                        "similarity_to_primary": entry["similarity_to_primary"],
                    }
                    for entry in selected_parents
                ],
                "selection_pool": selection_pool,
            }
        )
        return selected_parents

    def select_next(self) -> SearchNode | None:
        selected_parents = self.select_next_parents()
        if not selected_parents:
            return None
        return selected_parents[0]["node"]

    def _register_expansion_internal(
        self,
        *,
        parent_node_id: str,
        child_records: list[dict[str, Any]],
        success: bool,
        source_run_id: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        parent = self.nodes[parent_node_id]
        parent.expansions += 1

        previous_best = self.best_node
        child_nodes: list[SearchNode] = []
        for record in child_records:
            factor_name = str(record.get("factor_name", "")).strip()
            expression = str(record.get("expression", "")).strip()
            candidate_id = str(record.get("candidate_id", "")).strip() or make_seed_candidate_id(factor_name)
            if not factor_name or not expression:
                continue
            if self.policy.require_novel_expression and expression in self._seen_expressions:
                continue
            status = str(record.get("status", "")).strip()
            eligible_raw = record.get("eligible_for_best_node", True)
            eligible_for_best_node = str(eligible_raw).strip().lower() not in {"", "0", "false", "no"}
            if not self.policy.allow_keep_nodes and "keep" in status.lower():
                continue
            node = SearchNode(
                node_id=candidate_id,
                family=self.family,
                factor_name=factor_name,
                expression=expression,
                candidate_id=candidate_id,
                parent_candidate_id=str(record.get("parent_candidate_id", "")).strip() or parent.candidate_id,
                root_candidate_id=parent.root_candidate_id or parent.candidate_id or parent.node_id,
                branch_key=parent.branch_key or parent.root_candidate_id or parent.candidate_id or parent.node_id,
                motif_signature=expression_motif_signature(expression),
                source_run_id=str(record.get("run_id", "")).strip() or source_run_id,
                source_model=str(record.get("source_model", "")).strip(),
                source_provider=str(record.get("source_provider", "")).strip(),
                round_id=int(record.get("round_id") or parent.round_id + 1),
                depth=parent.depth + 1,
                node_kind="candidate",
                status=status,
                quick_rank_ic_mean=record.get("quick_rank_ic_mean"),
                quick_rank_icir=record.get("quick_rank_icir"),
                net_ann_return=record.get("net_ann_return"),
                net_excess_ann_return=record.get("net_excess_ann_return"),
                net_sharpe=record.get("net_sharpe"),
                mean_turnover=record.get("mean_turnover"),
                metrics_completeness=float(record.get("metrics_completeness") or 0.0),
                missing_core_metrics_count=int(record.get("missing_core_metrics_count") or 0),
                eligible_for_best_node=eligible_for_best_node,
                evaluated_at=str(record.get("evaluated_at", "")),
                created_at=utc_now_iso(),
                metadata={
                    "candidate_role": str(record.get("candidate_role", "")).strip(),
                    "parent_expression": parent.expression,
                    "parent_factor_name": parent.factor_name,
                },
            )
            child_nodes.append(self._register_node(node))
            self.edges.append(
                SearchEdge(
                    parent_node_id=parent.node_id,
                    child_node_id=node.node_id,
                    edge_type="refine",
                    source_run_id=node.source_run_id,
                    created_at=utc_now_iso(),
                )
            )

        improved = False
        if child_nodes:
            best_child = max(child_nodes, key=lambda item: item.base_score)
            if winner_improved(best_child, previous_best) or (previous_best is None):
                if bool(getattr(best_child, "eligible_for_best_node", True)):
                    self.best_node_id = best_child.node_id
                    improved = True
                    parent.successful_expansions += 1

            gains = [compute_parent_child_gain(parent, child) for child in child_nodes]
            keep_like_statuses = ("research_keep", "keep", "research_winner", "winner", "research_keep_exploratory")
            winner_like_statuses = ("research_winner", "winner")
            parent.children_total += len(child_nodes)
            parent.children_kept += sum(
                1 for child in child_nodes if str(child.status or "").lower() in keep_like_statuses
            )
            parent.children_winners += sum(
                1 for child in child_nodes if str(child.status or "").lower() in winner_like_statuses
            )
            parent.child_best_gain = max([parent.child_best_gain, *gains])
            total_before = max(parent.children_total - len(child_nodes), 0)
            prev_sum = float(parent.child_avg_gain) * float(total_before)
            parent.child_avg_gain = (prev_sum + sum(gains)) / float(max(parent.children_total, 1))
            gain_history = [float(item) for item in (parent.metadata.get("child_gain_history") or [])]
            gain_history.extend(float(item) for item in gains)
            gain_history_sorted = sorted(gain_history, reverse=True)
            parent.metadata["child_gain_history"] = gain_history
            parent.child_top3_mean_gain = (
                sum(gain_history_sorted[:3]) / float(min(len(gain_history_sorted), 3))
                if gain_history_sorted
                else 0.0
            )
            parent.child_median_gain = float(statistics.median(gain_history)) if gain_history else 0.0
            parent.child_positive_gain_rate = (
                float(sum(1 for item in gain_history if float(item) > 0.0)) / float(len(gain_history))
                if gain_history
                else 0.0
            )
            parent.child_gain_std = float(statistics.pstdev(gain_history)) if len(gain_history) > 1 else 0.0
            model_set = {
                str(item).strip()
                for item in (parent.metadata.get("child_models_seen") or [])
                if str(item).strip()
            }
            mutation_set = {
                str(item).strip()
                for item in (parent.metadata.get("child_mutation_classes_seen") or [])
                if str(item).strip()
            }
            model_set.update(
                str(child.source_model).strip() for child in child_nodes if str(child.source_model).strip()
            )
            mutation_set.update(
                str(child.metadata.get("mutation_class", "")).strip()
                for child in child_nodes
                if str(child.metadata.get("mutation_class", "")).strip()
            )
            parent.metadata["child_models_seen"] = sorted(model_set)
            parent.metadata["child_mutation_classes_seen"] = sorted(mutation_set)
            parent.child_model_support = len(model_set)
            parent.child_model_support_rate = min(float(parent.child_model_support) / 3.0, 1.0)
            parent.child_mutation_diversity = len(mutation_set)
            motif_model_map: dict[str, set[str]] = {}
            stored_motif_model_map = dict(parent.metadata.get("child_motif_model_map") or {})
            for motif_key, models in stored_motif_model_map.items():
                motif_model_map[str(motif_key)] = {
                    str(item).strip() for item in (models or []) if str(item).strip()
                }
            for child in child_nodes:
                motif_key = str(child.motif_signature or "literal")
                model_name = str(child.source_model or "").strip()
                if model_name:
                    motif_model_map.setdefault(motif_key, set()).add(model_name)
            parent.metadata["child_motif_model_map"] = {
                motif_key: sorted(models)
                for motif_key, models in motif_model_map.items()
            }
            max_motif_model_support = max((len(models) for models in motif_model_map.values()), default=0)
            if parent.child_model_support > 1:
                parent.child_cross_model_convergence = max(
                    0.0,
                    min(
                        float(max_motif_model_support - 1) / float(max(parent.child_model_support - 1, 1)),
                        1.0,
                    ),
                )
            else:
                parent.child_cross_model_convergence = 0.0

            positive_excess_count = int(parent.metadata.get("child_positive_excess_count") or 0)
            low_turnover_count = int(parent.metadata.get("child_low_turnover_count") or 0)
            full_metrics_count = int(parent.metadata.get("child_full_metrics_count") or 0)
            admission_friendly_count = int(parent.metadata.get("child_admission_friendly_count") or 0)
            high_quality_novel_count = int(parent.metadata.get("child_high_quality_novel_count") or 0)
            new_motif_total_count = int(parent.metadata.get("child_new_motif_total_count") or 0)
            new_motif_success_count = int(parent.metadata.get("child_new_motif_success_count") or 0)
            parent_motif_signature = str(parent.motif_signature or "literal")
            parent_turnover = float(parent.mean_turnover) if parent.mean_turnover is not None else float("nan")
            turnover_threshold = 0.20
            if parent_turnover == parent_turnover:
                turnover_threshold = min(max(parent_turnover * 1.10, 0.12), 0.30)

            last_success_round = int(parent.last_success_round or parent.metadata.get("last_success_round") or 0)
            current_round_marker = max([parent.round_id, *[int(child.round_id) for child in child_nodes]], default=parent.round_id)

            for child, gain in zip(child_nodes, gains):
                positive_excess = float(child.net_excess_ann_return) > 0.0 if child.net_excess_ann_return is not None else False
                low_turnover = float(child.mean_turnover) <= turnover_threshold if child.mean_turnover is not None else False
                full_metrics = int(getattr(child, "missing_core_metrics_count", 0) or 0) == 0
                admission_friendly = positive_excess and low_turnover and full_metrics
                if positive_excess:
                    positive_excess_count += 1
                if low_turnover:
                    low_turnover_count += 1
                if full_metrics:
                    full_metrics_count += 1
                if admission_friendly:
                    admission_friendly_count += 1

                child_similarity = float(pairwise_similarity(parent, child, self.policy))
                child_is_keep_like = str(child.status or "").lower() in keep_like_statuses
                if float(gain) > 0.15 and child_similarity < float(self.policy.dual_parent_similarity_threshold) and child_is_keep_like:
                    high_quality_novel_count += 1

                child_motif = str(child.motif_signature or "literal")
                if child_motif != parent_motif_signature:
                    new_motif_total_count += 1
                    if child_is_keep_like:
                        new_motif_success_count += 1

                if child_is_keep_like or float(gain) > 0.0:
                    last_success_round = max(last_success_round, int(child.round_id))

            parent.metadata["child_positive_excess_count"] = positive_excess_count
            parent.metadata["child_low_turnover_count"] = low_turnover_count
            parent.metadata["child_full_metrics_count"] = full_metrics_count
            parent.metadata["child_admission_friendly_count"] = admission_friendly_count
            parent.metadata["child_high_quality_novel_count"] = high_quality_novel_count
            parent.metadata["child_new_motif_total_count"] = new_motif_total_count
            parent.metadata["child_new_motif_success_count"] = new_motif_success_count
            parent.metadata["last_success_round"] = last_success_round
            parent.child_positive_excess_rate = float(positive_excess_count) / float(max(parent.children_total, 1))
            parent.child_low_turnover_rate = float(low_turnover_count) / float(max(parent.children_total, 1))
            parent.child_full_metrics_rate = float(full_metrics_count) / float(max(parent.children_total, 1))
            parent.child_admission_friendly_rate = float(admission_friendly_count) / float(max(parent.children_total, 1))
            parent.child_high_quality_novel_count = high_quality_novel_count
            parent.child_new_motif_success_rate = (
                float(new_motif_success_count) / float(max(new_motif_total_count, 1))
                if new_motif_total_count > 0
                else 0.0
            )
            parent.last_success_round = last_success_round
            parent.rounds_since_last_success = max(int(current_round_marker) - int(last_success_round), 0) if last_success_round > 0 else 0
            expandability_score, confidence, expandability_breakdown = compute_expandability_score(parent)
            parent.expandability_score = expandability_score
            parent.expandability_confidence = confidence
            parent.effective_expandability_score = expandability_breakdown["effective_expandability_score"]
            parent.metadata["expandability_breakdown"] = expandability_breakdown
            branch_value_score, branch_value_confidence, branch_value_breakdown = compute_branch_value_score(parent)
            parent.branch_value_score = branch_value_score
            parent.metadata["branch_value_confidence"] = branch_value_confidence
            parent.metadata["branch_value_breakdown"] = branch_value_breakdown

        self.event_log.append(
            {
                "event": "expand",
                "at": utc_now_iso(),
                "parent_node_id": parent.node_id,
                "parent_factor_name": parent.factor_name,
                "success": bool(success),
                "children_added": len(child_nodes),
                "improved": bool(improved),
                "parent_expandability_score": parent.expandability_score,
                "parent_expandability_confidence": parent.expandability_confidence,
                "parent_effective_expandability_score": parent.effective_expandability_score,
                "parent_branch_value_score": parent.branch_value_score,
                "source_run_id": source_run_id,
                "branch_key": parent.branch_key,
                "note": note,
            }
        )
        return {
            "parent": parent.to_dict(),
            "children": [item.to_dict() for item in child_nodes],
            "improved": bool(improved),
            "success": bool(success),
            "best_node": self.best_node.to_dict() if self.best_node is not None else None,
        }

    def register_expansion(
        self,
        *,
        parent_node_id: str,
        child_records: list[dict[str, Any]],
        success: bool,
        source_run_id: str = "",
        note: str = "",
        count_attempt: bool = True,
    ) -> dict[str, Any]:
        if count_attempt:
            self.attempts_used += 1
        result = self._register_expansion_internal(
            parent_node_id=parent_node_id,
            child_records=child_records,
            success=success,
            source_run_id=source_run_id,
            note=note,
        )
        if success:
            self.successful_rounds += 1
        if success and not bool(result.get("improved")):
            self.consecutive_no_improve += 1
        elif success and bool(result.get("improved")):
            self.consecutive_no_improve = 0
        return result

    def register_round_expansions(
        self,
        *,
        expansions: list[dict[str, Any]],
        note: str = "",
    ) -> dict[str, Any]:
        self.attempts_used += 1
        results: list[dict[str, Any]] = []
        any_success = False
        any_improved = False
        for payload in expansions:
            success = bool(payload.get("success"))
            result = self._register_expansion_internal(
                parent_node_id=str(payload.get("parent_node_id", "")),
                child_records=list(payload.get("child_records") or []),
                success=success,
                source_run_id=str(payload.get("source_run_id", "")),
                note=str(payload.get("note", "")),
            )
            results.append(result)
            any_success = any_success or success
            any_improved = any_improved or bool(result.get("improved"))
        if any_success:
            self.successful_rounds += 1
        if any_success and any_improved:
            self.consecutive_no_improve = 0
        elif any_success and not any_improved:
            self.consecutive_no_improve += 1
        self.event_log.append(
            {
                "event": "round_expand",
                "at": utc_now_iso(),
                "parent_node_ids": [str(item.get("parent_node_id", "")) for item in expansions],
                "success_count": sum(1 for item in expansions if bool(item.get("success"))),
                "expansion_count": len(expansions),
                "improved": bool(any_improved),
                "note": note,
            }
        )
        return {
            "results": results,
            "improved": bool(any_improved),
            "best_node": self.best_node.to_dict() if self.best_node is not None else None,
        }

    def can_continue(self) -> bool:
        if self.successful_rounds >= int(self.budget.max_rounds):
            return False
        if self.attempts_used >= int(self.budget.family_budget):
            return False
        if int(self.budget.stop_if_no_improve) > 0 and self.consecutive_no_improve >= int(self.budget.stop_if_no_improve):
            return False
        return bool(
            self.frontier.ranked(
                policy=self.policy,
                budget=self.budget,
                total_visits=self.total_visits,
                seen_expression_count=len(self._seen_expressions),
                branch_stats=self._branch_stats(),
                motif_counts=self._motif_counts(),
                normalizer=self.normalizer,
                reference_nodes=list(self.nodes.values()),
            )
        )

    def frontier_snapshot(self) -> list[dict[str, object]]:
        return self.frontier.snapshot(
            policy=self.policy,
            budget=self.budget,
            total_visits=self.total_visits,
            seen_expression_count=len(self._seen_expressions),
            branch_stats=self._branch_stats(),
            motif_counts=self._motif_counts(),
            normalizer=self.normalizer,
            reference_nodes=list(self.nodes.values()),
        )

    def summary(self) -> dict[str, Any]:
        branch_stats = self._branch_stats()
        motif_counts = self._motif_counts()
        return {
            "family": self.family,
            "policy": self.policy.to_dict(),
            "normalizer": self.normalizer.to_dict() if self.normalizer is not None else None,
            "budget": self.budget.to_dict(),
            "attempts_used": self.attempts_used,
            "successful_rounds": self.successful_rounds,
            "consecutive_no_improve": self.consecutive_no_improve,
            "best_node": self.best_node.to_dict() if self.best_node is not None else None,
            "branch_stats": branch_stats,
            "motif_counts": motif_counts,
            "frontier": self.frontier_snapshot(),
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "event_log": list(self.event_log),
        }
