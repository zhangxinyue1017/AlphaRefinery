from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


POLICY_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    # Basic policy identity and global interpretation mode.
    "identity": (
        "name",
        "target_profile",
        "selection_strategy",
        "metric_normalization",
    ),
    # Main alpha-quality objective and normalization scales.
    "performance_objective": (
        "rank_ic_weight",
        "rank_icir_weight",
        "ann_return_weight",
        "excess_return_weight",
        "sharpe_weight",
        "rank_ic_scale",
        "rank_icir_scale",
        "ann_return_scale",
        "excess_return_scale",
        "sharpe_scale",
    ),
    # Costs that make candidates harder to deploy or maintain.
    "risk_penalty": (
        "turnover_penalty_weight",
        "complexity_penalty_weight",
        "depth_penalty_weight",
        "turnover_scale",
        "complexity_scale",
        "depth_scale",
    ),
    # Redundancy controls that push search away from repeated motifs/branches.
    "redundancy_diversity": (
        "mmr_rerank",
        "mmr_lambda",
        "branch_penalty_weight",
        "redundancy_penalty_weight",
        "family_motif_saturation_weight",
        "correlation_redundancy_weight",
        "novelty_bonus_weight",
        "motif_novelty_weight",
    ),
    # Frontier-selection mechanics and caps.
    "frontier_control": (
        "frontier_rerank",
        "prefer_unexpanded",
        "allow_keep_nodes",
        "require_novel_expression",
        "branch_frontier_cap",
        "motif_frontier_cap",
        "selection_pool_size",
        "mmr_candidate_pool_size",
    ),
    # Feature weights used to estimate similarity between candidate nodes.
    "similarity_model": (
        "similarity_branch_weight",
        "similarity_motif_weight",
        "similarity_mutation_weight",
        "similarity_skeleton_weight",
        "similarity_economic_weight",
        "similarity_operator_weight",
        "similarity_token_weight",
    ),
    # Extra objective terms activated by raw_alpha/deployability/complementarity/robustness profiles.
    "target_profile_conditioning": (
        "target_conditioned_weight",
        "constraint_weight",
        "portfolio_weight",
        "regime_weight",
        "transfer_weight",
    ),
    # Continuation-value bonuses for promising branches and expandable parents.
    "continuation_value": (
        "exploration_weight",
        "expandability_weight",
        "branch_value_weight",
        "winner_status_bonus",
        "keep_status_bonus",
    ),
    # Optional two-parent expansion policy.
    "dual_parent": (
        "dual_parent_enabled",
        "dual_parent_max_parents",
        "dual_parent_delta_threshold",
        "dual_parent_similarity_threshold",
        "dual_parent_min_expandability_advantage",
    ),
}


@dataclass(frozen=True)
class SearchPolicy:
    name: str  # Policy name for logging/selection presets.
    target_profile: str = "raw_alpha"  # Optimization profile (raw_alpha/deployability/complementarity/robustness).
    selection_strategy: str = "ucb_lite"  # Parent selection strategy.
    mmr_rerank: bool = True  # Whether to run MMR reranking for diversity.
    mmr_lambda: float = 0.72  # MMR trade-off: larger -> quality, smaller -> diversity.
    exploration_weight: float = 0.18  # Exploration bonus strength in parent selection.
    rank_ic_weight: float = 1.8  # Weight of RankIC in objective score.
    rank_icir_weight: float = 0.6  # Weight of RankICIR in objective score.
    ann_return_weight: float = 0.7  # Weight of annualized return in objective score.
    excess_return_weight: float = 0.6  # Weight of excess return in objective score.
    sharpe_weight: float = 0.8  # Weight of Sharpe in objective score.
    rank_ic_scale: float = 0.08  # Normalization scale for RankIC contribution.
    rank_icir_scale: float = 0.6  # Normalization scale for RankICIR contribution.
    ann_return_scale: float = 1.8  # Normalization scale for annualized return contribution.
    excess_return_scale: float = 1.2  # Normalization scale for excess return contribution.
    sharpe_scale: float = 2.0  # Normalization scale for Sharpe contribution.
    turnover_penalty_weight: float = 0.35  # Penalty weight for high turnover.
    complexity_penalty_weight: float = 0.025  # Penalty weight for expression complexity.
    depth_penalty_weight: float = 0.05  # Penalty weight for expression tree depth.
    branch_penalty_weight: float = 0.14  # Penalty weight for overusing same branch lineage.
    redundancy_penalty_weight: float = 0.10  # Penalty weight for candidate redundancy.
    family_motif_saturation_weight: float = 0.06  # Penalty for over-saturated motifs within family.
    correlation_redundancy_weight: float = 0.20  # Penalty for high correlation to existing winners/targets.
    expandability_weight: float = 0.08  # Bonus for candidates with good downstream expandability.
    branch_value_weight: float = 0.12  # Bonus for historically valuable branch lineages.
    target_conditioned_weight: float = 0.0  # Extra weight from target-profile conditioned term.
    constraint_weight: float = 0.0  # Global weight for deployability constraints.
    portfolio_weight: float = 0.0  # Global weight for portfolio complementarity term.
    regime_weight: float = 0.0  # Global weight for regime robustness term.
    transfer_weight: float = 0.0  # Global weight for transfer/generalization term.
    turnover_scale: float = 0.45  # Scale used when converting turnover to penalty.
    complexity_scale: float = 8.0  # Scale used when converting complexity to penalty.
    depth_scale: float = 3.0  # Scale used when converting depth to penalty.
    novelty_bonus_weight: float = 0.10  # Bonus for overall novelty.
    motif_novelty_weight: float = 0.08  # Bonus for motif-level novelty.
    winner_status_bonus: float = 0.05  # Bonus if parent has winner status.
    keep_status_bonus: float = 0.02  # Bonus if parent has keep status.
    frontier_rerank: bool = True  # Whether to rerank candidates in frontier stage.
    prefer_unexpanded: bool = True  # Prefer nodes with less expansion history.
    allow_keep_nodes: bool = True  # Whether keep-status nodes can be selected as parents.
    require_novel_expression: bool = True  # Reject exact/replay expressions.
    branch_frontier_cap: int = 2  # Max selected candidates per branch family in frontier.
    motif_frontier_cap: int = 3  # Max selected candidates per motif in frontier.
    selection_pool_size: int = 5  # Number of parents pulled into selection pool.
    mmr_candidate_pool_size: int = 8  # Candidate pool size used by MMR rerank.
    similarity_branch_weight: float = 0.4  # Branch similarity component weight.
    similarity_motif_weight: float = 0.25  # Motif similarity component weight.
    similarity_mutation_weight: float = 0.15  # Mutation-type similarity component weight.
    similarity_skeleton_weight: float = 0.2  # Expression skeleton similarity component weight.
    similarity_economic_weight: float = 0.15  # Economic-intuition similarity component weight.
    similarity_operator_weight: float = 0.2  # Operator-usage similarity component weight.
    similarity_token_weight: float = 0.1  # Token-level similarity component weight.
    metric_normalization: str = "percentile"  # Metric normalization method before weighted sum.
    dual_parent_enabled: bool = False  # Enable dual-parent synthesis mode.
    dual_parent_max_parents: int = 2  # Max parents used in one dual-parent synthesis.
    dual_parent_delta_threshold: float = 0.12  # Minimum expected gain threshold to enable dual-parent.
    dual_parent_similarity_threshold: float = 0.65  # Similarity ceiling between dual parents.
    dual_parent_min_expandability_advantage: float = 0.02  # Min expandability advantage for dual-parent usage.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def grouped_dict(self) -> dict[str, dict[str, Any]]:
        payload = self.to_dict()
        grouped: dict[str, dict[str, Any]] = {}
        assigned: set[str] = set()
        for group_name, fields in POLICY_FIELD_GROUPS.items():
            grouped[group_name] = {field: payload[field] for field in fields if field in payload}
            assigned.update(grouped[group_name])
        remaining = {key: value for key, value in payload.items() if key not in assigned}
        if remaining:
            grouped["other"] = remaining
        return grouped

    def grouped_summary(self) -> dict[str, tuple[str, ...]]:
        return {group_name: tuple(values.keys()) for group_name, values in self.grouped_dict().items()}

    def with_mmr_rerank(self, enabled: bool) -> "SearchPolicy":
        return replace(self, mmr_rerank=bool(enabled))

    def with_dual_parent(
        self,
        enabled: bool,
        *,
        max_parents: int | None = None,
        delta_threshold: float | None = None,
        similarity_threshold: float | None = None,
        min_expandability_advantage: float | None = None,
    ) -> "SearchPolicy":
        return replace(
            self,
            dual_parent_enabled=bool(enabled),
            dual_parent_max_parents=int(max_parents) if max_parents is not None else self.dual_parent_max_parents,
            dual_parent_delta_threshold=(
                float(delta_threshold) if delta_threshold is not None else self.dual_parent_delta_threshold
            ),
            dual_parent_similarity_threshold=(
                float(similarity_threshold)
                if similarity_threshold is not None
                else self.dual_parent_similarity_threshold
            ),
            dual_parent_min_expandability_advantage=(
                float(min_expandability_advantage)
                if min_expandability_advantage is not None
                else self.dual_parent_min_expandability_advantage
            ),
        )

    def with_target_profile(self, target_profile: str) -> "SearchPolicy":
        normalized = str(target_profile or "raw_alpha").strip().lower() or "raw_alpha"
        if normalized == "raw_alpha":
            return replace(
                self,
                target_profile=normalized,
                target_conditioned_weight=0.08,
                constraint_weight=0.55,
                portfolio_weight=0.45,
                regime_weight=0.0,
                transfer_weight=0.0,
                ann_return_weight=max(float(self.ann_return_weight), 0.65),
                excess_return_weight=max(float(self.excess_return_weight), 0.6),
                sharpe_weight=min(float(self.sharpe_weight), 0.8),
                ann_return_scale=min(float(self.ann_return_scale), 1.8),
                excess_return_scale=min(float(self.excess_return_scale), 0.9),
                sharpe_scale=max(float(self.sharpe_scale), 2.0),
            )
        if normalized == "deployability":
            return replace(
                self,
                target_profile=normalized,
                target_conditioned_weight=0.16,
                constraint_weight=0.75,
                portfolio_weight=0.25,
                regime_weight=0.0,
                transfer_weight=0.0,
            )
        if normalized == "complementarity":
            return replace(
                self,
                target_profile=normalized,
                target_conditioned_weight=0.24,
                constraint_weight=0.25,
                portfolio_weight=0.85,
                regime_weight=0.0,
                transfer_weight=0.0,
                redundancy_penalty_weight=max(float(self.redundancy_penalty_weight), 0.12),
                family_motif_saturation_weight=max(float(self.family_motif_saturation_weight), 0.08),
                correlation_redundancy_weight=max(float(self.correlation_redundancy_weight), 0.14),
                dual_parent_delta_threshold=max(float(self.dual_parent_delta_threshold), 0.22),
                dual_parent_similarity_threshold=min(float(self.dual_parent_similarity_threshold), 0.55),
            )
        if normalized == "robustness":
            return replace(
                self,
                target_profile=normalized,
                target_conditioned_weight=0.14,
                constraint_weight=0.55,
                portfolio_weight=0.20,
                regime_weight=0.25,
                transfer_weight=0.0,
            )
        raise ValueError(f"unknown search target profile: {target_profile}")

    @classmethod
    def available_presets(cls) -> tuple[str, ...]:
        return ("balanced", "exploratory", "conservative")

    @classmethod
    def available_target_profiles(cls) -> tuple[str, ...]:
        return ("raw_alpha", "deployability", "complementarity", "robustness")

    @classmethod
    def from_preset(cls, preset: str = "balanced") -> "SearchPolicy":
        normalized = str(preset or "balanced").strip().lower()
        if normalized == "balanced":
            return cls(name="balanced")
        if normalized == "exploratory":
            return cls(
                name="exploratory",
                mmr_lambda=0.6,
                exploration_weight=0.24,
                rank_ic_weight=1.55,
                rank_icir_weight=0.55,
                ann_return_weight=0.65,
                excess_return_weight=0.6,
                sharpe_weight=0.82,
                turnover_penalty_weight=0.35,
                complexity_penalty_weight=0.018,
                depth_penalty_weight=0.04,
                branch_penalty_weight=0.12,
                redundancy_penalty_weight=0.06,
                family_motif_saturation_weight=0.04,
                correlation_redundancy_weight=0.04,
                expandability_weight=0.1,
                branch_value_weight=0.14,
                novelty_bonus_weight=0.14,
                motif_novelty_weight=0.1,
                branch_frontier_cap=2,
                motif_frontier_cap=2,
                selection_pool_size=7,
                mmr_candidate_pool_size=10,
            )
        if normalized == "conservative":
            return cls(
                name="conservative",
                mmr_lambda=0.82,
                exploration_weight=0.1,
                rank_ic_weight=1.7,
                rank_icir_weight=0.7,
                ann_return_weight=0.75,
                excess_return_weight=0.65,
                sharpe_weight=0.95,
                turnover_penalty_weight=0.62,
                complexity_penalty_weight=0.03,
                depth_penalty_weight=0.06,
                branch_penalty_weight=0.12,
                redundancy_penalty_weight=0.06,
                family_motif_saturation_weight=0.04,
                correlation_redundancy_weight=0.08,
                expandability_weight=0.05,
                branch_value_weight=0.08,
                novelty_bonus_weight=0.04,
                motif_novelty_weight=0.03,
                branch_frontier_cap=2,
                motif_frontier_cap=3,
                selection_pool_size=4,
                mmr_candidate_pool_size=6,
            )
        raise ValueError(f"unknown search policy preset: {preset}")

    @classmethod
    def balanced(cls) -> "SearchPolicy":
        return cls.from_preset("balanced")

    @classmethod
    def exploratory(cls) -> "SearchPolicy":
        return cls.from_preset("exploratory")

    @classmethod
    def conservative(cls) -> "SearchPolicy":
        return cls.from_preset("conservative")

    @classmethod
    def for_mode(cls, mode: str, *, preset: str = "balanced") -> "SearchPolicy":
        base = cls.from_preset(preset)
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode == "multi_model_best_first":
            return replace(
                base,
                name=f"multi_model_best_first:{base.name}",
                branch_frontier_cap=max(base.branch_frontier_cap, 2),
                motif_frontier_cap=max(base.motif_frontier_cap, 3),
                selection_pool_size=max(base.selection_pool_size, 5),
                mmr_candidate_pool_size=max(base.mmr_candidate_pool_size, 8),
            )
        if normalized_mode == "family_breadth_first":
            return replace(
                base,
                name=f"family_breadth_first:{base.name}",
                exploration_weight=base.exploration_weight + 0.02,
                novelty_bonus_weight=base.novelty_bonus_weight + 0.03,
                motif_novelty_weight=base.motif_novelty_weight + 0.03,
                branch_penalty_weight=max(base.branch_penalty_weight, 0.18),
                prefer_unexpanded=True,
                branch_frontier_cap=1,
                motif_frontier_cap=2,
                selection_pool_size=max(base.selection_pool_size, 8),
                mmr_candidate_pool_size=max(base.mmr_candidate_pool_size, 12),
            )
        if normalized_mode == "local_best_first":
            return replace(
                base,
                name=f"local_best_first:{base.name}",
                exploration_weight=max(base.exploration_weight - 0.04, 0.08),
                novelty_bonus_weight=max(base.novelty_bonus_weight - 0.03, 0.03),
                motif_novelty_weight=max(base.motif_novelty_weight - 0.02, 0.02),
                branch_penalty_weight=max(base.branch_penalty_weight - 0.04, 0.08),
                branch_frontier_cap=max(base.branch_frontier_cap, 3),
                motif_frontier_cap=max(base.motif_frontier_cap, 3),
                selection_pool_size=min(max(base.selection_pool_size, 4), 5),
                mmr_candidate_pool_size=min(max(base.mmr_candidate_pool_size, 6), 8),
            )
        raise ValueError(f"unknown search policy mode: {mode}")

    @classmethod
    def multi_model_best_first(cls, *, preset: str = "balanced") -> "SearchPolicy":
        return cls.for_mode("multi_model_best_first", preset=preset)

    @classmethod
    def family_breadth_first(cls, *, preset: str = "balanced") -> "SearchPolicy":
        return cls.for_mode("family_breadth_first", preset=preset)

    @classmethod
    def local_best_first(cls, *, preset: str = "balanced") -> "SearchPolicy":
        return cls.for_mode("local_best_first", preset=preset)
