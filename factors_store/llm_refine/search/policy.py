'''Search policy presets and target-profile weights.

Defines balanced, exploratory, and conservative settings plus raw-alpha, deployability, complementarity, and robustness modes.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SearchPolicy:
    # Basic policy identity and global interpretation mode.
    name: str
    target_profile: str = "raw_alpha"
    selection_strategy: str = "ucb_lite"
    metric_normalization: str = "percentile"

    # Main alpha-quality objective and normalization scales.
    rank_ic_weight: float = 1.8
    rank_icir_weight: float = 0.6
    ann_return_weight: float = 0.7
    excess_return_weight: float = 0.6
    sharpe_weight: float = 0.8
    rank_ic_scale: float = 0.08
    rank_icir_scale: float = 0.6
    ann_return_scale: float = 1.8
    excess_return_scale: float = 1.2
    sharpe_scale: float = 2.0

    # Costs that make candidates harder to deploy or maintain.
    turnover_penalty_weight: float = 0.35
    complexity_penalty_weight: float = 0.025
    depth_penalty_weight: float = 0.05
    turnover_scale: float = 0.45
    complexity_scale: float = 8.0
    depth_scale: float = 3.0

    # Redundancy controls that push search away from repeated motifs/branches.
    mmr_rerank: bool = True
    mmr_lambda: float = 0.72
    branch_penalty_weight: float = 0.14
    redundancy_penalty_weight: float = 0.10
    family_motif_saturation_weight: float = 0.06
    correlation_redundancy_weight: float = 0.20
    novelty_bonus_weight: float = 0.10
    motif_novelty_weight: float = 0.08

    # Frontier-selection mechanics and caps.
    frontier_rerank: bool = True
    prefer_unexpanded: bool = True
    allow_keep_nodes: bool = True
    require_novel_expression: bool = True
    branch_frontier_cap: int = 2
    motif_frontier_cap: int = 3
    selection_pool_size: int = 5
    mmr_candidate_pool_size: int = 8

    # Feature weights used to estimate similarity between candidate nodes.
    similarity_branch_weight: float = 0.4
    similarity_motif_weight: float = 0.25
    similarity_mutation_weight: float = 0.15
    similarity_skeleton_weight: float = 0.2
    similarity_economic_weight: float = 0.15
    similarity_operator_weight: float = 0.2
    similarity_token_weight: float = 0.1

    # Extra objective terms activated by raw_alpha/deployability/complementarity/robustness profiles.
    target_conditioned_weight: float = 0.0
    constraint_weight: float = 0.0
    portfolio_weight: float = 0.0
    regime_weight: float = 0.0
    transfer_weight: float = 0.0

    # Continuation-value bonuses for promising branches and expandable parents.
    exploration_weight: float = 0.18
    expandability_weight: float = 0.08
    branch_value_weight: float = 0.12
    winner_status_bonus: float = 0.05
    keep_status_bonus: float = 0.02

    # Optional two-parent expansion policy.
    dual_parent_enabled: bool = False
    dual_parent_max_parents: int = 2
    dual_parent_delta_threshold: float = 0.12
    dual_parent_similarity_threshold: float = 0.65
    dual_parent_min_expandability_advantage: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
