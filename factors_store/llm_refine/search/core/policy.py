'''Search policy presets and target-profile weights.

Defines balanced, exploratory, and conservative settings plus raw-alpha, deployability, complementarity, and robustness modes.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from ..policy_config import DEFAULT_POLICY_CONFIG


_SEARCH_DEFAULTS = DEFAULT_POLICY_CONFIG.search.base
_DECORRELATION_DEFAULTS = DEFAULT_POLICY_CONFIG.decorrelation


@dataclass(frozen=True)
class SearchPolicy:
    # Basic policy identity and global interpretation mode.
    name: str
    target_profile: str = "raw_alpha"
    selection_strategy: str = _SEARCH_DEFAULTS.selection_strategy
    metric_normalization: str = _SEARCH_DEFAULTS.metric_normalization

    # Main alpha-quality objective and normalization scales.
    rank_ic_weight: float = _SEARCH_DEFAULTS.rank_ic_weight
    rank_icir_weight: float = _SEARCH_DEFAULTS.rank_icir_weight
    ann_return_weight: float = _SEARCH_DEFAULTS.ann_return_weight
    excess_return_weight: float = _SEARCH_DEFAULTS.excess_return_weight
    sharpe_weight: float = _SEARCH_DEFAULTS.sharpe_weight
    rank_ic_scale: float = _SEARCH_DEFAULTS.rank_ic_scale
    rank_icir_scale: float = _SEARCH_DEFAULTS.rank_icir_scale
    ann_return_scale: float = _SEARCH_DEFAULTS.ann_return_scale
    excess_return_scale: float = _SEARCH_DEFAULTS.excess_return_scale
    sharpe_scale: float = _SEARCH_DEFAULTS.sharpe_scale

    # Costs that make candidates harder to deploy or maintain.
    turnover_penalty_weight: float = _SEARCH_DEFAULTS.turnover_penalty_weight
    complexity_penalty_weight: float = _SEARCH_DEFAULTS.complexity_penalty_weight
    depth_penalty_weight: float = _SEARCH_DEFAULTS.depth_penalty_weight
    turnover_scale: float = _SEARCH_DEFAULTS.turnover_scale
    complexity_scale: float = _SEARCH_DEFAULTS.complexity_scale
    depth_scale: float = _SEARCH_DEFAULTS.depth_scale

    # Redundancy controls that push search away from repeated motifs/branches.
    mmr_rerank: bool = _SEARCH_DEFAULTS.mmr_rerank
    mmr_lambda: float = _SEARCH_DEFAULTS.mmr_lambda
    branch_penalty_weight: float = _SEARCH_DEFAULTS.branch_penalty_weight
    redundancy_penalty_weight: float = _SEARCH_DEFAULTS.redundancy_penalty_weight
    family_motif_saturation_weight: float = _SEARCH_DEFAULTS.family_motif_saturation_weight
    correlation_redundancy_weight: float = _SEARCH_DEFAULTS.correlation_redundancy_weight
    novelty_bonus_weight: float = _SEARCH_DEFAULTS.novelty_bonus_weight
    motif_novelty_weight: float = _SEARCH_DEFAULTS.motif_novelty_weight

    # Frontier-selection mechanics and caps.
    frontier_rerank: bool = _SEARCH_DEFAULTS.frontier_rerank
    prefer_unexpanded: bool = _SEARCH_DEFAULTS.prefer_unexpanded
    allow_keep_nodes: bool = _SEARCH_DEFAULTS.allow_keep_nodes
    require_novel_expression: bool = _SEARCH_DEFAULTS.require_novel_expression
    branch_frontier_cap: int = _SEARCH_DEFAULTS.branch_frontier_cap
    motif_frontier_cap: int = _SEARCH_DEFAULTS.motif_frontier_cap
    selection_pool_size: int = _SEARCH_DEFAULTS.selection_pool_size
    mmr_candidate_pool_size: int = _SEARCH_DEFAULTS.mmr_candidate_pool_size

    # Feature weights used to estimate similarity between candidate nodes.
    similarity_branch_weight: float = _SEARCH_DEFAULTS.similarity_branch_weight
    similarity_motif_weight: float = _SEARCH_DEFAULTS.similarity_motif_weight
    similarity_mutation_weight: float = _SEARCH_DEFAULTS.similarity_mutation_weight
    similarity_skeleton_weight: float = _SEARCH_DEFAULTS.similarity_skeleton_weight
    similarity_economic_weight: float = _SEARCH_DEFAULTS.similarity_economic_weight
    similarity_operator_weight: float = _SEARCH_DEFAULTS.similarity_operator_weight
    similarity_token_weight: float = _SEARCH_DEFAULTS.similarity_token_weight

    # Extra objective terms activated by raw_alpha/deployability/complementarity/robustness profiles.
    target_conditioned_weight: float = _SEARCH_DEFAULTS.target_conditioned_weight
    constraint_weight: float = _SEARCH_DEFAULTS.constraint_weight
    portfolio_weight: float = _SEARCH_DEFAULTS.portfolio_weight
    regime_weight: float = _SEARCH_DEFAULTS.regime_weight
    transfer_weight: float = _SEARCH_DEFAULTS.transfer_weight

    # Continuation-value bonuses for promising branches and expandable parents.
    exploration_weight: float = _SEARCH_DEFAULTS.exploration_weight
    expandability_weight: float = _SEARCH_DEFAULTS.expandability_weight
    branch_value_weight: float = _SEARCH_DEFAULTS.branch_value_weight
    winner_status_bonus: float = _SEARCH_DEFAULTS.winner_status_bonus
    keep_status_bonus: float = _SEARCH_DEFAULTS.keep_status_bonus

    # Optional two-parent expansion policy.
    dual_parent_enabled: bool = _SEARCH_DEFAULTS.dual_parent_enabled
    dual_parent_max_parents: int = _SEARCH_DEFAULTS.dual_parent_max_parents
    dual_parent_delta_threshold: float = _SEARCH_DEFAULTS.dual_parent_delta_threshold
    dual_parent_similarity_threshold: float = _SEARCH_DEFAULTS.dual_parent_similarity_threshold
    dual_parent_min_expandability_advantage: float = _SEARCH_DEFAULTS.dual_parent_min_expandability_advantage

    # De-correlation scoring/gating knobs.
    decorrelation_excellent_corr: float = _DECORRELATION_DEFAULTS.excellent_corr
    decorrelation_good_corr: float = _DECORRELATION_DEFAULTS.good_corr
    decorrelation_acceptable_corr: float = _DECORRELATION_DEFAULTS.acceptable_corr
    decorrelation_weak_corr: float = _DECORRELATION_DEFAULTS.weak_corr
    decorrelation_suppress_winner_corr: float = _DECORRELATION_DEFAULTS.suppress_winner_corr
    decorrelation_soft_drop_corr: float = _DECORRELATION_DEFAULTS.soft_drop_corr
    decorrelation_hard_drop_corr: float = _DECORRELATION_DEFAULTS.hard_drop_corr
    decorrelation_excellent_bonus: float = _DECORRELATION_DEFAULTS.excellent_bonus
    decorrelation_good_bonus: float = _DECORRELATION_DEFAULTS.good_bonus
    decorrelation_acceptable_bonus: float = _DECORRELATION_DEFAULTS.acceptable_bonus
    decorrelation_weak_penalty: float = _DECORRELATION_DEFAULTS.weak_penalty
    decorrelation_failed_penalty: float = _DECORRELATION_DEFAULTS.failed_penalty
    decorrelation_avg_corr_penalty_weight: float = _DECORRELATION_DEFAULTS.avg_corr_penalty_weight
    decorrelation_quality_icir_floor: float = _DECORRELATION_DEFAULTS.quality_icir_floor
    decorrelation_quality_sharpe_floor: float = _DECORRELATION_DEFAULTS.quality_sharpe_floor
    decorrelation_quality_excess_floor: float = _DECORRELATION_DEFAULTS.quality_excess_floor
    decorrelation_quality_ann_floor: float = _DECORRELATION_DEFAULTS.quality_ann_floor
    decorrelation_strong_quality_icir: float = _DECORRELATION_DEFAULTS.strong_quality_icir
    decorrelation_strong_quality_sharpe: float = _DECORRELATION_DEFAULTS.strong_quality_sharpe

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_config_version"] = DEFAULT_POLICY_CONFIG.version
        payload["search_policy_config_version"] = DEFAULT_POLICY_CONFIG.search.version
        payload["search_policy_config_source"] = "DEFAULT_POLICY_CONFIG.search"
        return payload

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
        profile_updates = getattr(DEFAULT_POLICY_CONFIG.search.target_profiles, normalized, None)
        if profile_updates is None:
            raise ValueError(f"unknown search target profile: {target_profile}")
        return _apply_policy_operations(
            self,
            profile_updates.operations,
            initial_updates={"target_profile": normalized},
        )

    @classmethod
    def available_presets(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(DEFAULT_POLICY_CONFIG.search.presets))

    @classmethod
    def available_target_profiles(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(DEFAULT_POLICY_CONFIG.search.target_profiles))

    @classmethod
    def from_preset(cls, preset: str = "balanced") -> "SearchPolicy":
        normalized = str(preset or "balanced").strip().lower()
        preset_updates = getattr(DEFAULT_POLICY_CONFIG.search.presets, normalized, None)
        if preset_updates is None:
            raise ValueError(f"unknown search policy preset: {preset}")
        return cls(name=normalized, **dict(preset_updates.values))

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
        mode_updates = getattr(DEFAULT_POLICY_CONFIG.search.modes, normalized_mode, None)
        if mode_updates is None:
            raise ValueError(f"unknown search policy mode: {mode}")
        return _apply_policy_operations(
            base,
            mode_updates.operations,
            initial_updates={"name": f"{normalized_mode}:{base.name}"},
        )

    @classmethod
    def multi_model_best_first(cls, *, preset: str = "balanced") -> "SearchPolicy":
        return cls.for_mode("multi_model_best_first", preset=preset)

    @classmethod
    def family_breadth_first(cls, *, preset: str = "balanced") -> "SearchPolicy":
        return cls.for_mode("family_breadth_first", preset=preset)

    @classmethod
    def local_best_first(cls, *, preset: str = "balanced") -> "SearchPolicy":
        return cls.for_mode("local_best_first", preset=preset)


def _apply_policy_operations(
    policy: SearchPolicy,
    operations: tuple[tuple[str, str, Any], ...],
    *,
    initial_updates: dict[str, Any] | None = None,
) -> SearchPolicy:
    updates = dict(initial_updates or {})
    for field_name, operator, value in operations:
        current = getattr(policy, field_name)
        if operator == "set":
            updates[field_name] = value
        elif operator == "max":
            updates[field_name] = _typed_numeric_update(current, max(float(current), float(value)))
        elif operator == "min":
            updates[field_name] = _typed_numeric_update(current, min(float(current), float(value)))
        elif operator == "add":
            updates[field_name] = _typed_numeric_update(current, float(current) + float(value))
        elif operator == "add_min":
            delta, floor = value
            updates[field_name] = _typed_numeric_update(current, max(float(current) + float(delta), float(floor)))
        elif operator == "clamp":
            low, high = value
            updates[field_name] = _typed_numeric_update(current, min(max(float(current), float(low)), float(high)))
        else:
            raise ValueError(f"unknown search policy operation: {operator}")
    return replace(policy, **updates)


def _typed_numeric_update(current: Any, value: float) -> float | int:
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    return float(value)
