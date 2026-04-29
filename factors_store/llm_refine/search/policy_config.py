'''Central policy thresholds for auditable llm_refine decisions.

This module is the single place for human-tuned thresholds that should remain
visible, versioned, and reusable across signal extraction, table policy, and
decorrelation assessment. Runtime code should consume these dataclasses instead
of scattering numeric cutoffs through decision modules.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Any, TypeVar


@dataclass(frozen=True)
class StageAnchorThresholds:
    strong_passed_count: int = 2
    strong_quality_score: float = 0.70
    passed_count: int = 1
    weak_quality_score: float = 0.50


@dataclass(frozen=True)
class StageWinnerQualityThresholds:
    strong_icir: float = 0.50
    strong_sharpe: float = 3.00
    usable_icir: float = 0.40
    usable_sharpe: float = 2.00
    weak_icir: float = 0.30
    weak_sharpe: float = 1.50


@dataclass(frozen=True)
class StageMaterialGainThresholds:
    excess_gain_unit: float = 0.02
    icir_gain_unit: float = 0.05
    sharpe_gain_unit: float = 0.25


@dataclass(frozen=True)
class StageCorrelationPressureThresholds:
    critical_high_corr_count: int = 3
    high_corr_count: int = 1
    motif_usage_medium: int = 3
    motif_usage_high: int = 4
    motif_count_medium: int = 3
    motif_count_high: int = 4
    family_overlap_medium: float = 0.70
    family_overlap_high: float = 0.80
    portfolio_similarity_high: float = 0.85
    saturation_penalty_medium: float = 0.05


@dataclass(frozen=True)
class StageTurnoverPressureThresholds:
    critical_winner_turnover: float = 0.80
    high_winner_turnover: float = 0.65
    medium_winner_turnover: float = 0.50
    multiple_high_turnover_count: int = 2


@dataclass(frozen=True)
class StageFrontierHealthThresholds:
    high_children_added: int = 5
    medium_children_added: int = 2
    low_children_added: int = 1
    multiple_motif_count: int = 2
    multiple_branch_count: int = 2
    cross_model_count: int = 2


@dataclass(frozen=True)
class StageModelConsensusThresholds:
    high_model_support: int = 3
    medium_model_support: int = 2
    medium_motif_count: int = 3


@dataclass(frozen=True)
class StageTransitionPolicyConfig:
    version: str = "stage_policy_v1"
    anchor: StageAnchorThresholds = StageAnchorThresholds()
    winner_quality: StageWinnerQualityThresholds = StageWinnerQualityThresholds()
    material_gain: StageMaterialGainThresholds = StageMaterialGainThresholds()
    corr_pressure: StageCorrelationPressureThresholds = StageCorrelationPressureThresholds()
    turnover_pressure: StageTurnoverPressureThresholds = StageTurnoverPressureThresholds()
    frontier_health: StageFrontierHealthThresholds = StageFrontierHealthThresholds()
    model_consensus: StageModelConsensusThresholds = StageModelConsensusThresholds()
    validation_fail_repair_count: int = 2
    no_improve_reopen_count: int = 2


@dataclass(frozen=True)
class DecorrelationPolicyConfig:
    version: str = "decorrelation_policy_v1"
    excellent_corr: float = 0.35
    good_corr: float = 0.55
    acceptable_corr: float = 0.70
    weak_corr: float = 0.85
    suppress_winner_corr: float = 0.75
    soft_drop_corr: float = 0.85
    hard_drop_corr: float = 0.90
    excellent_bonus: float = 0.12
    good_bonus: float = 0.08
    acceptable_bonus: float = 0.03
    weak_penalty: float = 0.08
    failed_penalty: float = 0.16
    avg_corr_penalty_weight: float = 0.05
    quality_icir_floor: float = 0.15
    quality_sharpe_floor: float = 1.20
    quality_excess_floor: float = 0.05
    quality_ann_floor: float = 1.50
    strong_quality_icir: float = 0.50
    strong_quality_sharpe: float = 3.00
    complementarity_excellent_bonus: float = 0.25
    complementarity_good_bonus: float = 0.15
    complementarity_acceptable_bonus: float = 0.05
    complementarity_weak_penalty: float = 0.20
    complementarity_failed_penalty: float = 0.35
    complementarity_avg_corr_penalty_weight: float = 0.12


@dataclass(frozen=True)
class SaturationPolicyConfig:
    version: str = "saturation_policy_v1_1"
    low_score: float = 0.0
    medium_score: float = 0.10
    high_score: float = 0.25
    critical_score: float = 0.45
    corr_weight: float = 0.25
    motif_weight: float = 0.18
    turnover_weight: float = 0.25
    plateau_weight: float = 0.12
    frontier_weight: float = 0.08
    anchor_reuse_weight: float = 0.12
    plateau_no_improve_count: int = 2
    max_no_improve_count: int = 4
    motif_usage_reference: int = 4
    family_overlap_reference: float = 0.80
    high_corr_reference_count: int = 3
    frontier_medium_base: float = 0.15
    frontier_medium_single_motif_penalty: float = 0.10
    frontier_medium_single_branch_penalty: float = 0.10
    frontier_medium_single_model_penalty: float = 0.10
    frontier_medium_cap: float = 0.45


@dataclass(frozen=True)
class RoundTransitionPolicyConfig:
    version: str = "round_transition_policy_v1"
    default_authority: str = "advisory"
    max_policy_extensions: int = 2
    default_max_total_rounds: int = 4
    allowed_authorities: tuple[str, ...] = ("audit_only", "advisory", "guarded_control")
    extension_safe_saturation_grades: tuple[str, ...] = ("low", "medium")
    extension_max_turnover_pressure: str = "medium"
    extension_max_corr_pressure: str = "medium"
    extension_min_winner_quality: str = "usable"


@dataclass(frozen=True)
class SearchPolicyBaseConfig:
    version: str = "search_policy_base_v1"
    selection_strategy: str = "ucb_lite"
    metric_normalization: str = "percentile"
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
    turnover_penalty_weight: float = 0.35
    complexity_penalty_weight: float = 0.025
    depth_penalty_weight: float = 0.05
    turnover_scale: float = 0.45
    complexity_scale: float = 8.0
    depth_scale: float = 3.0
    mmr_rerank: bool = True
    mmr_lambda: float = 0.72
    branch_penalty_weight: float = 0.14
    redundancy_penalty_weight: float = 0.10
    family_motif_saturation_weight: float = 0.06
    correlation_redundancy_weight: float = 0.20
    novelty_bonus_weight: float = 0.10
    motif_novelty_weight: float = 0.08
    frontier_rerank: bool = True
    prefer_unexpanded: bool = True
    allow_keep_nodes: bool = True
    require_novel_expression: bool = True
    branch_frontier_cap: int = 2
    motif_frontier_cap: int = 3
    selection_pool_size: int = 5
    mmr_candidate_pool_size: int = 8
    similarity_branch_weight: float = 0.4
    similarity_motif_weight: float = 0.25
    similarity_mutation_weight: float = 0.15
    similarity_skeleton_weight: float = 0.2
    similarity_economic_weight: float = 0.15
    similarity_operator_weight: float = 0.2
    similarity_token_weight: float = 0.1
    target_conditioned_weight: float = 0.0
    constraint_weight: float = 0.0
    portfolio_weight: float = 0.0
    regime_weight: float = 0.0
    transfer_weight: float = 0.0
    exploration_weight: float = 0.18
    expandability_weight: float = 0.08
    branch_value_weight: float = 0.12
    winner_status_bonus: float = 0.05
    keep_status_bonus: float = 0.02
    dual_parent_enabled: bool = False
    dual_parent_max_parents: int = 2
    dual_parent_delta_threshold: float = 0.12
    dual_parent_similarity_threshold: float = 0.65
    dual_parent_min_expandability_advantage: float = 0.02


@dataclass(frozen=True)
class SearchPolicyUpdateSet:
    values: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class SearchPolicyOperationSet:
    operations: tuple[tuple[str, str, Any], ...] = ()


@dataclass(frozen=True)
class SearchPolicyPresetConfig:
    balanced: SearchPolicyUpdateSet = SearchPolicyUpdateSet()
    exploratory: SearchPolicyUpdateSet = SearchPolicyUpdateSet(
        values=(
            ("mmr_lambda", 0.60),
            ("exploration_weight", 0.24),
            ("rank_ic_weight", 1.55),
            ("rank_icir_weight", 0.55),
            ("ann_return_weight", 0.65),
            ("excess_return_weight", 0.60),
            ("sharpe_weight", 0.82),
            ("turnover_penalty_weight", 0.35),
            ("complexity_penalty_weight", 0.018),
            ("depth_penalty_weight", 0.04),
            ("branch_penalty_weight", 0.12),
            ("redundancy_penalty_weight", 0.06),
            ("family_motif_saturation_weight", 0.04),
            ("correlation_redundancy_weight", 0.04),
            ("expandability_weight", 0.10),
            ("branch_value_weight", 0.14),
            ("novelty_bonus_weight", 0.14),
            ("motif_novelty_weight", 0.10),
            ("branch_frontier_cap", 2),
            ("motif_frontier_cap", 2),
            ("selection_pool_size", 7),
            ("mmr_candidate_pool_size", 10),
        )
    )
    conservative: SearchPolicyUpdateSet = SearchPolicyUpdateSet(
        values=(
            ("mmr_lambda", 0.82),
            ("exploration_weight", 0.10),
            ("rank_ic_weight", 1.70),
            ("rank_icir_weight", 0.70),
            ("ann_return_weight", 0.75),
            ("excess_return_weight", 0.65),
            ("sharpe_weight", 0.95),
            ("turnover_penalty_weight", 0.62),
            ("complexity_penalty_weight", 0.03),
            ("depth_penalty_weight", 0.06),
            ("branch_penalty_weight", 0.12),
            ("redundancy_penalty_weight", 0.06),
            ("family_motif_saturation_weight", 0.04),
            ("correlation_redundancy_weight", 0.08),
            ("expandability_weight", 0.05),
            ("branch_value_weight", 0.08),
            ("novelty_bonus_weight", 0.04),
            ("motif_novelty_weight", 0.03),
            ("branch_frontier_cap", 2),
            ("motif_frontier_cap", 3),
            ("selection_pool_size", 4),
            ("mmr_candidate_pool_size", 6),
        )
    )


@dataclass(frozen=True)
class SearchPolicyTargetProfileConfig:
    raw_alpha: SearchPolicyOperationSet = SearchPolicyOperationSet(
        operations=(
            ("target_conditioned_weight", "set", 0.08),
            ("constraint_weight", "set", 0.55),
            ("portfolio_weight", "set", 0.45),
            ("regime_weight", "set", 0.0),
            ("transfer_weight", "set", 0.0),
            ("ann_return_weight", "max", 0.65),
            ("excess_return_weight", "max", 0.60),
            ("sharpe_weight", "min", 0.80),
            ("ann_return_scale", "min", 1.80),
            ("excess_return_scale", "min", 0.90),
            ("sharpe_scale", "max", 2.00),
        )
    )
    deployability: SearchPolicyOperationSet = SearchPolicyOperationSet(
        operations=(
            ("target_conditioned_weight", "set", 0.16),
            ("constraint_weight", "set", 0.75),
            ("portfolio_weight", "set", 0.25),
            ("regime_weight", "set", 0.0),
            ("transfer_weight", "set", 0.0),
        )
    )
    complementarity: SearchPolicyOperationSet = SearchPolicyOperationSet(
        operations=(
            ("target_conditioned_weight", "set", 0.24),
            ("constraint_weight", "set", 0.25),
            ("portfolio_weight", "set", 0.85),
            ("regime_weight", "set", 0.0),
            ("transfer_weight", "set", 0.0),
            ("redundancy_penalty_weight", "max", 0.12),
            ("family_motif_saturation_weight", "max", 0.08),
            ("correlation_redundancy_weight", "max", 0.14),
            ("dual_parent_delta_threshold", "max", 0.22),
            ("dual_parent_similarity_threshold", "min", 0.55),
            ("decorrelation_excellent_bonus", "max", 0.25),
            ("decorrelation_good_bonus", "max", 0.15),
            ("decorrelation_acceptable_bonus", "max", 0.05),
            ("decorrelation_weak_penalty", "max", 0.20),
            ("decorrelation_failed_penalty", "max", 0.35),
            ("decorrelation_avg_corr_penalty_weight", "max", 0.12),
        )
    )
    robustness: SearchPolicyOperationSet = SearchPolicyOperationSet(
        operations=(
            ("target_conditioned_weight", "set", 0.14),
            ("constraint_weight", "set", 0.55),
            ("portfolio_weight", "set", 0.20),
            ("regime_weight", "set", 0.25),
            ("transfer_weight", "set", 0.0),
        )
    )


@dataclass(frozen=True)
class SearchPolicyModeConfig:
    multi_model_best_first: SearchPolicyOperationSet = SearchPolicyOperationSet(
        operations=(
            ("branch_frontier_cap", "max", 2),
            ("motif_frontier_cap", "max", 3),
            ("selection_pool_size", "max", 5),
            ("mmr_candidate_pool_size", "max", 8),
        )
    )
    family_breadth_first: SearchPolicyOperationSet = SearchPolicyOperationSet(
        operations=(
            ("exploration_weight", "add", 0.02),
            ("novelty_bonus_weight", "add", 0.03),
            ("motif_novelty_weight", "add", 0.03),
            ("branch_penalty_weight", "max", 0.18),
            ("prefer_unexpanded", "set", True),
            ("branch_frontier_cap", "set", 1),
            ("motif_frontier_cap", "set", 2),
            ("selection_pool_size", "max", 8),
            ("mmr_candidate_pool_size", "max", 12),
        )
    )
    local_best_first: SearchPolicyOperationSet = SearchPolicyOperationSet(
        operations=(
            ("exploration_weight", "add_min", (-0.04, 0.08)),
            ("novelty_bonus_weight", "add_min", (-0.03, 0.03)),
            ("motif_novelty_weight", "add_min", (-0.02, 0.02)),
            ("branch_penalty_weight", "add_min", (-0.04, 0.08)),
            ("branch_frontier_cap", "max", 3),
            ("motif_frontier_cap", "max", 3),
            ("selection_pool_size", "clamp", (4, 5)),
            ("mmr_candidate_pool_size", "clamp", (6, 8)),
        )
    )


@dataclass(frozen=True)
class SearchPolicyConfig:
    version: str = "search_policy_config_v1"
    base: SearchPolicyBaseConfig = SearchPolicyBaseConfig()
    presets: SearchPolicyPresetConfig = SearchPolicyPresetConfig()
    target_profiles: SearchPolicyTargetProfileConfig = SearchPolicyTargetProfileConfig()
    modes: SearchPolicyModeConfig = SearchPolicyModeConfig()


@dataclass(frozen=True)
class RefinePolicyConfig:
    version: str = "refine_policy_config_v1"
    search: SearchPolicyConfig = SearchPolicyConfig()
    stage_transition: StageTransitionPolicyConfig = StageTransitionPolicyConfig()
    round_transition: RoundTransitionPolicyConfig = RoundTransitionPolicyConfig()
    decorrelation: DecorrelationPolicyConfig = DecorrelationPolicyConfig()
    saturation: SaturationPolicyConfig = SaturationPolicyConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_POLICY_CONFIG = RefinePolicyConfig()

T = TypeVar("T")


def get_default_policy_config() -> RefinePolicyConfig:
    return DEFAULT_POLICY_CONFIG


def policy_config_to_dict(config: RefinePolicyConfig | None = None) -> dict[str, Any]:
    return (config or DEFAULT_POLICY_CONFIG).to_dict()


def policy_config_from_mapping(mapping: dict[str, Any] | None) -> RefinePolicyConfig:
    if not mapping:
        return DEFAULT_POLICY_CONFIG
    return _merge_dataclass(DEFAULT_POLICY_CONFIG, mapping)


def _merge_dataclass(instance: T, mapping: dict[str, Any]) -> T:
    if not is_dataclass(instance):
        raise TypeError(f"expected dataclass instance, got {type(instance)!r}")
    updates: dict[str, Any] = {}
    field_names = {field.name for field in fields(instance)}
    for key, value in dict(mapping or {}).items():
        if key not in field_names:
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            updates[key] = _merge_dataclass(current, value)
        else:
            updates[key] = value
    return replace(instance, **updates)
