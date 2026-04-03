from __future__ import annotations

import math
import re
from typing import Any

from .normalization import SearchNormalizer
from .policy import SearchPolicy
from .state import SearchNode

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except Exception:
        return default
    if out != out:
        return default
    return out


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(float(value), upper))


def _positive_tanh(value: float, scale: float) -> float:
    return math.tanh(max(float(value), 0.0) / max(float(scale), 1e-12))


def _keyword_tokens(expression: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(str(expression or "")) if len(token) >= 3}


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def _profile_saturation_risk(*, profile_key: str, normalizer: SearchNormalizer | None) -> tuple[float, int, float]:
    if normalizer is None or not profile_key:
        return 0.0, 0, 0.0
    profile_count = int(normalizer.motif_profile_counts.get(profile_key, 0) or 0)
    corr_risk = float(normalizer.corr_risk_by_profile.get(profile_key, 0.0) or 0.0)
    saturation_risk = _clamp(max(profile_count - 1, 0) / 5.0)
    combined = max(saturation_risk, _clamp(corr_risk))
    return combined, profile_count, _clamp(corr_risk)


def compute_constraint_score(
    node: SearchNode,
    *,
    policy: SearchPolicy,
    normalizer: SearchNormalizer | None = None,
) -> tuple[float, dict[str, float]]:
    profile_key = str(node.metadata.get("profile_key", "") or "")
    redundancy_risk, profile_count, corr_risk = _profile_saturation_risk(
        profile_key=profile_key,
        normalizer=normalizer,
    )

    turnover_score = 1.0 - _positive_tanh(_safe_float(node.mean_turnover, 0.0), max(policy.turnover_scale, 0.12))
    complexity_score = 1.0 - _positive_tanh(max(float(node.complexity), 0.0), max(policy.complexity_scale, 4.0))
    completeness = _clamp(_safe_float(node.metrics_completeness, 0.0))
    eligible_score = 1.0 if bool(getattr(node, "eligible_for_best_node", True)) else 0.0
    reliability_score = _clamp(0.75 * completeness + 0.25 * eligible_score)
    missing_core = max(int(getattr(node, "missing_core_metrics_count", 0) or 0), 0)
    missing_penalty = _clamp(missing_core / 4.0)
    redundancy_score = 1.0 - redundancy_risk

    score = _clamp(
        0.30 * turnover_score
        + 0.20 * complexity_score
        + 0.30 * reliability_score
        + 0.20 * redundancy_score
        - 0.10 * missing_penalty
    )
    breakdown = {
        "constraint_turnover_score": turnover_score,
        "constraint_complexity_score": complexity_score,
        "constraint_reliability_score": reliability_score,
        "constraint_redundancy_score": redundancy_score,
        "constraint_missing_penalty": missing_penalty,
        "constraint_profile_count": float(profile_count),
        "constraint_corr_risk": corr_risk,
        "constraint_score": score,
    }
    return score, breakdown


def _reference_similarity(node: SearchNode, other: SearchNode) -> float:
    if node.node_id == other.node_id:
        return 1.0
    profile_similarity = 1.0 if str(node.metadata.get("profile_key", "")) == str(other.metadata.get("profile_key", "")) else 0.0
    skeleton_similarity = (
        1.0
        if str(node.metadata.get("operator_skeleton", "")) == str(other.metadata.get("operator_skeleton", ""))
        else 0.0
    )
    motif_similarity = 1.0 if str(node.motif_signature or "") == str(other.motif_signature or "") else 0.0
    mutation_similarity = (
        1.0
        if str(node.metadata.get("mutation_class", "")) == str(other.metadata.get("mutation_class", ""))
        else 0.0
    )
    economic_similarity = _token_jaccard(
        set(node.metadata.get("economic_family_tags") or ()),
        set(other.metadata.get("economic_family_tags") or ()),
    )
    token_similarity = _token_jaccard(_keyword_tokens(node.expression), _keyword_tokens(other.expression))
    similarity = (
        0.25 * profile_similarity
        + 0.20 * skeleton_similarity
        + 0.15 * motif_similarity
        + 0.10 * mutation_similarity
        + 0.15 * economic_similarity
        + 0.15 * token_similarity
    )
    return _clamp(similarity)


def compute_portfolio_score(
    node: SearchNode,
    *,
    policy: SearchPolicy,
    reference_nodes: list[SearchNode] | None = None,
    normalizer: SearchNormalizer | None = None,
) -> tuple[float, dict[str, float]]:
    refs = [item for item in (reference_nodes or []) if item.node_id != node.node_id]
    profile_key = str(node.metadata.get("profile_key", "") or "")
    profile_novelty = 1.0
    profile_count = 0
    corr_risk = 0.0
    if normalizer is not None and profile_key:
        profile_count = int(normalizer.motif_profile_counts.get(profile_key, 0) or 0)
        corr_risk = _clamp(float(normalizer.corr_risk_by_profile.get(profile_key, 0.0) or 0.0))
        profile_novelty = 1.0 - _clamp(profile_count / 6.0)
    complementarity_mode = str(policy.target_profile or "").strip().lower() == "complementarity"
    if not refs:
        if complementarity_mode:
            score = _clamp(0.70 + 0.30 * profile_novelty - 0.10 * corr_risk)
        else:
            score = _clamp(0.80 + 0.20 * profile_novelty)
        return score, {
            "portfolio_max_similarity": 0.0,
            "portfolio_avg_similarity": 0.0,
            "portfolio_orthogonality_score": 1.0,
            "portfolio_structural_complement_score": 1.0,
            "portfolio_profile_novelty_score": profile_novelty,
            "portfolio_profile_count": float(profile_count),
            "portfolio_corr_risk": corr_risk,
            "portfolio_similarity_guard_penalty": 0.0,
            "portfolio_saturation_penalty": _clamp(0.60 * corr_risk),
            "portfolio_score": score,
        }

    similarities = [_reference_similarity(node, ref) for ref in refs]
    max_similarity = max(similarities, default=0.0)
    avg_similarity = sum(similarities) / float(len(similarities))
    orthogonality_score = 1.0 - max_similarity
    diversity_support = 1.0 - avg_similarity

    best_ref = refs[similarities.index(max_similarity)]
    skeleton_diff = 0.0 if str(node.metadata.get("operator_skeleton", "")) == str(best_ref.metadata.get("operator_skeleton", "")) else 1.0
    motif_diff = 0.0 if str(node.motif_signature or "") == str(best_ref.motif_signature or "") else 1.0
    mutation_diff = 0.0 if str(node.metadata.get("mutation_class", "")) == str(best_ref.metadata.get("mutation_class", "")) else 1.0
    economic_overlap = _token_jaccard(
        set(node.metadata.get("economic_family_tags") or ()),
        set(best_ref.metadata.get("economic_family_tags") or ()),
    )
    structural_complement_score = _clamp(
        0.35 * skeleton_diff
        + 0.20 * motif_diff
        + 0.20 * mutation_diff
        + 0.25 * (1.0 - economic_overlap)
    )

    similarity_guard_penalty = 0.0
    saturation_penalty = 0.0
    if complementarity_mode:
        similarity_guard_penalty = _clamp((max_similarity - 0.35) / 0.45)
        saturation_penalty = _clamp(0.60 * corr_risk + 0.40 * max(profile_count - 1, 0) / 6.0)
        score = _clamp(
            0.50 * orthogonality_score
            + 0.15 * diversity_support
            + 0.25 * structural_complement_score
            + 0.10 * profile_novelty
            - 0.20 * similarity_guard_penalty
            - 0.10 * saturation_penalty
        )
    else:
        score = _clamp(
            0.45 * orthogonality_score
            + 0.25 * diversity_support
            + 0.20 * structural_complement_score
            + 0.10 * profile_novelty
        )
    return score, {
        "portfolio_max_similarity": max_similarity,
        "portfolio_avg_similarity": avg_similarity,
        "portfolio_orthogonality_score": orthogonality_score,
        "portfolio_diversity_support": diversity_support,
        "portfolio_structural_complement_score": structural_complement_score,
        "portfolio_profile_novelty_score": profile_novelty,
        "portfolio_profile_count": float(profile_count),
        "portfolio_corr_risk": corr_risk,
        "portfolio_similarity_guard_penalty": similarity_guard_penalty,
        "portfolio_saturation_penalty": saturation_penalty,
        "portfolio_score": score,
    }


def compute_target_conditioned_score(
    node: SearchNode,
    *,
    policy: SearchPolicy,
    reference_nodes: list[SearchNode] | None = None,
    normalizer: SearchNormalizer | None = None,
) -> tuple[float, dict[str, float]]:
    constraint_score, constraint_breakdown = compute_constraint_score(
        node,
        policy=policy,
        normalizer=normalizer,
    )
    portfolio_score, portfolio_breakdown = compute_portfolio_score(
        node,
        policy=policy,
        reference_nodes=reference_nodes,
        normalizer=normalizer,
    )
    regime_score = 0.0
    transfer_score = 0.0
    target_bonus = float(policy.target_conditioned_weight) * (
        float(policy.constraint_weight) * constraint_score
        + float(policy.portfolio_weight) * portfolio_score
        + float(policy.regime_weight) * regime_score
        + float(policy.transfer_weight) * transfer_score
    )
    breakdown = {
        "target_profile": str(policy.target_profile),
        "target_conditioned_weight": float(policy.target_conditioned_weight),
        "constraint_weight": float(policy.constraint_weight),
        "portfolio_weight": float(policy.portfolio_weight),
        "regime_weight": float(policy.regime_weight),
        "transfer_weight": float(policy.transfer_weight),
        "constraint_score": constraint_score,
        "portfolio_score": portfolio_score,
        "regime_score": regime_score,
        "transfer_score": transfer_score,
        "target_conditioned_score": target_bonus,
        **constraint_breakdown,
        **portfolio_breakdown,
    }
    return target_bonus, breakdown
