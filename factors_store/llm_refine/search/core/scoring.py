'''Search node scoring, similarity, and branch-value utilities.

Calculates base quality, frontier retrieval score, structural similarity, material gains, and expandability.
'''

from __future__ import annotations

import math
import re
import statistics
from typing import Any

from .normalization import SearchNormalizer
from .objectives import compute_target_conditioned_score
from .policy import SearchPolicy
from .state import SearchNode

_OPERATOR_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WINDOW_PATTERN = re.compile(r"\b(?:3|5|10|14|15|20|28|40|60|100|120|180|250|375)\b")
_RANK_TOKENS = {"ts_rank", "cs_rank", "rank"}
_SMOOTHING_TOKENS = {"ema", "decay_linear"}
_NORMALIZATION_TOKENS = {"ts_std", "std"}
_CONDITIONAL_TOKENS = {"where", "if_then_else", "bucket_sum"}
_RELATIVE_TOKENS = {"rel_amount", "rel_volume"}
_LIQUIDITY_FIELD_TOKENS = {"amount", "volume", "turnover", "rel_amount", "rel_volume"}
_PRICE_FIELD_TOKENS = {"open", "close", "high", "low", "vwap", "returns"}


def safe_float(value: Any, default: float = float("-inf")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def signed_tanh_normalize(value: Any, *, scale: float, default: float = 0.0) -> float:
    raw = safe_float(value, default=default)
    scale = max(float(scale), 1e-12)
    return math.tanh(raw / scale)


def positive_tanh_normalize(value: Any, *, scale: float, default: float = 0.0) -> float:
    raw = max(safe_float(value, default=default), 0.0)
    scale = max(float(scale), 1e-12)
    return math.tanh(raw / scale)


def expression_operator_names(expression: str) -> list[str]:
    text = str(expression or "").strip()
    if not text:
        return []
    return _OPERATOR_PATTERN.findall(text)


def expression_token_count(expression: str) -> int:
    text = str(expression or "").strip()
    if not text:
        return 0
    return len(_TOKEN_PATTERN.findall(text))


def expression_depth(expression: str) -> int:
    text = str(expression or "").strip()
    if not text:
        return 0
    depth = 0
    max_depth = 0
    for char in text:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(depth - 1, 0)
    return max_depth


def expression_complexity(expression: str) -> float:
    text = str(expression or "").strip()
    if not text:
        return 0.0
    operator_count = len(expression_operator_names(text))
    token_count = expression_token_count(text)
    nesting = expression_depth(text)
    return float(operator_count) + 0.08 * float(token_count) + 0.18 * float(nesting)


def expression_motif_signature(expression: str) -> str:
    operators = expression_operator_names(expression)
    if not operators:
        return "literal"
    return ">".join(operators[:3])


def expression_operator_skeleton(expression: str) -> str:
    operators = expression_operator_names(expression)
    if not operators:
        return "literal"
    normalized: list[str] = []
    for op in operators[:5]:
        low = op.lower()
        if low.startswith("ts_"):
            normalized.append(f"ts:{low[3:]}")
        elif low.startswith("cs_"):
            normalized.append(f"cs:{low[3:]}")
        else:
            normalized.append(low)
    return "|".join(normalized)


def expression_economic_family_tags(expression: str) -> tuple[str, ...]:
    tokens = expression_keyword_tokens(expression)
    tags: list[str] = []
    if {"amount", "volume", "turnover"} & tokens:
        tags.append("volume_liquidity")
    if {"returns", "open", "close", "high", "low", "vwap"} & tokens:
        tags.append("price_structure")
    if {"ema", "decay_linear"} & tokens:
        tags.append("smoothing")
    if {"ts_std", "std"} & tokens:
        tags.append("volatility_normalization")
    if {"ts_rank", "cs_rank", "rank"} & tokens:
        tags.append("ranking")
    if {"rel_amount", "rel_volume"} & tokens:
        tags.append("relative_participation")
    if {"where", "if_then_else", "bucket_sum"} & tokens:
        tags.append("conditional_state")
    return tuple(dict.fromkeys(tags)) or ("generic",)


def expression_window_tokens(expression: str) -> set[str]:
    return set(_WINDOW_PATTERN.findall(str(expression or "")))


def expression_profile_key(expression: str, parent_expression: str = "") -> str:
    mutation_class = expression_mutation_class(expression, parent_expression)
    operator_skeleton = expression_operator_skeleton(expression)
    economic_tags = expression_economic_family_tags(expression)
    return "||".join(
        [
            mutation_class or "unknown",
            operator_skeleton or "literal",
            ",".join(economic_tags) or "generic",
        ]
    )


def _classify_expression_shape(expression: str) -> str:
    text = str(expression or "").lower()
    tokens = expression_keyword_tokens(text)
    windows = len(expression_window_tokens(text))
    if _CONDITIONAL_TOKENS & tokens:
        return "conditionalization"
    if _RANK_TOKENS & tokens:
        return "rank_wrapper"
    if _SMOOTHING_TOKENS & tokens:
        return "smoothing_insertion"
    if _NORMALIZATION_TOKENS & tokens and _LIQUIDITY_FIELD_TOKENS & tokens:
        return "normalization_insertion"
    if _RELATIVE_TOKENS & tokens:
        return "relative_reweighting"
    if windows >= 3:
        return "window_rebalancing"
    return "structural_rewrite"


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return float(len(left & right)) / float(len(union))


def expression_keyword_tokens(expression: str) -> set[str]:
    tokens = {token.lower() for token in _TOKEN_PATTERN.findall(str(expression or ""))}
    return {token for token in tokens if len(token) >= 3}


def expression_mutation_class(expression: str, parent_expression: str = "") -> str:
    child = str(expression or "").strip()
    parent = str(parent_expression or "").strip()
    if not child:
        return "structural_rewrite"
    if not parent:
        return _classify_expression_shape(child)
    if child == parent:
        return "identity"

    child_tokens = expression_keyword_tokens(child)
    parent_tokens = expression_keyword_tokens(parent)
    child_ops = set(expression_operator_names(child))
    parent_ops = set(expression_operator_names(parent))
    child_windows = expression_window_tokens(child)
    parent_windows = expression_window_tokens(parent)
    child_fields = child_tokens & (_LIQUIDITY_FIELD_TOKENS | _PRICE_FIELD_TOKENS)
    parent_fields = parent_tokens & (_LIQUIDITY_FIELD_TOKENS | _PRICE_FIELD_TOKENS)

    new_tokens = child_tokens - parent_tokens
    removed_tokens = parent_tokens - child_tokens
    new_ops = child_ops - parent_ops
    removed_ops = parent_ops - child_ops

    if (_CONDITIONAL_TOKENS & (new_tokens | new_ops)) and not (_CONDITIONAL_TOKENS & parent_tokens):
        return "conditional_gate_added"
    if (_RANK_TOKENS & (new_tokens | new_ops)) and not (_RANK_TOKENS & parent_tokens):
        return "rank_wrapper_added"
    if (_SMOOTHING_TOKENS & (new_tokens | new_ops)) and not (_SMOOTHING_TOKENS & parent_tokens):
        return "smoothing_added"
    if (_NORMALIZATION_TOKENS & (new_tokens | new_ops)) and not (_NORMALIZATION_TOKENS & parent_tokens):
        return "normalization_added"
    if (_RELATIVE_TOKENS & (new_tokens | removed_tokens)) or (
        child_fields != parent_fields and (_LIQUIDITY_FIELD_TOKENS & (child_fields | parent_fields))
    ):
        return "liquidity_proxy_swap"
    if child_ops == parent_ops and child_fields == parent_fields and child_windows != parent_windows:
        return "window_tuning"
    if ("div" in child_ops and "div" in parent_ops) and (
        (_NORMALIZATION_TOKENS | _SMOOTHING_TOKENS | _RELATIVE_TOKENS) & (new_tokens | removed_tokens | new_ops | removed_ops)
    ):
        return "denominator_rewrite"
    if child_fields != parent_fields:
        return "field_proxy_swap"
    if child_ops != parent_ops and child_windows == parent_windows:
        return "operator_refactor"
    return "structural_rewrite"


def pairwise_similarity(node: SearchNode, other: SearchNode, policy: SearchPolicy) -> float:
    _ensure_expression_profile(node)
    _ensure_expression_profile(other)

    branch_similarity = 1.0 if (node.branch_key and node.branch_key == other.branch_key) else 0.0
    motif_similarity = 1.0 if (node.motif_signature and node.motif_signature == other.motif_signature) else 0.0

    operators_a = set(expression_operator_names(node.expression))
    operators_b = set(expression_operator_names(other.expression))
    operator_similarity = _token_jaccard(operators_a, operators_b)

    tokens_a = expression_keyword_tokens(node.expression)
    tokens_b = expression_keyword_tokens(other.expression)
    token_similarity = _token_jaccard(tokens_a, tokens_b)

    mutation_similarity = 1.0 if node.metadata.get("mutation_class") == other.metadata.get("mutation_class") else 0.0
    skeleton_similarity = 1.0 if node.metadata.get("operator_skeleton") == other.metadata.get("operator_skeleton") else 0.0
    economic_similarity = _token_jaccard(
        set(node.metadata.get("economic_family_tags") or ()),
        set(other.metadata.get("economic_family_tags") or ()),
    )

    similarity = (
        policy.similarity_branch_weight * branch_similarity
        + policy.similarity_motif_weight * motif_similarity
        + policy.similarity_mutation_weight * mutation_similarity
        + policy.similarity_skeleton_weight * skeleton_similarity
        + policy.similarity_economic_weight * economic_similarity
        + policy.similarity_operator_weight * operator_similarity
        + policy.similarity_token_weight * token_similarity
    )
    return max(0.0, min(float(similarity), 1.0))


def winner_signature(winner: dict[str, Any] | SearchNode | None) -> tuple[str, str]:
    if winner is None:
        return "", ""
    if isinstance(winner, SearchNode):
        return winner.factor_name.strip(), winner.expression.strip()
    return (
        str(winner.get("factor_name", "")).strip(),
        str(winner.get("expression", "")).strip(),
    )


def winner_improved(new_winner: dict[str, Any] | SearchNode | None, prev_winner: dict[str, Any] | SearchNode | None) -> bool:
    if new_winner is None:
        return False
    if prev_winner is None:
        return True

    new_candidate_id = (
        new_winner.candidate_id if isinstance(new_winner, SearchNode) else str(new_winner.get("candidate_id", ""))
    )
    prev_candidate_id = (
        prev_winner.candidate_id if isinstance(prev_winner, SearchNode) else str(prev_winner.get("candidate_id", ""))
    )
    if new_candidate_id and new_candidate_id == prev_candidate_id:
        return False
    if winner_signature(new_winner) == winner_signature(prev_winner):
        return False

    def _metric(item: dict[str, Any] | SearchNode, name: str, *, default: float = float("-inf")) -> float:
        if isinstance(item, SearchNode):
            return safe_float(getattr(item, name), default=default)
        return safe_float(item.get(name), default=default)

    if _metric(new_winner, "quick_rank_ic_mean") > _metric(prev_winner, "quick_rank_ic_mean"):
        return True
    if _metric(new_winner, "net_ann_return") > _metric(prev_winner, "net_ann_return"):
        return True
    if _metric(new_winner, "net_excess_ann_return") > _metric(prev_winner, "net_excess_ann_return"):
        return True
    if _metric(new_winner, "net_sharpe") > _metric(prev_winner, "net_sharpe"):
        return True
    if _metric(new_winner, "mean_turnover", default=float("inf")) < _metric(prev_winner, "mean_turnover", default=float("inf")) * 0.9:
        return True
    return False


def _ensure_expression_profile(node: SearchNode) -> None:
    if node.operator_count <= 0 and node.token_count <= 0 and node.expression_depth <= 0 and node.complexity <= 0.0:
        node.operator_count = len(expression_operator_names(node.expression))
        node.token_count = expression_token_count(node.expression)
        node.expression_depth = expression_depth(node.expression)
        node.complexity = expression_complexity(node.expression)
    node.motif_signature = node.motif_signature or expression_motif_signature(node.expression)
    metadata = dict(node.metadata or {})
    parent_expression = str(metadata.get("parent_expression", "") or "")
    metadata["mutation_class"] = metadata.get("mutation_class") or expression_mutation_class(node.expression, parent_expression)
    metadata["operator_skeleton"] = metadata.get("operator_skeleton") or expression_operator_skeleton(node.expression)
    metadata["economic_family_tags"] = tuple(metadata.get("economic_family_tags") or expression_economic_family_tags(node.expression))
    metadata["profile_key"] = metadata.get("profile_key") or expression_profile_key(node.expression, parent_expression)
    node.metadata = metadata


def _status_bonus(node: SearchNode, policy: SearchPolicy) -> float:
    status_bonus = 0.0
    status = str(node.status or "").lower()
    if "winner" in status:
        status_bonus += policy.winner_status_bonus
    elif "keep" in status:
        status_bonus += policy.keep_status_bonus
    return status_bonus


def _performance_score(node: SearchNode, policy: SearchPolicy) -> tuple[float, dict[str, float]]:
    raw_rank_ic = safe_float(node.quick_rank_ic_mean, default=0.0)
    raw_rank_icir = safe_float(node.quick_rank_icir, default=0.0)
    raw_net_ann = safe_float(node.net_ann_return, default=0.0)
    raw_net_excess = safe_float(node.net_excess_ann_return, default=0.0)
    raw_sharpe = safe_float(node.net_sharpe, default=0.0)

    rank_ic = signed_tanh_normalize(raw_rank_ic, scale=policy.rank_ic_scale)
    rank_icir = signed_tanh_normalize(raw_rank_icir, scale=policy.rank_icir_scale)
    net_ann = signed_tanh_normalize(raw_net_ann, scale=policy.ann_return_scale)
    net_excess = signed_tanh_normalize(raw_net_excess, scale=policy.excess_return_scale)
    sharpe = signed_tanh_normalize(raw_sharpe, scale=policy.sharpe_scale)

    parts = {
        "rank_ic_component": policy.rank_ic_weight * rank_ic,
        "rank_icir_component": policy.rank_icir_weight * rank_icir,
        "ann_return_component": policy.ann_return_weight * net_ann,
        "excess_return_component": policy.excess_return_weight * net_excess,
        "sharpe_component": policy.sharpe_weight * sharpe,
        "raw_rank_ic_mean": raw_rank_ic,
        "raw_rank_icir": raw_rank_icir,
        "raw_net_ann_return": raw_net_ann,
        "raw_net_excess_ann_return": raw_net_excess,
        "raw_net_sharpe": raw_sharpe,
        "normalized_rank_ic_mean": rank_ic,
        "normalized_rank_icir": rank_icir,
        "normalized_net_ann_return": net_ann,
        "normalized_net_excess_ann_return": net_excess,
        "normalized_net_sharpe": sharpe,
    }
    performance_score = (
        parts["rank_ic_component"]
        + parts["rank_icir_component"]
        + parts["ann_return_component"]
        + parts["excess_return_component"]
        + parts["sharpe_component"]
    )
    return performance_score, parts


def _normalize_signed_metric(
    *,
    metric_name: str,
    raw_value: float,
    scale: float,
    policy: SearchPolicy,
    normalizer: SearchNormalizer | None,
) -> float:
    if policy.metric_normalization == "percentile" and normalizer is not None:
        normalized = normalizer.normalize_signed(metric_name, raw_value)
        if normalized is not None:
            return normalized
    return signed_tanh_normalize(raw_value, scale=scale)


def _normalize_positive_metric(
    *,
    metric_name: str,
    raw_value: float,
    scale: float,
    policy: SearchPolicy,
    normalizer: SearchNormalizer | None,
) -> float:
    if policy.metric_normalization == "percentile" and normalizer is not None:
        normalized = normalizer.normalize_positive(metric_name, raw_value)
        if normalized is not None:
            return normalized
    return positive_tanh_normalize(raw_value, scale=scale)


def compute_base_score(node: SearchNode, policy: SearchPolicy, normalizer: SearchNormalizer | None = None) -> float:
    _ensure_expression_profile(node)
    raw_rank_ic = safe_float(node.quick_rank_ic_mean, default=0.0)
    raw_rank_icir = safe_float(node.quick_rank_icir, default=0.0)
    raw_net_ann = safe_float(node.net_ann_return, default=0.0)
    raw_net_excess = safe_float(node.net_excess_ann_return, default=0.0)
    raw_sharpe = safe_float(node.net_sharpe, default=0.0)

    normalized_rank_ic = _normalize_signed_metric(
        metric_name="rank_ic_mean",
        raw_value=raw_rank_ic,
        scale=policy.rank_ic_scale,
        policy=policy,
        normalizer=normalizer,
    )
    normalized_rank_icir = _normalize_signed_metric(
        metric_name="rank_icir",
        raw_value=raw_rank_icir,
        scale=policy.rank_icir_scale,
        policy=policy,
        normalizer=normalizer,
    )
    normalized_net_ann = _normalize_signed_metric(
        metric_name="net_ann_return",
        raw_value=raw_net_ann,
        scale=policy.ann_return_scale,
        policy=policy,
        normalizer=normalizer,
    )
    normalized_net_excess = _normalize_signed_metric(
        metric_name="net_excess_ann_return",
        raw_value=raw_net_excess,
        scale=policy.excess_return_scale,
        policy=policy,
        normalizer=normalizer,
    )
    normalized_sharpe = _normalize_signed_metric(
        metric_name="net_sharpe",
        raw_value=raw_sharpe,
        scale=policy.sharpe_scale,
        policy=policy,
        normalizer=normalizer,
    )

    performance_parts = {
        "rank_ic_component": policy.rank_ic_weight * normalized_rank_ic,
        "rank_icir_component": policy.rank_icir_weight * normalized_rank_icir,
        "ann_return_component": policy.ann_return_weight * normalized_net_ann,
        "excess_return_component": policy.excess_return_weight * normalized_net_excess,
        "sharpe_component": policy.sharpe_weight * normalized_sharpe,
        "raw_rank_ic_mean": raw_rank_ic,
        "raw_rank_icir": raw_rank_icir,
        "raw_net_ann_return": raw_net_ann,
        "raw_net_excess_ann_return": raw_net_excess,
        "raw_net_sharpe": raw_sharpe,
        "normalized_rank_ic_mean": normalized_rank_ic,
        "normalized_rank_icir": normalized_rank_icir,
        "normalized_net_ann_return": normalized_net_ann,
        "normalized_net_excess_ann_return": normalized_net_excess,
        "normalized_net_sharpe": normalized_sharpe,
        "metric_normalization": policy.metric_normalization,
    }
    performance_score = (
        performance_parts["rank_ic_component"]
        + performance_parts["rank_icir_component"]
        + performance_parts["ann_return_component"]
        + performance_parts["excess_return_component"]
        + performance_parts["sharpe_component"]
    )

    raw_turnover = max(safe_float(node.mean_turnover, default=0.0), 0.0)
    normalized_turnover = _normalize_positive_metric(
        metric_name="mean_turnover",
        raw_value=raw_turnover,
        scale=policy.turnover_scale,
        policy=policy,
        normalizer=normalizer,
    )
    turnover_penalty = policy.turnover_penalty_weight * normalized_turnover

    raw_complexity = max(node.complexity, 0.0)
    normalized_complexity = _normalize_positive_metric(
        metric_name="complexity",
        raw_value=raw_complexity,
        scale=policy.complexity_scale,
        policy=policy,
        normalizer=normalizer,
    )
    complexity_penalty = policy.complexity_penalty_weight * normalized_complexity

    depth_proxy = max(float(node.depth), float(max(node.expression_depth - 1, 0)))
    normalized_depth = _normalize_positive_metric(
        metric_name="expression_depth",
        raw_value=depth_proxy,
        scale=policy.depth_scale,
        policy=policy,
        normalizer=normalizer,
    )
    depth_penalty = policy.depth_penalty_weight * normalized_depth
    status_bonus = _status_bonus(node, policy)
    quality_score = performance_score + status_bonus - turnover_penalty - depth_penalty

    node.performance_score = performance_score
    node.quality_score = quality_score
    node.turnover_penalty_score = turnover_penalty
    node.complexity_penalty_score = complexity_penalty
    node.depth_penalty_score = depth_penalty
    node.status_bonus_score = status_bonus
    node.base_score = quality_score - complexity_penalty
    node.score_breakdown = {
        **performance_parts,
        "raw_mean_turnover": raw_turnover,
        "normalized_mean_turnover": normalized_turnover,
        "raw_complexity": raw_complexity,
        "normalized_complexity": normalized_complexity,
        "depth_proxy": depth_proxy,
        "normalized_depth": normalized_depth,
        "performance_score": performance_score,
        "quality_score": quality_score,
        "status_bonus": status_bonus,
        "turnover_penalty": turnover_penalty,
        "complexity_penalty": complexity_penalty,
        "depth_penalty": depth_penalty,
        "mutation_class": node.metadata.get("mutation_class"),
        "operator_skeleton": node.metadata.get("operator_skeleton"),
        "economic_family_tags": list(node.metadata.get("economic_family_tags") or ()),
        "base_score": node.base_score,
    }
    return node.base_score


def compute_frontier_score(
    node: SearchNode,
    *,
    policy: SearchPolicy,
    total_visits: int,
    seen_expression_count: int,
    branch_usage_count: int = 0,
    motif_usage_count: int = 0,
    branch_budget: int = 1,
    normalizer: SearchNormalizer | None = None,
    reference_nodes: list[SearchNode] | None = None,
) -> float:
    base = compute_base_score(node, policy, normalizer=normalizer)
    exploration = 0.0
    if policy.selection_strategy == "ucb_lite":
        exploration = policy.exploration_weight * math.sqrt(
            math.log(float(total_visits) + 2.0) / float(node.visits + 1)
        )

    novelty = 0.0
    underexplored_bonus = 0.0
    if node.expansions == 0 and policy.prefer_unexpanded:
        underexplored_bonus += policy.novelty_bonus_weight
    if motif_usage_count >= 0:
        novelty += policy.motif_novelty_weight / float(motif_usage_count + 1)
    if seen_expression_count >= 0 and node.visits == 0:
        novelty += 0.5 * policy.novelty_bonus_weight / math.sqrt(float(seen_expression_count) + 1.0)

    branch_pressure = float(branch_usage_count) / float(max(branch_budget, 1))
    branch_penalty = policy.branch_penalty_weight * max(branch_pressure - 0.5, 0.0)
    motif_redundancy = policy.redundancy_penalty_weight * max(float(motif_usage_count) - 1.0, 0.0) / float(motif_usage_count + 1.0)
    profile_key = str(node.metadata.get("profile_key", "") or "")
    historical_profile_count = 0
    corr_risk = 0.0
    if normalizer is not None:
        historical_profile_count = int(normalizer.motif_profile_counts.get(profile_key, 0))
        corr_risk = float(normalizer.corr_risk_by_profile.get(profile_key, 0.0))
    family_motif_saturation_penalty = (
        policy.family_motif_saturation_weight
        * max(float(historical_profile_count) - 1.0, 0.0)
        / float(historical_profile_count + 1.0)
        if historical_profile_count > 0
        else 0.0
    )
    corr_redundancy_penalty = policy.correlation_redundancy_weight * corr_risk
    redundancy_penalty = branch_penalty + motif_redundancy + family_motif_saturation_penalty + corr_redundancy_penalty

    quality_score = node.quality_score
    complexity_penalty = node.complexity_penalty_score
    underexplored_bonus += exploration
    effective_expandability = max(float(node.effective_expandability_score), 0.0)
    expandability_bonus = policy.expandability_weight * effective_expandability
    branch_value = max(float(getattr(node, "branch_value_score", 0.0) or 0.0), 0.0)
    branch_value_bonus = policy.branch_value_weight * branch_value
    target_conditioned_bonus, target_breakdown = compute_target_conditioned_score(
        node,
        policy=policy,
        reference_nodes=reference_nodes,
        normalizer=normalizer,
    )
    completeness = max(min(float(getattr(node, "metrics_completeness", 1.0) or 0.0), 1.0), 0.0)
    completeness_penalty = 0.0
    if completeness < 0.75:
        completeness_penalty = 0.6 * (0.75 - completeness) / 0.75
    retrieval_score = (
        quality_score
        + novelty
        + underexplored_bonus
        + expandability_bonus
        + branch_value_bonus
        + target_conditioned_bonus
        - redundancy_penalty
        - complexity_penalty
        - completeness_penalty
    )

    node.exploration_score = exploration
    node.novelty_score = novelty
    node.underexplored_bonus_score = underexplored_bonus
    node.branch_penalty_score = branch_penalty
    node.redundancy_penalty_score = redundancy_penalty
    node.retrieval_score = retrieval_score
    node.constraint_score = float(target_breakdown.get("constraint_score", 0.0) or 0.0)
    node.portfolio_score = float(target_breakdown.get("portfolio_score", 0.0) or 0.0)
    node.regime_score = float(target_breakdown.get("regime_score", 0.0) or 0.0)
    node.transfer_score = float(target_breakdown.get("transfer_score", 0.0) or 0.0)
    node.target_conditioned_score = float(target_conditioned_bonus)
    node.frontier_score = retrieval_score
    node.score_breakdown = {
        **node.score_breakdown,
        "branch_usage_count": int(branch_usage_count),
        "motif_usage_count": int(motif_usage_count),
        "branch_budget": int(branch_budget),
        "branch_pressure": branch_pressure,
        "exploration_score": exploration,
        "quality_score": quality_score,
        "novelty_score": novelty,
        "underexplored_bonus": underexplored_bonus,
        "branch_penalty": branch_penalty,
        "motif_redundancy_penalty": motif_redundancy,
        "family_motif_saturation_penalty": family_motif_saturation_penalty,
        "historical_profile_count": historical_profile_count,
        "corr_risk": corr_risk,
        "corr_redundancy_penalty": corr_redundancy_penalty,
        "redundancy_penalty": redundancy_penalty,
        "complexity_penalty": complexity_penalty,
        "expandability_weight": float(policy.expandability_weight),
        "expandability_score": float(node.expandability_score),
        "expandability_confidence": float(node.expandability_confidence),
        "effective_expandability_score": effective_expandability,
        "expandability_bonus": expandability_bonus,
        "branch_value_weight": float(policy.branch_value_weight),
        "branch_value_score": branch_value,
        "branch_value_bonus": branch_value_bonus,
        **target_breakdown,
        "metrics_completeness": completeness,
        "missing_core_metrics_count": int(getattr(node, "missing_core_metrics_count", 0) or 0),
        "eligible_for_best_node": bool(getattr(node, "eligible_for_best_node", True)),
        "completeness_penalty": completeness_penalty,
        "retrieval_score": retrieval_score,
        "frontier_score": node.frontier_score,
    }
    return node.frontier_score


def compute_parent_child_gain(parent: SearchNode, child: SearchNode) -> float:
    delta_sharpe = safe_float(child.net_sharpe, default=0.0) - safe_float(parent.net_sharpe, default=0.0)
    delta_excess = safe_float(child.net_excess_ann_return, default=0.0) - safe_float(parent.net_excess_ann_return, default=0.0)
    delta_ann = safe_float(child.net_ann_return, default=0.0) - safe_float(parent.net_ann_return, default=0.0)
    delta_icir = safe_float(child.quick_rank_icir, default=0.0) - safe_float(parent.quick_rank_icir, default=0.0)
    delta_ic = safe_float(child.quick_rank_ic_mean, default=0.0) - safe_float(parent.quick_rank_ic_mean, default=0.0)
    turnover_rise = max(
        safe_float(child.mean_turnover, default=0.0) - safe_float(parent.mean_turnover, default=0.0),
        0.0,
    )
    gain = (
        0.35 * signed_tanh_normalize(delta_sharpe, scale=1.5)
        + 0.25 * signed_tanh_normalize(delta_excess, scale=0.5)
        + 0.20 * signed_tanh_normalize(delta_ann, scale=1.0)
        + 0.10 * signed_tanh_normalize(delta_icir, scale=0.3)
        + 0.10 * signed_tanh_normalize(delta_ic, scale=0.05)
        - 0.15 * positive_tanh_normalize(turnover_rise, scale=0.15)
    )
    return max(min(float(gain), 1.0), -1.0)


def compute_expandability_score(node: SearchNode) -> tuple[float, float, dict[str, float]]:
    total = max(int(node.children_total), 0)
    kept = max(int(node.children_kept), 0)
    winners = max(int(node.children_winners), 0)
    model_support = max(int(node.child_model_support), 0)
    mutation_diversity = max(int(node.child_mutation_diversity), 0)

    success_rate_score = float(kept + 1) / float(total + 3)
    winner_rate_score = float(winners + 1) / float(total + 4)
    best_gain_score = max(min((float(node.child_best_gain) + 1.0) / 2.0, 1.0), 0.0)
    model_stability_score = min(float(model_support) / 3.0, 1.0)
    child_diversity_score = min(float(mutation_diversity) / 4.0, 1.0)

    expandability_score = (
        0.35 * success_rate_score
        + 0.25 * winner_rate_score
        + 0.20 * best_gain_score
        + 0.10 * model_stability_score
        + 0.10 * child_diversity_score
    )
    confidence = min(float(total) / 8.0, 1.0)
    effective_expandability = confidence * expandability_score
    breakdown = {
        "success_rate_score": success_rate_score,
        "winner_rate_score": winner_rate_score,
        "best_gain_score": best_gain_score,
        "model_stability_score": model_stability_score,
        "child_diversity_score": child_diversity_score,
        "expandability_score": expandability_score,
        "expandability_confidence": confidence,
        "effective_expandability_score": effective_expandability,
    }
    return expandability_score, confidence, breakdown


def compute_branch_value_score(node: SearchNode) -> tuple[float, float, dict[str, float]]:
    total = max(int(node.children_total), 0)
    if total <= 0:
        breakdown = {
            "desc_quality_score": 0.0,
            "desc_stability_score": 0.0,
            "desc_admission_score": 0.0,
            "desc_novelty_score": 0.0,
            "recency_adjustment": 0.0,
            "branch_value_confidence": 0.0,
            "branch_value_score": 0.0,
        }
        return 0.0, 0.0, breakdown

    top1_gain_score = max(min((float(node.child_best_gain) + 1.0) / 2.0, 1.0), 0.0)
    top3_gain_score = max(min((float(node.child_top3_mean_gain) + 1.0) / 2.0, 1.0), 0.0)
    median_gain_score = max(min((float(node.child_median_gain) + 1.0) / 2.0, 1.0), 0.0)
    positive_gain_rate = max(min(float(node.child_positive_gain_rate), 1.0), 0.0)
    desc_quality_score = (
        0.35 * top1_gain_score
        + 0.30 * top3_gain_score
        + 0.20 * median_gain_score
        + 0.15 * positive_gain_rate
    )

    gain_std = max(float(node.child_gain_std), 0.0)
    gain_stability_score = max(0.0, 1.0 - min(gain_std / 0.6, 1.0))
    model_support_rate = max(min(float(node.child_model_support_rate), 1.0), 0.0)
    cross_model_convergence = max(min(float(node.child_cross_model_convergence), 1.0), 0.0)
    desc_stability_score = (
        0.35 * model_support_rate
        + 0.35 * cross_model_convergence
        + 0.30 * gain_stability_score
    )

    positive_excess_rate = max(min(float(node.child_positive_excess_rate), 1.0), 0.0)
    low_turnover_rate = max(min(float(node.child_low_turnover_rate), 1.0), 0.0)
    full_metrics_rate = max(min(float(node.child_full_metrics_rate), 1.0), 0.0)
    admission_friendly_rate = max(min(float(node.child_admission_friendly_rate), 1.0), 0.0)
    desc_admission_score = (
        0.30 * positive_excess_rate
        + 0.20 * low_turnover_rate
        + 0.20 * full_metrics_rate
        + 0.30 * admission_friendly_rate
    )

    high_quality_novel_score = min(float(max(int(node.child_high_quality_novel_count), 0)) / 3.0, 1.0)
    new_motif_success_rate = max(min(float(node.child_new_motif_success_rate), 1.0), 0.0)
    child_diversity_score = min(float(max(int(node.child_mutation_diversity), 0)) / 4.0, 1.0)
    desc_novelty_score = (
        0.35 * high_quality_novel_score
        + 0.35 * new_motif_success_rate
        + 0.30 * child_diversity_score
    )

    rounds_since_last_success = max(int(getattr(node, "rounds_since_last_success", 0) or 0), 0)
    recency_adjustment = max(0.4, 1.0 - 0.12 * float(rounds_since_last_success))
    confidence = min(float(total) / 10.0, 1.0)
    raw_branch_value = (
        0.35 * desc_quality_score
        + 0.25 * desc_stability_score
        + 0.25 * desc_admission_score
        + 0.15 * desc_novelty_score
    )
    branch_value_score = confidence * recency_adjustment * raw_branch_value
    breakdown = {
        "top1_gain_score": top1_gain_score,
        "top3_gain_score": top3_gain_score,
        "median_gain_score": median_gain_score,
        "positive_gain_rate": positive_gain_rate,
        "desc_quality_score": desc_quality_score,
        "model_support_rate": model_support_rate,
        "cross_model_convergence": cross_model_convergence,
        "gain_stability_score": gain_stability_score,
        "desc_stability_score": desc_stability_score,
        "positive_excess_rate": positive_excess_rate,
        "low_turnover_rate": low_turnover_rate,
        "full_metrics_rate": full_metrics_rate,
        "admission_friendly_rate": admission_friendly_rate,
        "desc_admission_score": desc_admission_score,
        "high_quality_novel_score": high_quality_novel_score,
        "new_motif_success_rate": new_motif_success_rate,
        "child_diversity_score": child_diversity_score,
        "desc_novelty_score": desc_novelty_score,
        "recency_adjustment": recency_adjustment,
        "branch_value_confidence": confidence,
        "branch_value_score": branch_value_score,
    }
    return branch_value_score, confidence, breakdown
