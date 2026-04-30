'''Unified decorrelation scoring, gates, and rerank adjustments.'''

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from ..policy_config import DEFAULT_POLICY_CONFIG


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


@dataclass(frozen=True)
class DecorrelationPolicy:
    target_profile: str = "raw_alpha"
    strong_gate_enabled: bool = False
    excellent_corr_threshold: float = 0.35
    good_corr_threshold: float = 0.55
    acceptable_corr_threshold: float = 0.70
    weak_corr_threshold: float = 0.85
    suppress_winner_corr_threshold: float = 0.75
    soft_drop_corr_threshold: float = 0.85
    hard_drop_corr_threshold: float = 0.90
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

    @classmethod
    def from_search_policy(
        cls,
        policy: Any | None,
        *,
        target_profile: str = "",
        decorrelation_targets_present: bool = False,
    ) -> "DecorrelationPolicy":
        defaults = DEFAULT_POLICY_CONFIG.decorrelation
        profile = str(target_profile or getattr(policy, "target_profile", "raw_alpha") or "raw_alpha").strip().lower()
        strong_gate_enabled = bool(decorrelation_targets_present or profile == "complementarity")
        excellent_bonus = float(getattr(policy, "decorrelation_excellent_bonus", defaults.excellent_bonus))
        good_bonus = float(getattr(policy, "decorrelation_good_bonus", defaults.good_bonus))
        acceptable_bonus = float(getattr(policy, "decorrelation_acceptable_bonus", defaults.acceptable_bonus))
        weak_penalty = float(getattr(policy, "decorrelation_weak_penalty", defaults.weak_penalty))
        failed_penalty = float(getattr(policy, "decorrelation_failed_penalty", defaults.failed_penalty))
        avg_corr_penalty_weight = float(
            getattr(policy, "decorrelation_avg_corr_penalty_weight", defaults.avg_corr_penalty_weight)
        )
        if profile == "complementarity":
            excellent_bonus = max(excellent_bonus, defaults.complementarity_excellent_bonus)
            good_bonus = max(good_bonus, defaults.complementarity_good_bonus)
            acceptable_bonus = max(acceptable_bonus, defaults.complementarity_acceptable_bonus)
            weak_penalty = max(weak_penalty, defaults.complementarity_weak_penalty)
            failed_penalty = max(failed_penalty, defaults.complementarity_failed_penalty)
            avg_corr_penalty_weight = max(avg_corr_penalty_weight, defaults.complementarity_avg_corr_penalty_weight)
        return cls(
            target_profile=profile,
            strong_gate_enabled=strong_gate_enabled,
            excellent_corr_threshold=float(getattr(policy, "decorrelation_excellent_corr", defaults.excellent_corr)),
            good_corr_threshold=float(getattr(policy, "decorrelation_good_corr", defaults.good_corr)),
            acceptable_corr_threshold=float(getattr(policy, "decorrelation_acceptable_corr", defaults.acceptable_corr)),
            weak_corr_threshold=float(getattr(policy, "decorrelation_weak_corr", defaults.weak_corr)),
            suppress_winner_corr_threshold=float(
                getattr(policy, "decorrelation_suppress_winner_corr", defaults.suppress_winner_corr)
            ),
            soft_drop_corr_threshold=float(getattr(policy, "decorrelation_soft_drop_corr", defaults.soft_drop_corr)),
            hard_drop_corr_threshold=float(getattr(policy, "decorrelation_hard_drop_corr", defaults.hard_drop_corr)),
            excellent_bonus=excellent_bonus,
            good_bonus=good_bonus,
            acceptable_bonus=acceptable_bonus,
            weak_penalty=weak_penalty,
            failed_penalty=failed_penalty,
            avg_corr_penalty_weight=avg_corr_penalty_weight,
            quality_icir_floor=float(getattr(policy, "decorrelation_quality_icir_floor", defaults.quality_icir_floor)),
            quality_sharpe_floor=float(
                getattr(policy, "decorrelation_quality_sharpe_floor", defaults.quality_sharpe_floor)
            ),
            quality_excess_floor=float(
                getattr(policy, "decorrelation_quality_excess_floor", defaults.quality_excess_floor)
            ),
            quality_ann_floor=float(getattr(policy, "decorrelation_quality_ann_floor", defaults.quality_ann_floor)),
            strong_quality_icir=float(
                getattr(policy, "decorrelation_strong_quality_icir", defaults.strong_quality_icir)
            ),
            strong_quality_sharpe=float(
                getattr(policy, "decorrelation_strong_quality_sharpe", defaults.strong_quality_sharpe)
            ),
        )


@dataclass(frozen=True)
class DecorrelationAssessment:
    nearest_corr: float
    avg_corr: float
    nearest_target: str
    grade: str
    score: float
    rerank_adjustment: float
    gate_action: str
    gate_reason: str
    quality_gate_passed: bool
    strong_quality_passed: bool
    winner_allowed: bool
    keep_allowed: bool
    reference_allowed: bool
    arbitration_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decorrelation_rerank_enabled(records: list[dict[str, object]], *, decorrelation_targets_present: bool = False) -> bool:
    if decorrelation_targets_present:
        return True
    for item in records:
        if str(item.get("nearest_decorrelation_target", "") or "").strip():
            return True
        nearest_corr = _safe_float(item.get("corr_to_nearest_decorrelation_target"))
        avg_corr = _safe_float(item.get("avg_abs_decorrelation_target_corr"))
        if _finite(nearest_corr) or _finite(avg_corr):
            return True
    return False


def assess_decorrelation(
    item: dict[str, Any],
    policy: DecorrelationPolicy,
    *,
    material_gain: bool = False,
) -> DecorrelationAssessment:
    nearest_corr = abs(_safe_float(item.get("corr_to_nearest_decorrelation_target")))
    avg_corr = _safe_float(item.get("avg_abs_decorrelation_target_corr"))
    nearest_target = str(item.get("nearest_decorrelation_target", "") or "").strip()
    quality_gate_passed = _quality_gate_passed(item, policy)
    strong_quality_passed = _strong_quality_passed(item, policy)

    if not _finite(nearest_corr) and _finite(avg_corr):
        nearest_corr = avg_corr
    if not _finite(nearest_corr):
        return DecorrelationAssessment(
            nearest_corr=float("nan"),
            avg_corr=avg_corr,
            nearest_target=nearest_target,
            grade="unknown",
            score=0.0,
            rerank_adjustment=0.0,
            gate_action="pass",
            gate_reason="no decorrelation diagnostics available",
            quality_gate_passed=quality_gate_passed,
            strong_quality_passed=strong_quality_passed,
            winner_allowed=True,
            keep_allowed=True,
            reference_allowed=False,
            arbitration_reason="",
        )

    grade = _grade(nearest_corr, policy)
    score = _decorrelation_score(nearest_corr, avg_corr)
    adjustment = _rerank_adjustment(
        nearest_corr=nearest_corr,
        avg_corr=avg_corr,
        grade=grade,
        quality_gate_passed=quality_gate_passed,
        policy=policy,
    )
    gate_action = "pass"
    gate_reason = f"decorrelation {grade}: nearest_corr={nearest_corr:.4f}"
    winner_allowed = True
    keep_allowed = True
    reference_allowed = False
    arbitration_reason = ""

    if policy.strong_gate_enabled:
        if nearest_corr > policy.hard_drop_corr_threshold:
            if strong_quality_passed or bool(material_gain):
                gate_action = "suppress_high_corr_reference"
                arbitration_reason = (
                    "high-corr candidate is strong/material vs parent, but cannot be a decorrelated keep"
                )
                gate_reason = (
                    f"decorrelation hard gate reference-only: nearest_corr={nearest_corr:.4f} "
                    f"> {policy.hard_drop_corr_threshold:.2f}; {arbitration_reason}"
                )
                reference_allowed = True
            else:
                gate_action = "drop"
                gate_reason = (
                    f"decorrelation hard gate: nearest_corr={nearest_corr:.4f} "
                    f"> {policy.hard_drop_corr_threshold:.2f}"
                )
            winner_allowed = False
            keep_allowed = False
        elif nearest_corr > policy.soft_drop_corr_threshold and not bool(material_gain):
            gate_action = "drop"
            gate_reason = (
                f"decorrelation weak-fail gate: nearest_corr={nearest_corr:.4f} "
                f"> {policy.soft_drop_corr_threshold:.2f} and material_gain=false"
            )
            winner_allowed = False
            keep_allowed = False
        elif nearest_corr > policy.suppress_winner_corr_threshold and not strong_quality_passed:
            gate_action = "suppress_winner"
            gate_reason = (
                f"decorrelation winner suppression: nearest_corr={nearest_corr:.4f} "
                f"> {policy.suppress_winner_corr_threshold:.2f} and strong_quality=false"
            )
            winner_allowed = False
        elif nearest_corr > policy.acceptable_corr_threshold:
            gate_action = "soft_penalty"

    return DecorrelationAssessment(
        nearest_corr=nearest_corr,
        avg_corr=avg_corr,
        nearest_target=nearest_target,
        grade=grade,
        score=score,
        rerank_adjustment=adjustment,
        gate_action=gate_action,
        gate_reason=gate_reason,
        quality_gate_passed=quality_gate_passed,
        strong_quality_passed=strong_quality_passed,
        winner_allowed=winner_allowed,
        keep_allowed=keep_allowed,
        reference_allowed=reference_allowed,
        arbitration_reason=arbitration_reason,
    )


def decorate_with_decorrelation_assessment(
    item: dict[str, Any],
    policy: DecorrelationPolicy,
    *,
    material_gain: bool = False,
    base_quality_score: float = 0.0,
) -> dict[str, Any]:
    out = dict(item)
    assessment = assess_decorrelation(out, policy, material_gain=material_gain)
    out["decorrelation_grade"] = assessment.grade
    out["decorrelation_score"] = assessment.score
    out["decorrelation_gate_action"] = assessment.gate_action
    out["decorrelation_gate_reason"] = assessment.gate_reason
    out["decorrelation_winner_allowed"] = assessment.winner_allowed
    out["decorrelation_keep_allowed"] = assessment.keep_allowed
    out["decorrelation_reference_allowed"] = assessment.reference_allowed
    out["decorrelation_arbitration_reason"] = assessment.arbitration_reason
    out["decorrelation_quality_gate_passed"] = assessment.quality_gate_passed
    out["decorrelation_strong_quality_passed"] = assessment.strong_quality_passed
    out["decorrelation_adjustment"] = assessment.rerank_adjustment
    out["decorrelation_quality_score"] = float(base_quality_score)
    out["decorrelation_adjusted_score"] = float(base_quality_score) + float(assessment.rerank_adjustment)
    return out


def _quality_gate_passed(item: dict[str, Any], policy: DecorrelationPolicy) -> bool:
    icir = _safe_float(item.get("quick_rank_icir"))
    sharpe = _safe_float(item.get("net_sharpe"))
    excess = _safe_float(item.get("net_excess_ann_return"))
    ann = _safe_float(item.get("net_ann_return"))
    neutral_icir = _safe_float(item.get("neutral_quick_rank_icir"))
    neutral_sharpe = _safe_float(item.get("neutral_net_sharpe"))
    return bool(
        (_finite(icir) and _finite(sharpe) and icir >= policy.quality_icir_floor and sharpe >= policy.quality_sharpe_floor)
        or (_finite(excess) and excess >= policy.quality_excess_floor)
        or (_finite(ann) and ann >= policy.quality_ann_floor)
        or (_finite(neutral_icir) and _finite(neutral_sharpe) and neutral_icir >= 0.08 and neutral_sharpe >= 1.0)
    )


def _strong_quality_passed(item: dict[str, Any], policy: DecorrelationPolicy) -> bool:
    icir = _safe_float(item.get("quick_rank_icir"))
    sharpe = _safe_float(item.get("net_sharpe"))
    return bool(_finite(icir) and _finite(sharpe) and icir >= policy.strong_quality_icir and sharpe >= policy.strong_quality_sharpe)


def _grade(nearest_corr: float, policy: DecorrelationPolicy) -> str:
    if nearest_corr <= policy.excellent_corr_threshold:
        return "excellent"
    if nearest_corr <= policy.good_corr_threshold:
        return "good"
    if nearest_corr <= policy.acceptable_corr_threshold:
        return "acceptable"
    if nearest_corr <= policy.weak_corr_threshold:
        return "weak"
    return "failed"


def _decorrelation_score(nearest_corr: float, avg_corr: float) -> float:
    nearest_component = 1.0 - abs(float(nearest_corr))
    if _finite(avg_corr):
        avg_component = 1.0 - abs(float(avg_corr))
        return round(_clamp(0.75 * nearest_component + 0.25 * avg_component), 4)
    return round(_clamp(nearest_component), 4)


def _rerank_adjustment(
    *,
    nearest_corr: float,
    avg_corr: float,
    grade: str,
    quality_gate_passed: bool,
    policy: DecorrelationPolicy,
) -> float:
    adjustment = 0.0
    if quality_gate_passed:
        if grade == "excellent":
            adjustment += policy.excellent_bonus
        elif grade == "good":
            adjustment += policy.good_bonus
        elif grade == "acceptable":
            adjustment += policy.acceptable_bonus
    if grade == "weak":
        adjustment -= policy.weak_penalty
    elif grade == "failed":
        adjustment -= policy.failed_penalty
    if _finite(avg_corr) and avg_corr > policy.acceptable_corr_threshold:
        adjustment -= policy.avg_corr_penalty_weight * _clamp(
            (float(avg_corr) - policy.acceptable_corr_threshold)
            / max(1.0 - policy.acceptable_corr_threshold, 1e-9)
        )
    return round(float(adjustment), 4)
