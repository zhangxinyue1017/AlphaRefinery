'''Family saturation assessment for refinement runs.

The saturation layer is intentionally advisory-only in v1: it writes a compact
continuous score and component breakdown to artifacts, but it does not select
the next stage. Table/scoring policies can consume it later once enough runs
have been audited.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..policy_config import DEFAULT_POLICY_CONFIG, RefinePolicyConfig
from ..transition.signals import StageTransitionSignals


_ORDERED_PRESSURE = {
    "low": 0.0,
    "medium": 0.45,
    "high": 0.75,
    "critical": 1.0,
}

_ORDERED_FRONTIER = {
    "high": 0.0,
    "medium": 0.25,
    "low": 0.70,
    "exhausted": 1.0,
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(float(low), min(float(high), float(value)))


@dataclass(frozen=True)
class SaturationAssessment:
    score: float
    grade: str
    recommended_escape_mode: str
    components: dict[str, float]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SaturationAnalyzer:
    '''Compute one continuous saturation score from already-extracted signals.'''

    def __init__(self, config: RefinePolicyConfig | None = None) -> None:
        self.config = config or DEFAULT_POLICY_CONFIG

    @classmethod
    def from_evidence(
        cls,
        evidence: Any,
        signals: StageTransitionSignals,
        config: RefinePolicyConfig | None = None,
    ) -> SaturationAssessment:
        return cls(config=config).assess(evidence, signals)

    def assess(self, evidence: Any, signals: StageTransitionSignals) -> SaturationAssessment:
        thresholds = self.config.saturation
        corr_diag = dict((signals.diagnostics or {}).get("corr_pressure") or {})
        frontier_diag = dict((signals.diagnostics or {}).get("frontier") or {})
        anchor_diag = dict((signals.diagnostics or {}).get("anchor") or {})

        high_corr_count = _safe_int(corr_diag.get("high_corr_count"), _safe_int(getattr(evidence, "high_corr_count", 0)))
        motif_usage_count = _safe_int(corr_diag.get("motif_usage_count"))
        max_motif_count = _safe_int(corr_diag.get("max_motif_count"))
        family_overlap = _safe_float(corr_diag.get("family_overlap"))
        saturation_penalty = _safe_float(corr_diag.get("family_motif_saturation_penalty"))
        children_added = _safe_int(
            frontier_diag.get("children_added_to_search"),
            _safe_int(getattr(evidence, "children_added_to_search", 0)),
        )
        passed_anchor_count = _safe_int(anchor_diag.get("passed_anchor_count"))

        corr_component = max(
            _ORDERED_PRESSURE.get(str(signals.corr_pressure), 0.0),
            _clamp(high_corr_count / max(float(thresholds.high_corr_reference_count), 1.0)),
        )
        motif_component = max(
            _clamp(motif_usage_count / max(float(thresholds.motif_usage_reference), 1.0)),
            _clamp(max_motif_count / max(float(thresholds.motif_usage_reference), 1.0)),
            _clamp(family_overlap / max(float(thresholds.family_overlap_reference), 1e-9)),
            _clamp(saturation_penalty / 0.20),
        )
        plateau_component = _clamp(
            _safe_int(signals.no_improve_count) / max(float(thresholds.max_no_improve_count), 1.0)
        )
        frontier_component = _ORDERED_FRONTIER.get(str(signals.frontier_health), 0.0)
        if children_added <= 1 and signals.winner_quality in {"none", "weak"}:
            frontier_component = max(frontier_component, 0.60)

        anchor_reuse_component = 0.0
        if passed_anchor_count <= 1 and signals.anchor_strength in {"weak", "passed"} and signals.material_gain is False:
            anchor_reuse_component = 0.50
        if signals.anchor_strength == "none" and signals.winner_quality in {"none", "weak"}:
            anchor_reuse_component = max(anchor_reuse_component, 0.70)

        components = {
            "corr": round(corr_component, 4),
            "motif": round(motif_component, 4),
            "plateau": round(plateau_component, 4),
            "frontier": round(frontier_component, 4),
            "anchor_reuse": round(anchor_reuse_component, 4),
        }
        score = (
            components["corr"] * thresholds.corr_weight
            + components["motif"] * thresholds.motif_weight
            + components["plateau"] * thresholds.plateau_weight
            + components["frontier"] * thresholds.frontier_weight
            + components["anchor_reuse"] * thresholds.anchor_reuse_weight
        )
        score = round(_clamp(score), 4)
        grade = self._grade(score)
        recommendation = self._recommendation(
            grade=grade,
            signals=signals,
            high_corr_count=high_corr_count,
        )
        return SaturationAssessment(
            score=score,
            grade=grade,
            recommended_escape_mode=recommendation,
            components=components,
            diagnostics={
                "policy_config_version": thresholds.version,
                "advisory_only": True,
                "high_corr_count": high_corr_count,
                "motif_usage_count": motif_usage_count,
                "max_motif_count": max_motif_count,
                "family_overlap": round(family_overlap, 4),
                "family_motif_saturation_penalty": round(saturation_penalty, 4),
                "children_added_to_search": children_added,
                "passed_anchor_count": passed_anchor_count,
                "weights": {
                    "corr": thresholds.corr_weight,
                    "motif": thresholds.motif_weight,
                    "plateau": thresholds.plateau_weight,
                    "frontier": thresholds.frontier_weight,
                    "anchor_reuse": thresholds.anchor_reuse_weight,
                },
            },
        )

    def _grade(self, score: float) -> str:
        thresholds = self.config.saturation
        if score >= thresholds.critical_score:
            return "critical"
        if score >= thresholds.high_score:
            return "high"
        if score >= thresholds.medium_score:
            return "medium"
        return "low"

    def _recommendation(
        self,
        *,
        grade: str,
        signals: StageTransitionSignals,
        high_corr_count: int,
    ) -> str:
        if grade == "critical":
            if signals.frontier_exhausted or signals.frontier_health == "exhausted":
                return "retire_family"
            return "fork_new_seed"
        if grade == "high":
            if signals.corr_pressure in {"high", "critical"} or high_corr_count > 0:
                return "switch_to_complementarity"
            return "diversify_within_family"
        if grade == "medium":
            return "diversify_within_family"
        return "continue_local"

