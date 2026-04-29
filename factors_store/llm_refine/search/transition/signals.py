'''Signal extraction for auditable stage-transition decisions.

The extractor makes the implicit inputs behind the legacy transition resolver
explicit. It is intentionally side-effect free so it can run in shadow mode.
'''

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from ..policy_config import DEFAULT_POLICY_CONFIG, RefinePolicyConfig


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_finite(*values: object, default: float = float("nan")) -> float:
    for value in values:
        numeric = _safe_float(value)
        if _finite(numeric):
            return numeric
    return default


def _candidate_name(payload: dict[str, Any]) -> str:
    return str(payload.get("factor_name") or payload.get("candidate_name") or "").strip()


@dataclass(frozen=True)
class StageTransitionSignals:
    anchor_strength: str
    winner_quality: str
    material_gain: bool
    material_gain_score: float
    corr_pressure: str
    turnover_pressure: str
    frontier_health: str
    no_improve_count: int
    budget_exhausted: bool
    frontier_exhausted: bool
    model_consensus: str
    validation_fail_count: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalExtractor:
    '''Extract discrete transition signals from StageTransitionEvidence.'''

    def __init__(self, config: RefinePolicyConfig | None = None) -> None:
        self.config = config or DEFAULT_POLICY_CONFIG

    def extract(self, evidence: Any) -> StageTransitionSignals:
        stage = str(getattr(evidence, "current_stage", "") or "auto").strip() or "auto"
        winner = self._primary_winner(evidence)
        baseline = self._material_gain_baseline(evidence, stage=stage)
        frontier_nodes = self._frontier_nodes(evidence)
        diagnostics: dict[str, Any] = {
            "primary_winner_name": _candidate_name(winner),
            "baseline_name": _candidate_name(baseline),
            "stage": stage,
            "policy_config_version": self.config.stage_transition.version,
        }

        anchor_strength, anchor_diag = self._anchor_strength(evidence)
        diagnostics["anchor"] = anchor_diag

        winner_quality, winner_diag = self._winner_quality(winner)
        diagnostics["winner"] = winner_diag

        material_gain, material_score, material_diag = self._material_gain(winner, baseline)
        diagnostics["material_gain"] = material_diag

        corr_pressure, corr_diag = self._corr_pressure(evidence, winner, frontier_nodes)
        diagnostics["corr_pressure"] = corr_diag

        turnover_pressure, turnover_diag = self._turnover_pressure(evidence, winner, frontier_nodes)
        diagnostics["turnover_pressure"] = turnover_diag

        frontier_health, frontier_diag = self._frontier_health(evidence, frontier_nodes)
        diagnostics["frontier"] = frontier_diag

        model_consensus, consensus_diag = self._model_consensus(winner, frontier_nodes)
        diagnostics["model_consensus"] = consensus_diag

        return StageTransitionSignals(
            anchor_strength=anchor_strength,
            winner_quality=winner_quality,
            material_gain=material_gain,
            material_gain_score=round(float(material_score), 4),
            corr_pressure=corr_pressure,
            turnover_pressure=turnover_pressure,
            frontier_health=frontier_health,
            no_improve_count=_safe_int(getattr(evidence, "consecutive_no_improve", 0)),
            budget_exhausted=bool(getattr(evidence, "budget_exhausted", False)),
            frontier_exhausted=bool(getattr(evidence, "frontier_exhausted", False)),
            model_consensus=model_consensus,
            validation_fail_count=_safe_int(getattr(evidence, "validation_fail_count", 0)),
            diagnostics=diagnostics,
        )

    @classmethod
    def from_evidence(
        cls,
        evidence: Any,
        config: RefinePolicyConfig | None = None,
    ) -> StageTransitionSignals:
        return cls(config=config).extract(evidence)

    def _primary_winner(self, evidence: Any) -> dict[str, Any]:
        focused = dict(getattr(evidence, "focused_best_node", None) or {})
        winner = dict(getattr(evidence, "last_round_winner", None) or {})
        keep = dict(getattr(evidence, "last_round_keep", None) or {})
        return focused or winner or keep

    def _material_gain_baseline(self, evidence: Any, *, stage: str) -> dict[str, Any]:
        anchor = dict(getattr(evidence, "best_anchor", None) or {})
        keep = dict(getattr(evidence, "last_round_keep", None) or {})
        if stage == "focused_refine":
            return anchor or keep
        return keep or anchor

    def _frontier_nodes(self, evidence: Any) -> tuple[dict[str, Any], ...]:
        raw_nodes = getattr(evidence, "frontier_nodes", ()) or ()
        nodes: list[dict[str, Any]] = []
        for item in raw_nodes:
            if isinstance(item, dict):
                nodes.append(dict(item))
        for item in (
            getattr(evidence, "focused_best_node", None),
            getattr(evidence, "last_round_winner", None),
            getattr(evidence, "last_round_keep", None),
            getattr(evidence, "best_anchor", None),
        ):
            if isinstance(item, dict) and item:
                nodes.append(dict(item))
        return tuple(nodes)

    def _anchor_strength(self, evidence: Any) -> tuple[str, dict[str, Any]]:
        thresholds = self.config.stage_transition.anchor
        anchor = dict(getattr(evidence, "best_anchor", None) or {})
        passed_count = _safe_int(getattr(evidence, "passed_anchor_count", 0))
        score = _first_finite(
            anchor.get("anchor_quality_score"),
            anchor.get("anchor_score"),
            anchor.get("winner_score"),
        )
        if (
            passed_count >= thresholds.strong_passed_count
            and _finite(score)
            and score >= thresholds.strong_quality_score
        ):
            level = "strong"
        elif passed_count >= thresholds.passed_count:
            level = "passed"
        elif anchor and _finite(score) and score >= thresholds.weak_quality_score:
            level = "weak"
        else:
            level = "none"
        return level, {
            "passed_anchor_count": passed_count,
            "best_anchor_name": _candidate_name(anchor),
            "anchor_quality_score": score if _finite(score) else None,
            "thresholds": asdict(thresholds),
        }

    def _winner_quality(self, winner: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        thresholds = self.config.stage_transition.winner_quality
        icir = _safe_float(winner.get("quick_rank_icir"))
        sharpe = _safe_float(winner.get("net_sharpe"))
        if (
            winner
            and _finite(icir)
            and _finite(sharpe)
            and icir >= thresholds.strong_icir
            and sharpe >= thresholds.strong_sharpe
        ):
            level = "strong"
        elif (
            winner
            and _finite(icir)
            and _finite(sharpe)
            and icir >= thresholds.usable_icir
            and sharpe >= thresholds.usable_sharpe
        ):
            level = "usable"
        elif winner and (
            (_finite(icir) and icir >= thresholds.weak_icir)
            or (_finite(sharpe) and sharpe >= thresholds.weak_sharpe)
        ):
            level = "weak"
        else:
            level = "none"
        return level, {
            "factor_name": _candidate_name(winner),
            "quick_rank_icir": icir if _finite(icir) else None,
            "net_sharpe": sharpe if _finite(sharpe) else None,
            "thresholds": asdict(thresholds),
        }

    def _material_gain(
        self,
        candidate: dict[str, Any],
        baseline: dict[str, Any],
    ) -> tuple[bool, float, dict[str, Any]]:
        if not candidate or not baseline:
            return False, 0.0, {
                "available": False,
                "reason": "missing candidate or baseline",
            }
        thresholds = self.config.stage_transition.material_gain
        excess_gain = _safe_float(candidate.get("net_excess_ann_return"), 0.0) - _safe_float(
            baseline.get("net_excess_ann_return"), 0.0
        )
        icir_gain = _safe_float(candidate.get("quick_rank_icir"), 0.0) - _safe_float(
            baseline.get("quick_rank_icir"), 0.0
        )
        sharpe_gain = _safe_float(candidate.get("net_sharpe"), 0.0) - _safe_float(
            baseline.get("net_sharpe"), 0.0
        )
        score = max(
            excess_gain / thresholds.excess_gain_unit,
            icir_gain / thresholds.icir_gain_unit,
            sharpe_gain / thresholds.sharpe_gain_unit,
        )
        passed = bool(
            excess_gain >= thresholds.excess_gain_unit
            or icir_gain >= thresholds.icir_gain_unit
            or sharpe_gain >= thresholds.sharpe_gain_unit
        )
        return passed, score, {
            "available": True,
            "candidate_name": _candidate_name(candidate),
            "baseline_name": _candidate_name(baseline),
            "excess_gain": round(float(excess_gain), 6),
            "icir_gain": round(float(icir_gain), 6),
            "sharpe_gain": round(float(sharpe_gain), 6),
            "score": round(float(score), 4),
            "thresholds": asdict(thresholds),
        }

    def _corr_pressure(
        self,
        evidence: Any,
        winner: dict[str, Any],
        frontier_nodes: tuple[dict[str, Any], ...],
    ) -> tuple[str, dict[str, Any]]:
        thresholds = self.config.stage_transition.corr_pressure
        redundancy_state = dict(getattr(evidence, "redundancy_state", None) or {})
        motif_state = dict(getattr(evidence, "motif_state", None) or {})
        high_corr_count = _safe_int(getattr(evidence, "high_corr_count", 0))
        portfolio_max_similarity = max(
            [
                _safe_float(winner.get("portfolio_max_similarity"), 0.0),
                _safe_float(redundancy_state.get("portfolio_max_similarity"), 0.0),
                *[_safe_float(item.get("portfolio_max_similarity"), 0.0) for item in frontier_nodes],
            ],
            default=0.0,
        )
        motif_usage_count = max(
            [
                _safe_int(winner.get("motif_usage_count"), 0),
                _safe_int(motif_state.get("motif_usage_count"), 0),
                *[_safe_int(item.get("motif_usage_count"), 0) for item in frontier_nodes],
            ],
            default=0,
        )
        family_overlap = max(
            [
                _safe_float(winner.get("family_overlap"), 0.0),
                _safe_float(winner.get("family_overlap_score"), 0.0),
                _safe_float(redundancy_state.get("family_overlap"), 0.0),
                _safe_float(redundancy_state.get("family_overlap_score"), 0.0),
            ],
            default=0.0,
        )
        saturation_penalty = max(
            [
                _safe_float(winner.get("family_motif_saturation_penalty"), 0.0),
                _safe_float(motif_state.get("family_motif_saturation_penalty"), 0.0),
                *[_safe_float(item.get("family_motif_saturation_penalty"), 0.0) for item in frontier_nodes],
            ],
            default=0.0,
        )
        motif_counts = dict(motif_state.get("motif_counts") or {})
        max_motif_count = max([_safe_int(value, 0) for value in motif_counts.values()], default=0)
        motif_saturation = bool(
            motif_usage_count >= thresholds.motif_usage_medium
            or max_motif_count >= thresholds.motif_count_medium
            or family_overlap >= thresholds.family_overlap_medium
            or saturation_penalty >= thresholds.saturation_penalty_medium
        )
        high_family_overlap = bool(
            portfolio_max_similarity >= thresholds.portfolio_similarity_high
            or family_overlap >= thresholds.family_overlap_high
            or motif_usage_count >= thresholds.motif_usage_high
            or max_motif_count >= thresholds.motif_count_high
        )
        if high_corr_count >= thresholds.critical_high_corr_count and high_family_overlap:
            level = "critical"
        elif high_corr_count >= thresholds.high_corr_count:
            level = "high"
        elif motif_saturation:
            level = "medium"
        else:
            level = "low"
        return level, {
            "high_corr_count": high_corr_count,
            "portfolio_max_similarity": round(float(portfolio_max_similarity), 4),
            "motif_usage_count": motif_usage_count,
            "max_motif_count": max_motif_count,
            "family_overlap": round(float(family_overlap), 4),
            "family_motif_saturation_penalty": round(float(saturation_penalty), 4),
            "has_decorrelation_targets": bool(getattr(evidence, "has_decorrelation_targets", False)),
            "note": "decorrelation targets are reported but not treated as pressure by themselves",
            "thresholds": asdict(thresholds),
        }

    def _turnover_pressure(
        self,
        evidence: Any,
        winner: dict[str, Any],
        frontier_nodes: tuple[dict[str, Any], ...],
    ) -> tuple[str, dict[str, Any]]:
        thresholds = self.config.stage_transition.turnover_pressure
        winner_turnover = _safe_float(winner.get("mean_turnover"))
        high_turnover_count = _safe_int(getattr(evidence, "high_turnover_count", 0))
        inferred_high_count = sum(
            1
            for item in frontier_nodes
            if _finite(_safe_float(item.get("mean_turnover")))
            and _safe_float(item.get("mean_turnover")) > thresholds.high_winner_turnover
        )
        multiple_high = max(high_turnover_count, inferred_high_count) >= thresholds.multiple_high_turnover_count
        if (_finite(winner_turnover) and winner_turnover > thresholds.critical_winner_turnover) or multiple_high:
            level = "critical"
        elif (_finite(winner_turnover) and winner_turnover > thresholds.high_winner_turnover) or high_turnover_count > 0:
            level = "high"
        elif (
            _finite(winner_turnover)
            and thresholds.medium_winner_turnover <= winner_turnover <= thresholds.high_winner_turnover
        ):
            level = "medium"
        else:
            level = "low"
        return level, {
            "winner_turnover": winner_turnover if _finite(winner_turnover) else None,
            "high_turnover_count": high_turnover_count,
            "inferred_high_turnover_count": inferred_high_count,
            "thresholds": asdict(thresholds),
        }

    def _frontier_health(
        self,
        evidence: Any,
        frontier_nodes: tuple[dict[str, Any], ...],
    ) -> tuple[str, dict[str, Any]]:
        thresholds = self.config.stage_transition.frontier_health
        if bool(getattr(evidence, "frontier_exhausted", False)):
            return "exhausted", {"frontier_exhausted": True}
        added = _safe_int(getattr(evidence, "children_added_to_search", 0))
        motif_keys = {
            str(item.get("motif_signature") or item.get("operator_skeleton") or "").strip()
            for item in frontier_nodes
            if str(item.get("motif_signature") or item.get("operator_skeleton") or "").strip()
        }
        branch_keys = {
            str(item.get("parent_factor") or item.get("parent_name") or item.get("parent_id") or "").strip()
            for item in frontier_nodes
            if str(item.get("parent_factor") or item.get("parent_name") or item.get("parent_id") or "").strip()
        }
        model_keys = {
            str(item.get("source_model") or item.get("model") or "").strip()
            for item in frontier_nodes
            if str(item.get("source_model") or item.get("model") or "").strip()
        }
        multiple_motifs = len(motif_keys) >= thresholds.multiple_motif_count
        multiple_branches = len(branch_keys) >= thresholds.multiple_branch_count
        cross_model = len(model_keys) >= thresholds.cross_model_count
        if added >= thresholds.high_children_added and (multiple_motifs or multiple_branches or cross_model):
            level = "high"
        elif added >= thresholds.medium_children_added:
            level = "medium"
        else:
            level = "low"
        return level, {
            "children_added_to_search": added,
            "motif_count": len(motif_keys),
            "branch_count": len(branch_keys),
            "model_count": len(model_keys),
            "cross_model": cross_model,
            "thresholds": asdict(thresholds),
        }

    def _model_consensus(
        self,
        winner: dict[str, Any],
        frontier_nodes: tuple[dict[str, Any], ...],
    ) -> tuple[str, dict[str, Any]]:
        thresholds = self.config.stage_transition.model_consensus
        motif_model_map: dict[str, set[str]] = defaultdict(set)
        motif_counts: Counter[str] = Counter()
        for item in (winner, *frontier_nodes):
            if not item:
                continue
            motif = str(
                item.get("operator_skeleton")
                or item.get("motif_signature")
                or item.get("expression_hash")
                or ""
            ).strip()
            if not motif:
                continue
            model = str(item.get("source_model") or item.get("model") or "").strip()
            motif_counts[motif] += 1
            if model:
                motif_model_map[motif].add(model)
        max_model_support = max((len(models) for models in motif_model_map.values()), default=0)
        max_motif_count = max(motif_counts.values(), default=0)
        if max_model_support >= thresholds.high_model_support:
            level = "high"
        elif max_model_support >= thresholds.medium_model_support or max_motif_count >= thresholds.medium_motif_count:
            level = "medium"
        else:
            level = "low"
        return level, {
            "max_model_support": max_model_support,
            "max_motif_count": int(max_motif_count),
            "motif_count": len(motif_counts),
            "thresholds": asdict(thresholds),
        }
