'''Winner and keep selection engine for evaluated candidates.

Ranks child records using quality metrics, optional decorrelation adjustments, and stage-aware sort keys.
'''

from __future__ import annotations

import math
from typing import Any

from .decision_context import DecisionContext, FamilyDecisionState
from .decision_features import CandidateDecisionFeatures
from .scoring import safe_float


def _flag_value(item: dict[str, object], name: str, *, default: bool) -> float:
    value = item.get(name)
    if value is None:
        return 1.0 if default else 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1.0
    if text in {"0", "false", "no", "n"}:
        return 0.0
    return 1.0 if default else 0.0


def _signed_tanh_metric(value: object, *, scale: float) -> float:
    numeric = safe_float(value, default=float("nan"))
    if not math.isfinite(numeric):
        return 0.0
    return math.tanh(numeric / max(float(scale), 1e-9))


def _positive_tanh_metric(value: object, *, scale: float) -> float:
    numeric = safe_float(value, default=float("nan"))
    if not math.isfinite(numeric):
        return 0.0
    return math.tanh(max(numeric, 0.0) / max(float(scale), 1e-9))


def _default_child_sort_key(item: dict[str, object]) -> tuple[float, float, float, float, float, float]:
    status_rank = {
        "research_winner": 4,
        "winner": 3,
        "research_keep": 2,
        "keep": 1,
    }
    status = str(item.get("status", "")).strip().lower()
    return (
        float(status_rank.get(status, 0)),
        safe_float(item.get("quick_rank_ic_mean"), default=float("-inf")),
        safe_float(item.get("quick_rank_icir"), default=float("-inf")),
        safe_float(item.get("net_sharpe"), default=float("-inf")),
        safe_float(item.get("net_ann_return"), default=float("-inf")),
        -safe_float(item.get("mean_turnover"), default=float("inf")),
    )


def _global_new_family_broad_sort_key(
    item: dict[str, object],
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    status = str(item.get("status", "")).strip().lower()
    return (
        _flag_value(item, "stage_winner_guard_passed", default=status == "research_winner"),
        _flag_value(item, "neutral_winner_guard_passed", default=True),
        1.0 if status == "research_winner" else 0.0,
        safe_float(item.get("net_sharpe"), default=float("-inf")),
        safe_float(item.get("net_ann_return"), default=float("-inf")),
        safe_float(item.get("net_excess_ann_return"), default=float("-inf")),
        safe_float(item.get("quick_rank_icir"), default=float("-inf")),
        safe_float(item.get("neutral_net_sharpe"), default=float("-inf")),
        safe_float(item.get("neutral_quick_rank_icir"), default=float("-inf")),
        -safe_float(item.get("mean_turnover"), default=float("inf")),
    )


class DecisionEngine:
    def __init__(self, context: DecisionContext) -> None:
        self.context = context

    def decorrelation_rerank_enabled(self, records: list[dict[str, object]]) -> bool:
        if self.context.decorrelation_enabled:
            return True
        for item in records:
            if CandidateDecisionFeatures.from_record(item).has_decorrelation_metrics:
                return True
        return False

    def _decorrelation_quality_gate_passed(self, item: dict[str, object]) -> bool:
        features = CandidateDecisionFeatures.from_record(item)
        return bool(
            (
                math.isfinite(features.quick_rank_icir)
                and math.isfinite(features.net_sharpe)
                and features.quick_rank_icir >= 0.15
                and features.net_sharpe >= 1.2
            )
            or (math.isfinite(features.net_excess_ann_return) and features.net_excess_ann_return >= 0.05)
            or (math.isfinite(features.net_ann_return) and features.net_ann_return >= 1.5)
            or (
                math.isfinite(features.neutral_quick_rank_icir)
                and math.isfinite(features.neutral_net_sharpe)
                and features.neutral_quick_rank_icir >= 0.08
                and features.neutral_net_sharpe >= 1.0
            )
        )

    def _base_rerank_quality_score(self, item: dict[str, object]) -> float:
        status = str(item.get("status", "")).strip().lower()
        status_bonus = 0.05 if status in {"research_winner", "winner"} else 0.02 if status in {"research_keep", "keep"} else 0.0
        return (
            0.34 * _signed_tanh_metric(item.get("quick_rank_icir"), scale=0.35)
            + 0.26 * _signed_tanh_metric(item.get("net_sharpe"), scale=3.0)
            + 0.18 * _signed_tanh_metric(item.get("net_excess_ann_return"), scale=0.35)
            + 0.12 * _signed_tanh_metric(item.get("net_ann_return"), scale=3.0)
            + 0.06 * _positive_tanh_metric(item.get("neutral_quick_rank_icir"), scale=0.2)
            + 0.06 * _positive_tanh_metric(item.get("neutral_net_sharpe"), scale=1.8)
            - 0.08 * _positive_tanh_metric(item.get("mean_turnover"), scale=0.25)
            + status_bonus
        )

    def _decorrelation_adjustment(self, item: dict[str, object]) -> float:
        features = CandidateDecisionFeatures.from_record(item)
        nearest_corr = abs(features.corr_to_nearest_decorrelation_target)
        avg_corr = features.avg_abs_decorrelation_target_corr
        if not math.isfinite(nearest_corr) and not math.isfinite(avg_corr):
            return 0.0

        quality_gate_passed = self._decorrelation_quality_gate_passed(item)
        adjustment = 0.0

        if math.isfinite(nearest_corr):
            if quality_gate_passed:
                if nearest_corr <= 0.35:
                    adjustment += 0.12
                elif nearest_corr <= 0.60:
                    adjustment += 0.06
                elif nearest_corr <= 0.80:
                    adjustment += 0.02
                elif nearest_corr >= 0.95:
                    adjustment -= 0.08
                elif nearest_corr >= 0.85:
                    adjustment -= 0.04
            elif nearest_corr >= 0.90:
                adjustment -= 0.03

        if math.isfinite(avg_corr):
            if quality_gate_passed:
                if avg_corr <= 0.20:
                    adjustment += 0.05
                elif avg_corr <= 0.40:
                    adjustment += 0.02
                elif avg_corr >= 0.75:
                    adjustment -= 0.05
                elif avg_corr >= 0.60:
                    adjustment -= 0.02
            elif avg_corr >= 0.80:
                adjustment -= 0.02

        return adjustment

    def _decorate_record(self, item: dict[str, object]) -> dict[str, object]:
        out = dict(item)
        quality_score = self._base_rerank_quality_score(out)
        adjustment = self._decorrelation_adjustment(out)
        out["decorrelation_quality_gate_passed"] = self._decorrelation_quality_gate_passed(out)
        out["decorrelation_quality_score"] = quality_score
        out["decorrelation_adjustment"] = adjustment
        out["decorrelation_adjusted_score"] = quality_score + adjustment
        return out

    def _decorrelation_rerank_sort_key(self, item: dict[str, object]) -> tuple[float, float, float, float, float, float, float, float]:
        adjusted = safe_float(item.get("decorrelation_adjusted_score"), default=float("-inf"))
        quality = safe_float(item.get("decorrelation_quality_score"), default=float("-inf"))
        adjustment = safe_float(item.get("decorrelation_adjustment"), default=float("-inf"))
        if self.context.stage_mode == "new_family_broad":
            status = str(item.get("status", "")).strip().lower()
            return (
                _flag_value(item, "stage_winner_guard_passed", default=status == "research_winner"),
                _flag_value(item, "neutral_winner_guard_passed", default=True),
                adjusted,
                quality,
                adjustment,
                safe_float(item.get("quick_rank_icir"), default=float("-inf")),
                safe_float(item.get("net_sharpe"), default=float("-inf")),
                -safe_float(item.get("mean_turnover"), default=float("inf")),
            )
        return (
            adjusted,
            quality,
            adjustment,
            safe_float(item.get("quick_rank_icir"), default=float("-inf")),
            safe_float(item.get("net_sharpe"), default=float("-inf")),
            safe_float(item.get("net_excess_ann_return"), default=float("-inf")),
            safe_float(item.get("net_ann_return"), default=float("-inf")),
            -safe_float(item.get("mean_turnover"), default=float("inf")),
        )

    def rank_records(self, records: list[dict[str, object]]) -> list[dict[str, object]]:
        if not records:
            return []
        decorrelation_enabled = self.decorrelation_rerank_enabled(records)
        working = [self._decorate_record(item) if decorrelation_enabled else dict(item) for item in records]
        if decorrelation_enabled:
            return sorted(working, key=self._decorrelation_rerank_sort_key, reverse=True)
        if self.context.stage_mode == "new_family_broad":
            return sorted(working, key=_global_new_family_broad_sort_key, reverse=True)
        return sorted(working, key=_default_child_sort_key, reverse=True)

    def pick_best_candidate(self, records: list[dict[str, object]]) -> dict[str, object] | None:
        ranked = self.rank_records(records)
        return ranked[0] if ranked else None

    def pick_best_keep(self, records: list[dict[str, object]]) -> dict[str, object] | None:
        keep_records = [
            item
            for item in records
            if str(item.get("status", "")).strip().lower() in {"research_keep", "keep"}
        ]
        ranked = self.rank_records(keep_records)
        return ranked[0] if ranked else None

    def build_rerank_preview(self, records: list[dict[str, object]], *, limit: int = 5) -> list[dict[str, object]]:
        preview: list[dict[str, object]] = []
        for item in self.rank_records(records)[: max(int(limit), 0)]:
            preview.append(
                {
                    "factor_name": str(item.get("factor_name", "") or ""),
                    "status": str(item.get("status", "") or ""),
                    "source_model": str(item.get("source_model", "") or ""),
                    "quick_rank_icir": safe_float(item.get("quick_rank_icir"), default=float("nan")),
                    "net_ann_return": safe_float(item.get("net_ann_return"), default=float("nan")),
                    "net_excess_ann_return": safe_float(item.get("net_excess_ann_return"), default=float("nan")),
                    "net_sharpe": safe_float(item.get("net_sharpe"), default=float("nan")),
                    "mean_turnover": safe_float(item.get("mean_turnover"), default=float("nan")),
                    "nearest_decorrelation_target": str(item.get("nearest_decorrelation_target", "") or ""),
                    "corr_to_nearest_decorrelation_target": safe_float(
                        item.get("corr_to_nearest_decorrelation_target"), default=float("nan")
                    ),
                    "avg_abs_decorrelation_target_corr": safe_float(
                        item.get("avg_abs_decorrelation_target_corr"), default=float("nan")
                    ),
                    "decorrelation_quality_score": safe_float(
                        item.get("decorrelation_quality_score"), default=float("nan")
                    ),
                    "decorrelation_adjustment": safe_float(item.get("decorrelation_adjustment"), default=float("nan")),
                    "decorrelation_adjusted_score": safe_float(
                        item.get("decorrelation_adjusted_score"), default=float("nan")
                    ),
                }
            )
        return preview

    def metric_snapshot(self, item: dict[str, object] | None) -> dict[str, object] | None:
        if not item:
            return None
        return {
            "factor_name": str(item.get("factor_name", "") or ""),
            "status": str(item.get("status", "") or ""),
            "quick_rank_icir": safe_float(item.get("quick_rank_icir"), default=float("nan")),
            "net_ann_return": safe_float(item.get("net_ann_return"), default=float("nan")),
            "net_excess_ann_return": safe_float(item.get("net_excess_ann_return"), default=float("nan")),
            "net_sharpe": safe_float(item.get("net_sharpe"), default=float("nan")),
            "mean_turnover": safe_float(item.get("mean_turnover"), default=float("nan")),
            "nearest_decorrelation_target": str(item.get("nearest_decorrelation_target", "") or ""),
            "corr_to_nearest_decorrelation_target": safe_float(
                item.get("corr_to_nearest_decorrelation_target"), default=float("nan")
            ),
            "avg_abs_decorrelation_target_corr": safe_float(
                item.get("avg_abs_decorrelation_target_corr"), default=float("nan")
            ),
            "decorrelation_quality_gate_passed": bool(item.get("decorrelation_quality_gate_passed", False)),
            "decorrelation_quality_score": safe_float(item.get("decorrelation_quality_score"), default=float("nan")),
            "decorrelation_adjustment": safe_float(item.get("decorrelation_adjustment"), default=float("nan")),
            "decorrelation_adjusted_score": safe_float(
                item.get("decorrelation_adjusted_score"), default=float("nan")
            ),
        }

    def winner_selection_mode(self, records: list[dict[str, object]]) -> str:
        if self.decorrelation_rerank_enabled(records):
            return "decorrelation_soft_rerank"
        if self.context.stage_mode == "new_family_broad":
            return "global_new_family_broad_rerank"
        return "best_child_record"

    def select_anchor(
        self,
        *,
        collected: dict[str, Any],
        min_icir: float,
        min_sharpe: float,
        max_turnover: float,
        min_metrics_completeness: float,
    ) -> dict[str, Any]:
        passed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for raw in list(collected.get("candidates") or []):
            item = dict(raw)
            gate_reasons: list[str] = []
            status = str(item.get("status", "") or item.get("decision", "")).strip().lower()
            if status not in {"research_winner", "winner", "research_keep", "keep"}:
                gate_reasons.append("status_not_promotable")
            anchor_eligible = ("winner" in status) or _flag_value(item, "eligible_for_best_node", default=True) > 0.5
            if not anchor_eligible:
                gate_reasons.append("not_eligible_for_best_node")
            if safe_float(item.get("metrics_completeness"), default=0.0) < float(min_metrics_completeness):
                gate_reasons.append("metrics_incomplete")
            if safe_float(item.get("quick_rank_icir"), default=float("-inf")) < float(min_icir):
                gate_reasons.append("icir_below_threshold")
            if safe_float(item.get("net_sharpe"), default=float("-inf")) < float(min_sharpe):
                gate_reasons.append("sharpe_below_threshold")
            if safe_float(item.get("mean_turnover"), default=float("inf")) > float(max_turnover):
                gate_reasons.append("turnover_above_threshold")
            if _flag_value(item, "true_parent_corr_guard_blocked", default=False) > 0.5:
                gate_reasons.append("true_parent_corr_guard_blocked")
            if _flag_value(item, "stronger_candidate_corr_guard_blocked", default=False) > 0.5:
                gate_reasons.append("stronger_candidate_corr_guard_blocked")
            if _flag_value(item, "heuristic_parent_guard_blocked", default=False) > 0.5:
                gate_reasons.append("heuristic_parent_guard_blocked")
            if _flag_value(item, "corr_guard_blocked", default=False) > 0.5 and not any(
                reason.endswith("corr_guard_blocked") for reason in gate_reasons
            ):
                gate_reasons.append("corr_guard_blocked")

            item["graduation_gate_reasons"] = list(gate_reasons)
            item["graduation_passed"] = not gate_reasons
            status_rank = 1.0 if "winner" in status else 0.0
            anchor_quality_score = (
                0.32 * _signed_tanh_metric(item.get("net_excess_ann_return"), scale=0.35)
                + 0.24 * _signed_tanh_metric(item.get("quick_rank_icir"), scale=0.35)
                + 0.18 * _signed_tanh_metric(item.get("net_sharpe"), scale=3.0)
                + 0.10 * _signed_tanh_metric(item.get("quick_rank_ic_mean"), scale=0.08)
                + 0.08 * _signed_tanh_metric(item.get("winner_score"), scale=0.5)
                - 0.08 * _positive_tanh_metric(item.get("mean_turnover"), scale=0.25)
                + 0.08 * status_rank
            )
            item["anchor_quality_score"] = anchor_quality_score
            item["anchor_selection_trace"] = {
                "status": status,
                "status_rank": status_rank,
                "net_excess_ann_return": safe_float(item.get("net_excess_ann_return"), default=float("nan")),
                "quick_rank_icir": safe_float(item.get("quick_rank_icir"), default=float("nan")),
                "net_sharpe": safe_float(item.get("net_sharpe"), default=float("nan")),
                "quick_rank_ic_mean": safe_float(item.get("quick_rank_ic_mean"), default=float("nan")),
                "winner_score": safe_float(item.get("winner_score"), default=float("nan")),
                "mean_turnover": safe_float(item.get("mean_turnover"), default=float("nan")),
                "anchor_quality_score": anchor_quality_score,
                "graduation_gate_reasons": list(gate_reasons),
            }
            if item["graduation_passed"]:
                passed.append(item)
            else:
                rejected.append(item)

        passed_sorted = sorted(
            passed,
            key=lambda item: (
                safe_float(item.get("anchor_quality_score"), default=float("-inf")),
                safe_float(item.get("net_excess_ann_return"), default=float("-inf")),
                safe_float(item.get("quick_rank_icir"), default=float("-inf")),
                safe_float(item.get("net_sharpe"), default=float("-inf")),
                safe_float(item.get("winner_score"), default=float("-inf")),
                -safe_float(item.get("mean_turnover"), default=float("inf")),
            ),
            reverse=True,
        )
        best_anchor = dict(passed_sorted[0]) if passed_sorted else {}
        if best_anchor:
            best_anchor["selected_as_anchor"] = True

        return {
            "parent_name": collected.get("parent_name", ""),
            "parent_expression": collected.get("parent_expression", ""),
            "parent_metrics": dict(collected.get("parent_metrics") or {}),
            "corr_mode": collected.get("corr_mode", ""),
            "corr_context_error": collected.get("corr_context_error", ""),
            "true_parent_corr_threshold": safe_float(collected.get("true_parent_corr_threshold"), default=0.0),
            "true_sibling_corr_threshold": safe_float(collected.get("true_sibling_corr_threshold"), default=0.0),
            "passed_candidates": passed_sorted,
            "rejected_candidates": rejected,
            "best_anchor": best_anchor,
            "anchor_selection_mode": "decision_engine_v1",
            "anchor_selection_trace": {
                "target_profile": self.context.target_profile,
                "stage_mode": self.context.stage_mode,
                "policy_preset": self.context.policy_preset,
                "context_profile": self.context.context_profile.to_dict() if self.context.context_profile else None,
                "passed_count": len(passed_sorted),
                "rejected_count": len(rejected),
                "top_passed_anchor_scores": [
                    {
                        "factor_name": str(item.get("factor_name", "") or ""),
                        "anchor_quality_score": safe_float(item.get("anchor_quality_score"), default=float("nan")),
                    }
                    for item in passed_sorted[:5]
                ],
            },
        }

    def recommend_next_action(self, state: FamilyDecisionState) -> dict[str, Any]:
        best_anchor = dict(state.anchor_selection.get("best_anchor") or {})
        passed_candidates = list(state.anchor_selection.get("passed_candidates") or [])
        broad_snapshot = dict(state.broad_snapshot or {})
        broad_display = dict(state.broad_display_node or {})
        if not self._has_any_eval_metric(broad_display):
            strongest_evaluated = dict(broad_snapshot.get("strongest_evaluated") or {})
            if strongest_evaluated:
                broad_display = strongest_evaluated
        focused_display = dict(state.focused_display_node or {})

        def _strong_anchor(payload: dict[str, Any]) -> bool:
            return (
                safe_float(payload.get("net_excess_ann_return"), default=float("-inf")) > 0.0
                and safe_float(payload.get("quick_rank_icir"), default=float("-inf")) >= 0.60
                and safe_float(payload.get("net_sharpe"), default=float("-inf")) >= 4.0
            )

        delta = {}
        if best_anchor and focused_display:
            for metric in (
                "quick_rank_ic_mean",
                "quick_rank_icir",
                "net_ann_return",
                "net_excess_ann_return",
                "net_sharpe",
                "mean_turnover",
            ):
                delta[f"delta_anchor_to_focused_{metric}"] = safe_float(
                    focused_display.get(metric), default=0.0
                ) - safe_float(best_anchor.get(metric), default=0.0)

        recommended_stage_preset = ""
        if not best_anchor:
            recommendation = "return_to_broad"
            reason = "broad 阶段没有候选通过 anchor graduation gate"
            recommended_stage_preset = "new_family_broad"
        elif state.focused_returncode != 0 or not focused_display:
            if _strong_anchor(best_anchor):
                recommendation = "donor_mode"
                reason = "focused 阶段没有形成新的 best_node，但当前 anchor 已足够强，适合转 donor/confirmation"
                recommended_stage_preset = "donor_validation"
            else:
                recommendation = "freeze_anchor"
                reason = "anchor 已选出，但 focused 阶段没有形成可比 best candidate"
        else:
            improved = self._winner_improved(focused_display, best_anchor)
            delta_excess = safe_float(focused_display.get("net_excess_ann_return"), default=0.0) - safe_float(
                best_anchor.get("net_excess_ann_return"), default=0.0
            )
            delta_icir = safe_float(focused_display.get("quick_rank_icir"), default=0.0) - safe_float(
                best_anchor.get("quick_rank_icir"), default=0.0
            )
            delta_sharpe = safe_float(focused_display.get("net_sharpe"), default=0.0) - safe_float(
                best_anchor.get("net_sharpe"), default=0.0
            )
            material_improvement = delta_excess >= 0.02 or delta_icir >= 0.05 or delta_sharpe >= 0.25
            if improved and material_improvement:
                recommendation = "continue_focused"
                reason = "focused best node 相对 broad anchor 仍有实质提升"
                recommended_stage_preset = "focused_refine"
            elif improved:
                recommendation = "confirmation"
                reason = "focused best node 仍优于 anchor，但增益已转小，适合进入轻量确认阶段"
                recommended_stage_preset = "confirmation"
            else:
                if _strong_anchor(best_anchor):
                    recommendation = "donor_mode"
                    reason = "focused 阶段没有继续抬高 anchor，但当前 anchor 足够强，适合转 donor/confirmation"
                    recommended_stage_preset = "donor_validation"
                elif len(passed_candidates) >= 2:
                    recommendation = "return_to_broad"
                    reason = "focused 未继续改善，而且 broad 阶段仍有其他通过 gate 的候选，建议回到 broad 重新展开"
                    recommended_stage_preset = "new_family_broad"
                else:
                    recommendation = "freeze_anchor"
                    reason = "focused 阶段没有继续抬高 anchor 的综合质量"

        return {
            "recommended_next_step": recommendation,
            "recommended_next_stage_preset": recommended_stage_preset,
            "recommended_reason": reason,
            "comparison": delta,
            "next_action_mode": "decision_engine_v1",
            "next_action_trace": {
                "target_profile": self.context.target_profile,
                "stage_mode": self.context.stage_mode,
                "policy_preset": self.context.policy_preset,
                "context_profile": self.context.context_profile.to_dict() if self.context.context_profile else None,
                "broad_stop_reason": state.broad_stop_reason,
                "focused_stop_reason": state.focused_stop_reason,
                "passed_anchor_candidate_count": len(passed_candidates),
                "strong_anchor": _strong_anchor(best_anchor) if best_anchor else False,
                "focused_improved_vs_anchor": self._winner_improved(focused_display, best_anchor)
                if best_anchor and focused_display
                else False,
            },
        }

    @staticmethod
    def _has_any_eval_metric(row: dict[str, Any]) -> bool:
        for key in (
            "quick_rank_ic_mean",
            "quick_rank_icir",
            "net_ann_return",
            "net_excess_ann_return",
            "net_sharpe",
            "mean_turnover",
        ):
            value = row.get(key)
            if value not in (None, ""):
                return True
        return False

    @staticmethod
    def _winner_improved(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
        cand_ic = safe_float(candidate.get("quick_rank_ic_mean"), default=float("-inf"))
        base_ic = safe_float(baseline.get("quick_rank_ic_mean"), default=float("-inf"))
        cand_icir = safe_float(candidate.get("quick_rank_icir"), default=float("-inf"))
        base_icir = safe_float(baseline.get("quick_rank_icir"), default=float("-inf"))
        cand_ann = safe_float(candidate.get("net_ann_return"), default=float("-inf"))
        base_ann = safe_float(baseline.get("net_ann_return"), default=float("-inf"))
        cand_excess = safe_float(candidate.get("net_excess_ann_return"), default=float("-inf"))
        base_excess = safe_float(baseline.get("net_excess_ann_return"), default=float("-inf"))
        cand_sharpe = safe_float(candidate.get("net_sharpe"), default=float("-inf"))
        base_sharpe = safe_float(baseline.get("net_sharpe"), default=float("-inf"))
        cand_to = safe_float(candidate.get("mean_turnover"), default=float("inf"))
        base_to = safe_float(baseline.get("mean_turnover"), default=float("inf"))
        metric_improved = (
            cand_ic > base_ic
            or cand_icir > base_icir
            or cand_ann > base_ann
            or cand_excess > base_excess
            or cand_sharpe > base_sharpe
        )
        turnover_not_much_worse = cand_to <= (base_to + 0.05)
        return bool(metric_improved and turnover_not_much_worse)
