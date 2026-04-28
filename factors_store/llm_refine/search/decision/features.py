'''Typed feature extraction for candidate decision records.

Normalizes evaluation metrics and decorrelation diagnostics for decision-engine scoring.
'''

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..core.scoring import safe_float


@dataclass(frozen=True)
class CandidateDecisionFeatures:
    factor_name: str
    status: str
    source_model: str
    quick_rank_icir: float
    net_ann_return: float
    net_excess_ann_return: float
    net_sharpe: float
    mean_turnover: float
    neutral_quick_rank_icir: float
    neutral_net_sharpe: float
    nearest_decorrelation_target: str
    corr_to_nearest_decorrelation_target: float
    avg_abs_decorrelation_target_corr: float
    stage_winner_guard_passed: float
    neutral_winner_guard_passed: float

    @property
    def has_decorrelation_metrics(self) -> bool:
        return bool(
            self.nearest_decorrelation_target
            or math.isfinite(self.corr_to_nearest_decorrelation_target)
            or math.isfinite(self.avg_abs_decorrelation_target_corr)
        )

    @classmethod
    def from_record(cls, item: dict[str, Any]) -> "CandidateDecisionFeatures":
        return cls(
            factor_name=str(item.get("factor_name", "") or ""),
            status=str(item.get("status", "") or ""),
            source_model=str(item.get("source_model", "") or ""),
            quick_rank_icir=safe_float(item.get("quick_rank_icir"), default=float("nan")),
            net_ann_return=safe_float(item.get("net_ann_return"), default=float("nan")),
            net_excess_ann_return=safe_float(item.get("net_excess_ann_return"), default=float("nan")),
            net_sharpe=safe_float(item.get("net_sharpe"), default=float("nan")),
            mean_turnover=safe_float(item.get("mean_turnover"), default=float("nan")),
            neutral_quick_rank_icir=safe_float(item.get("neutral_quick_rank_icir"), default=float("nan")),
            neutral_net_sharpe=safe_float(item.get("neutral_net_sharpe"), default=float("nan")),
            nearest_decorrelation_target=str(item.get("nearest_decorrelation_target", "") or ""),
            corr_to_nearest_decorrelation_target=safe_float(
                item.get("corr_to_nearest_decorrelation_target"), default=float("nan")
            ),
            avg_abs_decorrelation_target_corr=safe_float(
                item.get("avg_abs_decorrelation_target_corr"), default=float("nan")
            ),
            stage_winner_guard_passed=safe_float(item.get("stage_winner_guard_passed"), default=float("nan")),
            neutral_winner_guard_passed=safe_float(item.get("neutral_winner_guard_passed"), default=float("nan")),
        )
