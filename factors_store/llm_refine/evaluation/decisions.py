'''Metric helpers, research keep/drop gates, and winner selection.

This module owns policy-like evaluation decisions after per-factor backtest metrics exist.
'''

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.models import SeedFamily
from ..core.seed_loader import resolve_preferred_refine_seed

WINNER_SCORE_WEIGHTS: dict[str, float] = {
    'quick_rank_ic_mean': 0.14,
    'quick_rank_icir': 0.16,
    'net_ann_return': 0.18,
    'net_excess_ann_return': 0.18,
    'net_sharpe': 0.26,
    'mean_turnover': 0.08,
}

CORE_FULL_METRICS: tuple[str, ...] = (
    'net_ann_return',
    'net_excess_ann_return',
    'net_sharpe',
    'mean_turnover',
)
NEW_FAMILY_BROAD_MATERIAL_THRESHOLDS: dict[str, float] = {
    'quick_rank_icir': 0.02,
    'net_ann_return': 0.20,
    'net_excess_ann_return': 0.05,
}

EXPLORATION_QUALITY_FLOOR_ICIR = 0.30
EXPLORATION_QUALITY_FLOOR_IC_MEAN = 0.03
EXPLORATION_BONUS_THRESHOLD = 0.50
MAX_EXPLORATORY_KEEPS_PER_ROUND = 2

EXPLORATION_MUTATION_NOVELTY_WEIGHT = 0.35
EXPLORATION_MOTIF_NOVELTY_WEIGHT = 0.30
EXPLORATION_CROSS_MODEL_WEIGHT = 0.25
EXPLORATION_DECORRELATION_WEIGHT = 0.10


def _parent_reference(
    summary_df: pd.DataFrame,
    family: SeedFamily,
    *,
    parent_factor_name: str = "",
) -> pd.Series | None:
    parent_rows = summary_df[summary_df["role"] == "parent"]
    if not parent_rows.empty:
        return parent_rows.iloc[0]
    effective_parent_name = str(parent_factor_name or "").strip() or resolve_preferred_refine_seed(family)
    row = summary_df[summary_df["factor_name"] == effective_parent_name]
    if row.empty and effective_parent_name != family.canonical_seed:
        row = summary_df[summary_df["factor_name"] == family.canonical_seed]
    if row.empty:
        return None
    return row.iloc[0]


def _num(row: pd.Series | dict[str, Any], key: str) -> float:
    value = row.get(key) if hasattr(row, "get") else row[key]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _core_metric_presence(row: pd.Series | dict[str, Any]) -> dict[str, bool]:
    return {key: not np.isnan(_num(row, key)) for key in CORE_FULL_METRICS}


def _has_reference_metrics(row: pd.Series | dict[str, Any] | None) -> bool:
    if row is None:
        return False
    for key in (
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "net_excess_ann_return",
        "net_sharpe",
        "mean_turnover",
    ):
        if not np.isnan(_num(row, key)):
            return True
    return False


def _metrics_completeness(row: pd.Series | dict[str, Any]) -> tuple[float, int]:
    presence = _core_metric_presence(row)
    present = sum(1 for ok in presence.values() if ok)
    missing = len(CORE_FULL_METRICS) - present
    return float(present) / float(len(CORE_FULL_METRICS)), int(missing)


def _prefixed_summary(summary: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    skip = {"factor_name", "candidate_rank", "expr", "direction"}
    return {f"{prefix}{key}": value for key, value in summary.items() if key not in skip}


def _metric_gap(
    row: pd.Series | dict[str, Any],
    *,
    raw_key: str,
    neutral_key: str,
) -> float:
    raw_value = _num(row, raw_key)
    neutral_value = _num(row, neutral_key)
    if np.isnan(raw_value) or np.isnan(neutral_value):
        return np.nan
    return raw_value - neutral_value


def _metric_retention(
    row: pd.Series | dict[str, Any],
    *,
    raw_key: str,
    neutral_key: str,
) -> float:
    raw_value = _num(row, raw_key)
    neutral_value = _num(row, neutral_key)
    if np.isnan(raw_value) or np.isnan(neutral_value) or abs(raw_value) <= 1e-12:
        return np.nan
    return neutral_value / raw_value


def _neutral_winner_guard_passed(row: pd.Series | dict[str, Any]) -> bool:
    if str(row.get("role", "")) != "candidate":
        return True
    raw_icir = _num(row, "quick_rank_icir")
    neutral_icir = _num(row, "neutral_quick_rank_icir")
    raw_sharpe = _num(row, "net_sharpe")
    neutral_sharpe = _num(row, "neutral_net_sharpe")
    avg_style_corr = _num(row, "avg_abs_style_corr")

    if np.isnan(neutral_icir) and np.isnan(neutral_sharpe):
        return True

    icir_retention = _metric_retention(row, raw_key="quick_rank_icir", neutral_key="neutral_quick_rank_icir")
    sharpe_retention = _metric_retention(row, raw_key="net_sharpe", neutral_key="neutral_net_sharpe")
    high_style_loading = (not np.isnan(avg_style_corr)) and avg_style_corr >= 0.20
    icir_collapse = (
        not np.isnan(raw_icir)
        and raw_icir > 0.0
        and not np.isnan(neutral_icir)
        and neutral_icir <= 0.0
        and (np.isnan(icir_retention) or icir_retention < 0.25)
    )
    sharpe_collapse = (
        not np.isnan(raw_sharpe)
        and raw_sharpe > 0.0
        and not np.isnan(neutral_sharpe)
        and neutral_sharpe <= 0.0
        and (np.isnan(sharpe_retention) or sharpe_retention < 0.25)
    )
    if high_style_loading and icir_collapse and sharpe_collapse:
        return False
    return True


def _metric_delta(
    row: pd.Series | dict[str, Any],
    parent: pd.Series | dict[str, Any] | None,
    *,
    key: str,
) -> float:
    if parent is None:
        return np.nan
    value = _num(row, key)
    parent_value = _num(parent, key)
    if np.isnan(value) or np.isnan(parent_value):
        return np.nan
    return value - parent_value


def _is_material_gain(
    row: pd.Series | dict[str, Any],
    parent: pd.Series | dict[str, Any] | None,
    *,
    key: str,
) -> bool:
    threshold = float(NEW_FAMILY_BROAD_MATERIAL_THRESHOLDS.get(key, 0.0))
    delta = _metric_delta(row, parent, key=key)
    return not np.isnan(delta) and delta >= threshold


def _material_gain_vs_parent(row: pd.Series | dict[str, Any], parent: pd.Series | dict[str, Any] | None) -> bool:
    if parent is None:
        return False
    excess_gain = _metric_delta(row, parent, key="net_excess_ann_return")
    icir_gain = _metric_delta(row, parent, key="quick_rank_icir")
    sharpe_gain = _metric_delta(row, parent, key="net_sharpe")
    return bool(
        (not np.isnan(excess_gain) and excess_gain >= 0.02)
        or (not np.isnan(icir_gain) and icir_gain >= 0.05)
        or (not np.isnan(sharpe_gain) and sharpe_gain >= 0.25)
    )


def _attach_decorrelation_assessment(
    summary_df: pd.DataFrame,
    parent: pd.Series | None,
    *,
    target_profile: str,
    decorrelation_targets_present: bool,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    from ..search.core.policy import SearchPolicy
    from ..search.decision.decorrelation_policy import DecorrelationPolicy, assess_decorrelation

    search_policy = SearchPolicy.balanced().with_target_profile(target_profile)
    policy = DecorrelationPolicy.from_search_policy(
        search_policy,
        target_profile=target_profile,
        decorrelation_targets_present=decorrelation_targets_present,
    )
    work = summary_df.copy()
    defaults = {
        "decorrelation_grade": "",
        "decorrelation_score": np.nan,
        "decorrelation_gate_action": "pass",
        "decorrelation_gate_reason": "",
        "decorrelation_winner_allowed": True,
        "decorrelation_keep_allowed": True,
        "decorrelation_reference_allowed": False,
        "decorrelation_arbitration_reason": "",
        "decorrelation_quality_gate_passed": False,
        "decorrelation_strong_quality_passed": False,
    }
    for key, value in defaults.items():
        if key not in work.columns:
            work[key] = value
    for idx, row in work.iterrows():
        if str(row.get("role", "") or "") != "candidate":
            continue
        material_gain = _material_gain_vs_parent(row, parent)
        assessment = assess_decorrelation(dict(row), policy, material_gain=material_gain)
        work.at[idx, "decorrelation_grade"] = assessment.grade
        work.at[idx, "decorrelation_score"] = assessment.score
        work.at[idx, "decorrelation_gate_action"] = assessment.gate_action
        work.at[idx, "decorrelation_gate_reason"] = assessment.gate_reason
        work.at[idx, "decorrelation_winner_allowed"] = assessment.winner_allowed
        work.at[idx, "decorrelation_keep_allowed"] = assessment.keep_allowed
        work.at[idx, "decorrelation_reference_allowed"] = assessment.reference_allowed
        work.at[idx, "decorrelation_arbitration_reason"] = assessment.arbitration_reason
        work.at[idx, "decorrelation_quality_gate_passed"] = assessment.quality_gate_passed
        work.at[idx, "decorrelation_strong_quality_passed"] = assessment.strong_quality_passed
    return work


def _new_family_broad_winner_guard(
    row: pd.Series | dict[str, Any],
    parent: pd.Series | dict[str, Any] | None,
) -> tuple[bool, str]:
    if str(row.get("role", "")) != "candidate":
        return True, "baseline row"
    if parent is None or not _has_reference_metrics(parent):
        return False, "missing parent baseline"

    net_sharpe_delta = _metric_delta(row, parent, key="net_sharpe")
    if np.isnan(net_sharpe_delta) or net_sharpe_delta <= 0.0:
        return False, "new_family_broad winner requires NetSharpe improvement"

    material_labels: list[str] = []
    if _is_material_gain(row, parent, key="quick_rank_icir"):
        material_labels.append("RankICIR")
    if _is_material_gain(row, parent, key="net_ann_return"):
        material_labels.append("NetAnn")
    if _is_material_gain(row, parent, key="net_excess_ann_return"):
        material_labels.append("NetExcess")
    if not material_labels:
        return False, "new_family_broad winner requires material gain on RankICIR / NetAnn / NetExcess"
    return (
        True,
        "new_family_broad winner guard passed: "
        f"NetSharpe improved and material gain on {', '.join(material_labels)}",
    )




def _passes_quality_floor(row: pd.Series) -> bool:
    """Minimum quality bar for exploratory keep rescue."""
    icir = _num(row, "quick_rank_icir")
    ic_mean = _num(row, "quick_rank_ic_mean")
    net_sharpe = _num(row, "net_sharpe")
    if not np.isnan(icir) and icir >= EXPLORATION_QUALITY_FLOOR_ICIR:
        return True
    if (
        not np.isnan(ic_mean)
        and ic_mean >= EXPLORATION_QUALITY_FLOOR_IC_MEAN
        and not np.isnan(net_sharpe)
        and net_sharpe >= 0.0
    ):
        return True
    return False


def _compute_exploration_score(
    row: pd.Series,
    *,
    family_motif_set: set[str],
    family_mutation_set: set[str],
    seen_models: set[str],
) -> float:
    """Compute exploration bonus [0, 1] for a candidate."""
    mutation_class = str(row.get("mutation_class", "")).strip()
    motif_signature = str(row.get("motif_signature", "")).strip()
    model = str(row.get("model", "")).strip()

    mutation_novelty = 1.0 if mutation_class and mutation_class not in family_mutation_set else 0.0
    motif_novelty = 1.0 if motif_signature and motif_signature not in family_motif_set else 0.0
    cross_model = 1.0 if model and model not in seen_models else 0.0

    # Decorrelation: use corr_to_nearest_decorrelation_target if available,
    # otherwise fall back to parent_corr (stored by redundancy check)
    corr_target = _num(row, "corr_to_nearest_decorrelation_target")
    if np.isnan(corr_target):
        parent_corr = _num(row, "parent_corr")
        if not np.isnan(parent_corr):
            corr_target = parent_corr
        else:
            corr_target = 0.0
    decorrelation = max(0.0, 1.0 - abs(corr_target))

    score = (
        EXPLORATION_MUTATION_NOVELTY_WEIGHT * mutation_novelty
        + EXPLORATION_MOTIF_NOVELTY_WEIGHT * motif_novelty
        + EXPLORATION_CROSS_MODEL_WEIGHT * cross_model
        + EXPLORATION_DECORRELATION_WEIGHT * decorrelation
    )
    return float(score)


def _candidate_decision(
    row: pd.Series,
    parent: pd.Series | None,
    *,
    stage_mode: str = "auto",
    target_profile: str = "raw_alpha",
    decorrelation_targets_present: bool = False,
    family_motif_set: set[str] | None = None,
    family_mutation_set: set[str] | None = None,
    seen_models: set[str] | None = None,
    exploration_budget: dict[str, int] | None = None,
) -> tuple[str, str]:
    existing_decision = str(row.get("decision", "")).strip()
    existing_reason = str(row.get("decision_reason", "")).strip()
    # Absolute drops (exact duplicate) are never overridden
    if existing_decision == "drop_redundant_family_exact":
        return existing_decision, existing_reason
    # Soft redundancy drops can be overridden by exploration bonus
    was_soft_redundant = existing_decision.startswith("drop_redundant")

    if str(row.get("role")) != "candidate":
        return str(row.get("role", "")), "baseline row"
    if parent is None or not _has_reference_metrics(parent):
        return "research_drop", "missing parent baseline"
    if pd.notna(row.get("error")) and str(row.get("error")).strip():
        return "research_drop", "evaluation failed"
    if str(row.get("expression", "")).strip() == str(parent.get("expression", "")).strip():
        return "research_drop", "same expression as parent"

    improved: list[str] = []
    material_gain_labels: list[str] = []
    for key, label in (
        ("quick_rank_ic_mean", "RankIC"),
        ("quick_rank_icir", "RankICIR"),
        ("net_ann_return", "NetAnn"),
        ("net_excess_ann_return", "NetExcess"),
        ("net_sharpe", "NetSharpe"),
    ):
        if _num(row, key) > _num(parent, key):
            improved.append(label)
        if stage_mode == "new_family_broad" and key in NEW_FAMILY_BROAD_MATERIAL_THRESHOLDS:
            if _is_material_gain(row, parent, key=key):
                material_gain_labels.append(label)

    completeness, missing_count = _metrics_completeness(row)
    presence = _core_metric_presence(row)
    if missing_count > 1:
        present_count = len(CORE_FULL_METRICS) - missing_count
        return "research_drop", f"insufficient full metrics completeness ({present_count}/{len(CORE_FULL_METRICS)})"
    if not presence["net_sharpe"] and not presence["mean_turnover"]:
        return "research_drop", "missing both NetSharpe and Turnover"

    decorrelation_gate_action = str(row.get("decorrelation_gate_action", "") or "pass").strip()
    decorrelation_gate_reason = str(row.get("decorrelation_gate_reason", "") or "").strip()
    decorrelation_strong_gate_active = bool(
        str(target_profile or "").strip().lower() == "complementarity"
        or decorrelation_targets_present
    )
    if decorrelation_strong_gate_active and decorrelation_gate_action == "drop":
        return "research_drop", decorrelation_gate_reason or "decorrelation strong gate dropped candidate"
    if decorrelation_strong_gate_active and decorrelation_gate_action == "suppress_high_corr_reference":
        return (
            "research_drop",
            decorrelation_gate_reason
            or "decorrelation high-corr reference only; not eligible as decorrelated keep",
        )

    turnover_parent = _num(parent, "mean_turnover")
    turnover_candidate = _num(row, "mean_turnover")
    turnover_cap = max(turnover_parent * 2.0, 0.50) if not np.isnan(turnover_parent) else np.nan
    turnover_ok = np.isnan(turnover_parent) or np.isnan(turnover_candidate) or turnover_candidate <= turnover_cap

    keep = False
    if turnover_ok and improved:
        keep = True
    elif "RankIC" in improved and "NetAnn" in improved:
        keep = True
    elif ("NetExcess" in improved or "NetSharpe" in improved) and (
        np.isnan(turnover_candidate) or np.isnan(turnover_parent) or turnover_candidate <= turnover_parent * 1.25
    ):
        keep = True

    if keep:
        reason = f"broad gate: beats parent on {', '.join(improved)}"
        if stage_mode == "new_family_broad" and material_gain_labels:
            reason += f"; material_gain={', '.join(material_gain_labels)}"
        if decorrelation_strong_gate_active and decorrelation_gate_action == "suppress_winner":
            reason += f"; {decorrelation_gate_reason}"
        reason += f"; full_metrics={len(CORE_FULL_METRICS) - missing_count}/{len(CORE_FULL_METRICS)}"
        return "research_keep", reason

    # --- Exploration rescue (second-stage gate) ---
    budget = exploration_budget or {"remaining": 0}
    if budget.get("remaining", 0) > 0 and _passes_quality_floor(row):
        exploration_score = _compute_exploration_score(
            row,
            family_motif_set=family_motif_set or set(),
            family_mutation_set=family_mutation_set or set(),
            seen_models=seen_models or set(),
        )
        if exploration_score >= EXPLORATION_BONUS_THRESHOLD:
            budget["remaining"] -= 1
            reason = (
                f"exploration rescue (score={exploration_score:.2f}): "
                f"does not beat parent on main metrics but passes quality floor with novel structure"
            )
            if was_soft_redundant:
                reason += f"; overrides {existing_decision}"
            return "research_keep_exploratory", reason

    if was_soft_redundant:
        return existing_decision, existing_reason
    if improved:
        return "research_drop", f"partially improves {', '.join(improved)} but trade-off is too large"
    return "research_drop", "does not beat parent on the main metrics"


def _sorted_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    work = summary_df.copy()
    sort_cols = [col for col in ("quick_rank_icir", "net_ann_return", "quick_rank_ic_mean") if col in work.columns]
    if not sort_cols:
        return work
    return work.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last").reset_index(drop=True)


def _winner_percentile_score(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    scores = pd.Series(0.0, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return scores
    if len(valid) == 1:
        scores.loc[valid.index] = 1.0
        return scores
    ranks = valid.rank(method="average", ascending=True)
    scaled = (ranks - 1.0) / float(len(valid) - 1)
    if not higher_is_better:
        scaled = 1.0 - scaled
    scores.loc[valid.index] = scaled.astype(float)
    return scores


def _attach_winner_scores(summary_df: pd.DataFrame) -> pd.DataFrame:
    work = summary_df.copy()
    candidate_mask = work["role"] == "candidate"
    keep_mask = candidate_mask & work["decision"].isin(["research_keep", "research_winner", "research_keep_exploratory"])
    work["winner_score"] = np.nan
    for key in WINNER_SCORE_WEIGHTS:
        work[f"winner_component_{key}"] = np.nan
    if not keep_mask.any():
        return work

    keep_df = work.loc[keep_mask].copy()
    component_scores: dict[str, pd.Series] = {
        "quick_rank_ic_mean": _winner_percentile_score(
            keep_df.get("quick_rank_ic_mean", pd.Series(index=keep_df.index, dtype=float)),
            higher_is_better=True,
        ),
        "quick_rank_icir": _winner_percentile_score(
            keep_df.get("quick_rank_icir", pd.Series(index=keep_df.index, dtype=float)),
            higher_is_better=True,
        ),
        "net_ann_return": _winner_percentile_score(
            keep_df.get("net_ann_return", pd.Series(index=keep_df.index, dtype=float)),
            higher_is_better=True,
        ),
        "net_excess_ann_return": _winner_percentile_score(
            keep_df.get("net_excess_ann_return", pd.Series(index=keep_df.index, dtype=float)),
            higher_is_better=True,
        ),
        "net_sharpe": _winner_percentile_score(
            keep_df.get("net_sharpe", pd.Series(index=keep_df.index, dtype=float)),
            higher_is_better=True,
        ),
        "mean_turnover": _winner_percentile_score(
            keep_df.get("mean_turnover", pd.Series(index=keep_df.index, dtype=float)),
            higher_is_better=False,
        ),
    }

    winner_score = pd.Series(0.0, index=keep_df.index, dtype=float)
    for key, weight in WINNER_SCORE_WEIGHTS.items():
        component = component_scores[key]
        winner_score = winner_score.add(component * float(weight), fill_value=0.0)
        work.loc[keep_df.index, f"winner_component_{key}"] = component.astype(float)
    work.loc[keep_df.index, "winner_score"] = winner_score.astype(float)
    return work


def _ensure_keep_floor(
    summary_df: pd.DataFrame,
    *,
    stage_mode: str = "auto",
) -> pd.DataFrame:
    if summary_df.empty or stage_mode != "new_family_broad":
        return summary_df

    work = summary_df.copy()
    candidate_mask = work["role"] == "candidate"
    if not candidate_mask.any():
        return work
    if work.loc[candidate_mask, "decision"].isin(["research_keep", "research_winner", "research_keep_exploratory"]).any():
        return work

    error_text = work.get("error", pd.Series(index=work.index, dtype=object)).fillna("").astype(str).str.strip()
    decision_text = work.get("decision", pd.Series(index=work.index, dtype=object)).fillna("").astype(str)
    decorrelation_gate = work.get("decorrelation_gate_action", pd.Series(index=work.index, dtype=object)).fillna("").astype(str)
    decorrelation_keep_allowed = (
        work.get("decorrelation_keep_allowed", pd.Series(True, index=work.index, dtype=bool))
        .fillna(True)
        .astype(bool)
    )
    fallback_mask = (
        candidate_mask
        & error_text.eq("")
        & ~decision_text.str.startswith("drop_redundant")
        & decorrelation_gate.ne("drop")
        & decorrelation_gate.ne("suppress_high_corr_reference")
        & decorrelation_keep_allowed
    )
    fallback_df = work.loc[fallback_mask].copy()
    if fallback_df.empty:
        return work

    sort_cols = [
        col
        for col in (
            "metrics_completeness",
            "net_sharpe",
            "net_ann_return",
            "net_excess_ann_return",
            "quick_rank_icir",
            "quick_rank_ic_mean",
            "mean_turnover",
            "candidate_rank",
        )
        if col in fallback_df.columns
    ]
    if not sort_cols:
        return work
    ascending = [False, False, False, False, False, False, True, True][: len(sort_cols)]
    fallback_df = fallback_df.sort_values(sort_cols, ascending=ascending, na_position="last")
    fallback_idx = fallback_df.index[0]
    base_reason = str(work.at[fallback_idx, "decision_reason"] or "").strip()
    work.at[fallback_idx, "decision"] = "research_keep"
    work.at[fallback_idx, "decision_reason"] = (
        "new_family_broad fallback keep to preserve at least one branch for global rerank"
        + (f"; {base_reason}" if base_reason else "")
    )
    return work


def _assign_winners(
    summary_df: pd.DataFrame,
    *,
    stage_mode: str = "auto",
    parent: pd.Series | None = None,
) -> pd.DataFrame:
    work = _attach_winner_scores(summary_df)
    candidate_mask = work["role"] == "candidate"
    existing_winner_mask = candidate_mask & (work["decision"] == "research_winner")
    work.loc[existing_winner_mask, "decision"] = "research_keep"
    keep_mask = candidate_mask & (work["decision"].isin(["research_keep", "research_keep_exploratory"]))
    if not keep_mask.any():
        return work
    if stage_mode == "new_family_broad":
        stage_guard = work.apply(
            lambda row: _new_family_broad_winner_guard(row, parent),
            axis=1,
            result_type="expand",
        )
        work["stage_winner_guard_passed"] = stage_guard[0]
        work["stage_winner_guard_reason"] = stage_guard[1]
    else:
        # Absolute quality floor for non-broad modes (focused_refine, confirmation, etc.)
        icir_ok = pd.to_numeric(work.get("quick_rank_icir"), errors="coerce").fillna(float("-inf")) >= 0.30
        sharpe_ok = pd.to_numeric(work.get("net_sharpe"), errors="coerce").fillna(float("-inf")) >= 1.5
        turnover_ok = pd.to_numeric(work.get("mean_turnover"), errors="coerce").fillna(float("inf")) <= 0.70
        quality_gate_passed = icir_ok & sharpe_ok & turnover_ok
        work["stage_winner_guard_passed"] = quality_gate_passed
        work["stage_winner_guard_reason"] = work.apply(
            lambda row: (
                "winner quality gate passed"
                if row.get("stage_winner_guard_passed")
                else (
                    f"winner quality gate failed: "
                    f"ICIR={_num(row, 'quick_rank_icir'):.3f} (need>=0.30), "
                    f"Sharpe={_num(row, 'net_sharpe'):.2f} (need>=1.5), "
                    f"TO={_num(row, 'mean_turnover'):.3f} (need<=0.70)"
                )
            ),
            axis=1,
        )
    winner_eligible_mask = keep_mask & (work["decision"] == "research_keep") & (
        pd.to_numeric(work.get("missing_core_metrics_count"), errors="coerce").fillna(99).astype(int) <= 0
    )
    if "neutral_winner_guard_passed" in work.columns:
        winner_eligible_mask = winner_eligible_mask & work["neutral_winner_guard_passed"].fillna(True).astype(bool)
    if "decorrelation_winner_allowed" in work.columns:
        winner_eligible_mask = winner_eligible_mask & work["decorrelation_winner_allowed"].fillna(True).astype(bool)
    winner_eligible_mask = winner_eligible_mask & work["stage_winner_guard_passed"].fillna(False).astype(bool)
    keep_df = work.loc[winner_eligible_mask].copy()
    if keep_df.empty:
        return work
    if stage_mode == "new_family_broad":
        sort_by = [
            col
            for col in (
                "net_sharpe",
                "net_ann_return",
                "net_excess_ann_return",
                "quick_rank_icir",
                "neutral_net_sharpe",
                "neutral_quick_rank_icir",
                "winner_score",
                "mean_turnover",
                "candidate_rank",
            )
            if col in keep_df.columns
        ]
    else:
        sort_by = [
            col
            for col in (
                "winner_score",
                "net_sharpe",
                "net_excess_ann_return",
                "net_ann_return",
                "quick_rank_icir",
                "quick_rank_ic_mean",
                "mean_turnover",
                "candidate_rank",
            )
            if col in keep_df.columns
        ]
    ascending = [False, False, False, False, False, False, False, True, True][: len(sort_by)]
    keep_df = keep_df.sort_values(by=sort_by, ascending=ascending, na_position="last")
    winner_idx = keep_df.index[0]
    base_reason = str(work.at[winner_idx, "decision_reason"] or "").strip()
    winner_score = _num(work.loc[winner_idx], "winner_score")
    component_bits: list[str] = []
    for key, label in (
        ("winner_component_net_sharpe", "sharpe"),
        ("winner_component_net_excess_ann_return", "excess"),
        ("winner_component_net_ann_return", "ann"),
        ("winner_component_quick_rank_icir", "icir"),
        ("winner_component_quick_rank_ic_mean", "ic"),
        ("winner_component_mean_turnover", "turnover"),
    ):
        value = _num(work.loc[winner_idx], key)
        if not np.isnan(value):
            component_bits.append(f"{label}={value:.2f}")
    work.at[winner_idx, "decision"] = "research_winner"
    if stage_mode == "new_family_broad":
        stage_reason = str(work.at[winner_idx, "stage_winner_guard_reason"] or "").strip()
        work.at[winner_idx, "decision_reason"] = (
            f"best new_family_broad candidate by stage-aware multi-metric ranking"
            + (f"; winner_score={winner_score:.4f}" if not np.isnan(winner_score) else "")
            + (f" ({', '.join(component_bits)})" if component_bits else "")
            + (f"; {stage_reason}" if stage_reason else "")
            + (f"; {base_reason}" if base_reason else "")
        )
    else:
        work.at[winner_idx, "decision_reason"] = (
            f"best research keep candidate by composite winner_score={winner_score:.4f}"
            + (f" ({', '.join(component_bits)})" if component_bits else "")
            + (f"; {base_reason}" if base_reason else "")
        )
    return work

