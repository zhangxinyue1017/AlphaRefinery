'''Family-level candidate evaluation and keep/drop decision logic.

Builds factor series, applies redundancy checks, runs raw and neutralized backtests, and writes reports.
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...data import build_data_bundle
from ...data_paths import DEFAULT_BENCHMARK_PATH
from ...eval import prepare_backtest_inputs, run_factor_backtest_dual, summarize_backtest_result
from ...registry import FactorRegistry, create_default_registry
from ..core.archive import (
    DEFAULT_ARCHIVE_DB,
    ensure_run_subdir,
    insert_evaluations,
    load_family_reference_candidates,
    make_seed_candidate_id,
    update_candidate_filter_metadata,
    update_candidate_statuses,
    utc_now_iso,
)
from ..parsing.expression_engine import ExpressionEvaluationError, WideExpressionEngine, guess_required_fields
from ..parsing.parser import expression_dedup_key
from ..core.models import LLMProposal, RefinementCandidate, SeedFamily, SeedPool
from ..core.seed_loader import resolve_family_formula, resolve_preferred_refine_seed
from .promotion import write_pending_curated_manifest
from .redundancy import factor_series_correlation, factor_series_correlations
from ..search.decision.decorrelation_policy import DecorrelationPolicy, assess_decorrelation
from ..search.core.policy import SearchPolicy
from ..search.core.scoring import (
    expression_motif_signature,
    expression_mutation_class,
    expression_operator_skeleton,
)

PARENT_CORR_THRESHOLD = 0.95
FAMILY_CORR_THRESHOLD = 0.92
WINNER_SCORE_WEIGHTS: dict[str, float] = {
    "quick_rank_ic_mean": 0.14,
    "quick_rank_icir": 0.16,
    "net_ann_return": 0.18,
    "net_excess_ann_return": 0.18,
    "net_sharpe": 0.26,
    "mean_turnover": 0.08,
}

CORE_FULL_METRICS: tuple[str, ...] = (
    "net_ann_return",
    "net_excess_ann_return",
    "net_sharpe",
    "mean_turnover",
)
NEW_FAMILY_BROAD_MATERIAL_THRESHOLDS: dict[str, float] = {
    "quick_rank_icir": 0.02,
    "net_ann_return": 0.20,
    "net_excess_ann_return": 0.05,
}
_BENCHMARK_REQUIRED_FIELDS = {"market_return", "benchmark_open", "benchmark_close"}

# --- Exploratory keep configuration ---
EXPLORATION_QUALITY_FLOOR_ICIR = 0.30
EXPLORATION_QUALITY_FLOOR_IC_MEAN = 0.03
EXPLORATION_BONUS_THRESHOLD = 0.50
MAX_EXPLORATORY_KEEPS_PER_ROUND = 2

# Correlation thresholds (exact vs soft redundancy)
PARENT_CORR_THRESHOLD_HARD = 0.98
FAMILY_CORR_THRESHOLD_HARD = 0.98
PARENT_CORR_THRESHOLD_SOFT = 0.95
FAMILY_CORR_THRESHOLD_SOFT = 0.92

# Exploration bonus component weights
EXPLORATION_MUTATION_NOVELTY_WEIGHT = 0.35
EXPLORATION_MOTIF_NOVELTY_WEIGHT = 0.30
EXPLORATION_CROSS_MODEL_WEIGHT = 0.25
EXPLORATION_DECORRELATION_WEIGHT = 0.10


def _expression_needs_benchmark(expression: str) -> bool:
    text = str(expression or "").strip()
    if not text:
        return False
    if "benchmark_" in text:
        return True
    return bool(set(guess_required_fields(text)) & _BENCHMARK_REQUIRED_FIELDS)


def _needs_benchmark(family: SeedFamily, proposal: LLMProposal) -> bool:
    if family.canonical_seed.startswith("alpha191."):
        return True
    if any(alias.startswith("alpha191.") for alias in family.aliases):
        return True

    expressions: list[str] = [candidate.expression for candidate in proposal.candidates]
    expressions.extend(str(expr or "") for expr in family.formulas.values())
    return any(_expression_needs_benchmark(expr) for expr in expressions)


def _build_data_for_family(
    *,
    seed_pool: SeedPool,
    family: SeedFamily,
    proposal: LLMProposal,
    panel_path: str | None = None,
    benchmark_path: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> tuple[dict[str, pd.Series], dict[str, object], dict[str, Any]]:
    defaults = dict(seed_pool.evaluation_defaults or {})
    effective_panel_path = panel_path or defaults.get("panel_path")
    if not effective_panel_path:
        raise ValueError("panel_path is required either from seed pool defaults or CLI override")
    effective_benchmark = benchmark_path
    if effective_benchmark is None and _needs_benchmark(family, proposal) and DEFAULT_BENCHMARK_PATH.exists():
        effective_benchmark = str(DEFAULT_BENCHMARK_PATH)
    data, meta = build_data_bundle(
        effective_panel_path,
        benchmark_path=effective_benchmark,
        start=start or defaults.get("start"),
        end=end or defaults.get("end"),
        apply_filters=bool(defaults.get("apply_filters", False)),
        stock_only=bool(defaults.get("stock_only", False)),
        exclude_st=bool(defaults.get("exclude_st", False)),
        exclude_suspended=bool(defaults.get("exclude_suspended", False)),
        min_listed_days=defaults.get("min_listed_days"),
        drop_limit_move=bool(defaults.get("drop_limit_move", False)),
        min_turnover_quantile=defaults.get("min_turnover_quantile"),
        min_liquidity_quantile=defaults.get("min_liquidity_quantile"),
    )
    settings = {
        "panel_path": effective_panel_path,
        "benchmark_path": effective_benchmark,
        "start": start or defaults.get("start"),
        "end": end or defaults.get("end"),
        "horizon": int(defaults.get("horizon", 5)),
        "n_groups": int(defaults.get("n_groups", 5)),
        "pure_mode": str(defaults.get("pure_mode", "none")),
        "cost_bps": float(defaults.get("cost_bps", 10.0)),
        "min_stocks": int(defaults.get("min_stocks", 10)),
    }
    return data, meta, settings


def _candidate_full_name(candidate: RefinementCandidate, *, name_prefix: str) -> str:
    return f"{name_prefix}.{candidate.name}"


def register_proposal_candidates(
    registry: FactorRegistry,
    *,
    proposal: LLMProposal,
    name_prefix: str,
) -> dict[str, str]:
    name_map: dict[str, str] = {}
    for idx, candidate in enumerate(proposal.candidates, start=1):
        full_name = _candidate_full_name(candidate, name_prefix=name_prefix)
        required_fields = guess_required_fields(candidate.expression)
        expression = candidate.expression

        def _make_factor(expr: str, factor_name: str):
            def _factor(data: dict[str, pd.Series]) -> pd.Series:
                engine = WideExpressionEngine(data)
                return engine.evaluate_series(expr, name=factor_name)

            return _factor

        registry.register(
            full_name,
            _make_factor(expression, full_name),
            source="llm_refine_generated",
            required_fields=required_fields,
            expr=expression,
            notes=f"parent={candidate.parent_factor}; family={candidate.family}; source_model={candidate.source_model}; candidate_rank={idx}",
        )
        name_map[candidate.name] = full_name
    return name_map


def _baseline_items(
    family: SeedFamily,
    registry: FactorRegistry,
    *,
    parent_factor_name: str = "",
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    effective_parent_name = str(parent_factor_name or "").strip() or resolve_preferred_refine_seed(family)

    def _resolve_compute_name(name: str) -> str:
        candidates = [name]
        if name.startswith("llmgen."):
            candidates.append(f"llm_refined.{name.split('.', 1)[1]}")
        elif name.startswith("llm_refined."):
            candidates.append(f"llmgen.{name.split('.', 1)[1]}")
        for candidate_name in candidates:
            try:
                registry.get(candidate_name)
                return candidate_name
            except KeyError:
                continue
        return name

    def _expr_for(name: str, *, compute_name: str) -> str:
        if name in family.formulas:
            return resolve_family_formula(family, name)
        try:
            return registry.get(compute_name).expr or ""
        except KeyError:
            return ""

    def _baseline_item(name: str, *, role: str) -> dict[str, Any]:
        compute_name = _resolve_compute_name(name)
        return {
            "factor_name": name,
            "compute_factor_name": compute_name,
            "role": role,
            "model": "Baseline",
            "provider": "baseline_registry",
            "expression": _expr_for(name, compute_name=compute_name),
            "candidate_rank": 0,
            "validation_warnings": (),
            "candidate_id": make_seed_candidate_id(name),
            "round_id": 0,
            "parent_candidate_id": "",
        }

    items.append(
        _baseline_item(
            family.canonical_seed,
            role="parent" if family.canonical_seed == effective_parent_name else "canonical_seed",
        )
    )
    if effective_parent_name != family.canonical_seed and effective_parent_name not in family.aliases:
        items.append(_baseline_item(effective_parent_name, role="parent"))
    for alias in family.aliases:
        items.append(_baseline_item(alias, role="parent" if alias == effective_parent_name else "peer"))
    return items


def _candidate_items(
    proposal: LLMProposal,
    *,
    name_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(proposal.candidates, start=1):
        rows.append(
            {
                "factor_name": _candidate_full_name(candidate, name_prefix=name_prefix),
                "role": "candidate",
                "model": candidate.source_model or "LLM",
                "provider": candidate.source_provider or "",
                "expression": candidate.expression,
                "candidate_rank": idx,
                "validation_warnings": tuple(candidate.validation_warnings),
                "candidate_meta": candidate,
                "candidate_id": candidate.candidate_id,
                "round_id": candidate.round_id,
                "parent_candidate_id": candidate.parent_candidate_id,
            }
        )
    return rows


def _augment_summary(
    summary: dict[str, Any],
    *,
    item: dict[str, Any],
    family: SeedFamily,
    parent_expression: str = "",
) -> dict[str, Any]:
    row = dict(summary)
    row["family"] = family.family
    row["role"] = item["role"]
    row["model"] = item["model"]
    row["provider"] = item["provider"]
    row["expression"] = item["expression"]
    row["candidate_rank"] = item["candidate_rank"]
    row["validation_warnings"] = " | ".join(item.get("validation_warnings", ()))
    row["candidate_id"] = item.get("candidate_id", "")
    row["round_id"] = item.get("round_id", 0)
    row["parent_candidate_id"] = item.get("parent_candidate_id", "")
    # Extract proposal metadata for exploration scoring
    expr = str(item.get("expression", "")).strip()
    parent_expr = str(parent_expression).strip()
    candidate_meta = item.get("candidate_meta")
    if candidate_meta is not None:
        row["mutation_class"] = getattr(candidate_meta, "mutation_class", "") or expression_mutation_class(expr, parent_expr)
        row["motif_signature"] = getattr(candidate_meta, "motif_signature", "") or expression_motif_signature(expr)
        row["operator_skeleton"] = getattr(candidate_meta, "operator_skeleton", "") or expression_operator_skeleton(expr)
    else:
        row["mutation_class"] = expression_mutation_class(expr, parent_expr)
        row["motif_signature"] = expression_motif_signature(expr)
        row["operator_skeleton"] = expression_operator_skeleton(expr)
    return row


def _error_row(*, item: dict[str, Any], family: SeedFamily, error: Exception) -> dict[str, Any]:
    return {
        "factor_name": item["factor_name"],
        "family": family.family,
        "role": item["role"],
        "model": item["model"],
        "provider": item["provider"],
        "expression": item["expression"],
        "candidate_rank": item["candidate_rank"],
        "validation_warnings": " | ".join(item.get("validation_warnings", ())),
        "candidate_id": item.get("candidate_id", ""),
        "round_id": item.get("round_id", 0),
        "parent_candidate_id": item.get("parent_candidate_id", ""),
        "error": str(error),
        "decision": "drop" if item["role"] == "candidate" else item["role"],
        "decision_reason": "evaluation failed",
    }


def _filtered_row(
    *,
    item: dict[str, Any],
    family: SeedFamily,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "factor_name": item["factor_name"],
        "family": family.family,
        "role": item["role"],
        "model": item["model"],
        "provider": item["provider"],
        "expression": item["expression"],
        "candidate_rank": item["candidate_rank"],
        "validation_warnings": " | ".join(item.get("validation_warnings", ())),
        "candidate_id": item.get("candidate_id", ""),
        "round_id": item.get("round_id", 0),
        "parent_candidate_id": item.get("parent_candidate_id", ""),
        "decision": decision,
        "decision_reason": reason,
    }


def _compute_factor_series(
    *,
    item: dict[str, Any],
    registry: FactorRegistry,
    data: dict[str, pd.Series],
) -> pd.Series:
    compute_name = str(item.get("compute_factor_name") or item["factor_name"])
    return registry.compute(compute_name, data)


def _compute_item_series(
    *,
    item: dict[str, Any],
    registry: FactorRegistry,
    data: dict[str, pd.Series],
) -> pd.Series:
    try:
        return _compute_factor_series(item=item, registry=registry, data=data)
    except KeyError:
        expression = str(item.get("expression", "")).strip()
        if not expression:
            raise
        return _compute_expression_series(
            expression=expression,
            data=data,
            factor_name=str(item.get("factor_name", "")),
        )


def _compute_expression_series(
    *,
    expression: str,
    data: dict[str, pd.Series],
    factor_name: str,
) -> pd.Series:
    engine = WideExpressionEngine(data)
    return engine.evaluate_series(expression, name=factor_name)


def _load_archive_reference_series(
    *,
    archive_db: str | Path,
    family: SeedFamily,
    run_id: str,
    data: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    refs = load_family_reference_candidates(
        db_path=archive_db,
        family=family.family,
        exclude_run_id=run_id,
        statuses=("research_winner", "research_keep", "winner", "keep"),
        limit=8,
    )
    if not refs:
        return []
    registry = create_default_registry()
    out: list[dict[str, Any]] = []
    for ref in refs:
        series: pd.Series | None = None
        try:
            try:
                series = registry.compute(ref["factor_name"], data)
            except KeyError:
                series = _compute_expression_series(
                    expression=ref["expression"],
                    data=data,
                    factor_name=ref["factor_name"],
                )
        except Exception:
            series = None
        if series is None:
            continue
        expr = str(ref.get("expression", "")).strip()
        out.append(
            {
                "candidate_id": ref["candidate_id"],
                "factor_name": ref["factor_name"],
                "status": ref["status"],
                "series": series,
                "model": str(ref.get("source_model", "")).strip(),
                "motif_signature": expression_motif_signature(expr),
                "mutation_class": expression_mutation_class(expr, ""),
            }
        )
    return out


def _load_archive_exact_expression_refs(
    *,
    archive_db: str | Path,
    family: SeedFamily,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    refs = load_family_reference_candidates(
        db_path=archive_db,
        family=family.family,
        exclude_run_id=run_id,
        statuses=("research_winner", "research_keep", "winner", "keep"),
        limit=None,
    )
    out: dict[str, dict[str, Any]] = {}
    for ref in refs:
        key = expression_dedup_key(str(ref.get("expression", "")))
        if key and key not in out:
            out[key] = ref
    return out


def _load_decorrelation_target_series(
    *,
    target_names: tuple[str, ...],
    data: dict[str, pd.Series],
) -> list[dict[str, Any]]:
    if not target_names:
        return []
    registry = create_default_registry()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in target_names:
        target_name = str(name or "").strip()
        if not target_name or target_name in seen:
            continue
        seen.add(target_name)
        try:
            series = registry.compute(target_name, data)
        except Exception:
            continue
        out.append({"factor_name": target_name, "series": series})
    return out


def _decorrelation_diagnostics(
    factor: pd.Series,
    *,
    factor_name: str,
    target_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not target_refs:
        return {
            "decorrelation_target_count": 0,
            "nearest_decorrelation_target": "",
            "corr_to_nearest_decorrelation_target": np.nan,
            "signed_corr_to_nearest_decorrelation_target": np.nan,
            "avg_abs_decorrelation_target_corr": np.nan,
        }

    valid_targets: list[tuple[str, pd.Series]] = []
    for ref in target_refs:
        target_name = str(ref.get("factor_name", "") or "").strip()
        target_series = ref.get("series")
        if not target_name or target_name == factor_name or not isinstance(target_series, pd.Series):
            continue
        valid_targets.append((target_name, target_series))

    if not valid_targets:
        return {
            "decorrelation_target_count": 0,
            "nearest_decorrelation_target": "",
            "corr_to_nearest_decorrelation_target": np.nan,
            "signed_corr_to_nearest_decorrelation_target": np.nan,
            "avg_abs_decorrelation_target_corr": np.nan,
        }

    aligned = pd.concat(
        [factor.rename("__candidate__")] + [series.rename(name) for name, series in valid_targets],
        axis=1,
    ).dropna()
    if aligned.empty or len(aligned) < 50:
        return {
            "decorrelation_target_count": 0,
            "nearest_decorrelation_target": "",
            "corr_to_nearest_decorrelation_target": np.nan,
            "signed_corr_to_nearest_decorrelation_target": np.nan,
            "avg_abs_decorrelation_target_corr": np.nan,
        }

    if len(aligned) > 200_000:
        aligned = aligned.sample(200_000, random_state=42)

    corr_series = aligned.drop(columns="__candidate__").corrwith(aligned["__candidate__"]).dropna()
    if corr_series.empty:
        return {
            "decorrelation_target_count": 0,
            "nearest_decorrelation_target": "",
            "corr_to_nearest_decorrelation_target": np.nan,
            "signed_corr_to_nearest_decorrelation_target": np.nan,
            "avg_abs_decorrelation_target_corr": np.nan,
        }

    abs_corr = corr_series.abs()
    nearest_name = str(abs_corr.idxmax())
    nearest_abs = float(abs_corr.loc[nearest_name])
    nearest_signed = float(corr_series.loc[nearest_name])
    return {
        "decorrelation_target_count": int(len(corr_series)),
        "nearest_decorrelation_target": nearest_name,
        "corr_to_nearest_decorrelation_target": nearest_abs,
        "signed_corr_to_nearest_decorrelation_target": nearest_signed,
        "avg_abs_decorrelation_target_corr": float(abs_corr.mean()),
    }


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
    fallback_mask = (
        candidate_mask
        & error_text.eq("")
        & ~decision_text.str.startswith("drop_redundant")
        & decorrelation_gate.ne("drop")
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


def _slim_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    work = summary_df.copy()
    rename_map = {
        "quick_rank_ic_mean": "quick_rank_ic",
        "quick_ic_mean": "quick_ic",
    }
    work = work.rename(columns=rename_map)
    desired = [
        "factor_name",
        "role",
        "model",
        "expression",
        "winner_score",
        "quick_rank_ic",
        "quick_rank_icir",
        "neutral_quick_rank_icir",
        "quick_ic",
        "quick_icir",
        "net_ann_return",
        "neutral_net_ann_return",
        "net_excess_ann_return",
        "neutral_net_excess_ann_return",
        "long_only_net_ann_return",
        "net_sharpe",
        "neutral_net_sharpe",
        "mean_turnover",
        "neutral_mean_turnover",
        "avg_abs_style_corr",
        "max_abs_style_corr",
        "top_style_exposure",
        "top_style_corr",
        "nearest_decorrelation_target",
        "corr_to_nearest_decorrelation_target",
        "avg_abs_decorrelation_target_corr",
        "decorrelation_grade",
        "decorrelation_score",
        "decorrelation_gate_action",
        "decorrelation_gate_reason",
        "decorrelation_winner_allowed",
        "decorrelation_quality_gate_passed",
        "decorrelation_strong_quality_passed",
        "raw_neutral_rank_icir_gap",
        "raw_neutral_net_sharpe_gap",
        "neutral_winner_guard_passed",
        "neutralization_status",
        "stage_winner_guard_passed",
        "stage_winner_guard_reason",
        "metrics_completeness",
        "missing_core_metrics_count",
        "eligible_for_best_node",
        "decision",
        "decision_reason",
    ]
    existing = [col for col in desired if col in work.columns]
    return work[existing].copy()


def _markdown_report(
    *,
    family: SeedFamily,
    summary_df: pd.DataFrame,
    settings: dict[str, Any],
    data_meta: dict[str, object],
) -> str:
    parent = _parent_reference(summary_df, family)
    lines = [
        "# LLM Refine Evaluation Report",
        "",
        f"- family: `{family.family}`",
        f"- canonical_seed: `{family.canonical_seed}`",
        f"- panel_path: `{settings.get('panel_path')}`",
        f"- benchmark_path: `{settings.get('benchmark_path') or 'none'}`",
        f"- start: `{settings.get('start')}`",
        f"- end: `{settings.get('end') or 'latest'}`",
        f"- horizon: `{settings.get('horizon')}`",
        f"- n_groups: `{settings.get('n_groups')}`",
        f"- cost_bps: `{settings.get('cost_bps')}`",
        f"- rows_after_filter: `{data_meta.get('rows_after_filter')}`",
        f"- instruments_after_filter: `{data_meta.get('instruments_after_filter')}`",
        "",
        "## Results",
        "",
        "| Factor | Role | Model | WinnerScore | RankICIR | NRankICIR | Net Sharpe | NSharpe | StyleCorr | Decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('factor_name', '')}`",
                    str(row.get("role", "")),
                    str(row.get("model", "")),
                    _fmt_md_num(row.get("winner_score")),
                    _fmt_md_num(row.get("quick_rank_icir")),
                    _fmt_md_num(row.get("neutral_quick_rank_icir")),
                    _fmt_md_num(row.get("net_sharpe")),
                    _fmt_md_num(row.get("neutral_net_sharpe")),
                    _fmt_md_num(row.get("avg_abs_style_corr")),
                    str(row.get("decision", "")),
                ]
            )
            + " |"
        )
    if parent is not None:
        lines.extend(["", "## Parent Reference", ""])
        lines.append(f"- expression: `{parent.get('expression', '')}`")
        lines.append(f"- RankIC: `{_fmt_md_num(parent.get('quick_rank_ic_mean'))}`")
        lines.append(f"- RankICIR: `{_fmt_md_num(parent.get('quick_rank_icir'))}`")
        lines.append(f"- Neutral RankICIR: `{_fmt_md_num(parent.get('neutral_quick_rank_icir'))}`")
        lines.append(f"- Net Ann Return: `{_fmt_md_num(parent.get('net_ann_return'))}`")
        lines.append(f"- Net Excess Ann Return: `{_fmt_md_num(parent.get('net_excess_ann_return'))}`")
        lines.append(f"- Neutral Net Sharpe: `{_fmt_md_num(parent.get('neutral_net_sharpe'))}`")
        lines.append(f"- Mean Turnover: `{_fmt_md_num(parent.get('mean_turnover'))}`")
    lines.extend(["", "## Research Gate Notes", ""])
    for _, row in summary_df[summary_df["role"] == "candidate"].iterrows():
        lines.append(f"### {row.get('factor_name')}")
        lines.append(f"- model: {row.get('model')}")
        lines.append(f"- expression: `{row.get('expression', '')}`")
        lines.append(f"- research_decision: {row.get('decision', '')}")
        lines.append(f"- reason: {row.get('decision_reason', '')}")
        lines.append(f"- neutral RankICIR: {_fmt_md_num(row.get('neutral_quick_rank_icir'))}")
        lines.append(f"- neutral NetSharpe: {_fmt_md_num(row.get('neutral_net_sharpe'))}")
        lines.append(f"- avg_abs_style_corr: {_fmt_md_num(row.get('avg_abs_style_corr'))}")
        lines.append(f"- nearest_decorrelation_target: {row.get('nearest_decorrelation_target', '')}")
        lines.append(
            f"- corr_to_nearest_decorrelation_target: "
            f"{_fmt_md_num(row.get('corr_to_nearest_decorrelation_target'))}"
        )
        lines.append(
            f"- avg_abs_decorrelation_target_corr: "
            f"{_fmt_md_num(row.get('avg_abs_decorrelation_target_corr'))}"
        )
        if str(row.get("top_style_exposure", "")).strip():
            lines.append(
                f"- top_style_exposure: {row.get('top_style_exposure')} "
                f"({_fmt_md_num(row.get('top_style_corr'))})"
            )
        lines.append(f"- raw_neutral_icir_gap: {_fmt_md_num(row.get('raw_neutral_rank_icir_gap'))}")
        lines.append(f"- raw_neutral_sharpe_gap: {_fmt_md_num(row.get('raw_neutral_net_sharpe_gap'))}")
        lines.append(f"- neutral_winner_guard_passed: {row.get('neutral_winner_guard_passed')}")
        if str(row.get("stage_winner_guard_reason", "")).strip():
            lines.append(f"- stage_winner_guard: {row.get('stage_winner_guard_reason')}")
        if row.get("validation_warnings"):
            lines.append(f"- validation_warnings: {row.get('validation_warnings')}")
        if row.get("error"):
            lines.append(f"- error: {row.get('error')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _fmt_md_num(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return "NA"
        return f"{float(value):.4f}"
    return str(value)


def _evaluate_one_window(
    *,
    seed_pool: SeedPool,
    family: SeedFamily,
    proposal: LLMProposal,
    name_prefix: str,
    panel_path: str | None,
    benchmark_path: str | None,
    start: str | None,
    end: str | None,
    run_id: str,
    archive_db: str | Path,
    stage_mode: str = "auto",
    target_profile: str = "raw_alpha",
    decorrelation_targets: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, Any], dict[str, Any]]:
    registry = create_default_registry()
    register_proposal_candidates(registry, proposal=proposal, name_prefix=name_prefix)
    data, data_meta, settings = _build_data_for_family(
        seed_pool=seed_pool,
        family=family,
        proposal=proposal,
        panel_path=panel_path,
        benchmark_path=benchmark_path,
        start=start,
        end=end,
    )
    prepared = prepare_backtest_inputs(data, horizon=int(settings["horizon"]))

    parent_factor_name = str(proposal.parent_factor or "").strip() or resolve_preferred_refine_seed(family)
    items = _baseline_items(family, registry, parent_factor_name=parent_factor_name) + _candidate_items(
        proposal, name_prefix=name_prefix
    )
    item_by_name = {item["factor_name"]: item for item in items}
    series_cache: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    parent_item = item_by_name.get(parent_factor_name) or item_by_name.get(family.canonical_seed)
    parent_series: pd.Series | None = None
    if parent_item is not None:
        try:
            parent_series = _compute_item_series(item=parent_item, registry=registry, data=data)
            series_cache[parent_item["factor_name"]] = parent_series
        except Exception:
            parent_series = None

    archive_refs = _load_archive_reference_series(
        archive_db=archive_db,
        family=family,
        run_id=run_id,
        data=data,
    )
    archive_exact_refs = _load_archive_exact_expression_refs(
        archive_db=archive_db,
        family=family,
        run_id=run_id,
    )
    decorrelation_target_refs = _load_decorrelation_target_series(
        target_names=decorrelation_targets,
        data=data,
    )
    accepted_candidate_refs: list[dict[str, Any]] = []

    for item in items:
        try:
            if item["role"] == "candidate":
                expr_key = expression_dedup_key(str(item.get("expression", "")))
                matched_exact_ref = archive_exact_refs.get(expr_key) if expr_key else None
                if matched_exact_ref is not None:
                    rows.append(
                        _filtered_row(
                            item=item,
                            family=family,
                            decision="drop_redundant_family_exact",
                            reason=(
                                f"same expression as archived family factor {matched_exact_ref['factor_name']}"
                                f" ({matched_exact_ref.get('status', '')})"
                            ),
                        )
                    )
                    continue

            factor = series_cache.get(item["factor_name"])
            if factor is None:
                factor = _compute_item_series(item=item, registry=registry, data=data)
                series_cache[item["factor_name"]] = factor

            # Redundancy checks: exact is absolute; soft redundancy is recorded
            # but candidate still backtested so exploration rescue can override.
            redundancy_decision: str | None = None
            redundancy_reason: str = ""
            parent_corr = np.nan
            if item["role"] == "candidate" and parent_series is not None:
                corr_refs: list[tuple[str, pd.Series]] = [("__parent__", parent_series)]
                corr_refs.extend(
                    (f"archive::{idx}", ref["series"])
                    for idx, ref in enumerate(archive_refs)
                    if isinstance(ref.get("series"), pd.Series)
                )
                corr_refs.extend(
                    (f"accepted::{idx}", ref["series"])
                    for idx, ref in enumerate(accepted_candidate_refs)
                    if isinstance(ref.get("series"), pd.Series)
                )
                corr_map = factor_series_correlations(factor, corr_refs)
                parent_corr = abs(corr_map.get("__parent__", np.nan))
                if not np.isnan(parent_corr) and parent_corr >= PARENT_CORR_THRESHOLD_SOFT:
                    redundancy_decision = "drop_redundant_parent"
                    redundancy_reason = f"corr with parent {parent_factor_name} = {parent_corr:.4f} >= {PARENT_CORR_THRESHOLD_SOFT:.3f}"

                if redundancy_decision is None:
                    matched_family_ref: tuple[str, float, str] | None = None
                    for idx, ref in enumerate(archive_refs):
                        corr = abs(corr_map.get(f"archive::{idx}", np.nan))
                        if not np.isnan(corr) and corr >= FAMILY_CORR_THRESHOLD_SOFT:
                            matched_family_ref = (ref["factor_name"], corr, str(ref.get("status", "")))
                            break
                    if matched_family_ref is None:
                        for idx, ref in enumerate(accepted_candidate_refs):
                            corr = abs(corr_map.get(f"accepted::{idx}", np.nan))
                            if not np.isnan(corr) and corr >= FAMILY_CORR_THRESHOLD_SOFT:
                                matched_family_ref = (ref["factor_name"], corr, "same_run_candidate")
                                break
                    if matched_family_ref is not None:
                        matched_name, corr_value, matched_status = matched_family_ref
                        redundancy_decision = "drop_redundant_family"
                        redundancy_reason = (
                            f"family corr with {matched_name} = {corr_value:.4f} >= {FAMILY_CORR_THRESHOLD_SOFT:.2f}"
                            f" ({matched_status})"
                        )

            dual_result = run_factor_backtest_dual(
                factor,
                prepared=prepared,
                factor_name=item["factor_name"],
                horizon=int(settings["horizon"]),
                min_stocks=int(settings["min_stocks"]),
                pure_mode=str(settings["pure_mode"]),
                n_groups=int(settings["n_groups"]),
                cost_bps=float(settings["cost_bps"]),
                enable_alphalens=False,
            )
            result = dual_result["raw_result"]
            result["metrics"]["candidate_rank"] = item["candidate_rank"]
            result["metrics"]["expr"] = item["expression"]
            result["metrics"]["direction"] = family.direction
            summary = summarize_backtest_result(result)
            neutralized_result = dual_result.get("neutralized_result")
            if isinstance(neutralized_result, dict):
                neutral_summary = summarize_backtest_result(neutralized_result)
                summary.update(_prefixed_summary(neutral_summary, prefix="neutral_"))
            summary.update(dict(dual_result.get("style_diagnostics") or {}))
            summary.update(
                _decorrelation_diagnostics(
                    factor,
                    factor_name=str(item["factor_name"]),
                    target_refs=decorrelation_target_refs,
                )
            )
            summary["neutralization_status"] = dual_result.get("neutralization_status", "")
            summary["neutralized_factor_nonnull"] = dual_result.get("neutralized_factor_nonnull", 0)
            summary["neutralized_factor_std"] = dual_result.get("neutralized_factor_std", np.nan)
            parent_expr = str(parent_item.get("expression", "")) if parent_item is not None else ""
            aug = _augment_summary(summary, item=item, family=family, parent_expression=parent_expr)
            aug["parent_corr"] = parent_corr
            if redundancy_decision is not None:
                aug["decision"] = redundancy_decision
                aug["decision_reason"] = redundancy_reason
            rows.append(aug)
            if item["role"] == "candidate":
                accepted_candidate_refs.append(
                    {
                        "factor_name": item["factor_name"],
                        "candidate_id": item.get("candidate_id", ""),
                        "series": factor,
                    }
                )
        except Exception as exc:
            rows.append(_error_row(item=item, family=family, error=exc))

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        completeness = summary_df.apply(lambda row: _metrics_completeness(row), axis=1)
        completeness_df = pd.DataFrame(completeness.tolist(), index=summary_df.index)
        summary_df["metrics_completeness"] = completeness_df[0]
        summary_df["missing_core_metrics_count"] = completeness_df[1]
        summary_df["raw_neutral_rank_icir_gap"] = summary_df.apply(
            lambda row: _metric_gap(row, raw_key="quick_rank_icir", neutral_key="neutral_quick_rank_icir"),
            axis=1,
        )
        summary_df["raw_neutral_net_sharpe_gap"] = summary_df.apply(
            lambda row: _metric_gap(row, raw_key="net_sharpe", neutral_key="neutral_net_sharpe"),
            axis=1,
        )
        summary_df["neutral_icir_retention"] = summary_df.apply(
            lambda row: _metric_retention(row, raw_key="quick_rank_icir", neutral_key="neutral_quick_rank_icir"),
            axis=1,
        )
        summary_df["neutral_sharpe_retention"] = summary_df.apply(
            lambda row: _metric_retention(row, raw_key="net_sharpe", neutral_key="neutral_net_sharpe"),
            axis=1,
        )
        summary_df["neutral_winner_guard_passed"] = summary_df.apply(_neutral_winner_guard_passed, axis=1)
        summary_df["eligible_for_best_node"] = summary_df.apply(
            lambda row: bool(
                str(row.get("role", "")) != "candidate"
                or int(row.get("missing_core_metrics_count", 99) or 99) <= 0
            ),
            axis=1,
        )
    parent = _parent_reference(summary_df, family, parent_factor_name=parent_factor_name)
    summary_df = _attach_decorrelation_assessment(
        summary_df,
        parent,
        target_profile=target_profile,
        decorrelation_targets_present=bool(decorrelation_targets),
    )

    # Build family novelty context from accepted archive + this run's kept candidates
    family_motif_set: set[str] = set()
    family_mutation_set: set[str] = set()
    seen_models: set[str] = set()
    for ref in archive_refs:
        family_motif_set.add(str(ref.get("motif_signature", "")).strip())
        family_mutation_set.add(str(ref.get("mutation_class", "")).strip())
        seen_models.add(str(ref.get("model", "")).strip())
    # Also include parent info
    if parent is not None:
        seen_models.add(str(parent.get("model", "")).strip())
        family_motif_set.add(str(parent.get("motif_signature", "")).strip())
        family_mutation_set.add(str(parent.get("mutation_class", "")).strip())

    exploration_budget = {"remaining": MAX_EXPLORATORY_KEEPS_PER_ROUND}

    decisions = summary_df.apply(
        lambda row: _candidate_decision(
            row,
            parent,
            stage_mode=stage_mode,
            target_profile=target_profile,
            decorrelation_targets_present=bool(decorrelation_targets),
            family_motif_set=family_motif_set,
            family_mutation_set=family_mutation_set,
            seen_models=seen_models,
            exploration_budget=exploration_budget,
        ),
        axis=1,
        result_type="expand",
    )
    summary_df["decision"] = decisions[0]
    summary_df["decision_reason"] = decisions[1]
    summary_df = _ensure_keep_floor(summary_df, stage_mode=stage_mode)
    summary_df = _assign_winners(summary_df, stage_mode=stage_mode, parent=parent)
    summary_df["run_id"] = run_id
    summary_df["evaluated_at"] = utc_now_iso()
    return summary_df, data, settings, data_meta


def _write_window_outputs(
    *,
    run_path: Path,
    family: SeedFamily,
    stage_name: str,
    summary_df: pd.DataFrame,
    settings: dict[str, Any],
    data_meta: dict[str, object],
    make_canonical_alias: bool = False,
) -> dict[str, Path]:
    evaluation_dir = ensure_run_subdir(run_path, "evaluation")
    base_name = f"family_backtest_{stage_name}"
    summary_path = evaluation_dir / f"{base_name}_summary.csv"
    ranked_path = evaluation_dir / f"{base_name}_ranked.csv"
    summary_full_path = evaluation_dir / f"{base_name}_summary_full.csv"
    ranked_full_path = evaluation_dir / f"{base_name}_ranked_full.csv"

    ranked_full_df = _sorted_summary(summary_df)
    summary_full_df = summary_df.copy()
    slim_summary_df = _slim_summary(summary_full_df)
    slim_ranked_df = _slim_summary(ranked_full_df)

    summary_full_df.to_csv(summary_full_path, index=False)
    ranked_full_df.to_csv(ranked_full_path, index=False)
    slim_summary_df.to_csv(summary_path, index=False)
    slim_ranked_df.to_csv(ranked_path, index=False)

    outputs = {
        "evaluation_dir": evaluation_dir,
        f"{stage_name}_summary": summary_path,
        f"{stage_name}_ranked": ranked_path,
        f"{stage_name}_summary_full": summary_full_path,
        f"{stage_name}_ranked_full": ranked_full_path,
    }

    if make_canonical_alias:
        canonical_summary = evaluation_dir / "family_backtest_summary.csv"
        canonical_ranked = evaluation_dir / "family_backtest_ranked.csv"
        canonical_summary_full = evaluation_dir / "family_backtest_summary_full.csv"
        canonical_ranked_full = evaluation_dir / "family_backtest_ranked_full.csv"
        canonical_report = evaluation_dir / "keep_drop_report.md"
        research_gate_report = evaluation_dir / "research_gate_report.md"

        report_text = _markdown_report(
            family=family,
            summary_df=summary_df,
            settings=settings,
            data_meta=data_meta,
        )

        slim_summary_df.to_csv(canonical_summary, index=False)
        slim_ranked_df.to_csv(canonical_ranked, index=False)
        summary_full_df.to_csv(canonical_summary_full, index=False)
        ranked_full_df.to_csv(canonical_ranked_full, index=False)
        canonical_report.write_text(report_text, encoding="utf-8")
        research_gate_report.write_text(report_text, encoding="utf-8")
        outputs.update(
            {
                "family_backtest_summary": canonical_summary,
                "family_backtest_ranked": canonical_ranked,
                "family_backtest_summary_full": canonical_summary_full,
                "family_backtest_ranked_full": canonical_ranked_full,
                "keep_drop_report": canonical_report,
                "research_gate_report": research_gate_report,
            }
        )
    return outputs


def evaluate_refinement_run(
    *,
    run_dir: str | Path,
    seed_pool: SeedPool,
    family: SeedFamily,
    proposal: LLMProposal,
    name_prefix: str = "llmgen",
    panel_path: str | None = None,
    benchmark_path: str | None = None,
    start: str | None = None,
    end: str | None = None,
    run_id: str = "",
    round_id: int = 1,
    parent_candidate_id: str = "",
    archive_db: str | Path = DEFAULT_ARCHIVE_DB,
    auto_apply_promotion: bool = False,
    stage_mode: str = "auto",
    target_profile: str = "raw_alpha",
    decorrelation_targets: tuple[str, ...] = (),
) -> dict[str, Path]:
    run_path = Path(run_dir)
    use_protocol = seed_pool.evaluation_protocol is not None and not (start or end)
    metadata_dir = ensure_run_subdir(run_path, "metadata")
    meta_path = metadata_dir / "evaluation_meta.json"
    outputs: dict[str, Path] = {}

    if use_protocol:
        protocol = seed_pool.evaluation_protocol
        assert protocol is not None
        stage_windows: dict[str, Any] = {
            "search": protocol.search,
            "selection": protocol.selection,
        }
        if protocol.final_oos is not None:
            stage_windows["final_oos"] = protocol.final_oos
        decision_stage = protocol.keep_decision_stage
        stage_details: dict[str, Any] = {}
        decision_summary_df: pd.DataFrame | None = None
        decision_data: dict[str, pd.Series] | None = None
        decision_settings: dict[str, Any] | None = None
        decision_data_meta: dict[str, Any] | None = None

        for stage_name in tuple(stage_windows.keys()):
            window = stage_windows[stage_name]
            summary_df, data, settings, data_meta = _evaluate_one_window(
                seed_pool=seed_pool,
                family=family,
                proposal=proposal,
                name_prefix=name_prefix,
                panel_path=panel_path,
                benchmark_path=benchmark_path,
                start=window.start or None,
                end=window.end or None,
                run_id=run_id,
                archive_db=archive_db,
                stage_mode=stage_mode,
                target_profile=target_profile,
                decorrelation_targets=decorrelation_targets,
            )
            stage_outputs = _write_window_outputs(
                run_path=run_path,
                family=family,
                stage_name=stage_name,
                summary_df=summary_df,
                settings=settings,
                data_meta=data_meta,
                make_canonical_alias=stage_name == decision_stage,
            )
            outputs.update(stage_outputs)
            stage_details[stage_name] = {
                "settings": settings,
                "data_meta": data_meta,
            }
            if stage_name == decision_stage:
                decision_summary_df = summary_df
                decision_data = data
                decision_settings = settings
                decision_data_meta = data_meta

        if (
            decision_summary_df is None
            or decision_data is None
            or decision_settings is None
            or decision_data_meta is None
        ):
            raise ValueError(f"decision stage `{decision_stage}` did not produce evaluation output")

        meta_payload = {
            "family": family.family,
            "canonical_seed": family.canonical_seed,
            "name_prefix": name_prefix,
            "evaluation_mode": "protocol",
            "protocol": protocol.to_dict(),
            "decision_stage": decision_stage,
            "stage_mode": str(stage_mode or "auto"),
            "decision_policy": (
                "system-internal research decisions are based on selection. "
                "If final_oos is configured, it is review-only and excluded from optimization and auto-parenting."
            ),
            "decorrelation_targets": list(decorrelation_targets),
            "stage_details": stage_details,
            "stage_files": {key: str(value) for key, value in outputs.items()},
        }
        summary_df_for_archive = decision_summary_df
    else:
        summary_df, data, settings, data_meta = _evaluate_one_window(
            seed_pool=seed_pool,
            family=family,
            proposal=proposal,
            name_prefix=name_prefix,
            panel_path=panel_path,
            benchmark_path=benchmark_path,
            start=start,
            end=end,
            run_id=run_id,
            archive_db=archive_db,
            stage_mode=stage_mode,
            target_profile=target_profile,
            decorrelation_targets=decorrelation_targets,
        )
        outputs.update(
            _write_window_outputs(
                run_path=run_path,
                family=family,
                stage_name="selection",
                summary_df=summary_df,
                settings=settings,
                data_meta=data_meta,
                make_canonical_alias=True,
            )
        )
        meta_payload = {
            "family": family.family,
            "canonical_seed": family.canonical_seed,
            "name_prefix": name_prefix,
            "evaluation_mode": "single_window",
            "stage_mode": str(stage_mode or "auto"),
            "decorrelation_targets": list(decorrelation_targets),
            "settings": settings,
            "data_meta": data_meta,
            "stage_files": {key: str(value) for key, value in outputs.items()},
        }
        summary_df_for_archive = summary_df

    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    insert_evaluations(
        db_path=archive_db,
        run_id=run_id,
        rows=summary_df_for_archive.to_dict(orient="records"),
    )
    update_candidate_filter_metadata(
        db_path=archive_db,
        updates=[
            (
                str(row.get("candidate_id", "")),
                str(row.get("decision", "")),
                "evaluation" if str(row.get("decision", "")).startswith("drop_redundant") else "",
                str(row.get("decision_reason", "")),
            )
            for row in summary_df_for_archive.to_dict(orient="records")
            if str(row.get("role", "")) == "candidate" and str(row.get("decision", "")).startswith("drop_redundant")
        ],
    )
    update_candidate_statuses(
        db_path=archive_db,
        statuses=[
            (str(row.get("candidate_id", "")), str(row.get("decision", "")))
            for row in summary_df_for_archive.to_dict(orient="records")
            if str(row.get("role", "")) == "candidate"
        ],
    )
    promote_outputs = write_pending_curated_manifest(
        family=family,
        summary_df=summary_df_for_archive,
        run_id=run_id,
        round_id=round_id,
        run_dir=run_path,
        decision_stage=str(meta_payload.get("decision_stage", "selection")),
        name_prefix=name_prefix,
        metadata_dir=metadata_dir,
        auto_apply=auto_apply_promotion,
        data=decision_data if use_protocol else data,
    )
    if promote_outputs:
        outputs.update(promote_outputs)
    outputs["evaluation_meta"] = meta_path
    return outputs
