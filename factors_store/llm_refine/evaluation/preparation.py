'''Data, factor registration, and reference-series helpers for evaluator.

Keeps the orchestration code focused by isolating input preparation and series construction.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...data import build_data_bundle
from ...data_paths import DEFAULT_BENCHMARK_PATH
from ...registry import FactorRegistry, create_default_registry
from ..core.archive import load_family_reference_candidates, make_seed_candidate_id
from ..core.models import LLMProposal, RefinementCandidate, SeedFamily, SeedPool
from ..core.seed_loader import resolve_family_formula, resolve_preferred_refine_seed
from ..parsing.expression_engine import WideExpressionEngine, guess_required_fields
from ..parsing.parser import expression_dedup_key

_BENCHMARK_REQUIRED_FIELDS = {'market_return', 'benchmark_open', 'benchmark_close'}


def expression_motif_signature(expression: str) -> str:
    from ..search.core.scoring import expression_motif_signature as _impl

    return _impl(expression)


def expression_mutation_class(expression: str, parent_expression: str) -> str:
    from ..search.core.scoring import expression_mutation_class as _impl

    return _impl(expression, parent_expression)


def expression_operator_skeleton(expression: str) -> str:
    from ..search.core.scoring import expression_operator_skeleton as _impl

    return _impl(expression)


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
    parent_expression: str = "",
    parent_candidate_id: str = "",
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    effective_parent_name = str(parent_factor_name or "").strip() or resolve_preferred_refine_seed(family)
    explicit_parent_expression = str(parent_expression or "").strip()

    def _resolve_compute_name(name: str) -> str:
        if name == effective_parent_name and explicit_parent_expression:
            return ""
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
        if name == effective_parent_name and explicit_parent_expression:
            return explicit_parent_expression
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
            "candidate_id": str(parent_candidate_id or "").strip() if role == "parent" and name == effective_parent_name else make_seed_candidate_id(name),
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
    expression = str(item.get("expression", "")).strip()
    if not str(item.get("compute_factor_name") or "").strip() and expression:
        return _compute_expression_series(
            expression=expression,
            data=data,
            factor_name=str(item.get("factor_name", "")),
        )
    try:
        return _compute_factor_series(item=item, registry=registry, data=data)
    except KeyError:
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

