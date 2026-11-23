'''Single-window evaluation orchestration.

Evaluates baselines and proposal candidates for one date window, then attaches decisions.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...eval import prepare_backtest_inputs, run_factor_backtest_dual, summarize_backtest_result
from ...registry import create_default_registry
from ..core.archive import utc_now_iso
from ..core.models import LLMProposal, SeedFamily, SeedPool
from ..core.seed_loader import resolve_preferred_refine_seed
from ..parsing.parser import expression_dedup_key
from .decisions import (
    MAX_EXPLORATORY_KEEPS_PER_ROUND,
    _assign_winners,
    _attach_decorrelation_assessment,
    _candidate_decision,
    _ensure_keep_floor,
    _metric_gap,
    _metric_retention,
    _metrics_completeness,
    _neutral_winner_guard_passed,
    _parent_reference,
    _prefixed_summary,
)
from .preparation import (
    _augment_summary,
    _baseline_items,
    _build_data_for_family,
    _candidate_items,
    _compute_item_series,
    _decorrelation_diagnostics,
    _error_row,
    _filtered_row,
    _load_archive_exact_expression_refs,
    _load_archive_reference_series,
    _load_decorrelation_target_series,
    register_proposal_candidates,
)
from .redundancy import factor_series_correlations

PARENT_CORR_THRESHOLD = 0.95
FAMILY_CORR_THRESHOLD = 0.92
PARENT_CORR_THRESHOLD_HARD = 0.98
FAMILY_CORR_THRESHOLD_HARD = 0.98
PARENT_CORR_THRESHOLD_SOFT = 0.95
FAMILY_CORR_THRESHOLD_SOFT = 0.92


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
    parent_factor_name_override: str = "",
    parent_expression: str = "",
    parent_candidate_id: str = "",
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

    parent_factor_name = (
        str(parent_factor_name_override or "").strip()
        or str(proposal.parent_factor or "").strip()
        or resolve_preferred_refine_seed(family)
    )
    items = _baseline_items(
        family,
        registry,
        parent_factor_name=parent_factor_name,
        parent_expression=parent_expression,
        parent_candidate_id=parent_candidate_id,
    ) + _candidate_items(proposal, name_prefix=name_prefix)
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
