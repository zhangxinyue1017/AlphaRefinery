'''Fast re-evaluation helpers for selected refinement candidates.

Reloads family-loop outputs and recomputes lightweight metrics for focused verification.
'''

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from ..cli.run_refine_multi_model import _pick_best_child_record
from ..cli.run_refine_multi_model_scheduler import _build_scheduler_summary_payload
from ..config import DEFAULT_ARCHIVE_DB_PATH, DEFAULT_SEED_POOL_PATH
from ..core.archive import (
    insert_evaluations,
    update_candidate_filter_metadata,
    update_candidate_statuses,
    utc_now_iso,
)
from ..core.seed_loader import load_seed_pool
from ..evaluation.evaluator import (
    _assign_winners,
    _candidate_decision,
    _has_reference_metrics,
    _metrics_completeness,
    _parent_reference,
    _write_window_outputs,
)
from ..evaluation.promotion import write_pending_curated_manifest
from ..knowledge.family_loop import _read_json, _write_json, build_family_loop_summary, render_family_loop_markdown
from ..search.run_ingest import load_candidate_records_from_completed_runs
from ..search.scoring import safe_float


def _summary_sort_key(row: dict[str, Any], *, path: Path) -> tuple[float, float, float, float, float, float]:
    role = str(row.get("role", "") or "").strip().lower()
    role_score = 1.0 if role == "candidate" else 0.0
    return (
        role_score,
        safe_float(row.get("net_excess_ann_return"), default=float("-inf")),
        safe_float(row.get("quick_rank_icir"), default=float("-inf")),
        safe_float(row.get("net_sharpe"), default=float("-inf")),
        safe_float(row.get("quick_rank_ic_mean"), default=float("-inf")),
        path.stat().st_mtime if path.exists() else 0.0,
    )


def _has_any_metric(row: dict[str, Any]) -> bool:
    for key in (
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "net_excess_ann_return",
        "net_sharpe",
        "mean_turnover",
    ):
        if str(row.get(key, "") or "").strip():
            return True
    return False


def _iter_summary_csvs(search_root: Path) -> list[Path]:
    if not search_root.exists():
        return []
    return sorted(search_root.rglob("family_backtest_summary_full.csv"))


def find_parent_metric_source(
    *,
    search_root: Path,
    factor_name: str,
    exclude_paths: set[Path] | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    target = str(factor_name or "").strip()
    if not target:
        return None, None
    excludes = {p.resolve() for p in (exclude_paths or set())}
    best_row: dict[str, Any] | None = None
    best_path: Path | None = None
    best_key: tuple[float, float, float, float, float, float] | None = None

    for csv_path in _iter_summary_csvs(search_root):
        resolved = csv_path.resolve()
        if resolved in excludes:
            continue
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fp:
                for row in csv.DictReader(fp):
                    if str(row.get("factor_name", "") or "").strip() != target:
                        continue
                    if not _has_any_metric(row):
                        continue
                    key = _summary_sort_key(row, path=csv_path)
                    if best_key is None or key > best_key:
                        best_row = dict(row)
                        best_path = csv_path
                        best_key = key
        except Exception:
            continue
    return best_row, best_path


def _coerce_like(series: pd.Series, raw: Any) -> Any:
    if raw in (None, ""):
        return raw
    if is_bool_dtype(series.dtype):
        text = str(raw).strip().lower()
        return text in {"1", "true", "yes", "y"}
    if is_integer_dtype(series.dtype):
        return int(float(raw))
    if is_numeric_dtype(series.dtype):
        return float(raw)
    return raw


def _patch_parent_baseline(
    *,
    summary_df: pd.DataFrame,
    parent_factor_name: str,
    parent_source: dict[str, Any],
) -> bool:
    parent_mask = (summary_df["role"].astype(str) == "parent") & (
        summary_df["factor_name"].astype(str) == str(parent_factor_name or "").strip()
    )
    if not parent_mask.any():
        return False
    parent_idx = summary_df[parent_mask].index[0]
    for column in summary_df.columns:
        if column in {"role", "model", "provider", "candidate_id", "round_id", "parent_candidate_id"}:
            continue
        raw = parent_source.get(column)
        if raw in (None, ""):
            continue
        summary_df.at[parent_idx, column] = _coerce_like(summary_df[column], raw)
    if "error" in summary_df.columns and is_numeric_dtype(summary_df["error"].dtype):
        summary_df["error"] = summary_df["error"].astype(object)
    summary_df.at[parent_idx, "error"] = ""
    summary_df.at[parent_idx, "decision"] = "parent"
    summary_df.at[parent_idx, "decision_reason"] = "baseline row"
    return True


def _refresh_summary_df(
    *,
    summary_df: pd.DataFrame,
    family: Any,
    parent_factor_name: str,
) -> pd.DataFrame:
    work = summary_df.copy()
    work["metrics_completeness"] = work.apply(lambda row: _metrics_completeness(row)[0], axis=1)
    work["missing_core_metrics_count"] = work.apply(lambda row: _metrics_completeness(row)[1], axis=1)
    work["eligible_for_best_node"] = work.apply(
        lambda row: bool(
            str(row.get("role", "")) != "candidate"
            or int(row.get("missing_core_metrics_count", 99) or 99) <= 0
        ),
        axis=1,
    )
    parent = _parent_reference(work, family, parent_factor_name=parent_factor_name)
    decisions = work.apply(lambda row: _candidate_decision(row, parent), axis=1, result_type="expand")
    work["decision"] = decisions[0]
    work["decision_reason"] = decisions[1]
    return _assign_winners(work)


def _pending_entry_count(metadata_dir: Path) -> int:
    payload = _read_json(metadata_dir / "pending_curated_manifest.json")
    return len(list(payload.get("entries") or []))


def _auto_apply_changed(metadata_dir: Path) -> bool:
    payload = _read_json(metadata_dir / "auto_applied_promotion.json")
    if isinstance(payload, dict):
        return bool(payload.get("changed"))
    if isinstance(payload, list):
        return any(isinstance(item, dict) and item.get("changed") for item in payload)
    return False


def fast_reevaluate_child_run(
    *,
    child_run_dir: Path,
    family: Any,
    archive_db: str | Path = DEFAULT_ARCHIVE_DB_PATH,
    parent_source_root: Path,
    auto_apply_promotion: bool = True,
) -> dict[str, Any]:
    summary_path = child_run_dir / "summary.json"
    proposal_path = child_run_dir / "proposal" / "parsed_proposal.json"
    meta_path = child_run_dir / "metadata" / "evaluation_meta.json"
    selection_csv_path = child_run_dir / "evaluation" / "family_backtest_summary_full.csv"

    if not (summary_path.exists() and proposal_path.exists() and meta_path.exists() and selection_csv_path.exists()):
        return {
            "child_dir": str(child_run_dir),
            "status": "skipped",
            "reason": "missing required files",
        }

    proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    child_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parent_factor_name = str(
        proposal_payload.get("parent_factor")
        or child_summary.get("parent_factor")
        or child_summary.get("current_parent")
        or ""
    ).strip()
    if not parent_factor_name:
        return {
            "child_dir": str(child_run_dir),
            "status": "skipped",
            "reason": "missing parent factor",
        }

    parent_source, parent_source_path = find_parent_metric_source(
        search_root=parent_source_root,
        factor_name=parent_factor_name,
        exclude_paths={selection_csv_path},
    )
    if parent_source is None:
        return {
            "child_dir": str(child_run_dir),
            "status": "skipped",
            "reason": f"parent metric source not found for {parent_factor_name}",
        }

    summary_df = pd.read_csv(selection_csv_path)
    patched_parent = _patch_parent_baseline(
        summary_df=summary_df,
        parent_factor_name=parent_factor_name,
        parent_source=parent_source,
    )
    refreshed_df = _refresh_summary_df(
        summary_df=summary_df,
        family=family,
        parent_factor_name=parent_factor_name,
    )
    refreshed_parent = _parent_reference(refreshed_df, family, parent_factor_name=parent_factor_name)

    if refreshed_parent is None or not _has_reference_metrics(refreshed_parent):
        return {
            "child_dir": str(child_run_dir),
            "status": "skipped",
            "reason": f"missing refreshed parent baseline for {parent_factor_name}",
            "parent_source_path": str(parent_source_path or ""),
        }

    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
    stage_details = meta_payload.get("stage_details", {}) if isinstance(meta_payload, dict) else {}
    decision_stage = str(meta_payload.get("decision_stage", "selection") or "selection")
    decision_settings = (stage_details.get(decision_stage) or {}).get("settings") or meta_payload.get("settings") or {}
    decision_data_meta = (stage_details.get(decision_stage) or {}).get("data_meta") or meta_payload.get("data_meta") or {}

    outputs = _write_window_outputs(
        run_path=child_run_dir,
        family=family,
        stage_name=decision_stage,
        summary_df=refreshed_df,
        settings=decision_settings,
        data_meta=decision_data_meta,
        make_canonical_alias=True,
    )

    run_id = str(child_summary.get("run_id", "") or "")
    round_id = int(child_summary.get("round_id") or 0)
    insert_evaluations(db_path=archive_db, run_id=run_id, rows=refreshed_df.to_dict(orient="records"))
    update_candidate_filter_metadata(
        db_path=archive_db,
        updates=[
            (
                str(row.get("candidate_id", "")),
                str(row.get("decision", "")),
                "evaluation" if str(row.get("decision", "")).startswith("drop_redundant") else "",
                str(row.get("decision_reason", "")),
            )
            for row in refreshed_df.to_dict(orient="records")
            if str(row.get("role", "")) == "candidate" and str(row.get("decision", "")).startswith("drop_redundant")
        ],
    )
    update_candidate_statuses(
        db_path=archive_db,
        statuses=[
            (str(row.get("candidate_id", "")), str(row.get("decision", "")))
            for row in refreshed_df.to_dict(orient="records")
            if str(row.get("role", "")) == "candidate"
        ],
    )
    promote_outputs = write_pending_curated_manifest(
        family=family,
        summary_df=refreshed_df,
        run_id=run_id,
        round_id=round_id,
        run_dir=child_run_dir,
        decision_stage=decision_stage,
        name_prefix=str(meta_payload.get("name_prefix", "llmgen") or "llmgen"),
        metadata_dir=child_run_dir / "metadata",
        auto_apply=auto_apply_promotion,
    ) or {}

    winner_df = refreshed_df[
        (refreshed_df["role"].astype(str) == "candidate")
        & (refreshed_df["decision"].astype(str) == "research_winner")
    ]
    winner_name = str(winner_df.iloc[0]["factor_name"]) if not winner_df.empty else ""
    winner_expression = str(winner_df.iloc[0]["expression"]) if not winner_df.empty else ""
    child_summary["winner_name"] = winner_name
    child_summary["winner_expression"] = winner_expression
    child_summary["reevaluated_fast_path"] = True
    child_summary["reevaluated_at"] = utc_now_iso()
    child_summary["reevaluated_parent_factor"] = parent_factor_name
    child_summary["reevaluated_parent_source_path"] = str(parent_source_path or "")
    child_summary["pending_manifest_path"] = str(promote_outputs.get("pending_curated_manifest_json", ""))
    _write_json(summary_path, child_summary)

    return {
        "child_dir": str(child_run_dir),
        "status": "ok",
        "patched_parent": bool(patched_parent),
        "parent_factor_name": parent_factor_name,
        "parent_source_path": str(parent_source_path or ""),
        "winner_name": winner_name,
        "winner_count": int((refreshed_df["decision"].astype(str) == "research_winner").sum()),
        "keep_count": int((refreshed_df["decision"].astype(str) == "research_keep").sum()),
        "pending_count": _pending_entry_count(child_run_dir / "metadata"),
        "auto_applied": _auto_apply_changed(child_run_dir / "metadata"),
    }


def refresh_round_summaries(
    *,
    scheduler_dir: Path,
    family: str,
    archive_db: str | Path = DEFAULT_ARCHIVE_DB_PATH,
) -> list[dict[str, Any]]:
    round_records: list[dict[str, Any]] = []
    for round_summary_path in sorted(scheduler_dir.glob("multi_runs/round_*/*/*/summary.json")):
        payload = json.loads(round_summary_path.read_text(encoding="utf-8"))
        child_records = load_candidate_records_from_completed_runs(
            archive_db=str(archive_db),
            completed=list(payload.get("completed") or []),
            family=family,
            statuses=("research_winner", "winner", "research_keep", "keep"),
        )
        payload["winner"] = _pick_best_child_record(child_records)
        payload["winner_name"] = str((payload["winner"] or {}).get("factor_name", "") or "")
        payload["winner_expression"] = str((payload["winner"] or {}).get("expression", "") or "")
        payload["children_collected"] = len(child_records)
        _write_json(round_summary_path, payload)
        round_records.append(payload)
    return round_records


def refresh_scheduler_summary(
    *,
    scheduler_dir: Path,
    family: str,
    archive_db: str | Path = DEFAULT_ARCHIVE_DB_PATH,
    round_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary_path = scheduler_dir / "summary.json"
    old_summary = _read_json(summary_path)
    records = list(round_records or [])
    current_search = dict(old_summary.get("search") or {})
    winner_payloads = [dict(item.get("winner") or {}) for item in records if dict(item.get("winner") or {})]
    best_winner = _pick_best_child_record(winner_payloads)
    if best_winner:
        current_search["best_node"] = best_winner
    summary = _build_scheduler_summary_payload(
        family=family,
        target_profile=str(old_summary.get("target_profile", "raw_alpha") or "raw_alpha"),
        scheduler_dir=scheduler_dir,
        archive_db=str(archive_db),
        round_records=records,
        current_search=current_search,
        stop_reason=old_summary.get("stop_reason"),
    )
    _write_json(summary_path, summary)
    return summary


def refresh_family_loop_summary(
    *,
    family_loop_dir: Path,
    broad_run_dir: Path | None,
    focused_run_dir: Path | None,
    focused_summary: dict[str, Any],
) -> dict[str, Any] | None:
    summary_json_path = family_loop_dir / "family_loop_summary.json"
    summary_md_path = family_loop_dir / "family_loop_summary.md"
    old_summary = _read_json(summary_json_path)
    if not old_summary:
        return None
    broad_summary = _read_json((broad_run_dir or Path()) / "summary.json") if broad_run_dir else {}
    refreshed = build_family_loop_summary(
        family=str(old_summary.get("family", "") or ""),
        target_profile=str(old_summary.get("target_profile", "raw_alpha") or "raw_alpha"),
        loop_dir=family_loop_dir,
        broad_stage_preset=str(old_summary.get("broad_stage_preset", "new_family_broad") or "new_family_broad"),
        focused_stage_preset=str(old_summary.get("focused_stage_preset", "focused_refine") or "focused_refine"),
        broad_run_dir=broad_run_dir,
        broad_summary=broad_summary,
        anchor_selection=dict(old_summary.get("anchor_selection") or {}),
        focused_run_dir=focused_run_dir,
        focused_summary=focused_summary,
        broad_returncode=int(old_summary.get("broad_returncode", 0) or 0),
        focused_returncode=int(old_summary.get("focused_returncode", 0) or 0),
    )
    _write_json(summary_json_path, refreshed)
    summary_md_path.write_text(render_family_loop_markdown(refreshed), encoding="utf-8")
    return refreshed


def fast_reevaluate_focused_run(
    *,
    focused_run_dir: str | Path,
    family_loop_dir: str | Path | None = None,
    seed_pool_path: str | Path = DEFAULT_SEED_POOL_PATH,
    archive_db: str | Path = DEFAULT_ARCHIVE_DB_PATH,
    auto_apply_promotion: bool = True,
) -> dict[str, Any]:
    focused_dir = Path(focused_run_dir).expanduser().resolve()
    focused_summary = _read_json(focused_dir / "summary.json")
    family_name = str(focused_summary.get("family", "") or "").strip()
    if not family_name:
        raise ValueError(f"cannot infer family from {focused_dir / 'summary.json'}")

    inferred_loop_dir = (
        Path(family_loop_dir).expanduser().resolve()
        if family_loop_dir
        else focused_dir.parent.parent if focused_dir.parent.name == "focused_runs" else None
    )
    parent_source_root = inferred_loop_dir or focused_dir
    seed_pool = load_seed_pool(seed_pool_path)
    family = seed_pool.get_family(family_name)

    child_reports: list[dict[str, Any]] = []
    for child_summary_path in sorted(focused_dir.glob("multi_runs/round_*/*/*/child_runs/*/*/summary.json")):
        report = fast_reevaluate_child_run(
            child_run_dir=child_summary_path.parent,
            family=family,
            archive_db=archive_db,
            parent_source_root=parent_source_root,
            auto_apply_promotion=auto_apply_promotion,
        )
        child_reports.append(report)

    round_records = refresh_round_summaries(
        scheduler_dir=focused_dir,
        family=family_name,
        archive_db=archive_db,
    )
    refreshed_focused_summary = refresh_scheduler_summary(
        scheduler_dir=focused_dir,
        family=family_name,
        archive_db=archive_db,
        round_records=round_records,
    )

    refreshed_loop_summary = None
    broad_run_dir = None
    if inferred_loop_dir is not None:
        loop_summary = _read_json(inferred_loop_dir / "family_loop_summary.json")
        broad_run_dir_str = str(loop_summary.get("broad_run_dir", "") or "").strip()
        broad_run_dir = Path(broad_run_dir_str) if broad_run_dir_str else None
        refreshed_loop_summary = refresh_family_loop_summary(
            family_loop_dir=inferred_loop_dir,
            broad_run_dir=broad_run_dir,
            focused_run_dir=focused_dir,
            focused_summary=refreshed_focused_summary,
        )

    report = {
        "family": family_name,
        "focused_run_dir": str(focused_dir),
        "family_loop_dir": str(inferred_loop_dir) if inferred_loop_dir else "",
        "reevaluated_at": utc_now_iso(),
        "auto_apply_promotion": bool(auto_apply_promotion),
        "children": child_reports,
        "round_count": len(round_records),
        "focused_last_winner_name": str(refreshed_focused_summary.get("last_winner_name", "") or ""),
        "focused_best_archive_winner_name": str(
            (refreshed_focused_summary.get("best_archive_winner") or {}).get("factor_name", "") or ""
        ),
        "recommended_next_step": str((refreshed_loop_summary or {}).get("recommended_next_step", "") or ""),
    }
    report_path = focused_dir / "reeval_fast_report.json"
    report["report_path"] = str(report_path)
    _write_json(report_path, report)
    return report
