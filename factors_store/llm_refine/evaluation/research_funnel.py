'''Research funnel aggregation for refinement outcomes.

Combines archive, scheduler, and admission artifacts into family-level progress summaries.
'''

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import (
    DEFAULT_AUTOFACTORSET_RUNS_DIR,
    DEFAULT_EVALUATOR_OUTPUT_DIR,
    DEFAULT_MULTI_SCHEDULER_RUNS_DIR,
    DEFAULT_SINGLE_RUNS_DIR,
)
from ..core.seed_loader import DEFAULT_SEED_POOL, load_seed_pool, resolve_preferred_refine_seed

DEFAULT_SCHEDULER_RUNS_DIR = DEFAULT_MULTI_SCHEDULER_RUNS_DIR

CORE_METRICS = (
    "quick_rank_ic_mean",
    "quick_rank_icir",
    "net_ann_return",
    "net_excess_ann_return",
    "net_sharpe",
    "mean_turnover",
)

STATUS_PRIORITY = {
    "research_winner": 4,
    "winner": 3,
    "research_keep": 2,
    "keep": 1,
}

METRIC_RENAME = {
    "quick_rank_ic_mean": "rank_ic",
    "quick_rank_icir": "rank_icir",
    "net_ann_return": "net_ann_return",
    "net_excess_ann_return": "net_excess_ann_return",
    "net_sharpe": "net_sharpe",
    "mean_turnover": "mean_turnover",
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_scheduler_top_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if not str(payload.get("family", "")).strip():
        return False
    return "latest_winner" in payload and ("rounds_completed" in payload or "scheduler_dir" in payload)


def _load_eval_df(run_dir: Path) -> pd.DataFrame:
    full_path = run_dir / "evaluation" / "family_backtest_summary_full.csv"
    slim_path = run_dir / "evaluation" / "family_backtest_summary.csv"
    path = full_path if full_path.exists() else slim_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalize_row(row: dict[str, Any] | pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, pd.Series):
        payload = row.to_dict()
    else:
        payload = dict(row)
    out = dict(payload)
    for key in CORE_METRICS:
        out[key] = _safe_float(out.get(key))
    out["winner_score"] = _safe_float(out.get("winner_score"))
    return out


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    status = str(row.get("decision") or row.get("status") or "").strip().lower()
    return (
        float(STATUS_PRIORITY.get(status, 0)),
        _safe_float(row.get("winner_score")) if _safe_float(row.get("winner_score")) is not None else float("-inf"),
        _safe_float(row.get("quick_rank_icir")) if _safe_float(row.get("quick_rank_icir")) is not None else float("-inf"),
        _safe_float(row.get("net_excess_ann_return")) if _safe_float(row.get("net_excess_ann_return")) is not None else float("-inf"),
        _safe_float(row.get("net_sharpe")) if _safe_float(row.get("net_sharpe")) is not None else float("-inf"),
        -(_safe_float(row.get("mean_turnover")) if _safe_float(row.get("mean_turnover")) is not None else float("inf")),
    )


def _metric_dict(row: dict[str, Any] | pd.Series | None) -> dict[str, Any]:
    payload = _normalize_row(row)
    return {key: payload.get(key) for key in CORE_METRICS}


def _find_factor_row(df: pd.DataFrame, factor_name: str) -> dict[str, Any]:
    if df.empty or not factor_name or "factor_name" not in df.columns:
        return {}
    matched = df[df["factor_name"] == factor_name]
    if matched.empty:
        return {}
    return _normalize_row(matched.iloc[0])


def _find_parent_row(df: pd.DataFrame, parent_factor_name: str) -> dict[str, Any]:
    if df.empty:
        return {}
    if "role" in df.columns:
        parent_rows = df[df["role"] == "parent"]
        if not parent_rows.empty:
            return _normalize_row(parent_rows.iloc[0])
    return _find_factor_row(df, parent_factor_name)


def _candidate_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "role" not in df.columns:
        return []
    work = df[df["role"] == "candidate"].copy()
    if work.empty:
        return []
    return [_normalize_row(row) for _, row in work.iterrows()]


def _top_k_candidates(rows: list[dict[str, Any]], k: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=_candidate_sort_key, reverse=True)
    return ranked[: max(int(k), 1)]


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _positive_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(1 for value in values if value > 0) / len(values))


def _metric_std(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [_safe_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    mean = float(sum(values) / len(values))
    variance = float(sum((value - mean) ** 2 for value in values) / len(values))
    return float(math.sqrt(variance))


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(right - left)


def _delta_abs(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(abs(right) - abs(left))


def _join_names(names: list[str]) -> str:
    clean = [str(name).strip() for name in names if str(name).strip()]
    return " | ".join(dict.fromkeys(clean))


def _series_median(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = [_safe_float(value) for value in df[column].tolist()]
    values = [value for value in values if value is not None]
    if not values:
        return None
    values.sort()
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _series_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns:
        return None
    values = [_safe_float(value) for value in df[column].tolist()]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _normalize_target_profile(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "default"


def _uplift_score(
    *,
    delta_rank_icir: float | None,
    delta_net_excess: float | None,
    delta_net_sharpe: float | None,
    delta_net_ann: float | None,
    delta_turnover: float | None,
) -> float | None:
    parts = (
        (0.25, delta_rank_icir),
        (0.35, delta_net_excess),
        (0.20, delta_net_sharpe),
        (0.20, delta_net_ann),
    )
    score = 0.0
    seen = False
    for weight, value in parts:
        if value is None:
            continue
        score += weight * value
        seen = True
    if delta_turnover is not None and delta_turnover > 0:
        score -= 0.15 * delta_turnover
        seen = True
    return float(score) if seen else None


def _collect_promotion_info(run_root: Path) -> dict[str, Any]:
    pending_names: list[str] = []
    applied_names: list[str] = []
    changed_modules = 0

    for path in run_root.rglob("pending_curated_manifest.json"):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        for entry in list(payload.get("entries", []) or []):
            name = str(entry.get("suggested_registry_name", "") or entry.get("factor_name", "")).strip()
            if name:
                pending_names.append(name)

    for path in run_root.rglob("auto_applied_promotion.json"):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        records = payload if isinstance(payload, list) else [payload]
        for item in records:
            if str(item.get("changed", "")).lower() == "true" or bool(item.get("changed")):
                changed_modules += 1
            for entry in list(item.get("entries", []) or []):
                name = str(entry).strip()
                if name:
                    applied_names.append(name)

    pending_names = list(dict.fromkeys(pending_names))
    applied_names = list(dict.fromkeys(applied_names))
    promotion_names = list(dict.fromkeys([*applied_names, *pending_names]))
    return {
        "promotion_pending_names": pending_names,
        "promotion_pending_count": len(pending_names),
        "formalized_applied_names": applied_names,
        "formalized_applied_count": len(applied_names),
        "promotion_candidate_names": promotion_names,
        "promotion_candidate_count": len(promotion_names),
        "auto_apply_changed_modules": changed_modules,
    }


def _extract_failed_checks(row: dict[str, Any]) -> list[str]:
    payload = row.get("promotion_checks") or {}
    checks = payload.get("checks") or []
    failed: list[str] = []
    for item in checks:
        if bool(item.get("pass")):
            continue
        check_id = str(item.get("id", "")).strip()
        if check_id and check_id != "hard_gate_summary":
            failed.append(check_id)
    return failed


def build_latest_admission_index(autofactorset_runs_dir: str | Path = DEFAULT_AUTOFACTORSET_RUNS_DIR) -> dict[str, dict[str, Any]]:
    root = Path(autofactorset_runs_dir)
    latest: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return latest

    for results_path in root.rglob("results.json"):
        try:
            payload = _read_json(results_path)
        except Exception:
            continue
        stamp = results_path.stat().st_mtime
        for row in list(payload.get("rows", []) or []):
            factor_name = str(row.get("factor_name", "")).strip()
            if not factor_name:
                continue
            row_payload = dict(row)
            row_payload["_source_results_path"] = str(results_path)
            row_payload["_observed_at"] = stamp
            row_payload["_failed_checks"] = _extract_failed_checks(row_payload)
            previous = latest.get(factor_name)
            if previous is None or float(previous.get("_observed_at", 0.0)) <= stamp:
                latest[factor_name] = row_payload
    return latest


def _collect_admission_names(
    *,
    candidate_names: list[str],
    admission_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    promoted_names: list[str] = []
    inserted_names: list[str] = []
    failed_counter: Counter[str] = Counter()

    for name in candidate_names:
        row = admission_index.get(name)
        if row is None:
            continue
        if bool(row.get("promoted")):
            promoted_names.append(name)
        if bool(row.get("inserted_into_library")):
            inserted_names.append(name)
        for failed in list(row.get("_failed_checks", []) or []):
            failed_counter[failed] += 1

    promoted_names = list(dict.fromkeys(promoted_names))
    inserted_names = list(dict.fromkeys(inserted_names))
    return {
        "promoted_names": promoted_names,
        "promoted_count": len(promoted_names),
        "inserted_names": inserted_names,
        "inserted_count": len(inserted_names),
        "failed_check_counter": dict(failed_counter),
    }


def _attach_metric_block(record: dict[str, Any], prefix: str, row: dict[str, Any] | None) -> None:
    payload = _normalize_row(row)
    for key, label in METRIC_RENAME.items():
        record[f"{prefix}_{label}"] = payload.get(key)
    rank_ic = payload.get("quick_rank_ic_mean")
    rank_icir = payload.get("quick_rank_icir")
    record[f"{prefix}_abs_rank_ic"] = abs(rank_ic) if rank_ic is not None else None
    record[f"{prefix}_abs_rank_icir"] = abs(rank_icir) if rank_icir is not None else None
    record[f"{prefix}_decision"] = payload.get("decision")
    record[f"{prefix}_status"] = payload.get("status")
    record[f"{prefix}_factor_name"] = payload.get("factor_name")


def _attach_delta_block(record: dict[str, Any], prefix: str, left: dict[str, Any] | None, right: dict[str, Any] | None) -> None:
    left_payload = _normalize_row(left)
    right_payload = _normalize_row(right)
    for key, label in METRIC_RENAME.items():
        out_label = "turnover" if label == "mean_turnover" and prefix.endswith("top3_mean") else label
        record[f"{prefix}_{out_label}"] = _delta(left_payload.get(key), right_payload.get(key))
    record[f"{prefix}_abs_rank_ic"] = _delta_abs(left_payload.get("quick_rank_ic_mean"), right_payload.get("quick_rank_ic_mean"))
    record[f"{prefix}_abs_rank_icir"] = _delta_abs(left_payload.get("quick_rank_icir"), right_payload.get("quick_rank_icir"))


def _top3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    top3 = _top_k_candidates(rows, k=3)
    payload: dict[str, Any] = {
        "top3_factor_names": [str(row.get("factor_name", "")).strip() for row in top3 if str(row.get("factor_name", "")).strip()],
        "top3_count": len(top3),
    }
    for key, label in METRIC_RENAME.items():
        out_label = "turnover" if label == "mean_turnover" else label
        payload[f"top3_mean_{out_label}"] = _mean_metric(top3, key)
    payload["top3_mean_abs_rank_ic"] = _mean_metric(
        [{"abs_rank_ic": abs(_safe_float(row.get("quick_rank_ic_mean")) or 0.0)} for row in top3],
        "abs_rank_ic",
    )
    payload["top3_mean_abs_rank_icir"] = _mean_metric(
        [{"abs_rank_icir": abs(_safe_float(row.get("quick_rank_icir")) or 0.0)} for row in top3],
        "abs_rank_icir",
    )
    payload["top3_positive_rank_ic_rate"] = _positive_rate(top3, "quick_rank_ic_mean")
    payload["top3_positive_rank_icir_rate"] = _positive_rate(top3, "quick_rank_icir")
    payload["top3_positive_net_excess_ann_return_rate"] = _positive_rate(top3, "net_excess_ann_return")
    payload["top3_positive_net_sharpe_rate"] = _positive_rate(top3, "net_sharpe")
    payload["top3_std_rank_ic"] = _metric_std(top3, "quick_rank_ic_mean")
    payload["top3_std_rank_icir"] = _metric_std(top3, "quick_rank_icir")
    payload["top3_std_net_excess_ann_return"] = _metric_std(top3, "net_excess_ann_return")
    payload["top3_std_net_sharpe"] = _metric_std(top3, "net_sharpe")
    return payload


def _record_has_research_metrics(record: dict[str, Any]) -> bool:
    if _safe_int(record.get("evaluated_candidates")) > 0:
        return True
    metric_keys = (
        "delta_seed_to_winner_rank_ic",
        "delta_seed_to_winner_rank_icir",
        "delta_seed_to_winner_net_ann_return",
        "delta_seed_to_winner_net_excess_ann_return",
        "delta_seed_to_winner_net_sharpe",
        "delta_seed_to_winner_abs_rank_ic",
        "delta_seed_to_winner_abs_rank_icir",
        "delta_seed_to_best_candidate_rank_ic",
        "delta_seed_to_best_candidate_rank_icir",
        "delta_seed_to_best_candidate_net_ann_return",
        "delta_seed_to_best_candidate_net_excess_ann_return",
        "delta_seed_to_best_candidate_net_sharpe",
        "delta_seed_to_top3_mean_net_excess_ann_return",
        "delta_seed_to_top3_mean_net_sharpe",
        "delta_seed_to_top3_mean_rank_ic",
        "delta_seed_to_top3_mean_rank_icir",
        "uplift_score",
    )
    return any(record.get(key) is not None for key in metric_keys)


def _load_child_run_summaries(run_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in run_root.rglob("summary.json"):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if payload.get("run_id") and isinstance(payload.get("children"), list):
            payload["_summary_path"] = str(path)
            out.append(payload)
    return out


def _build_record_from_single_run(
    *,
    summary_path: Path,
    family_config: Any,
    admission_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    run_dir = Path(str(summary.get("run_dir") or summary_path.parent))
    eval_df = _load_eval_df(run_dir)
    canonical_seed = str(family_config.canonical_seed)
    preferred_seed = str(resolve_preferred_refine_seed(family_config))
    selected_parent = dict(summary.get("selected_parent") or {})
    selected_parent_name = str(selected_parent.get("factor_name", "")).strip() or preferred_seed
    winner = _normalize_row(summary.get("winner") or {})
    candidate_rows = _candidate_rows(eval_df)
    best_candidate = _top_k_candidates(candidate_rows, k=1)
    best_candidate = best_candidate[0] if best_candidate else winner
    top3 = _top3_summary(candidate_rows)
    promotion_info = _collect_promotion_info(run_dir)
    admission_info = _collect_admission_names(
        candidate_names=promotion_info["promotion_candidate_names"],
        admission_index=admission_index,
    )

    seed_row = _find_factor_row(eval_df, canonical_seed)
    preferred_seed_row = _find_factor_row(eval_df, preferred_seed)
    sign_aware_seed_row = preferred_seed_row or seed_row
    parent_row = _find_parent_row(eval_df, selected_parent_name) or _normalize_row(selected_parent)

    record: dict[str, Any] = {
        "run_kind": "single",
        "family": str(summary.get("family", "")),
        "target_profile": str(summary.get("target_profile", "")),
        "run_path": str(run_dir),
        "summary_path": str(summary_path),
        "completed": True,
        "canonical_seed": canonical_seed,
        "preferred_refine_seed": preferred_seed,
        "selected_parent_factor_name": selected_parent_name,
        "winner_factor_name": str(winner.get("factor_name", "")).strip(),
        "best_candidate_factor_name": str(best_candidate.get("factor_name", "")).strip(),
        "evaluated_candidates": len(candidate_rows),
        "keep_count": sum(1 for row in candidate_rows if str(row.get("decision", "")).strip().lower() in {"research_keep", "keep", "research_winner", "winner", "research_keep_exploratory"}),
        "winner_count": sum(1 for row in candidate_rows if str(row.get("decision", "")).strip().lower() in {"research_winner", "winner"}),
        "promotion_pending_count": promotion_info["promotion_pending_count"],
        "formalized_applied_count": promotion_info["formalized_applied_count"],
        "promotion_candidate_count": promotion_info["promotion_candidate_count"],
        "promoted_count": admission_info["promoted_count"],
        "inserted_count": admission_info["inserted_count"],
        "promotion_candidate_names": _join_names(promotion_info["promotion_candidate_names"]),
        "formalized_applied_names": _join_names(promotion_info["formalized_applied_names"]),
        "promoted_names": _join_names(admission_info["promoted_names"]),
        "inserted_names": _join_names(admission_info["inserted_names"]),
        "top3_factor_names": _join_names(top3["top3_factor_names"]),
        "top3_count": _safe_int(top3.get("top3_count")),
    }
    record["keep_rate"] = (record["keep_count"] / record["evaluated_candidates"]) if int(record["evaluated_candidates"]) > 0 else None
    record["winner_rate"] = (record["winner_count"] / record["evaluated_candidates"]) if int(record["evaluated_candidates"]) > 0 else None
    _attach_metric_block(record, "seed", seed_row)
    _attach_metric_block(record, "preferred_seed", preferred_seed_row or seed_row)
    _attach_metric_block(record, "sign_aware_seed", sign_aware_seed_row)
    _attach_metric_block(record, "parent", parent_row)
    _attach_metric_block(record, "winner", winner)
    _attach_metric_block(record, "best_candidate", best_candidate)
    for key, value in top3.items():
        if key.startswith("top3_mean_") or key.startswith("top3_positive_") or key.startswith("top3_std_"):
            record[key] = value
    _attach_delta_block(record, "delta_seed_to_winner", seed_row, winner)
    _attach_delta_block(record, "delta_sign_aware_seed_to_winner", sign_aware_seed_row, winner)
    _attach_delta_block(record, "delta_parent_to_winner", parent_row, winner)
    _attach_delta_block(record, "delta_seed_to_best_candidate", seed_row, best_candidate)
    _attach_delta_block(record, "delta_sign_aware_seed_to_best_candidate", sign_aware_seed_row, best_candidate)

    top3_row = {
        "quick_rank_ic_mean": top3.get("top3_mean_rank_ic"),
        "quick_rank_icir": top3.get("top3_mean_rank_icir"),
        "net_ann_return": top3.get("top3_mean_net_ann_return"),
        "net_excess_ann_return": top3.get("top3_mean_net_excess_ann_return"),
        "net_sharpe": top3.get("top3_mean_net_sharpe"),
        "mean_turnover": top3.get("top3_mean_turnover"),
    }
    _attach_delta_block(record, "delta_seed_to_top3_mean", seed_row, top3_row)
    _attach_delta_block(record, "delta_sign_aware_seed_to_top3_mean", sign_aware_seed_row, top3_row)
    record["uplift_score"] = _uplift_score(
        delta_rank_icir=record.get("delta_seed_to_winner_rank_icir"),
        delta_net_excess=record.get("delta_seed_to_winner_net_excess_ann_return"),
        delta_net_sharpe=record.get("delta_seed_to_winner_net_sharpe"),
        delta_net_ann=record.get("delta_seed_to_winner_net_ann_return"),
        delta_turnover=record.get("delta_seed_to_winner_mean_turnover"),
    )
    record["sign_aware_uplift_score"] = _uplift_score(
        delta_rank_icir=record.get("delta_sign_aware_seed_to_winner_rank_icir"),
        delta_net_excess=record.get("delta_sign_aware_seed_to_winner_net_excess_ann_return"),
        delta_net_sharpe=record.get("delta_sign_aware_seed_to_winner_net_sharpe"),
        delta_net_ann=record.get("delta_sign_aware_seed_to_winner_net_ann_return"),
        delta_turnover=record.get("delta_sign_aware_seed_to_winner_mean_turnover"),
    )
    record["search_efficiency_score"] = (
        record["uplift_score"] / record["evaluated_candidates"]
        if record.get("uplift_score") is not None and int(record["evaluated_candidates"]) > 0
        else None
    )
    record["sign_aware_search_efficiency_score"] = (
        record["sign_aware_uplift_score"] / record["evaluated_candidates"]
        if record.get("sign_aware_uplift_score") is not None and int(record["evaluated_candidates"]) > 0
        else None
    )
    return record


def _build_record_from_scheduler_run(
    *,
    summary_path: Path,
    family_config: Any,
    admission_index: dict[str, dict[str, Any]],
    include_incomplete: bool = False,
) -> dict[str, Any] | None:
    summary = _read_json(summary_path)
    if not include_incomplete and summary.get("stop_reason") in (None, ""):
        return None
    scheduler_dir = Path(str(summary.get("scheduler_dir") or summary_path.parent))
    plan_path = scheduler_dir / "plan.json"
    plan = _read_json(plan_path) if plan_path.exists() else {}
    canonical_seed = str(family_config.canonical_seed)
    preferred_seed = str(resolve_preferred_refine_seed(family_config))
    initial_parent = dict(plan.get("initial_parent") or {})
    selected_parent_name = str(summary.get("last_selected_parent_name", "")).strip() or str(initial_parent.get("factor_name", "")).strip() or preferred_seed
    latest_winner = _normalize_row(summary.get("latest_winner") or {})

    child_summaries = _load_child_run_summaries(scheduler_dir)
    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for child in child_summaries:
        run_dir = Path(str(child.get("run_dir") or ""))
        if not run_dir:
            continue
        df = _load_eval_df(run_dir)
        if df.empty:
            continue
        for _, row in df.iterrows():
            payload = _normalize_row(row)
            if str(payload.get("role", "")) == "candidate":
                candidate_rows.append(payload)
            else:
                baseline_rows.append(payload)

    baseline_df = pd.DataFrame(baseline_rows)
    seed_row = _find_factor_row(baseline_df, canonical_seed)
    preferred_seed_row = _find_factor_row(baseline_df, preferred_seed)
    sign_aware_seed_row = preferred_seed_row or seed_row
    parent_row = _find_parent_row(baseline_df, selected_parent_name) or _normalize_row(initial_parent)
    best_candidate = _top_k_candidates(candidate_rows, k=1)
    best_candidate = best_candidate[0] if best_candidate else latest_winner
    top3 = _top3_summary(candidate_rows)
    promotion_info = _collect_promotion_info(scheduler_dir)
    admission_info = _collect_admission_names(
        candidate_names=promotion_info["promotion_candidate_names"],
        admission_index=admission_index,
    )

    record: dict[str, Any] = {
        "run_kind": "scheduler",
        "family": str(summary.get("family", "")),
        "target_profile": str(summary.get("target_profile", "")),
        "run_path": str(scheduler_dir),
        "summary_path": str(summary_path),
        "completed": bool(summary.get("stop_reason")),
        "canonical_seed": canonical_seed,
        "preferred_refine_seed": preferred_seed,
        "selected_parent_factor_name": selected_parent_name,
        "winner_factor_name": str(latest_winner.get("factor_name", "")).strip(),
        "best_candidate_factor_name": str(best_candidate.get("factor_name", "")).strip(),
        "rounds_completed": _safe_int(summary.get("rounds_completed")),
        "stop_reason": str(summary.get("stop_reason", "") or ""),
        "evaluated_candidates": len(candidate_rows),
        "keep_count": sum(1 for row in candidate_rows if str(row.get("decision", "")).strip().lower() in {"research_keep", "keep", "research_winner", "winner", "research_keep_exploratory"}),
        "winner_count": sum(1 for row in candidate_rows if str(row.get("decision", "")).strip().lower() in {"research_winner", "winner"}),
        "promotion_pending_count": promotion_info["promotion_pending_count"],
        "formalized_applied_count": promotion_info["formalized_applied_count"],
        "promotion_candidate_count": promotion_info["promotion_candidate_count"],
        "promoted_count": admission_info["promoted_count"],
        "inserted_count": admission_info["inserted_count"],
        "promotion_candidate_names": _join_names(promotion_info["promotion_candidate_names"]),
        "formalized_applied_names": _join_names(promotion_info["formalized_applied_names"]),
        "promoted_names": _join_names(admission_info["promoted_names"]),
        "inserted_names": _join_names(admission_info["inserted_names"]),
        "top3_factor_names": _join_names(top3["top3_factor_names"]),
        "top3_count": _safe_int(top3.get("top3_count")),
    }
    record["keep_rate"] = (record["keep_count"] / record["evaluated_candidates"]) if int(record["evaluated_candidates"]) > 0 else None
    record["winner_rate"] = (record["winner_count"] / record["evaluated_candidates"]) if int(record["evaluated_candidates"]) > 0 else None
    _attach_metric_block(record, "seed", seed_row)
    _attach_metric_block(record, "preferred_seed", preferred_seed_row or seed_row)
    _attach_metric_block(record, "sign_aware_seed", sign_aware_seed_row)
    _attach_metric_block(record, "parent", parent_row)
    _attach_metric_block(record, "winner", latest_winner)
    _attach_metric_block(record, "best_candidate", best_candidate)
    for key, value in top3.items():
        if key.startswith("top3_mean_") or key.startswith("top3_positive_") or key.startswith("top3_std_"):
            record[key] = value
    _attach_delta_block(record, "delta_seed_to_winner", seed_row, latest_winner)
    _attach_delta_block(record, "delta_sign_aware_seed_to_winner", sign_aware_seed_row, latest_winner)
    _attach_delta_block(record, "delta_parent_to_winner", parent_row, latest_winner)
    _attach_delta_block(record, "delta_seed_to_best_candidate", seed_row, best_candidate)
    _attach_delta_block(record, "delta_sign_aware_seed_to_best_candidate", sign_aware_seed_row, best_candidate)
    top3_row = {
        "quick_rank_ic_mean": top3.get("top3_mean_rank_ic"),
        "quick_rank_icir": top3.get("top3_mean_rank_icir"),
        "net_ann_return": top3.get("top3_mean_net_ann_return"),
        "net_excess_ann_return": top3.get("top3_mean_net_excess_ann_return"),
        "net_sharpe": top3.get("top3_mean_net_sharpe"),
        "mean_turnover": top3.get("top3_mean_turnover"),
    }
    _attach_delta_block(record, "delta_seed_to_top3_mean", seed_row, top3_row)
    _attach_delta_block(record, "delta_sign_aware_seed_to_top3_mean", sign_aware_seed_row, top3_row)
    record["uplift_score"] = _uplift_score(
        delta_rank_icir=record.get("delta_seed_to_winner_rank_icir"),
        delta_net_excess=record.get("delta_seed_to_winner_net_excess_ann_return"),
        delta_net_sharpe=record.get("delta_seed_to_winner_net_sharpe"),
        delta_net_ann=record.get("delta_seed_to_winner_net_ann_return"),
        delta_turnover=record.get("delta_seed_to_winner_mean_turnover"),
    )
    record["sign_aware_uplift_score"] = _uplift_score(
        delta_rank_icir=record.get("delta_sign_aware_seed_to_winner_rank_icir"),
        delta_net_excess=record.get("delta_sign_aware_seed_to_winner_net_excess_ann_return"),
        delta_net_sharpe=record.get("delta_sign_aware_seed_to_winner_net_sharpe"),
        delta_net_ann=record.get("delta_sign_aware_seed_to_winner_net_ann_return"),
        delta_turnover=record.get("delta_sign_aware_seed_to_winner_mean_turnover"),
    )
    record["search_efficiency_score"] = (
        record["uplift_score"] / record["evaluated_candidates"]
        if record.get("uplift_score") is not None and int(record["evaluated_candidates"]) > 0
        else None
    )
    record["sign_aware_search_efficiency_score"] = (
        record["sign_aware_uplift_score"] / record["evaluated_candidates"]
        if record.get("sign_aware_uplift_score") is not None and int(record["evaluated_candidates"]) > 0
        else None
    )
    return record


def build_run_uplift_records(
    *,
    seed_pool_path: str | Path = DEFAULT_SEED_POOL,
    single_runs_dir: str | Path = DEFAULT_SINGLE_RUNS_DIR,
    scheduler_runs_dir: str | Path = DEFAULT_SCHEDULER_RUNS_DIR,
    autofactorset_runs_dir: str | Path = DEFAULT_AUTOFACTORSET_RUNS_DIR,
    family: str = "",
    include_incomplete: bool = False,
) -> list[dict[str, Any]]:
    seed_pool = load_seed_pool(seed_pool_path)
    family_map = {item.family: item for item in seed_pool.families}
    admission_index = build_latest_admission_index(autofactorset_runs_dir)
    records: list[dict[str, Any]] = []

    single_root = Path(single_runs_dir)
    if single_root.exists():
        for summary_path in sorted(single_root.glob("*/summary.json")):
            try:
                payload = _read_json(summary_path)
            except Exception:
                continue
            family_name = str(payload.get("family", "")).strip()
            if family and family_name != family:
                continue
            family_config = family_map.get(family_name)
            if family_config is None:
                continue
            records.append(
                _build_record_from_single_run(
                    summary_path=summary_path,
                    family_config=family_config,
                    admission_index=admission_index,
                )
            )

    scheduler_root = Path(scheduler_runs_dir)
    if scheduler_root.exists():
        seen_scheduler_summaries: set[str] = set()
        for summary_path in sorted(scheduler_root.rglob("summary.json")):
            try:
                payload = _read_json(summary_path)
            except Exception:
                continue
            if not _is_scheduler_top_summary(payload):
                continue
            summary_key = str(summary_path.resolve())
            if summary_key in seen_scheduler_summaries:
                continue
            seen_scheduler_summaries.add(summary_key)
            family_name = str(payload.get("family", "")).strip()
            if family and family_name != family:
                continue
            family_config = family_map.get(family_name)
            if family_config is None:
                continue
            record = _build_record_from_scheduler_run(
                summary_path=summary_path,
                family_config=family_config,
                admission_index=admission_index,
                include_incomplete=include_incomplete,
            )
            if record is not None and _record_has_research_metrics(record):
                records.append(record)

    records.sort(key=lambda item: str(item.get("run_path", "")))
    return records


def _family_admission_summary(admission_index: dict[str, dict[str, Any]], family: str) -> dict[str, Any]:
    rows = [row for row in admission_index.values() if str(row.get("family", "")).strip() == family]
    evaluated = len(rows)
    promoted = sum(1 for row in rows if bool(row.get("promoted")))
    inserted = sum(1 for row in rows if bool(row.get("inserted_into_library")))
    failed_counter: Counter[str] = Counter()
    blocked_rows = 0
    for row in rows:
        failed = list(row.get("_failed_checks", []) or [])
        if failed:
            blocked_rows += 1
        for item in failed:
            failed_counter[item] += 1
    corr_blocks = failed_counter.get("corr_with_library_max", 0)
    turnover_blocks = failed_counter.get("daily_turnover", 0)
    return {
        "admission_evaluated_count": evaluated,
        "admission_promoted_count": promoted,
        "admission_inserted_count": inserted,
        "corr_block_count": corr_blocks,
        "corr_block_rate": (corr_blocks / blocked_rows) if blocked_rows else None,
        "turnover_block_count": turnover_blocks,
        "turnover_block_rate": (turnover_blocks / blocked_rows) if blocked_rows else None,
        "failed_check_counter": dict(failed_counter),
    }


def _summarize_record_group(df: pd.DataFrame, *, family: str, profile_label: str | None = None) -> dict[str, Any]:
    sign_aware_median_uplift = _series_median(df, "sign_aware_uplift_score")
    sign_aware_median_delta_excess = _series_median(df, "delta_sign_aware_seed_to_winner_net_excess_ann_return")
    sign_aware_median_delta_sharpe = _series_median(df, "delta_sign_aware_seed_to_winner_net_sharpe")
    sign_aware_median_delta_icir = _series_median(df, "delta_sign_aware_seed_to_winner_rank_icir")
    sign_aware_top3_mean_delta_excess = _series_mean(df, "delta_sign_aware_seed_to_top3_mean_net_excess_ann_return")
    sign_aware_top3_mean_delta_sharpe = _series_mean(df, "delta_sign_aware_seed_to_top3_mean_net_sharpe")

    canonical_median_uplift = _series_median(df, "uplift_score")
    canonical_median_delta_excess = _series_median(df, "delta_seed_to_winner_net_excess_ann_return")
    canonical_median_delta_sharpe = _series_median(df, "delta_seed_to_winner_net_sharpe")
    canonical_median_delta_icir = _series_median(df, "delta_seed_to_winner_rank_icir")
    canonical_top3_mean_delta_excess = _series_mean(df, "delta_seed_to_top3_mean_net_excess_ann_return")
    canonical_top3_mean_delta_sharpe = _series_mean(df, "delta_seed_to_top3_mean_net_sharpe")

    formalized_count = int(df["formalized_applied_count"].fillna(0).sum()) if "formalized_applied_count" in df.columns else 0
    total_evaluated = int(df["evaluated_candidates"].fillna(0).sum()) if "evaluated_candidates" in df.columns else 0
    total_keep = int(df["keep_count"].fillna(0).sum()) if "keep_count" in df.columns else 0
    total_winner = int(df["winner_count"].fillna(0).sum()) if "winner_count" in df.columns else 0

    if sign_aware_median_delta_excess is not None and sign_aware_median_delta_excess > 0.5 and formalized_count >= 3:
        recommended_status = "donor_mode"
    elif sign_aware_median_uplift is not None and sign_aware_median_uplift > 0 and formalized_count > 0:
        recommended_status = "keep_refining"
    elif sign_aware_median_delta_excess is not None and sign_aware_median_delta_excess > 0:
        recommended_status = "focused_refine_only"
    else:
        recommended_status = "pause"

    payload = {
        "family": family,
        "canonical_seed": str(df["canonical_seed"].dropna().iloc[0]) if "canonical_seed" in df.columns and not df["canonical_seed"].dropna().empty else "",
        "preferred_refine_seed": str(df["preferred_refine_seed"].dropna().iloc[0]) if "preferred_refine_seed" in df.columns and not df["preferred_refine_seed"].dropna().empty else "",
        "run_count": int(len(df)),
        "target_profiles": _join_names(sorted({_normalize_target_profile(item) for item in df["target_profile"].dropna().tolist()})) if "target_profile" in df.columns else "",
        "total_evaluated_candidates": total_evaluated,
        "total_keep_count": total_keep,
        "total_winner_count": total_winner,
        "mean_evaluated_candidates": _series_mean(df, "evaluated_candidates"),
        "mean_keep_count": _series_mean(df, "keep_count"),
        "mean_winner_count": _series_mean(df, "winner_count"),
        "mean_keep_rate": _series_mean(df, "keep_rate"),
        "mean_winner_rate": _series_mean(df, "winner_rate"),
        "mean_top3_count": _series_mean(df, "top3_count"),
        "mean_top3_positive_rank_icir_rate": _series_mean(df, "top3_positive_rank_icir_rate"),
        "mean_top3_positive_net_excess_ann_return_rate": _series_mean(df, "top3_positive_net_excess_ann_return_rate"),
        "mean_top3_positive_net_sharpe_rate": _series_mean(df, "top3_positive_net_sharpe_rate"),
        "mean_top3_std_rank_icir": _series_mean(df, "top3_std_rank_icir"),
        "mean_top3_std_net_excess_ann_return": _series_mean(df, "top3_std_net_excess_ann_return"),
        "mean_top3_std_net_sharpe": _series_mean(df, "top3_std_net_sharpe"),
        "formalized_count": formalized_count,
        "sign_aware_median_uplift_score": sign_aware_median_uplift,
        "sign_aware_median_delta_excess": sign_aware_median_delta_excess,
        "sign_aware_median_delta_sharpe": sign_aware_median_delta_sharpe,
        "sign_aware_median_delta_rank_icir": sign_aware_median_delta_icir,
        "sign_aware_top3_mean_delta_excess": sign_aware_top3_mean_delta_excess,
        "sign_aware_top3_mean_delta_sharpe": sign_aware_top3_mean_delta_sharpe,
        "canonical_median_uplift_score": canonical_median_uplift,
        "canonical_median_delta_excess": canonical_median_delta_excess,
        "canonical_median_delta_sharpe": canonical_median_delta_sharpe,
        "canonical_median_delta_rank_icir": canonical_median_delta_icir,
        "canonical_top3_mean_delta_excess": canonical_top3_mean_delta_excess,
        "canonical_top3_mean_delta_sharpe": canonical_top3_mean_delta_sharpe,
        "recommended_status": recommended_status,
    }
    if profile_label is not None:
        payload["target_profile"] = profile_label
    return payload


def summarize_family_funnel(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not run_records:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        grouped[str(record.get("family", ""))].append(record)

    out: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        df = pd.DataFrame(rows)
        out.append(_summarize_record_group(df, family=family))
    return out


def summarize_family_profile_funnel(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not run_records:
        return []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in run_records:
        family = str(record.get("family", ""))
        profile = _normalize_target_profile(record.get("target_profile"))
        grouped[(family, profile)].append(record)

    out: list[dict[str, Any]] = []
    for (family, profile), rows in sorted(grouped.items()):
        df = pd.DataFrame(rows)
        out.append(_summarize_record_group(df, family=family, profile_label=profile))
    return out


def _clean_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_research_funnel_outputs(
    *,
    run_records: list[dict[str, Any]],
    family_records: list[dict[str, Any]],
    family_profile_records: list[dict[str, Any]],
    output_dir: str | Path = DEFAULT_EVALUATOR_OUTPUT_DIR,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    run_df = pd.DataFrame(run_records)
    family_df = pd.DataFrame(family_records)

    run_csv = output_root / "run_uplift_summary.csv"
    run_json = output_root / "run_uplift_summary.json"
    family_csv = output_root / "family_funnel_summary.csv"
    family_json = output_root / "family_funnel_summary.json"
    family_profile_csv = output_root / "family_profile_funnel_summary.csv"
    family_profile_json = output_root / "family_profile_funnel_summary.json"
    report_md = output_root / "research_funnel_report.md"

    if not run_df.empty:
        run_df.to_csv(run_csv, index=False)
    else:
        run_csv.write_text("", encoding="utf-8")
    run_json.write_text(json.dumps(_clean_jsonable(run_records), ensure_ascii=False, indent=2), encoding="utf-8")

    if not family_df.empty:
        family_df.to_csv(family_csv, index=False)
    else:
        family_csv.write_text("", encoding="utf-8")
    family_json.write_text(json.dumps(_clean_jsonable(family_records), ensure_ascii=False, indent=2), encoding="utf-8")

    family_profile_df = pd.DataFrame(family_profile_records)
    if not family_profile_df.empty:
        family_profile_df.to_csv(family_profile_csv, index=False)
    else:
        family_profile_csv.write_text("", encoding="utf-8")
    family_profile_json.write_text(json.dumps(_clean_jsonable(family_profile_records), ensure_ascii=False, indent=2), encoding="utf-8")

    report_md.write_text(
        render_research_funnel_markdown(
            run_records=run_records,
            family_records=family_records,
            family_profile_records=family_profile_records,
        ),
        encoding="utf-8",
    )
    return {
        "run_csv": run_csv,
        "run_json": run_json,
        "family_csv": family_csv,
        "family_json": family_json,
        "family_profile_csv": family_profile_csv,
        "family_profile_json": family_profile_json,
        "report_md": report_md,
    }


def _fmt(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "-"
    return f"{num:.4f}"


def render_research_funnel_markdown(
    *,
    run_records: list[dict[str, Any]],
    family_records: list[dict[str, Any]],
    family_profile_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# LLM Refine Research Funnel",
        "",
        "这份评估默认遵循：先看分项指标，再看总分；总分只作辅助，不单独决定好坏。",
        "",
        "当前主口径以 `sign-aware uplift` 为准：",
        "- canonical seed 保留原始方向，方便回溯文献/注册来源",
        "- preferred refine seed 代表当前系统默认 refine 方向",
        "- family 总结与 profile 分层总结优先看 sign-aware 指标",
        "",
        "admission/library 本版先不作为核心评判标准，避免把未定型的库门槛混入主评估。",
        "",
        "## Run Summary",
        "",
        f"- total_runs: `{len(run_records)}`",
        f"- total_families: `{len({str(item.get('family', '')) for item in run_records})}`",
        "",
    ]

    if run_records:
        top_runs = sorted(
            run_records,
            key=lambda item: (_safe_float(item.get("delta_sign_aware_seed_to_winner_net_excess_ann_return")) or float("-inf")),
            reverse=True,
        )[:10]
        lines.extend(
            [
                "### Top Runs by `delta_sign_aware_seed_to_winner_net_excess_ann_return`",
                "",
                "| Family | Run Kind | Target | Winner | SignAware ΔExcess | SignAware ΔSharpe | SignAware ΔICIR | SignAware Uplift | Formalized |",
                "|---|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in top_runs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("family", "")),
                        str(item.get("run_kind", "")),
                        _normalize_target_profile(item.get("target_profile")),
                        f"`{item.get('winner_factor_name', '')}`",
                        _fmt(item.get("delta_sign_aware_seed_to_winner_net_excess_ann_return")),
                        _fmt(item.get("delta_sign_aware_seed_to_winner_net_sharpe")),
                        _fmt(item.get("delta_sign_aware_seed_to_winner_rank_icir")),
                        _fmt(item.get("sign_aware_uplift_score")),
                        str(_safe_int(item.get("formalized_applied_count"))),
                    ]
                )
                + " |"
            )
        lines.append("")

    if family_records:
        lines.extend(
            [
                "## Family Summary",
                "",
                "| Family | Canonical Seed | Preferred Seed | Runs | SignAware Median ΔExcess | KeepRate | WinnerRate | Top3 +ExcessRate | Formalized | Status |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in sorted(
            family_records,
            key=lambda row: (_safe_float(row.get("sign_aware_median_delta_excess")) or float("-inf")),
            reverse=True,
        ):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("family", "")),
                        f"`{item.get('canonical_seed', '')}`",
                        f"`{item.get('preferred_refine_seed', '')}`",
                        str(_safe_int(item.get("run_count"))),
                        _fmt(item.get("sign_aware_median_delta_excess")),
                        _fmt(item.get("mean_keep_rate")),
                        _fmt(item.get("mean_winner_rate")),
                        _fmt(item.get("mean_top3_positive_net_excess_ann_return_rate")),
                        str(_safe_int(item.get("formalized_count"))),
                        str(item.get("recommended_status", "")),
                    ]
                )
                + " |"
            )
        lines.append("")

    if family_profile_records:
        lines.extend(
            [
                "## Family x Profile Summary",
                "",
                "| Family | Profile | Runs | SignAware Median ΔExcess | KeepRate | WinnerRate | Top3 +ExcessRate | Formalized | Status |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in sorted(
            family_profile_records,
            key=lambda row: (
                str(row.get("family", "")),
                -((_safe_float(row.get("sign_aware_median_delta_excess")) or float("-inf"))),
            ),
        ):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("family", "")),
                        str(item.get("target_profile", "")),
                        str(_safe_int(item.get("run_count"))),
                        _fmt(item.get("sign_aware_median_delta_excess")),
                        _fmt(item.get("mean_keep_rate")),
                        _fmt(item.get("mean_winner_rate")),
                        _fmt(item.get("mean_top3_positive_net_excess_ann_return_rate")),
                        str(_safe_int(item.get("formalized_count"))),
                        str(item.get("recommended_status", "")),
                    ]
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"
