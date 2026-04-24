'''Build a manual evaluation CSV for stage-transition advisory quality.

Samples scheduler summaries and emits labels, decisions, metrics, and review fields for human audit.
'''

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factors_store.llm_refine.config import DEFAULT_MULTI_SCHEDULER_RUNS_DIR, PROJECT_ROOT
from factors_store.llm_refine.search import EvaluationFeedback, FamilyState, RefinementAction
from factors_store.llm_refine.search.stage_transition import resolve_stage_transition_from_state


LABELS = (
    "continue_focused",
    "return_to_broad",
    "switch_to_complementarity",
    "confirmation",
    "terminate",
    "reopen_branch",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric(payload: dict[str, Any], key: str) -> str:
    value = _safe_float(payload.get(key))
    return "" if value != value else f"{value:.6g}"


def _summary_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(root.rglob("summary.json")):
        rel = path.relative_to(root)
        if "multi_runs" in rel.parts or "child_runs" in rel.parts:
            continue
        payload = _read_json(path)
        if isinstance(payload.get("rounds"), list) and payload.get("scheduler_dir"):
            paths.append(path)
    return paths


def _decision_to_label(decision: dict[str, Any]) -> str:
    action = str(decision.get("action", "") or "")
    next_stage = str(decision.get("next_stage", "") or "")
    target_bias = str(decision.get("target_profile_bias", "") or "")
    branch_reopen = list(decision.get("branch_reopen_candidates") or [])
    parent_bias = str(decision.get("parent_selection_bias", "") or "")
    if next_stage == "terminate" or action in {"freeze_or_switch_family", "freeze_or_promote"}:
        return "terminate"
    if branch_reopen:
        return "reopen_branch"
    if action == "reopen_broad" and parent_bias in {"diversify_branch", "low_corr_parent"}:
        return "reopen_branch"
    if target_bias == "complementarity" or action == "decorrelate_or_reopen_branch":
        return "switch_to_complementarity"
    if next_stage == "confirmation" or action in {"confirm_and_freeze", "deployability_confirmation"}:
        return "confirmation"
    if action in {"continue_focused", "exploit_mainline"}:
        return "continue_focused"
    if next_stage == "broad_followup" or action in {"continue_broad_search", "reopen_broad", "repair_or_retry"}:
        return "return_to_broad"
    return ""


def _case_bucket(summary: dict[str, Any], decision: dict[str, Any]) -> str:
    stage = str(summary.get("stage_mode", "") or "")
    target = str(summary.get("target_profile", "") or "")
    stop = str(summary.get("stop_reason", "") or "")
    label = _decision_to_label(decision)
    if stage in {"auto", "new_family_broad", "broad_followup"}:
        if label == "return_to_broad":
            return "broad_should_continue"
        if label in {"continue_focused", "confirmation"}:
            return "broad_should_close"
    if stage == "focused_refine":
        if label == "continue_focused":
            return "focused_should_continue"
        if label == "switch_to_complementarity":
            return "focused_to_complementarity"
        if label == "return_to_broad":
            return "branch_reopen_or_return_broad"
        if label == "confirmation":
            return "confirmation_or_freeze"
    if target == "complementarity":
        return "complementarity_case"
    if stop in {"max_rounds", "no_new_search_improvement", "frontier_exhausted"}:
        return f"stop_{stop}"
    return "other"


def _build_state_action_feedback(summary: dict[str, Any]) -> tuple[FamilyState, RefinementAction, EvaluationFeedback]:
    rounds = list(summary.get("rounds") or [])
    last_round = dict(rounds[-1] or {}) if rounds else {}
    winner = dict(summary.get("last_round_best_candidate") or summary.get("last_round_winner") or {})
    keep = dict(summary.get("last_round_best_keep") or {})
    best_node = dict(summary.get("best_node") or {})
    search = dict(summary.get("search") or {})
    frontier_nodes = tuple(dict(item) for item in list(search.get("frontier") or [])[:8])
    state = FamilyState(
        family_id=str(summary.get("family", "") or ""),
        stage=str(last_round.get("child_stage_mode") or summary.get("stage_mode") or "auto"),
        target_profile=str(summary.get("target_profile") or "raw_alpha"),
        parent_set=tuple(dict(item) for item in list(last_round.get("selected_parents") or [])[:4]),
        best_node=best_node,
        frontier_nodes=frontier_nodes,
        redundancy_state={
            "has_decorrelation_targets": bool(
                dict(summary.get("orchestration_context_evidence") or {}).get("has_decorrelation_targets")
            )
        },
        failure_state={
            "high_corr_count": 0,
            "high_turnover_count": 0,
            "validation_fail_count": 0,
        },
        budget_state={
            "consecutive_no_improve": int(dict(search.get("budget") or {}).get("consecutive_no_improve") or 0),
            "children_collected": int(last_round.get("children_collected") or 0),
            "children_added_to_search": int(last_round.get("children_added_to_search") or 0),
            "budget_exhausted": str(summary.get("stop_reason") or "") == "max_rounds",
            "frontier_exhausted": str(summary.get("stop_reason") or "") == "frontier_exhausted",
        },
    )
    action = RefinementAction(
        stage_mode=state.stage,
        target_profile=state.target_profile,
        policy_preset=str(dict(summary.get("search", {}).get("policy") or {}).get("name") or ""),
        parent_selection=str(summary.get("last_selected_parent_name") or ""),
        decorrelation_targets=("__present__",) if state.redundancy_state.get("has_decorrelation_targets") else (),
        models=tuple(str(item) for item in list(dict(summary.get("search", {}).get("budget") or {}).get("models") or [])),
        n_candidates=int(dict(summary.get("prompt_trace") or {}).get("requested_candidate_count") or 0),
        max_rounds=int(summary.get("rounds_completed") or 0),
    )
    feedback = EvaluationFeedback(
        status=str(summary.get("last_round_status") or ""),
        search_improved=bool(last_round.get("search_improved")),
        winner=winner,
        keep=keep,
        best_anchor=dict(summary.get("anchor_selection", {}).get("best_anchor") or {}),
        passed_anchor_count=len(summary.get("anchor_selection", {}).get("passed_candidates") or []),
        focused_best_node=best_node,
        consecutive_no_improve=int(dict(search.get("budget") or {}).get("consecutive_no_improve") or 0),
        children_collected=int(last_round.get("children_collected") or 0),
        children_added_to_search=int(last_round.get("children_added_to_search") or 0),
        budget_exhausted=str(summary.get("stop_reason") or "") == "max_rounds",
        frontier_exhausted=str(summary.get("stop_reason") or "") == "frontier_exhausted",
    )
    return state, action, feedback


def _row(path: Path, root: Path) -> dict[str, Any] | None:
    summary = _read_json(path)
    if not summary:
        return None
    state, action, feedback = _build_state_action_feedback(summary)
    decision = resolve_stage_transition_from_state(state, action, feedback).to_dict()
    predicted = _decision_to_label(decision)
    winner = dict(summary.get("last_round_best_candidate") or summary.get("last_round_winner") or {})
    keep = dict(summary.get("last_round_best_keep") or {})
    best = dict(summary.get("best_node") or {})
    rounds = list(summary.get("rounds") or [])
    last_round = dict(rounds[-1] or {}) if rounds else {}
    return {
        "expected_label": "",
        "predicted_label": predicted,
        "decision_action": decision.get("action", ""),
        "decision_next_stage": decision.get("next_stage", ""),
        "decision_confidence": decision.get("confidence", ""),
        "decision_reason": decision.get("reason", ""),
        "decision_tags": "|".join(str(item) for item in decision.get("rationale_tags") or []),
        "case_bucket": _case_bucket(summary, decision),
        "family": summary.get("family", ""),
        "run_dir": str(path.parent),
        "relative_summary": str(path.relative_to(root)),
        "stage_mode": summary.get("stage_mode", ""),
        "target_profile": summary.get("target_profile", ""),
        "rounds_completed": summary.get("rounds_completed", ""),
        "stop_reason": summary.get("stop_reason", ""),
        "last_round_status": summary.get("last_round_status", ""),
        "search_improved": last_round.get("search_improved", ""),
        "children_collected": last_round.get("children_collected", ""),
        "children_added_to_search": last_round.get("children_added_to_search", ""),
        "successful_model_count": last_round.get("successful_model_count", ""),
        "failed_model_count": last_round.get("failed_model_count", ""),
        "selected_parent": summary.get("last_selected_parent_name", ""),
        "winner": winner.get("factor_name", ""),
        "winner_status": winner.get("status", ""),
        "winner_ic": _metric(winner, "quick_rank_ic_mean"),
        "winner_icir": _metric(winner, "quick_rank_icir"),
        "winner_excess": _metric(winner, "net_excess_ann_return"),
        "winner_sharpe": _metric(winner, "net_sharpe"),
        "winner_turnover": _metric(winner, "mean_turnover"),
        "keep": keep.get("factor_name", ""),
        "best_node": best.get("candidate_name") or best.get("factor_name", ""),
        "human_notes": "",
    }


def _balanced_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("case_bucket", "other"))].append(row)
    for values in buckets.values():
        values.sort(key=lambda item: str(item.get("relative_summary", "")), reverse=True)
    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets, key=lambda key: (-len(buckets[key]), key))
    while len(selected) < limit and any(buckets.values()):
        for key in bucket_names:
            if len(selected) >= limit:
                break
            if buckets[key]:
                selected.append(buckets[key].pop(0))
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small manual eval set for stage-transition advisory quality.")
    parser.add_argument("--runs-root", default=str(DEFAULT_MULTI_SCHEDULER_RUNS_DIR))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--family", action="append", default=[], help="optional family filter; may be repeated")
    parser.add_argument("--out", default="")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(args.runs_root).expanduser().resolve()
    families = {str(item).strip() for item in args.family if str(item).strip()}
    rows: list[dict[str, Any]] = []
    for path in _summary_paths(root):
        row = _row(path, root)
        if not row:
            continue
        if families and str(row.get("family", "")) not in families:
            continue
        rows.append(row)
    rows = _balanced_sample(rows, max(int(args.limit), 1))
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = (
        Path(args.out).expanduser().resolve()
        if str(args.out or "").strip()
        else PROJECT_ROOT / "artifacts" / "reports" / "stage_transition_eval" / f"stage_transition_eval_{ts}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "expected_label",
        "predicted_label",
        "decision_action",
        "decision_next_stage",
        "decision_confidence",
        "decision_reason",
        "decision_tags",
        "case_bucket",
        "family",
        "run_dir",
        "relative_summary",
        "stage_mode",
        "target_profile",
        "rounds_completed",
        "stop_reason",
        "last_round_status",
        "search_improved",
        "children_collected",
        "children_added_to_search",
        "successful_model_count",
        "failed_model_count",
        "selected_parent",
        "winner",
        "winner_status",
        "winner_ic",
        "winner_icir",
        "winner_excess",
        "winner_sharpe",
        "winner_turnover",
        "keep",
        "best_node",
        "human_notes",
    ]
    with out.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    print(f"rows={len(rows)}")
    print("labels=" + ",".join(LABELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
