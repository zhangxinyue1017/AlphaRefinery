'''Multi-model refinement orchestrator for one parent and one round.

Launches child model runs, collects evaluated candidates, reranks winners, and writes aggregate summaries.
'''

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from ..config import (
    DEFAULT_AUTO_APPLY_PROMOTION,
    DEFAULT_MAX_PARALLEL,
    DEFAULT_MULTI_RUNS_DIR,
    DEFAULT_POLICY_PRESET,
    DEFAULT_TARGET_PROFILE,
    PROJECT_ROOT,
)
from ..core.archive import (
    DEFAULT_ARCHIVE_DB,
    get_candidate_record,
    get_latest_family_round,
    get_latest_family_winner,
    make_seed_candidate_id,
)
from ..core.seed_loader import load_seed_pool, resolve_family_formula, resolve_preferred_refine_seed
from ..knowledge.round1 import build_bootstrap_frontier, is_seed_stage_node_kind, select_bootstrap_parent
from ..search import SearchBudget, SearchEngine, SearchPolicy, build_search_normalizer
from ..search.transition.context_resolver import ContextEvidence, resolve_context_profile
from ..search.decision.decorrelation_policy import (
    DecorrelationPolicy,
    assess_decorrelation,
    decorate_with_decorrelation_assessment,
    decorrelation_rerank_enabled,
)
from ..search.decision.context import DecisionContext
from ..search.decision.engine import DecisionEngine
from ..search.io.run_ingest import load_candidate_records_from_completed_runs
from ..search.core.scoring import safe_float
from .run_refine_loop import build_arg_parser as build_single_round_parser

_STAGE_MODE_CHOICES = (
    "auto",
    "new_family_broad",
    "broad_followup",
    "focused_refine",
    "confirmation",
    "donor_validation",
)


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _sanitize_slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("._-") or "model"


def _parse_models(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            model = part.strip()
            if model and model not in out:
                out.append(model)
    return out


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


def _decorrelation_rerank_enabled(records: list[dict[str, object]]) -> bool:
    return decorrelation_rerank_enabled(records)


def _decorrelation_quality_gate_passed(item: dict[str, object]) -> bool:
    policy = DecorrelationPolicy.from_search_policy(None)
    return bool(assess_decorrelation(dict(item), policy).quality_gate_passed)


def _base_rerank_quality_score(item: dict[str, object]) -> float:
    status = str(item.get("status", "")).strip().lower()
    status_bonus = 0.05 if status in {"research_winner", "winner"} else 0.02 if status in {"research_keep", "keep"} else 0.01 if status in {"research_keep_exploratory"} else 0.0
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


def _decorrelation_adjustment(item: dict[str, object]) -> float:
    policy = DecorrelationPolicy.from_search_policy(None)
    return safe_float(assess_decorrelation(dict(item), policy).rerank_adjustment, default=0.0)


def _decorate_with_decorrelation_rerank(item: dict[str, object]) -> dict[str, object]:
    out = dict(item)
    quality_score = _base_rerank_quality_score(out)
    policy = DecorrelationPolicy.from_search_policy(None)
    return decorate_with_decorrelation_assessment(out, policy, base_quality_score=quality_score)


def _decorrelation_rerank_sort_key(
    item: dict[str, object],
    *,
    stage_mode: str,
) -> tuple[float, float, float, float, float, float, float, float]:
    adjusted = safe_float(item.get("decorrelation_adjusted_score"), default=float("-inf"))
    quality = safe_float(item.get("decorrelation_quality_score"), default=float("-inf"))
    adjustment = safe_float(item.get("decorrelation_adjustment"), default=float("-inf"))
    if stage_mode == "new_family_broad":
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


def _global_new_family_broad_sort_key(item: dict[str, object]) -> tuple[float, float, float, float, float, float, float, float, float, float]:
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


def _build_global_rerank_preview(
    records: list[dict[str, object]],
    *,
    stage_mode: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    if not records:
        return []
    decorrelation_enabled = _decorrelation_rerank_enabled(records)
    working = [
        _decorate_with_decorrelation_rerank(item) if decorrelation_enabled else dict(item)
        for item in records
    ]
    if decorrelation_enabled:
        ranked = sorted(working, key=lambda item: _decorrelation_rerank_sort_key(item, stage_mode=stage_mode), reverse=True)
    elif stage_mode == "new_family_broad":
        ranked = sorted(working, key=_global_new_family_broad_sort_key, reverse=True)
    else:
        ranked = sorted(working, key=_default_child_sort_key, reverse=True)
    preview: list[dict[str, object]] = []
    for item in ranked[: max(int(limit), 0)]:
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
                "decorrelation_quality_score": safe_float(item.get("decorrelation_quality_score"), default=float("nan")),
                "decorrelation_grade": str(item.get("decorrelation_grade", "") or ""),
                "decorrelation_score": safe_float(item.get("decorrelation_score"), default=float("nan")),
                "decorrelation_gate_action": str(item.get("decorrelation_gate_action", "") or ""),
                "decorrelation_gate_reason": str(item.get("decorrelation_gate_reason", "") or ""),
                "decorrelation_winner_allowed": bool(item.get("decorrelation_winner_allowed", True)),
                "decorrelation_adjustment": safe_float(item.get("decorrelation_adjustment"), default=float("nan")),
                "decorrelation_adjusted_score": safe_float(
                    item.get("decorrelation_adjusted_score"), default=float("nan")
                ),
            }
        )
    return preview


def _pick_best_child_record(
    records: list[dict[str, object]],
    *,
    stage_mode: str = "auto",
) -> dict[str, object] | None:
    if not records:
        return None

    decorrelation_enabled = _decorrelation_rerank_enabled(records)
    working = [
        _decorate_with_decorrelation_rerank(item) if decorrelation_enabled else dict(item)
        for item in records
    ]
    if decorrelation_enabled:
        return max(working, key=lambda item: _decorrelation_rerank_sort_key(item, stage_mode=stage_mode))
    if stage_mode == "new_family_broad":
        return max(working, key=_global_new_family_broad_sort_key)
    return max(working, key=_default_child_sort_key)


def _pick_best_keep_record(
    records: list[dict[str, object]],
    *,
    stage_mode: str = "auto",
) -> dict[str, object] | None:
    keep_records = [
        item
        for item in records
        if str(item.get("status", "")).strip().lower() in {"research_keep", "keep", "research_keep_exploratory"}
    ]
    if not keep_records:
        return None
    return _pick_best_child_record(keep_records, stage_mode=stage_mode)


def _metric_snapshot(item: dict[str, object] | None) -> dict[str, object] | None:
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
        "decorrelation_grade": str(item.get("decorrelation_grade", "") or ""),
        "decorrelation_score": safe_float(item.get("decorrelation_score"), default=float("nan")),
        "decorrelation_gate_action": str(item.get("decorrelation_gate_action", "") or ""),
        "decorrelation_gate_reason": str(item.get("decorrelation_gate_reason", "") or ""),
        "decorrelation_winner_allowed": bool(item.get("decorrelation_winner_allowed", True)),
        "decorrelation_quality_score": safe_float(item.get("decorrelation_quality_score"), default=float("nan")),
        "decorrelation_adjustment": safe_float(item.get("decorrelation_adjustment"), default=float("nan")),
        "decorrelation_adjusted_score": safe_float(item.get("decorrelation_adjusted_score"), default=float("nan")),
    }


def _classify_round_outcome(completed: list[dict[str, object]]) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    successful = [item for item in completed if int(item["returncode"]) == 0]
    failed = [item for item in completed if int(item["returncode"]) != 0]
    if not successful:
        return "failed", successful, failed
    if failed:
        return "partial_success", successful, failed
    return "ok", successful, failed


def _read_child_run_summary(child_runs_dir: str | Path) -> dict[str, object]:
    root = Path(child_runs_dir)
    summary_paths = sorted(root.glob("*/summary.json"))
    if not summary_paths:
        return {}
    try:
        return json.loads(summary_paths[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_child_runtime_status(child_runs_dir: str | Path) -> dict[str, object]:
    root = Path(child_runs_dir)
    status_paths = sorted(root.glob("*/metadata/runtime_status.json"))
    if not status_paths:
        return {}
    try:
        return json.loads(status_paths[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_prompt_trace(completed: list[dict[str, object]]) -> dict[str, object]:
    for item in completed:
        summary = dict(item.get("child_summary") or {})
        trace = dict(summary.get("prompt_trace") or {})
        if trace:
            return trace
    return {}


def build_arg_parser() -> argparse.ArgumentParser:
    single = build_single_round_parser()
    parser = argparse.ArgumentParser(
        description="Run multiple llm_refine rounds in parallel, one model per process."
    )
    parser.add_argument("--family", required=True, help="seed family name from the seed pool")
    parser.add_argument(
        "--models",
        action="append",
        required=True,
        help="comma-separated model list; may be repeated, e.g. --models gpt-5.4,deepseek-v3.1",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="max concurrent child runs; 0 means all",
    )
    parser.add_argument("--multi-runs-dir", default=str(DEFAULT_MULTI_RUNS_DIR), help="orchestrator log/output root")

    passthrough_defaults = {
        "--seed-pool": str(single.get_default("seed_pool")),
        "--n-candidates": single.get_default("n_candidates"),
        "--runs-dir": str(single.get_default("runs_dir")),
        "--archive-db": str(single.get_default("archive_db")),
        "--name-prefix": str(single.get_default("name_prefix")),
        "--parent-candidate-id": str(single.get_default("parent_candidate_id")),
        "--provider-name": str(single.get_default("provider_name")),
        "--base-url": str(single.get_default("base_url")),
        "--api-key": str(single.get_default("api_key")),
        "--temperature": single.get_default("temperature"),
        "--max-tokens": single.get_default("max_tokens"),
        "--timeout": single.get_default("timeout"),
        "--additional-notes": str(single.get_default("additional_notes")),
        "--current-parent-name": str(single.get_default("current_parent_name")),
        "--current-parent-expression": str(single.get_default("current_parent_expression")),
        "--panel-path": str(single.get_default("panel_path")),
        "--benchmark-path": str(single.get_default("benchmark_path")),
        "--start": str(single.get_default("start")),
        "--end": str(single.get_default("end")),
        "--prompt-template-version": str(single.get_default("prompt_template_version")),
        "--primary-objective": "",
        "--secondary-objective": "",
    }
    for arg, default in passthrough_defaults.items():
        name = arg.lstrip("-").replace("-", "_")
        parser.add_argument(arg, default=default)
    parser.add_argument(
        "--decorrelation-target",
        action="append",
        default=list(single.get_default("decorrelation_target") or []),
        help="explicit factor(s) to decorrelate from; may be repeated or passed as comma-separated names",
    )

    parser.add_argument("--auto-parent", action="store_true", help="resolve parent from latest family research winner")
    parser.add_argument(
        "--round1-seed-stage",
        action="store_true",
        help="propagate seed-stage round1 prompt augmentation to child runs",
    )
    parser.add_argument(
        "--stage-mode",
        default="auto",
        choices=_STAGE_MODE_CHOICES,
        help="explicit orchestration stage label propagated to child runs",
    )
    parser.add_argument("--skip-eval", action="store_true", help="skip backtest stage in each child run")
    parser.add_argument("--dry-run", action="store_true", help="dry-run each child without provider call")
    parser.add_argument(
        "--policy-preset",
        default=DEFAULT_POLICY_PRESET,
        choices=SearchPolicy.available_presets(),
        help="search scoring preset for SearchEngine",
    )
    parser.add_argument(
        "--target-profile",
        default=DEFAULT_TARGET_PROFILE,
        choices=SearchPolicy.available_target_profiles(),
        help="target-conditioned search profile",
    )
    parser.add_argument(
        "--disable-mmr-rerank",
        action="store_true",
        help="disable MMR rerank and use raw frontier score ordering only",
    )
    parser.add_argument(
        "--auto-apply-promotion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_APPLY_PROMOTION,
        help="automatically apply pending curated promotion patches in each child run after evaluation",
    )
    return parser


def _build_child_cmd(args: argparse.Namespace, *, model: str, round_id: int, child_runs_dir: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "factors_store.llm_refine.cli.run_refine_loop",
        "--family",
        args.family,
        "--seed-pool",
        args.seed_pool,
        "--n-candidates",
        str(args.n_candidates),
        "--runs-dir",
        child_runs_dir,
        "--archive-db",
        args.archive_db,
        "--name-prefix",
        args.name_prefix,
        "--round-id",
        str(round_id),
        "--provider-name",
        args.provider_name,
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--model",
        model,
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
    ]
    if str(args.additional_notes).strip():
        cmd.extend(["--additional-notes", args.additional_notes])
    for target in list(args.decorrelation_target or []):
        if str(target).strip():
            cmd.extend(["--decorrelation-target", str(target)])
    if str(args.panel_path).strip():
        cmd.extend(["--panel-path", args.panel_path])
    if str(args.benchmark_path).strip():
        cmd.extend(["--benchmark-path", args.benchmark_path])
    if str(args.start).strip():
        cmd.extend(["--start", args.start])
    if str(args.end).strip():
        cmd.extend(["--end", args.end])
    if args.skip_eval:
        cmd.append("--skip-eval")
    if args.dry_run:
        cmd.append("--dry-run")
    if str(args.policy_preset).strip():
        cmd.extend(["--policy-preset", str(args.policy_preset)])
    if str(args.target_profile).strip():
        cmd.extend(["--target-profile", str(args.target_profile)])
    if str(args.stage_mode).strip():
        cmd.extend(["--stage-mode", str(args.stage_mode)])
    if str(args.prompt_template_version).strip():
        cmd.extend(["--prompt-template-version", str(args.prompt_template_version)])
    if str(args.primary_objective or "").strip():
        cmd.extend(["--primary-objective", str(args.primary_objective).strip()])
    if str(args.secondary_objective or "").strip():
        cmd.extend(["--secondary-objective", str(args.secondary_objective).strip()])
    if args.disable_mmr_rerank:
        cmd.append("--disable-mmr-rerank")
    cmd.append("--auto-apply-promotion" if args.auto_apply_promotion else "--no-auto-apply-promotion")
    if args.round1_seed_stage:
        cmd.append("--round1-seed-stage")
    return cmd


def _resolve_initial_parent(args: argparse.Namespace) -> dict[str, str]:
    seed_pool = load_seed_pool(args.seed_pool)
    family = seed_pool.get_family(args.family)
    explicit_name = str(args.current_parent_name or "").strip()
    explicit_expression = str(args.current_parent_expression or "").strip()
    explicit_candidate_id = str(args.parent_candidate_id or "").strip()

    if explicit_name:
        payload = (
            get_candidate_record(db_path=args.archive_db, candidate_id=explicit_candidate_id)
            if explicit_candidate_id
            else None
        )
        resolved = dict(payload or {})
        resolved.update(
            {
                "factor_name": explicit_name,
                "expression": explicit_expression
                or str(resolved.get("expression", "") or "")
                or resolve_family_formula(family, explicit_name),
                "candidate_id": explicit_candidate_id
                or str(resolved.get("candidate_id", "") or "")
                or make_seed_candidate_id(explicit_name),
                "status": str(resolved.get("status", "") or "explicit_seed"),
                "node_kind": "explicit_parent",
            }
        )
        return resolved

    latest_winner = get_latest_family_winner(db_path=args.archive_db, family=args.family) if args.auto_parent else None
    if latest_winner is not None:
        payload = dict(latest_winner)
        payload["node_kind"] = "archive_winner"
        return payload

    bootstrap_parent = select_bootstrap_parent(build_bootstrap_frontier(seed_pool=seed_pool, family=family))
    if bootstrap_parent:
        payload = dict(bootstrap_parent)
        payload["status"] = str(payload.get("status", "") or "seed")
        payload["node_kind"] = str(
            payload.get(
                "node_kind",
                "preferred_seed" if str(payload.get("factor_name", "")) != family.canonical_seed else "canonical_seed",
            )
        )
        return payload

    default_parent_name = resolve_preferred_refine_seed(family)
    return {
        "factor_name": default_parent_name,
        "expression": resolve_family_formula(family, default_parent_name),
        "candidate_id": make_seed_candidate_id(default_parent_name),
        "status": "seed",
        "node_kind": "preferred_seed" if default_parent_name != family.canonical_seed else "canonical_seed",
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    args = build_arg_parser().parse_args()
    models = _parse_models(args.models)
    if not models:
        raise SystemExit("至少提供一个 model")

    max_parallel = int(args.max_parallel) if int(args.max_parallel) > 0 else len(models)
    archive_db = Path(args.archive_db).expanduser().resolve()
    base_round = get_latest_family_round(db_path=archive_db, family=args.family) + 1
    initial_parent = _resolve_initial_parent(args)
    stage_mode = str(args.stage_mode or "auto").strip() or "auto"
    policy = (
        SearchPolicy.multi_model_best_first(preset=args.policy_preset)
        .with_target_profile(args.target_profile)
        .with_mmr_rerank(not bool(args.disable_mmr_rerank))
    )
    budget = SearchBudget(
        max_rounds=1,
        family_budget=1,
        branch_budget=1,
        max_frontier_size=max(len(models) * max(int(args.n_candidates), 1), 4),
        max_depth=2,
        stop_if_no_improve=0,
    )
    engine = SearchEngine(
        family=args.family,
        budget=budget,
        policy=policy,
        normalizer=build_search_normalizer(db_path=args.archive_db, family=args.family),
    )
    seed_node = engine.register_seed(
        factor_name=str(initial_parent.get("factor_name", "")),
        expression=str(initial_parent.get("expression", "")),
        candidate_id=str(initial_parent.get("candidate_id", "")),
        node_kind=str(initial_parent.get("node_kind", "seed")),
        status=str(initial_parent.get("status", "seed")),
        source_run_id=str(initial_parent.get("run_id", "")),
        source_model=str(initial_parent.get("source_model", "")),
        source_provider=str(initial_parent.get("source_provider", "")),
        round_id=int(initial_parent.get("round_id") or 0),
        metrics=initial_parent,
    )
    selected_parent = engine.select_next() or seed_node
    args.round1_seed_stage = bool(
        args.round1_seed_stage
        or stage_mode == "new_family_broad"
        or is_seed_stage_node_kind(selected_parent.node_kind)
    )

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    multi_run_dir = Path(args.multi_runs_dir).expanduser().resolve() / f"{ts}_{args.family}"
    multi_run_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for idx, model in enumerate(models):
        round_id = base_round + idx
        model_slug = _sanitize_slug(model)
        log_path = multi_run_dir / f"{idx + 1:02d}_{model_slug}.log"
        child_runs_dir = multi_run_dir / "child_runs" / f"{idx + 1:02d}_{model_slug}"
        child_runs_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_child_cmd(
            args,
            model=model,
            round_id=round_id,
            child_runs_dir=str(child_runs_dir),
        )
        cmd.extend(["--current-parent-name", selected_parent.factor_name])
        cmd.extend(["--current-parent-expression", selected_parent.expression])
        if selected_parent.candidate_id:
            cmd.extend(["--parent-candidate-id", selected_parent.candidate_id])
        plan.append(
            {
                "model": model,
                "round_id": round_id,
                "log_path": str(log_path),
                "child_runs_dir": str(child_runs_dir),
                "cmd": cmd,
            }
        )

    (multi_run_dir / "plan.json").write_text(
        json.dumps(
            {
                "family": args.family,
                "stage_mode": stage_mode,
                "target_profile": str(args.target_profile),
                "archive_db": str(archive_db),
                "base_round": base_round,
                "max_parallel": max_parallel,
                "models": models,
                "selection_mode": "single_parent_multi_model",
                "search_policy": policy.to_dict(),
                "search_budget": budget.to_dict(),
                "initial_parent": seed_node.to_dict(),
                "selected_parent": selected_parent.to_dict(),
                "runs": plan,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pending = list(plan)
    running: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    next_progress_emit_at = time.monotonic() + 30.0
    engine.record_attempt(parent_node_id=selected_parent.node_id, note="run_refine_multi_model_orchestrator")

    while pending or running:
        while pending and len(running) < max_parallel:
            item = pending.pop(0)
            log_path = Path(str(item["log_path"]))
            log_fp = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                list(item["cmd"]),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=os.environ.copy(),
                text=True,
            )
            item["pid"] = proc.pid
            item["process"] = proc
            item["log_fp"] = log_fp
            print(
                f"[launch] model={item['model']} round_id={item['round_id']} "
                f"pid={proc.pid} log={log_path}"
            )
            running.append(item)

        still_running: list[dict[str, object]] = []
        now = time.monotonic()
        emit_running_progress = now >= next_progress_emit_at
        for item in running:
            proc = item["process"]
            assert isinstance(proc, subprocess.Popen)
            ret = proc.poll()
            if ret is None:
                runtime_status = _read_child_runtime_status(str(item["child_runs_dir"]))
                phase = str(runtime_status.get("phase", "") or "starting")
                message = str(runtime_status.get("message", "") or "").strip()
                elapsed = runtime_status.get("elapsed_seconds")
                status_signature = (phase, message)
                if emit_running_progress or status_signature != item.get("last_status_signature"):
                    elapsed_text = (
                        f"{float(elapsed):.1f}s"
                        if isinstance(elapsed, (int, float))
                        else str(elapsed)
                        if elapsed is not None
                        else "?"
                    )
                    suffix = f" message={message}" if message else ""
                    print(
                        f"[progress] model={item['model']} round_id={item['round_id']} "
                        f"phase={phase} elapsed={elapsed_text}{suffix}"
                    )
                    item["last_status_signature"] = status_signature
                still_running.append(item)
                continue
            log_fp = item.pop("log_fp")
            log_fp.close()
            item["returncode"] = ret
            item["child_summary"] = _read_child_run_summary(str(item["child_runs_dir"]))
            completed.append(item)
            status = "ok" if ret == 0 else "failed"
            print(
                f"[done] model={item['model']} round_id={item['round_id']} "
                f"status={status} returncode={ret}"
            )
        running = still_running
        if emit_running_progress:
            next_progress_emit_at = time.monotonic() + 30.0
        if running:
            time.sleep(1.0)

    child_records = load_candidate_records_from_completed_runs(
        archive_db=str(archive_db),
        completed=[
            {
                "child_runs_dir": item["child_runs_dir"],
            }
            for item in completed
        ],
        family=args.family,
        statuses=() if args.skip_eval else ("research_winner", "winner", "research_keep", "keep", "research_keep_exploratory"),
    )
    decision_context = DecisionContext.from_runtime(
        family=args.family,
        stage_mode=stage_mode,
        target_profile=str(args.target_profile),
        policy_preset=str(args.policy_preset),
        decorrelation_targets=list(args.decorrelation_target or []),
        neutralized_eval_enabled=not bool(args.skip_eval),
        context_profile=resolve_context_profile(
            ContextEvidence.from_runtime(
                family=args.family,
                stage_mode=stage_mode,
                target_profile=str(args.target_profile),
                policy_preset=str(args.policy_preset),
                is_seed_stage=bool(stage_mode == "new_family_broad"),
                has_bootstrap_frontier=False,
                has_donor_motifs=False,
                has_decorrelation_targets=bool(args.decorrelation_target),
                selected_parent_kind=str(getattr(selected_parent, "node_kind", "") or ""),
                requested_candidate_count=int(getattr(args, "n_candidates", 0) or 0),
                final_candidate_target=int(getattr(args, "n_candidates", 0) or 0),
            )
        ),
    )
    decision_engine = DecisionEngine(decision_context)
    round_status, successful, failed = _classify_round_outcome(completed)
    expansion = engine.register_expansion(
        parent_node_id=selected_parent.node_id,
        child_records=child_records,
        success=bool(successful),
        source_run_id=str(multi_run_dir),
        note=f"multi_run_dir={multi_run_dir}",
        count_attempt=False,
    )
    global_best_candidate = decision_engine.pick_best_candidate(child_records)
    global_best_keep = decision_engine.pick_best_keep(child_records)
    decorrelation_rerank_enabled = decision_engine.decorrelation_rerank_enabled(child_records)
    summary = {
        "family": args.family,
        "stage_mode": stage_mode,
        "target_profile": str(args.target_profile),
        "base_round": base_round,
        "max_parallel": max_parallel,
        "status": round_status,
        "partial_success": round_status == "partial_success",
        "selected_parent": selected_parent.to_dict(),
        "selected_parent_name": selected_parent.factor_name,
        "selected_parent_expression": selected_parent.expression,
        "winner": global_best_candidate,
        "global_best_candidate": global_best_candidate,
        "global_best_keep": global_best_keep,
        "decision_context": {
            "family": decision_context.family,
            "stage_mode": decision_context.stage_mode,
            "target_profile": decision_context.target_profile,
            "policy_preset": decision_context.policy_preset,
            "context_profile": decision_context.context_profile.to_dict() if decision_context.context_profile else None,
            "decorrelation_targets": list(decision_context.decorrelation_targets),
            "decorrelation_enabled": decision_context.decorrelation_enabled,
            "neutralized_eval_enabled": decision_context.neutralized_eval_enabled,
            "admission_intent": decision_context.admission_intent,
        },
        "decorrelation_rerank_enabled": decorrelation_rerank_enabled,
        "decorrelation_rerank_mode": "soft_adjustment" if decorrelation_rerank_enabled else "disabled",
        "winner_selection_mode": decision_engine.winner_selection_mode(child_records),
        "global_rerank_preview": decision_engine.build_rerank_preview(child_records),
        "successful_models": [str(item["model"]) for item in successful],
        "failed_models": [str(item["model"]) for item in failed],
        "successful_model_count": len(successful),
        "failed_model_count": len(failed),
        "search_policy": policy.to_dict(),
        "search_budget": budget.to_dict(),
        "prompt_trace": _pick_prompt_trace(completed)
        or {
            "stage_mode": stage_mode,
            "seed_stage_active": bool(args.round1_seed_stage),
            "selected_parent_kind": str(selected_parent.node_kind),
            "selected_parent_factor_name": str(selected_parent.factor_name),
        },
        "child_runs": [
            {
                "model": str(item["model"]),
                "status": "ok" if int(item["returncode"]) == 0 else "failed",
                "returncode": int(item["returncode"]),
                "log_path": str(item["log_path"]),
                "child_runs_dir": str(item["child_runs_dir"]),
                "prompt_trace": dict((item.get("child_summary") or {}).get("prompt_trace") or {}),
            }
            for item in completed
        ],
        "completed": [
            {
                "model": item["model"],
                "round_id": item["round_id"],
                "returncode": item["returncode"],
                "log_path": item["log_path"],
                "child_runs_dir": item["child_runs_dir"],
            }
            for item in completed
        ],
    }
    winner_record = summary.get("winner") or {}
    summary["winner_name"] = str(winner_record.get("factor_name", "") or "")
    summary["winner_expression"] = str(winner_record.get("expression", "") or "")
    summary["winner_status"] = str(winner_record.get("status", "") or "")
    summary["winner_is_keep"] = str(winner_record.get("status", "") or "").strip().lower() in {"research_keep", "keep", "research_keep_exploratory"}
    best_candidate = summary.get("global_best_candidate") or {}
    summary["global_best_candidate_name"] = str(best_candidate.get("factor_name", "") or "")
    summary["global_best_candidate_expression"] = str(best_candidate.get("expression", "") or "")
    summary["global_best_candidate_status"] = str(best_candidate.get("status", "") or "")
    summary["global_best_candidate_metrics"] = decision_engine.metric_snapshot(best_candidate)
    best_keep = summary.get("global_best_keep") or {}
    summary["global_best_keep_name"] = str(best_keep.get("factor_name", "") or "")
    summary["global_best_keep_expression"] = str(best_keep.get("expression", "") or "")
    summary["global_best_keep_status"] = str(best_keep.get("status", "") or "")
    summary["global_best_keep_metrics"] = decision_engine.metric_snapshot(best_keep)
    summary["children_collected"] = len(child_records)
    summary["children_added_to_search"] = len(expansion.get("children", []))
    summary["search"] = engine.summary()
    (multi_run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if round_status == "failed":
        print(f"[summary] all {len(completed)} model run(s) failed")
        return 1
    if round_status == "partial_success":
        print(
            f"[summary] partial success: {len(successful)} succeeded, "
            f"{len(failed)} failed; continuing with collected results"
        )
        print(f"[multi_run_dir] {multi_run_dir}")
        return 0

    print(f"[summary] all {len(completed)} model run(s) completed")
    print(f"[multi_run_dir] {multi_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
