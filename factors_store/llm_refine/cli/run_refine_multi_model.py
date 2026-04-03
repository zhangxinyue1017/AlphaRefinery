from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from ..config import (
    DEFAULT_MAX_PARALLEL,
    DEFAULT_MULTI_RUNS_DIR,
    DEFAULT_POLICY_PRESET,
    DEFAULT_TARGET_PROFILE,
)
from ..core.archive import DEFAULT_ARCHIVE_DB, get_latest_family_round, get_latest_family_winner, make_seed_candidate_id
from ..core.seed_loader import load_seed_pool, resolve_family_formula, resolve_preferred_refine_seed
from ..knowledge.round1 import build_bootstrap_frontier, is_seed_stage_node_kind, select_bootstrap_parent
from ..search import SearchBudget, SearchEngine, SearchPolicy, build_search_normalizer
from ..search.run_ingest import load_candidate_records_from_completed_runs
from ..search.scoring import safe_float
from .run_refine_loop import build_arg_parser as build_single_round_parser


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


def _pick_best_child_record(records: list[dict[str, object]]) -> dict[str, object] | None:
    if not records:
        return None

    status_rank = {
        "research_winner": 4,
        "winner": 3,
        "research_keep": 2,
        "keep": 1,
    }

    def _score(item: dict[str, object]) -> tuple[float, float, float, float, float, float]:
        status = str(item.get("status", "")).strip().lower()
        return (
            float(status_rank.get(status, 0)),
            safe_float(item.get("quick_rank_ic_mean"), default=float("-inf")),
            safe_float(item.get("quick_rank_icir"), default=float("-inf")),
            safe_float(item.get("net_sharpe"), default=float("-inf")),
            safe_float(item.get("net_ann_return"), default=float("-inf")),
            -safe_float(item.get("mean_turnover"), default=float("inf")),
        )

    return max(records, key=_score)


def _classify_round_outcome(completed: list[dict[str, object]]) -> tuple[str, list[dict[str, object]], list[dict[str, object]]]:
    successful = [item for item in completed if int(item["returncode"]) == 0]
    failed = [item for item in completed if int(item["returncode"]) != 0]
    if not successful:
        return "failed", successful, failed
    if failed:
        return "partial_success", successful, failed
    return "ok", successful, failed


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
    }
    for arg, default in passthrough_defaults.items():
        name = arg.lstrip("-").replace("-", "_")
        parser.add_argument(arg, default=default)

    parser.add_argument("--auto-parent", action="store_true", help="resolve parent from latest family research winner")
    parser.add_argument(
        "--round1-seed-stage",
        action="store_true",
        help="propagate seed-stage round1 prompt augmentation to child runs",
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
        action="store_true",
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
    if args.disable_mmr_rerank:
        cmd.append("--disable-mmr-rerank")
    if args.auto_apply_promotion:
        cmd.append("--auto-apply-promotion")
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
        return {
            "factor_name": explicit_name,
            "expression": explicit_expression or resolve_family_formula(family, explicit_name),
            "candidate_id": explicit_candidate_id or make_seed_candidate_id(explicit_name),
            "status": "explicit_seed",
            "node_kind": "explicit_parent",
        }

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
    args = build_arg_parser().parse_args()
    models = _parse_models(args.models)
    if not models:
        raise SystemExit("至少提供一个 model")

    max_parallel = int(args.max_parallel) if int(args.max_parallel) > 0 else len(models)
    archive_db = Path(args.archive_db).expanduser().resolve()
    base_round = get_latest_family_round(db_path=archive_db, family=args.family) + 1
    initial_parent = _resolve_initial_parent(args)
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
    args.round1_seed_stage = bool(args.round1_seed_stage or is_seed_stage_node_kind(selected_parent.node_kind))

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
                cwd="/root/workspace/zxy_workspace/AlphaRefinery",
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
        for item in running:
            proc = item["process"]
            assert isinstance(proc, subprocess.Popen)
            ret = proc.poll()
            if ret is None:
                still_running.append(item)
                continue
            log_fp = item.pop("log_fp")
            log_fp.close()
            item["returncode"] = ret
            completed.append(item)
            status = "ok" if ret == 0 else "failed"
            print(
                f"[done] model={item['model']} round_id={item['round_id']} "
                f"status={status} returncode={ret}"
            )
        running = still_running
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
        statuses=() if args.skip_eval else ("research_winner", "winner", "research_keep", "keep"),
    )
    round_status, successful, failed = _classify_round_outcome(completed)
    expansion = engine.register_expansion(
        parent_node_id=selected_parent.node_id,
        child_records=child_records,
        success=bool(successful),
        source_run_id=str(multi_run_dir),
        note=f"multi_run_dir={multi_run_dir}",
        count_attempt=False,
    )
    summary = {
        "family": args.family,
        "target_profile": str(args.target_profile),
        "base_round": base_round,
        "max_parallel": max_parallel,
        "status": round_status,
        "partial_success": round_status == "partial_success",
        "selected_parent": selected_parent.to_dict(),
        "selected_parent_name": selected_parent.factor_name,
        "selected_parent_expression": selected_parent.expression,
        "winner": _pick_best_child_record(child_records),
        "successful_models": [str(item["model"]) for item in successful],
        "failed_models": [str(item["model"]) for item in failed],
        "successful_model_count": len(successful),
        "failed_model_count": len(failed),
        "search_policy": policy.to_dict(),
        "search_budget": budget.to_dict(),
        "child_runs": [
            {
                "model": str(item["model"]),
                "status": "ok" if int(item["returncode"]) == 0 else "failed",
                "returncode": int(item["returncode"]),
                "log_path": str(item["log_path"]),
                "child_runs_dir": str(item["child_runs_dir"]),
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
