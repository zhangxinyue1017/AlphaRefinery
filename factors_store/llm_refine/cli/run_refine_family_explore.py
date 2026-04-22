from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_AUTO_APPLY_PROMOTION,
    DEFAULT_FAMILY_EXPLORE_ALIAS_LIMIT,
    DEFAULT_FAMILY_EXPLORE_MAX_SEEDS,
    DEFAULT_FAMILY_EXPLORE_RECENT_REFERENCE_LIMIT,
    DEFAULT_FAMILY_EXPLORE_RUNS_DIR,
    DEFAULT_FAMILY_EXPLORE_SLEEP_BETWEEN_SEEDS,
    PROJECT_ROOT,
)
from ..core.archive import (
    DEFAULT_ARCHIVE_DB,
    get_latest_family_winner,
    load_family_reference_candidates,
    make_seed_candidate_id,
    utc_now_iso,
)
from .run_refine_multi_model import build_arg_parser as build_multi_model_arg_parser
from ..core.seed_loader import load_seed_pool, resolve_family_formula, resolve_preferred_refine_seed
from ..search import SearchBudget, SearchEngine, SearchPolicy, build_search_normalizer
from ..search.run_ingest import load_multi_run_candidate_records, resolve_materialized_multi_run_dir


def _normalize_models(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if item and item not in out:
                out.append(item)
    return out


def _sanitize_slug(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("._-") or "seed"


def _seed_record(
    *,
    factor_name: str,
    expression: str,
    candidate_id: str,
    seed_kind: str,
    source_run_id: str = "",
    source_model: str = "",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "factor_name": str(factor_name),
        "expression": str(expression or ""),
        "candidate_id": str(candidate_id),
        "seed_kind": str(seed_kind),
        "source_run_id": str(source_run_id or ""),
        "source_model": str(source_model or ""),
    }
    if metrics:
        payload.update(metrics)
    return payload


def _build_seed_frontier(
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    seed_pool = load_seed_pool(args.seed_pool)
    family = seed_pool.get_family(args.family)

    frontier: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_exprs: set[str] = set()

    def _add(item: dict[str, Any]) -> None:
        name = str(item.get("factor_name", "")).strip()
        expr = str(item.get("expression", "")).strip()
        if not name:
            return
        if name in seen_names:
            return
        if expr and expr in seen_exprs:
            return
        frontier.append(item)
        seen_names.add(name)
        if expr:
            seen_exprs.add(expr)

    default_parent_name = resolve_preferred_refine_seed(family)
    _add(
        _seed_record(
            factor_name=default_parent_name,
            expression=resolve_family_formula(family, default_parent_name),
            candidate_id=make_seed_candidate_id(default_parent_name),
            seed_kind="preferred_seed" if default_parent_name != family.canonical_seed else "canonical_seed",
        )
    )
    if default_parent_name != family.canonical_seed:
        _add(
            _seed_record(
                factor_name=family.canonical_seed,
                expression=resolve_family_formula(family, family.canonical_seed),
                candidate_id=make_seed_candidate_id(family.canonical_seed),
                seed_kind="canonical_seed",
            )
        )

    alias_limit = max(int(args.alias_limit), 0)
    for alias in family.aliases[:alias_limit]:
        _add(
            _seed_record(
                factor_name=alias,
                expression=resolve_family_formula(family, alias),
                candidate_id=make_seed_candidate_id(alias),
                seed_kind="alias",
            )
        )

    if not args.no_latest_winner:
        latest_winner = get_latest_family_winner(db_path=args.archive_db, family=args.family)
        if latest_winner is not None:
            _add(
                _seed_record(
                    factor_name=str(latest_winner.get("factor_name", "")),
                    expression=str(latest_winner.get("expression", "")),
                    candidate_id=str(latest_winner.get("candidate_id", "")) or make_seed_candidate_id(str(latest_winner.get("factor_name", ""))),
                    seed_kind="latest_winner",
                    source_run_id=str(latest_winner.get("run_id", "")),
                    source_model=str(latest_winner.get("source_model", "")),
                    metrics=dict(latest_winner),
                )
            )

    recent_limit = max(int(args.recent_reference_limit), 0)
    if recent_limit > 0:
        refs = load_family_reference_candidates(
            db_path=args.archive_db,
            family=args.family,
            limit=recent_limit + 4,
        )
        for ref in refs:
            _add(
                _seed_record(
                    factor_name=str(ref.get("factor_name", "")),
                    expression=str(ref.get("expression", "")),
                    candidate_id=str(ref.get("candidate_id", "")) or make_seed_candidate_id(str(ref.get("factor_name", ""))),
                    seed_kind=f"recent_{str(ref.get('status', '') or 'reference')}",
                    source_run_id=str(ref.get("run_id", "")),
                )
            )
            if len([item for item in frontier if str(item.get("seed_kind", "")).startswith("recent_")]) >= recent_limit:
                break

    max_seeds = max(int(args.max_seeds), 1)
    return frontier[:max_seeds]


def build_arg_parser() -> argparse.ArgumentParser:
    multi = build_multi_model_arg_parser()
    parser = argparse.ArgumentParser(
        description="Breadth-first family exploration: run one multi-model refine round from multiple seeds in the same family."
    )
    parser.add_argument("--family", required=True, help="seed family name from the seed pool")
    parser.add_argument("--models", action="append", required=True, help="comma-separated model list; may be repeated")
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=DEFAULT_FAMILY_EXPLORE_MAX_SEEDS,
        help="maximum seeds to include in the frontier",
    )
    parser.add_argument(
        "--alias-limit",
        type=int,
        default=DEFAULT_FAMILY_EXPLORE_ALIAS_LIMIT,
        help="maximum alias baselines to include before truncation",
    )
    parser.add_argument(
        "--recent-reference-limit",
        type=int,
        default=DEFAULT_FAMILY_EXPLORE_RECENT_REFERENCE_LIMIT,
        help="number of recent archive winners/keeps to include in the frontier",
    )
    parser.add_argument(
        "--no-latest-winner",
        action="store_true",
        help="do not include the latest family research winner in the frontier",
    )
    parser.add_argument(
        "--family-explore-runs-dir",
        default=str(DEFAULT_FAMILY_EXPLORE_RUNS_DIR),
        help="artifact/log root for breadth-first family exploration",
    )
    parser.add_argument(
        "--sleep-between-seeds",
        type=float,
        default=DEFAULT_FAMILY_EXPLORE_SLEEP_BETWEEN_SEEDS,
        help="optional sleep in seconds between seed dispatches",
    )

    passthrough_defaults = {
        "--seed-pool": str(multi.get_default("seed_pool")),
        "--n-candidates": multi.get_default("n_candidates"),
        "--runs-dir": str(multi.get_default("runs_dir")),
        "--archive-db": str(multi.get_default("archive_db")),
        "--name-prefix": str(multi.get_default("name_prefix")),
        "--provider-name": str(multi.get_default("provider_name")),
        "--base-url": str(multi.get_default("base_url")),
        "--api-key": str(multi.get_default("api_key")),
        "--temperature": multi.get_default("temperature"),
        "--max-tokens": multi.get_default("max_tokens"),
        "--timeout": multi.get_default("timeout"),
        "--additional-notes": str(multi.get_default("additional_notes")),
        "--policy-preset": str(multi.get_default("policy_preset")),
        "--target-profile": str(multi.get_default("target_profile")),
        "--panel-path": str(multi.get_default("panel_path")),
        "--benchmark-path": str(multi.get_default("benchmark_path")),
        "--start": str(multi.get_default("start")),
        "--end": str(multi.get_default("end")),
        "--max-parallel": multi.get_default("max_parallel"),
    }
    for arg, default in passthrough_defaults.items():
        parser.add_argument(arg, default=default)

    parser.add_argument("--skip-eval", action="store_true", help="skip evaluation stage in each seed dispatch")
    parser.add_argument("--dry-run", action="store_true", help="dry-run each seed dispatch without provider calls")
    parser.add_argument(
        "--auto-apply-promotion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_APPLY_PROMOTION,
        help="automatically apply pending curated promotion patches in each child run after evaluation",
    )
    parser.add_argument(
        "--disable-mmr-rerank",
        action="store_true",
        help="disable MMR rerank and use raw frontier score ordering only",
    )
    return parser


def _build_seed_cmd(args: argparse.Namespace, *, seed: dict[str, Any], seed_multi_runs_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "factors_store.llm_refine.cli.run_refine_multi_model",
        "--family",
        args.family,
        "--seed-pool",
        str(args.seed_pool),
        "--n-candidates",
        str(args.n_candidates),
        "--runs-dir",
        str(args.runs_dir),
        "--archive-db",
        str(args.archive_db),
        "--name-prefix",
        str(args.name_prefix),
        "--provider-name",
        str(args.provider_name),
        "--base-url",
        str(args.base_url),
        "--api-key",
        str(args.api_key),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--max-parallel",
        str(args.max_parallel),
        "--multi-runs-dir",
        str(seed_multi_runs_root),
        "--current-parent-name",
        str(seed["factor_name"]),
        "--current-parent-expression",
        str(seed["expression"]),
        "--parent-candidate-id",
        str(seed["candidate_id"]),
    ]
    for raw in args.models:
        cmd.extend(["--models", raw])
    if str(args.additional_notes).strip():
        cmd.extend(["--additional-notes", str(args.additional_notes)])
    if str(args.panel_path).strip():
        cmd.extend(["--panel-path", str(args.panel_path)])
    if str(args.benchmark_path).strip():
        cmd.extend(["--benchmark-path", str(args.benchmark_path)])
    if str(args.start).strip():
        cmd.extend(["--start", str(args.start)])
    if str(args.end).strip():
        cmd.extend(["--end", str(args.end)])
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
    cmd.append("--auto-apply-promotion" if args.auto_apply_promotion else "--no-auto-apply-promotion")
    return cmd


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = build_arg_parser().parse_args()
    models = _normalize_models(args.models)
    if not models:
        raise SystemExit("至少提供一个 model")

    frontier = _build_seed_frontier(args=args)
    if not frontier:
        raise SystemExit(f"未能为 family={args.family} 组装出可用 seed frontier")

    explore_dir = Path(args.family_explore_runs_dir).expanduser().resolve() / (
        time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_{args.family}"
    )
    explore_dir.mkdir(parents=True, exist_ok=True)

    summary_path = explore_dir / "summary.json"
    plan_path = explore_dir / "plan.json"
    policy = (
        SearchPolicy.family_breadth_first(preset=args.policy_preset)
        .with_target_profile(args.target_profile)
        .with_mmr_rerank(not bool(args.disable_mmr_rerank))
    )
    budget = SearchBudget(
        max_rounds=int(args.max_seeds),
        family_budget=int(args.max_seeds),
        branch_budget=1,
        max_frontier_size=max(int(args.max_seeds), 1),
        max_depth=1,
        stop_if_no_improve=0,
    )
    engine = SearchEngine(
        family=args.family,
        budget=budget,
        policy=policy,
        normalizer=build_search_normalizer(db_path=args.archive_db, family=args.family),
    )
    for seed in frontier:
        engine.register_seed(
            factor_name=str(seed.get("factor_name", "")),
            expression=str(seed.get("expression", "")),
            candidate_id=str(seed.get("candidate_id", "")),
            node_kind=str(seed.get("seed_kind", "seed")),
            status=str(seed.get("status", "seed")),
            source_run_id=str(seed.get("source_run_id", "")),
            source_model=str(seed.get("source_model", "")),
            source_provider=str(seed.get("source_provider", "")),
            round_id=int(seed.get("round_id") or 0),
            metrics=seed,
        )
    plan = {
        "family": args.family,
        "target_profile": str(args.target_profile),
        "started_at": utc_now_iso(),
        "models": models,
        "n_candidates": int(args.n_candidates),
        "max_parallel": int(args.max_parallel),
        "max_seeds": int(args.max_seeds),
        "frontier": frontier,
        "mode": "breadth_first_single_round_per_seed",
        "selection_mode": "search_engine_family_breadth_first",
        "search_policy": policy.to_dict(),
        "search_budget": budget.to_dict(),
        "initial_frontier": engine.frontier_snapshot(),
    }
    _write_json(plan_path, plan)

    dispatches: list[dict[str, Any]] = []
    while engine.can_continue():
        idx = len(dispatches) + 1
        selected = engine.select_next()
        if selected is None:
            break
        seed = selected.to_dict()
        seed_slug = _sanitize_slug(selected.factor_name)
        seed_root = explore_dir / "seed_runs" / f"{idx:02d}_{seed_slug}"
        seed_root.mkdir(parents=True, exist_ok=True)
        seed_multi_runs_root = seed_root / "multi_runs"
        seed_multi_runs_root.mkdir(parents=True, exist_ok=True)
        log_path = explore_dir / f"{idx:02d}_{seed_slug}.log"
        cmd = _build_seed_cmd(args, seed=seed, seed_multi_runs_root=seed_multi_runs_root)

        print(
            f"[dispatch] family={args.family} seed_idx={idx} seed={selected.factor_name} "
            f"kind={seed.get('seed_kind', selected.node_kind)} log={log_path}"
        )
        with log_path.open("w", encoding="utf-8") as log_fp:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=os.environ.copy(),
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )

        multi_run_dir = resolve_materialized_multi_run_dir(seed_multi_runs_root)
        child_records = load_multi_run_candidate_records(
            archive_db=str(args.archive_db),
            multi_run_dir=multi_run_dir,
            family=args.family,
        ) if multi_run_dir is not None else []
        expansion = engine.register_expansion(
            parent_node_id=selected.node_id,
            child_records=child_records,
            success=(proc.returncode == 0),
            source_run_id=str(multi_run_dir or ""),
            note=f"multi_run_dir={multi_run_dir}" if multi_run_dir is not None else "multi_run_dir_missing",
        )
        latest_winner = get_latest_family_winner(db_path=args.archive_db, family=args.family)
        dispatch_record = {
            "seed_index": idx,
            "seed": seed,
            "returncode": int(proc.returncode),
            "log_path": str(log_path),
            "selected_parent": selected.to_dict(),
            "multi_run_dir": str(multi_run_dir) if multi_run_dir is not None else "",
            "children_collected": len(child_records),
            "children_added_to_search": len(expansion.get("children", [])),
            "latest_winner_after_dispatch": latest_winner,
            "search_improved": bool(expansion.get("improved")),
            "search_summary": engine.summary(),
        }
        dispatches.append(dispatch_record)
        _write_json(
            summary_path,
            {
                "family": args.family,
                "target_profile": str(args.target_profile),
                "explore_dir": str(explore_dir),
                "frontier_size": len(frontier),
                "dispatches_completed": len(dispatches),
                "latest_winner": latest_winner,
                "dispatches": dispatches,
                "search": engine.summary(),
            },
        )

        if float(args.sleep_between_seeds) > 0 and engine.can_continue():
            time.sleep(float(args.sleep_between_seeds))

    failed = [item for item in dispatches if int(item["returncode"]) != 0]
    if failed:
        print(f"[summary] {len(failed)} / {len(dispatches)} seed dispatch(es) failed")
        print(f"[explore_dir] {explore_dir}")
        return 1

    _write_json(
        summary_path,
        {
            "family": args.family,
            "target_profile": str(args.target_profile),
            "explore_dir": str(explore_dir),
            "frontier_size": len(frontier),
            "dispatches_completed": len(dispatches),
            "latest_winner": get_latest_family_winner(db_path=args.archive_db, family=args.family),
            "dispatches": dispatches,
            "search": engine.summary(),
        },
    )
    print(f"[summary] all {len(dispatches)} seed dispatch(es) completed")
    print(f"[explore_dir] {explore_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
