'''CLI for staged broad-to-focused family refinement loops.

Coordinates anchor graduation, focused runs, correlation guards, and family-loop reports.
'''

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..config import (
    DEFAULT_AUTO_APPLY_PROMOTION,
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_PARENT_SIMILARITY,
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TURNOVER,
    DEFAULT_FAMILY_LOOP_ANCHOR_MIN_ICIR,
    DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_EXCESS_GAIN,
    DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_ICIR_GAIN,
    DEFAULT_FAMILY_LOOP_ANCHOR_MIN_METRICS_COMPLETENESS,
    DEFAULT_FAMILY_LOOP_ANCHOR_MIN_SHARPE,
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_PARENT_CORR,
    DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_SIBLING_CORR,
    DEFAULT_FAMILY_LOOP_BROAD_STAGE_PRESET,
    DEFAULT_FAMILY_LOOP_BROAD_MAX_ROUNDS,
    DEFAULT_FAMILY_LOOP_BROAD_POLICY_PRESET,
    DEFAULT_FAMILY_LOOP_BROAD_STOP_IF_NO_NEW_WINNER,
    DEFAULT_FAMILY_LOOP_FOCUSED_STAGE_PRESET,
    DEFAULT_FAMILY_LOOP_FOCUSED_MAX_ROUNDS,
    DEFAULT_FAMILY_LOOP_FOCUSED_N_CANDIDATES,
    DEFAULT_FAMILY_LOOP_FOCUSED_POLICY_PRESET,
    DEFAULT_FAMILY_LOOP_FOCUSED_STOP_IF_NO_NEW_WINNER,
    DEFAULT_FAMILY_LOOP_RUNS_DIR,
    FAMILY_LOOP_STAGE_PRESETS,
)
from ..core.archive import utc_now_iso
from ..knowledge.family_loop import (
    _read_json,
    _write_json,
    build_family_loop_summary,
    build_scheduler_cmd,
    collect_anchor_candidates,
    render_family_loop_markdown,
    resolve_stage_protocol,
    run_scheduler_stage,
    select_best_anchor,
)
from ..search import SearchPolicy
from .run_refine_multi_model_scheduler import build_arg_parser as build_scheduler_arg_parser


def _normalize_models(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if item and item not in out:
                out.append(item)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    scheduler = build_scheduler_arg_parser()
    parser = argparse.ArgumentParser(
        description="Run a family-level broad -> anchor selection -> focused refine loop."
    )
    parser.add_argument("--family", required=True, help="seed family name from the seed pool")
    parser.add_argument("--models", action="append", required=True, help="comma-separated model list; may be repeated")
    parser.add_argument(
        "--family-loop-runs-dir",
        default=str(DEFAULT_FAMILY_LOOP_RUNS_DIR),
        help="artifact root for family-loop orchestration logs",
    )

    parser.add_argument(
        "--broad-stage-preset",
        default=DEFAULT_FAMILY_LOOP_BROAD_STAGE_PRESET,
        choices=tuple(FAMILY_LOOP_STAGE_PRESETS.keys()),
        help="stage-aware protocol preset for the broad stage",
    )
    parser.add_argument(
        "--focused-stage-preset",
        default=DEFAULT_FAMILY_LOOP_FOCUSED_STAGE_PRESET,
        choices=tuple(FAMILY_LOOP_STAGE_PRESETS.keys()),
        help="stage-aware protocol preset for the focused stage",
    )
    parser.add_argument(
        "--broad-policy-preset",
        default=None,
        choices=SearchPolicy.available_presets(),
        help="optional override for the broad-stage policy preset",
    )
    parser.add_argument(
        "--focused-policy-preset",
        default=None,
        choices=SearchPolicy.available_presets(),
        help="optional override for the focused-stage policy preset",
    )
    parser.add_argument(
        "--target-profile",
        default=str(scheduler.get_default("target_profile")),
        choices=SearchPolicy.available_target_profiles(),
        help="target-conditioned profile shared by broad and focused stages",
    )
    parser.add_argument("--n-candidates", type=int, default=None, help="optional override for broad-stage candidate budget")
    parser.add_argument(
        "--focused-n-candidates",
        type=int,
        default=None,
        help="optional override for focused-stage candidate budget",
    )
    parser.add_argument("--broad-max-rounds", type=int, default=None, help="optional override for broad-stage max rounds")
    parser.add_argument(
        "--focused-max-rounds",
        type=int,
        default=None,
        help="optional override for focused-stage max rounds",
    )
    parser.add_argument(
        "--broad-stop-if-no-new-winner",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--focused-stop-if-no-new-winner",
        type=int,
        default=None,
    )

    parser.add_argument("--anchor-min-icir", type=float, default=DEFAULT_FAMILY_LOOP_ANCHOR_MIN_ICIR)
    parser.add_argument("--anchor-min-sharpe", type=float, default=DEFAULT_FAMILY_LOOP_ANCHOR_MIN_SHARPE)
    parser.add_argument("--anchor-max-turnover", type=float, default=DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TURNOVER)
    parser.add_argument(
        "--anchor-min-metrics-completeness",
        type=float,
        default=DEFAULT_FAMILY_LOOP_ANCHOR_MIN_METRICS_COMPLETENESS,
    )
    parser.add_argument(
        "--anchor-max-parent-similarity",
        type=float,
        default=DEFAULT_FAMILY_LOOP_ANCHOR_MAX_PARENT_SIMILARITY,
        help="corr-like redundancy guard: reject near-parent candidates without material gain",
    )
    parser.add_argument(
        "--anchor-max-true-parent-corr",
        type=float,
        default=DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_PARENT_CORR,
        help="true series correlation guard against near-parent candidates",
    )
    parser.add_argument(
        "--anchor-max-true-sibling-corr",
        type=float,
        default=DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_SIBLING_CORR,
        help="true series correlation guard against stronger sibling candidates",
    )
    parser.add_argument(
        "--anchor-min-material-excess-gain",
        type=float,
        default=DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_EXCESS_GAIN,
    )
    parser.add_argument(
        "--anchor-min-material-icir-gain",
        type=float,
        default=DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_ICIR_GAIN,
    )

    passthrough_defaults = {
        "--seed-pool": str(scheduler.get_default("seed_pool")),
        "--runs-dir": str(scheduler.get_default("runs_dir")),
        "--archive-db": str(scheduler.get_default("archive_db")),
        "--name-prefix": str(scheduler.get_default("name_prefix")),
        "--provider-name": str(scheduler.get_default("provider_name")),
        "--base-url": str(scheduler.get_default("base_url")),
        "--api-key": str(scheduler.get_default("api_key")),
        "--temperature": scheduler.get_default("temperature"),
        "--max-tokens": scheduler.get_default("max_tokens"),
        "--timeout": scheduler.get_default("timeout"),
        "--additional-notes": str(scheduler.get_default("additional_notes")),
        "--current-parent-name": str(scheduler.get_default("current_parent_name")),
        "--current-parent-expression": str(scheduler.get_default("current_parent_expression")),
        "--panel-path": str(scheduler.get_default("panel_path")),
        "--benchmark-path": str(scheduler.get_default("benchmark_path")),
        "--start": str(scheduler.get_default("start")),
        "--end": str(scheduler.get_default("end")),
        "--max-parallel": scheduler.get_default("max_parallel"),
    }
    for arg, default in passthrough_defaults.items():
        parser.add_argument(arg, default=default)

    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-apply-promotion", action=argparse.BooleanOptionalAction, default=DEFAULT_AUTO_APPLY_PROMOTION)
    parser.add_argument("--disable-mmr-rerank", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    models = _normalize_models(args.models)
    if not models:
        raise SystemExit("至少提供一个 model")
    broad_protocol = resolve_stage_protocol(
        stage_preset=str(args.broad_stage_preset),
        policy_preset=args.broad_policy_preset,
        n_candidates=args.n_candidates,
        max_rounds=args.broad_max_rounds,
        stop_if_no_new_winner=args.broad_stop_if_no_new_winner,
    )
    focused_protocol = resolve_stage_protocol(
        stage_preset=str(args.focused_stage_preset),
        policy_preset=args.focused_policy_preset,
        n_candidates=args.focused_n_candidates,
        max_rounds=args.focused_max_rounds,
        stop_if_no_new_winner=args.focused_stop_if_no_new_winner,
    )

    family_loop_dir = Path(args.family_loop_runs_dir).expanduser().resolve() / (
        time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_{args.family}"
    )
    family_loop_dir.mkdir(parents=True, exist_ok=True)
    plan_path = family_loop_dir / "plan.json"
    summary_json_path = family_loop_dir / "family_loop_summary.json"
    summary_md_path = family_loop_dir / "family_loop_summary.md"

    plan = {
        "family": args.family,
        "generated_at": utc_now_iso(),
        "models": models,
        "target_profile": str(args.target_profile),
        "broad_stage_preset": str(args.broad_stage_preset),
        "focused_stage_preset": str(args.focused_stage_preset),
        "broad_protocol": dict(broad_protocol),
        "focused_protocol": dict(focused_protocol),
        "anchor_gate": {
            "min_icir": float(args.anchor_min_icir),
            "min_sharpe": float(args.anchor_min_sharpe),
            "max_turnover": float(args.anchor_max_turnover),
            "min_metrics_completeness": float(args.anchor_min_metrics_completeness),
            "max_parent_similarity": float(args.anchor_max_parent_similarity),
            "max_true_parent_corr": float(args.anchor_max_true_parent_corr),
            "max_true_sibling_corr": float(args.anchor_max_true_sibling_corr),
            "min_material_excess_gain": float(args.anchor_min_material_excess_gain),
            "min_material_icir_gain": float(args.anchor_min_material_icir_gain),
        },
    }
    _write_json(plan_path, plan)

    broad_root = family_loop_dir / "broad_runs"
    broad_log = family_loop_dir / "broad.launch.log"
    broad_cmd = build_scheduler_cmd(
        python_executable=sys.executable,
        family=args.family,
        stage_mode=str(broad_protocol["stage_preset"]),
        scheduler_runs_dir=broad_root,
        models=models,
        seed_pool=args.seed_pool,
        n_candidates=int(broad_protocol["n_candidates"]),
        runs_dir=args.runs_dir,
        archive_db=args.archive_db,
        name_prefix=args.name_prefix,
        provider_name=args.provider_name,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        timeout=float(args.timeout),
        additional_notes=args.additional_notes,
        policy_preset=str(broad_protocol["policy_preset"]),
        target_profile=args.target_profile,
        panel_path=args.panel_path,
        benchmark_path=args.benchmark_path,
        start=args.start,
        end=args.end,
        max_parallel=int(args.max_parallel),
        max_rounds=int(broad_protocol["max_rounds"]),
        stop_if_no_new_winner=int(broad_protocol["stop_if_no_new_winner"]),
        skip_eval=bool(args.skip_eval),
        dry_run=bool(args.dry_run),
        auto_apply_promotion=bool(args.auto_apply_promotion),
        disable_mmr_rerank=bool(args.disable_mmr_rerank),
        current_parent_name=str(args.current_parent_name or ""),
        current_parent_expression=str(args.current_parent_expression or ""),
    )
    print(f"[family_loop] launch broad family={args.family} log={broad_log}")
    broad_returncode, broad_run_dir = run_scheduler_stage(cmd=broad_cmd, stage_root=broad_root, log_path=broad_log)
    broad_summary = _read_json((broad_run_dir or family_loop_dir) / "summary.json") if broad_run_dir else {}

    if broad_returncode != 0 or not broad_run_dir:
        summary = {
            "family": args.family,
            "target_profile": str(args.target_profile),
            "loop_dir": str(family_loop_dir),
            "generated_at": utc_now_iso(),
            "broad_run_dir": str(broad_run_dir) if broad_run_dir else "",
            "focused_run_dir": "",
            "broad_best_node": dict(broad_summary.get("best_node") or {}),
            "anchor_selection": {},
            "focused_best_node": {},
            "comparison": {},
            "recommended_next_step": "return_to_broad",
            "recommended_reason": "broad stage failed before anchor graduation",
            "broad_returncode": int(broad_returncode),
        }
        _write_json(summary_json_path, summary)
        summary_md_path.write_text(render_family_loop_markdown(summary), encoding="utf-8")
        return int(broad_returncode or 1)

    broad_parent_name = str(
        broad_summary.get("last_selected_parent_name")
        or broad_summary.get("search", {}).get("seed_node", {}).get("factor_name", "")
        or args.current_parent_name
        or ""
    )
    broad_parent_expression = str(
        broad_summary.get("last_selected_parent_expression")
        or broad_summary.get("search", {}).get("seed_node", {}).get("expression", "")
        or args.current_parent_expression
        or ""
    )
    anchor_policy = SearchPolicy.multi_model_best_first(preset=str(broad_protocol["policy_preset"])).with_target_profile(
        args.target_profile
    )
    collected = collect_anchor_candidates(
        archive_db=args.archive_db,
        scheduler_dir=broad_run_dir,
        seed_pool_path=args.seed_pool,
        family=args.family,
        parent_name=broad_parent_name,
        parent_expression=broad_parent_expression,
        panel_path=str(args.panel_path or ""),
        benchmark_path=str(args.benchmark_path or ""),
        start=str(args.start or ""),
        end=str(args.end or ""),
        policy=anchor_policy,
        max_parent_similarity=float(args.anchor_max_parent_similarity),
        max_true_parent_corr=float(args.anchor_max_true_parent_corr),
        max_true_sibling_corr=float(args.anchor_max_true_sibling_corr),
        min_material_excess_gain=float(args.anchor_min_material_excess_gain),
        min_material_icir_gain=float(args.anchor_min_material_icir_gain),
    )
    anchor_selection = select_best_anchor(
        collected=collected,
        target_profile=str(args.target_profile),
        min_icir=float(args.anchor_min_icir),
        min_sharpe=float(args.anchor_min_sharpe),
        max_turnover=float(args.anchor_max_turnover),
        min_metrics_completeness=float(args.anchor_min_metrics_completeness),
    )

    best_anchor = dict(anchor_selection.get("best_anchor") or {})
    focused_run_dir: Path | None = None
    focused_summary: dict[str, Any] = {}
    focused_returncode = 0
    if best_anchor:
        focused_root = family_loop_dir / "focused_runs"
        focused_log = family_loop_dir / "focused.launch.log"
        focused_cmd = build_scheduler_cmd(
            python_executable=sys.executable,
            family=args.family,
            stage_mode=str(focused_protocol["stage_preset"]),
            scheduler_runs_dir=focused_root,
            models=models,
            seed_pool=args.seed_pool,
            n_candidates=int(focused_protocol["n_candidates"]),
            runs_dir=args.runs_dir,
            archive_db=args.archive_db,
            name_prefix=args.name_prefix,
            provider_name=args.provider_name,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout=float(args.timeout),
            additional_notes=args.additional_notes,
            policy_preset=str(focused_protocol["policy_preset"]),
            target_profile=args.target_profile,
            panel_path=args.panel_path,
            benchmark_path=args.benchmark_path,
            start=args.start,
            end=args.end,
            max_parallel=int(args.max_parallel),
            max_rounds=int(focused_protocol["max_rounds"]),
            stop_if_no_new_winner=int(focused_protocol["stop_if_no_new_winner"]),
            skip_eval=bool(args.skip_eval),
            dry_run=bool(args.dry_run),
            auto_apply_promotion=bool(args.auto_apply_promotion),
            disable_mmr_rerank=bool(args.disable_mmr_rerank),
            current_parent_name=str(best_anchor.get("factor_name", "")),
            current_parent_expression=str(best_anchor.get("expression", "")),
        )
        print(f"[family_loop] launch focused family={args.family} log={focused_log}")
        focused_returncode, focused_run_dir = run_scheduler_stage(
            cmd=focused_cmd, stage_root=focused_root, log_path=focused_log
        )
        focused_summary = _read_json((focused_run_dir or family_loop_dir) / "summary.json") if focused_run_dir else {}

    summary = build_family_loop_summary(
        family=args.family,
        target_profile=str(args.target_profile),
        loop_dir=family_loop_dir,
        broad_stage_preset=str(broad_protocol["stage_preset"]),
        focused_stage_preset=str(focused_protocol["stage_preset"]),
        broad_run_dir=broad_run_dir,
        broad_summary=broad_summary,
        anchor_selection=anchor_selection,
        focused_run_dir=focused_run_dir,
        focused_summary=focused_summary,
        broad_returncode=int(broad_returncode),
        focused_returncode=int(focused_returncode),
    )
    _write_json(summary_json_path, summary)
    summary_md_path.write_text(render_family_loop_markdown(summary), encoding="utf-8")
    return int(focused_returncode if best_anchor else 0)


if __name__ == "__main__":
    raise SystemExit(main())
