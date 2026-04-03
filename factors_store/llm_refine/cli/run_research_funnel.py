from __future__ import annotations

import argparse

from ..core.seed_loader import DEFAULT_SEED_POOL
from ..evaluation.research_funnel import (
    DEFAULT_AUTOFACTORSET_RUNS_DIR,
    DEFAULT_EVALUATOR_OUTPUT_DIR,
    DEFAULT_SCHEDULER_RUNS_DIR,
    DEFAULT_SINGLE_RUNS_DIR,
    build_run_uplift_records,
    summarize_family_funnel,
    summarize_family_profile_funnel,
    write_research_funnel_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build llm_refine research funnel evaluation artifacts.")
    parser.add_argument("--seed-pool", default=str(DEFAULT_SEED_POOL), help="seed pool yaml path")
    parser.add_argument("--single-runs-dir", default=str(DEFAULT_SINGLE_RUNS_DIR), help="llm_refine_single runs root")
    parser.add_argument("--scheduler-runs-dir", default=str(DEFAULT_SCHEDULER_RUNS_DIR), help="llm_refine_multi_scheduler runs root")
    parser.add_argument("--autofactorset-runs-dir", default=str(DEFAULT_AUTOFACTORSET_RUNS_DIR), help="autofactorset ingest runs root")
    parser.add_argument("--output-dir", default=str(DEFAULT_EVALUATOR_OUTPUT_DIR), help="output report directory")
    parser.add_argument("--family", default="", help="optional family filter")
    parser.add_argument("--include-incomplete", action="store_true", help="include scheduler runs whose top-level summary has no stop_reason yet")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_records = build_run_uplift_records(
        seed_pool_path=args.seed_pool,
        single_runs_dir=args.single_runs_dir,
        scheduler_runs_dir=args.scheduler_runs_dir,
        autofactorset_runs_dir=args.autofactorset_runs_dir,
        family=args.family,
        include_incomplete=bool(args.include_incomplete),
    )
    family_records = summarize_family_funnel(
        run_records,
    )
    family_profile_records = summarize_family_profile_funnel(run_records)
    outputs = write_research_funnel_outputs(
        run_records=run_records,
        family_records=family_records,
        family_profile_records=family_profile_records,
        output_dir=args.output_dir,
    )
    for key, path in outputs.items():
        print(f"[saved] {key}={path}")
    print(f"[runs] {len(run_records)}")
    print(f"[families] {len(family_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
