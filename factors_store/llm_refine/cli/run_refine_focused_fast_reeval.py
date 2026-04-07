from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import DEFAULT_ARCHIVE_DB_PATH, DEFAULT_SEED_POOL_PATH
from ..knowledge.fast_reeval import fast_reevaluate_focused_run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast re-evaluate a focused family-loop scheduler run by patching parent baselines and refreshing gate/promotion summaries."
    )
    parser.add_argument(
        "--focused-run-dir",
        required=True,
        help="focused scheduler run dir under family_loop/focused_runs/<timestamp_family>",
    )
    parser.add_argument(
        "--family-loop-dir",
        default="",
        help="optional family-loop root; inferred automatically when omitted",
    )
    parser.add_argument(
        "--seed-pool",
        default=str(DEFAULT_SEED_POOL_PATH),
        help="seed pool yaml used to resolve family metadata",
    )
    parser.add_argument(
        "--archive-db",
        default=str(DEFAULT_ARCHIVE_DB_PATH),
        help="llm_refine archive sqlite path",
    )
    parser.add_argument(
        "--no-auto-apply-promotion",
        action="store_true",
        help="rejudge and rewrite summaries, but do not auto-apply pending promotion entries",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    report = fast_reevaluate_focused_run(
        focused_run_dir=Path(args.focused_run_dir),
        family_loop_dir=Path(args.family_loop_dir).expanduser().resolve() if str(args.family_loop_dir).strip() else None,
        seed_pool_path=args.seed_pool,
        archive_db=args.archive_db,
        auto_apply_promotion=not bool(args.no_auto_apply_promotion),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
