from __future__ import annotations

import argparse
import json

from ..config import (
    DEFAULT_NEXT_EXPERIMENTS_MAX_SUGGESTIONS_PER_FAMILY,
    DEFAULT_NEXT_EXPERIMENTS_OUTPUT_DIR,
)
from ..core.archive import DEFAULT_ARCHIVE_DB
from ..core.seed_loader import DEFAULT_SEED_POOL
from ..knowledge.next_experiments import write_next_experiments_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a lightweight motif-transfer / next-experiments plan from seed pool + archive."
    )
    parser.add_argument("--seed-pool", default=str(DEFAULT_SEED_POOL), help="Seed pool YAML path")
    parser.add_argument("--archive-db", default=str(DEFAULT_ARCHIVE_DB), help="llm_refine archive sqlite path")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_NEXT_EXPERIMENTS_OUTPUT_DIR),
        help="Output directory for markdown/json plan files",
    )
    parser.add_argument(
        "--max-suggestions-per-family",
        type=int,
        default=DEFAULT_NEXT_EXPERIMENTS_MAX_SUGGESTIONS_PER_FAMILY,
        help="How many motif-transfer suggestions to keep per family",
    )
    args = parser.parse_args()

    paths = write_next_experiments_report(
        seed_pool_path=args.seed_pool,
        db_path=args.archive_db,
        out_dir=args.out_dir,
        max_suggestions_per_family=max(1, int(args.max_suggestions_per_family)),
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
