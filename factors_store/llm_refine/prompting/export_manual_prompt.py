from __future__ import annotations

import argparse
from pathlib import Path

from .prompt_builder import PROMPT_TEMPLATE_VERSIONS, export_manual_full_prompt
from ..core.seed_loader import DEFAULT_SEED_POOL, load_seed_pool


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a full manual prompt using llm_refine.prompt_builder.")
    parser.add_argument("--family", required=True, help="seed family name")
    parser.add_argument("--output", required=True, help="output txt path")
    parser.add_argument("--seed-pool", default=str(DEFAULT_SEED_POOL), help="seed pool yaml path")
    parser.add_argument("--n-candidates", type=int, default=3, help="number of candidates requested in prompt")
    parser.add_argument("--parent-name", default="", help="override current parent factor name")
    parser.add_argument("--parent-expression", default="", help="override current parent expression")
    parser.add_argument("--additional-notes", default="", help="extra note block appended into user prompt")
    parser.add_argument(
        "--prompt-template-version",
        default="current_compact",
        choices=PROMPT_TEMPLATE_VERSIONS,
        help="prompt template variant to export",
    )
    args = parser.parse_args()

    seed_pool = load_seed_pool(args.seed_pool)
    family = seed_pool.get_family(args.family)
    export_manual_full_prompt(
        seed_pool=seed_pool,
        family=family,
        output_path=Path(args.output),
        n_candidates=int(args.n_candidates),
        additional_notes=args.additional_notes,
        current_parent_name=args.parent_name or None,
        current_parent_expression=args.parent_expression or None,
        prompt_template_version=str(args.prompt_template_version),
    )
    print(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
