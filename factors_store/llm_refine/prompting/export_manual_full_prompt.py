from __future__ import annotations

import argparse
from pathlib import Path

from .prompt_builder import build_refinement_prompt
from ..core.seed_loader import DEFAULT_SEED_POOL, load_seed_pool, resolve_preferred_refine_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one manual full prompt using protocol-safe history stage.")
    parser.add_argument("--family", required=True, help="seed family name from the seed pool")
    parser.add_argument("--parent-name", default="", help="optional parent factor name; defaults to the family preferred refine seed")
    parser.add_argument("--seed-pool", default=str(DEFAULT_SEED_POOL), help="seed pool yaml path")
    parser.add_argument("--n-candidates", type=int, default=3, help="number of requested candidates")
    parser.add_argument("--output", required=True, help="output txt path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seed_pool = load_seed_pool(args.seed_pool)
    family = seed_pool.get_family(args.family)
    parent_name = args.parent_name.strip() or resolve_preferred_refine_seed(family)
    prompt = build_refinement_prompt(
        seed_pool=seed_pool,
        family=family,
        n_candidates=int(args.n_candidates),
        current_parent_name=parent_name,
    )
    text = f"系统提示词：\n{prompt.system_prompt}\n\n用户提示词：\n{prompt.user_prompt}\n"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
