from __future__ import annotations

import argparse
from pathlib import Path

from ..llm_refine.config import DEFAULT_AUTOFACTORSET_MANIFESTS_DIR
from .build_seed_baseline_manifest import build_seed_baseline_manifest_payload, write_seed_baseline_manifest
from .evaluate_registry_manifest import evaluate_registry_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click seed-baseline admission: build manifest from seed pool, then evaluate and optionally insert promoted baselines."
    )
    parser.add_argument("--families", nargs="*", default=None, help="Optional family filter list")
    parser.add_argument("--manifest-out", default=None, help="Optional manifest output path")
    parser.add_argument("--run-root", default=None, help="Output root for autofactorset ingest run artifacts")
    parser.add_argument("--seed-pool", default=None, help="Optional override seed pool yaml path")
    parser.add_argument("--benchmark-path", default=None, help="Optional benchmark csv path")
    parser.add_argument("--label-horizon", type=int, default=1, help="Forward return horizon for bridge evaluation")
    parser.add_argument("--insert-promoted", action="store_true", help="Insert promoted factors into autofactorset SQLite library")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on candidate count for smoke testing")
    parser.add_argument("--manifest-only", action="store_true", help="Only build/write the manifest, skip evaluation")
    args = parser.parse_args()

    payload = build_seed_baseline_manifest_payload(
        seed_pool_path=args.seed_pool,
        families=args.families,
    )
    manifest_out = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out
        else (DEFAULT_AUTOFACTORSET_MANIFESTS_DIR / f"{payload['manifest_name']}.yaml").resolve()
    )
    manifest_path = write_seed_baseline_manifest(
        output_path=manifest_out,
        seed_pool_path=args.seed_pool,
        families=args.families,
    )

    print(f"[manifest] {manifest_path}")
    print(f"[families] {len({str(item.get('family','')) for item in payload['candidates']})}")
    print(f"[candidates] {len(payload['candidates'])}")

    if args.manifest_only:
        return

    result = evaluate_registry_manifest(
        manifest_path=manifest_path,
        run_root=args.run_root,
        benchmark_path=args.benchmark_path,
        label_horizon=int(args.label_horizon),
        insert_promoted=bool(args.insert_promoted),
        limit=args.limit,
    )
    print(f"[ok] {result['ok']}")
    print(f"[errors] {result['errors']}")
    print(f"[promoted] {result['promoted']}")
    print(f"[inserted] {result['inserted']}")
    print(f"[summary] {result['summary_path']}")
    print(f"[results] {result['results_path']}")


if __name__ == "__main__":
    main()
