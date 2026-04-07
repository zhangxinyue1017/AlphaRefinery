from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import yaml

from ..factors.seed_baselines import seed_baseline_catalog
from ..llm_refine.config import DEFAULT_AUTOFACTORSET_MANIFESTS_DIR, DEFAULT_SEED_POOL_PATH
from ..llm_refine.core.seed_loader import load_seed_pool


def _default_eval_defaults(seed_pool_path: str | Path = DEFAULT_SEED_POOL_PATH) -> dict[str, Any]:
    seed_pool_path = seed_pool_path or DEFAULT_SEED_POOL_PATH
    pool = load_seed_pool(seed_pool_path)
    protocol = pool.evaluation_protocol
    selection_start = ""
    selection_end = ""
    if protocol is not None:
        selection_start = str(protocol.selection.start or "")
        selection_end = str(protocol.selection.end or "")
    defaults = dict(pool.evaluation_defaults or {})
    data_dir = str(defaults.get("panel_path", "/root/dmd/BaoStock/panel.parquet"))
    return {
        "data_source": "baostock_parquet",
        "data_dir": data_dir,
        "eval_options": {
            "data_begin": selection_start,
            "data_end": selection_end,
            "n_days": 0,
            "n_sh_stocks": 0,
            "n_sz_stocks": 0,
            "sampling_mode": "latest_mcap_262",
            "sampling_seed": 42,
            "eval_backend": "python",
        },
    }


def build_seed_baseline_manifest_payload(
    *,
    seed_pool_path: str | Path | None = DEFAULT_SEED_POOL_PATH,
    families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    seed_pool_path = seed_pool_path or DEFAULT_SEED_POOL_PATH
    family_filter = {str(item).strip() for item in (families or ()) if str(item).strip()}
    catalog = [
        row
        for row in seed_baseline_catalog(seed_pool_path=seed_pool_path)
        if not family_filter or str(row.get("family", "")).strip() in family_filter
    ]
    family_names = tuple(dict.fromkeys(str(row["family"]) for row in catalog))
    ts_day = time.strftime("%Y%m%d", time.gmtime())
    manifest_stub = "all_families" if not family_filter else "_".join(family_names)
    manifest_name = f"seed_baselines_{manifest_stub}_{ts_day}"
    notes = [
        "Seed-baseline admission manifest built from refinement_seed_pool.yaml.",
        "Includes sign-aware canonical, alias, and preferred/oriented seed baseline registry factors.",
        "Use with evaluate_registry_manifest.py or evaluate_seed_baselines.py; insertion only happens with --insert-promoted.",
    ]
    if family_filter:
        notes.append(f"Family filter: {', '.join(family_names)}.")

    candidates = [
        {
            "factor_name": str(row["registry_factor_name"]),
            "bucket": "seed_baseline",
            "family": str(row["family"]),
            "role": str(row["role"]),
            "source_stage": "seed_baseline_registry",
            "source_model": "seed_pool",
            "source_report": "refinement_seed_pool",
            "seed_factor_name": str(row["seed_factor_name"]),
            "preferred_refine_seed": str(row["preferred_refine_seed"]),
            "is_preferred": bool(row["is_preferred"]),
            "direction": str(row["direction"]),
        }
        for row in catalog
    ]

    return {
        "manifest_name": manifest_name,
        "created_at": time.strftime("%Y-%m-%d", time.gmtime()),
        "source_reports": [str(Path(seed_pool_path).expanduser().resolve())],
        "notes": notes,
        "eval_defaults": _default_eval_defaults(seed_pool_path),
        "candidates": candidates,
    }


def write_seed_baseline_manifest(
    *,
    output_path: str | Path,
    seed_pool_path: str | Path | None = DEFAULT_SEED_POOL_PATH,
    families: list[str] | tuple[str, ...] | None = None,
) -> Path:
    seed_pool_path = seed_pool_path or DEFAULT_SEED_POOL_PATH
    payload = build_seed_baseline_manifest_payload(
        seed_pool_path=seed_pool_path,
        families=families,
    )
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a seed-baseline autofactorset manifest from refinement_seed_pool.yaml.")
    parser.add_argument("--output", default=None, help="Output manifest path; defaults to artifacts/autofactorset_ingest/manifests/<generated>.yaml")
    parser.add_argument("--seed-pool", default=str(DEFAULT_SEED_POOL_PATH), help="Seed pool yaml path")
    parser.add_argument("--families", nargs="*", default=None, help="Optional family filter list")
    args = parser.parse_args()

    payload = build_seed_baseline_manifest_payload(
        seed_pool_path=args.seed_pool,
        families=args.families,
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (DEFAULT_AUTOFACTORSET_MANIFESTS_DIR / f"{payload['manifest_name']}.yaml").resolve()
    )
    path = write_seed_baseline_manifest(
        output_path=output,
        seed_pool_path=args.seed_pool,
        families=args.families,
    )
    print(f"[manifest_name] {payload['manifest_name']}")
    print(f"[families] {len({str(item.get('family','')) for item in payload['candidates']})}")
    print(f"[candidates] {len(payload['candidates'])}")
    print(f"[output] {path}")


if __name__ == "__main__":
    main()
