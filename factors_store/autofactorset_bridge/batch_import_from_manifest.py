from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..registry import create_default_registry


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("manifest 顶层必须是 YAML object")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("manifest.candidates 必须是 list")
    return payload


def _validate_candidate(candidate: dict[str, Any], registry_names: set[str], registry_summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    name = str(candidate.get("factor_name", "")).strip()
    if not name:
        raise ValueError("candidate.factor_name 不能为空")
    exists = name in registry_names
    row = dict(candidate)
    row["registry_exists"] = exists
    if exists:
        spec = registry_summary[name]
        row["registry_source"] = spec.get("source", "")
        row["required_fields"] = spec.get("required_fields", "")
        row["expr_available"] = bool(spec.get("expr"))
        row["registry_expr"] = spec.get("expr", "")
        row["registry_notes"] = spec.get("notes", "")
    else:
        row["registry_source"] = ""
        row["required_fields"] = ""
        row["expr_available"] = False
        row["registry_expr"] = ""
        row["registry_notes"] = ""
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket = Counter()
    by_family = Counter()
    by_role = Counter()
    expr_ready = 0
    missing = []
    for row in rows:
        by_bucket[str(row.get("bucket", ""))] += 1
        by_family[str(row.get("family", ""))] += 1
        by_role[str(row.get("role", ""))] += 1
        expr_ready += int(bool(row.get("expr_available")))
        if not row.get("registry_exists"):
            missing.append(str(row.get("factor_name", "")))
    return {
        "total": len(rows),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_role": dict(sorted(by_role.items())),
        "registry_expr_ready": expr_ready,
        "registry_function_only": len(rows) - expr_ready,
        "missing_factor_names": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize factors_store -> autofactorset ingest manifest.")
    parser.add_argument("--manifest", required=True, help="YAML manifest path")
    parser.add_argument("--output", default=None, help="Optional JSON output path for validated manifest")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    payload = _load_manifest(manifest_path)

    registry = create_default_registry()
    summary_df = registry.summary()
    registry_rows = summary_df.to_dict("records")
    registry_names = {str(r["name"]) for r in registry_rows}
    registry_summary = {str(r["name"]): r for r in registry_rows}

    validated_rows = [
        _validate_candidate(dict(candidate), registry_names, registry_summary)
        for candidate in payload["candidates"]
    ]
    summary = _summarize(validated_rows)

    print(f"[manifest] {manifest_path}")
    print(f"[total] {summary['total']}")
    print(f"[expr_ready] {summary['registry_expr_ready']}")
    print(f"[function_only] {summary['registry_function_only']}")
    print("[by_bucket]")
    for k, v in summary["by_bucket"].items():
        print(f"  - {k}: {v}")
    print("[by_family]")
    for k, v in summary["by_family"].items():
        print(f"  - {k}: {v}")
    print("[by_role]")
    for k, v in summary["by_role"].items():
        print(f"  - {k}: {v}")
    if summary["missing_factor_names"]:
        print("[missing_factor_names]")
        for name in summary["missing_factor_names"]:
            print(f"  - {name}")
    else:
        print("[missing_factor_names] none")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "manifest_name": payload.get("manifest_name", manifest_path.stem),
            "manifest_path": str(manifest_path),
            "summary": summary,
            "candidates": validated_rows,
        }
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
