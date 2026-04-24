'''Artifact ingestion for evaluated multi-model runs.

Loads child summaries, resolved run directories, candidate records, and evaluation metrics into search nodes.
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.archive import get_run_record_by_dir, load_run_candidate_records


def resolve_materialized_single_run_dir(round_runs_root: Path) -> Path | None:
    if not round_runs_root.exists():
        return None
    children = sorted(p for p in round_runs_root.iterdir() if p.is_dir())
    return children[-1] if children else None


def resolve_materialized_multi_run_dir(round_multi_root: Path) -> Path | None:
    if not round_multi_root.exists():
        return None
    children = sorted(p for p in round_multi_root.iterdir() if p.is_dir())
    return children[-1] if children else None


def resolve_materialized_child_run_dir(child_runs_dir: str | Path) -> Path | None:
    root = Path(child_runs_dir)
    if not root.exists():
        return None
    children = sorted(p for p in root.iterdir() if p.is_dir())
    return children[-1] if children else None


def load_multi_run_candidate_records(
    *,
    archive_db: str,
    multi_run_dir: Path,
    family: str,
    statuses: tuple[str, ...] = ("research_winner", "winner", "research_keep", "keep"),
) -> list[dict[str, Any]]:
    summary_path = multi_run_dir / "summary.json"
    if not summary_path.exists():
        return []

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return load_candidate_records_from_completed_runs(
        archive_db=archive_db,
        completed=payload.get("completed", []),
        family=family,
        statuses=statuses,
    )


def load_candidate_records_from_completed_runs(
    *,
    archive_db: str,
    completed: list[dict[str, Any]],
    family: str,
    statuses: tuple[str, ...] = ("research_winner", "winner", "research_keep", "keep"),
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in completed:
        child_run_dir = resolve_materialized_child_run_dir(item.get("child_runs_dir", ""))
        if child_run_dir is None:
            continue
        run_record = get_run_record_by_dir(db_path=archive_db, run_dir=child_run_dir)
        if run_record is None or str(run_record.get("family", "")) != family:
            continue
        rows = load_run_candidate_records(
            db_path=archive_db,
            run_id=str(run_record.get("run_id", "")),
            statuses=statuses,
        )
        for row in rows:
            candidate_id = str(row.get("candidate_id", "")).strip()
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            out.append(row)
    return out


def load_single_run_candidate_records(
    *,
    archive_db: str,
    run_dir: Path,
    family: str,
    statuses: tuple[str, ...] = ("research_winner", "winner", "research_keep", "keep"),
) -> list[dict[str, Any]]:
    run_record = get_run_record_by_dir(db_path=archive_db, run_dir=run_dir)
    if run_record is None or str(run_record.get("family", "")) != family:
        return []
    return load_run_candidate_records(
        db_path=archive_db,
        run_id=str(run_record.get("run_id", "")),
        statuses=statuses,
    )
