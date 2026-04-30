'''SQLite archive access for refinement runs and candidates.

Stores evaluated candidates, fetches family history, and resolves best or latest parent nodes.
'''

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..config import DEFAULT_ARCHIVE_DB_PATH, DEFAULT_SINGLE_RUNS_DIR
from .models import LLMProposal, PromptBundle, SeedFamily

DEFAULT_RUNS_DIR = DEFAULT_SINGLE_RUNS_DIR
DEFAULT_ARCHIVE_DB = DEFAULT_ARCHIVE_DB_PATH


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def create_run_dir(*, family: str, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_family = family.replace("/", "_").replace(" ", "_")
    run_dir = Path(runs_dir) / f"{stamp}_{safe_family}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_run_subdir(run_dir: str | Path, name: str) -> Path:
    path = Path(run_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_seed_candidate_id(factor_name: str) -> str:
    return f"seed::{factor_name}"


def make_candidate_id(*, family: str, round_id: int, expression: str, name: str) -> str:
    digest = hashlib.sha1(f"{family}|{round_id}|{name}|{expression}".encode("utf-8")).hexdigest()[:16]
    return f"cand::{family}::r{round_id}::{digest}"


def expression_hash(expression: str) -> str:
    return hashlib.sha1(expression.encode("utf-8")).hexdigest()


def make_run_id(*, family: str, round_id: int, started_at: str | None = None) -> str:
    stamp = started_at or utc_now_iso()
    digest = hashlib.sha1(f"{family}|{round_id}|{stamp}".encode("utf-8")).hexdigest()[:12]
    return f"run::{family}::r{round_id}::{digest}"


def init_archive_db(db_path: str | Path = DEFAULT_ARCHIVE_DB) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                family TEXT NOT NULL,
                canonical_seed TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                parent_candidate_id TEXT,
                provider TEXT,
                model TEXT,
                prompt_hash TEXT,
                run_dir TEXT,
                status TEXT,
                started_at TEXT,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                family TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                parent_candidate_id TEXT,
                factor_name TEXT NOT NULL,
                expression TEXT NOT NULL,
                expression_hash TEXT NOT NULL,
                candidate_role TEXT,
                source_model TEXT,
                source_provider TEXT,
                explanation TEXT,
                rationale TEXT,
                validation_warnings TEXT,
                filter_stage TEXT,
                filter_reason TEXT,
                status TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                candidate_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                quick_rank_ic_mean REAL,
                quick_rank_icir REAL,
                net_ann_return REAL,
                net_excess_ann_return REAL,
                net_sharpe REAL,
                mean_turnover REAL,
                decision TEXT,
                decision_reason TEXT,
                evaluation_stage TEXT,
                decision_scope TEXT,
                decorrelation_grade TEXT,
                decorrelation_score REAL,
                nearest_decorrelation_target TEXT,
                corr_to_nearest_decorrelation_target REAL,
                avg_abs_decorrelation_target_corr REAL,
                evaluated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS lineage (
                family TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                parent_candidate_id TEXT NOT NULL,
                child_candidate_id TEXT NOT NULL,
                PRIMARY KEY (parent_candidate_id, child_candidate_id)
            );
            """
        )
        if not _has_column(conn, "candidates", "candidate_role"):
            conn.execute("ALTER TABLE candidates ADD COLUMN candidate_role TEXT")
        if not _has_column(conn, "candidates", "filter_stage"):
            conn.execute("ALTER TABLE candidates ADD COLUMN filter_stage TEXT")
        if not _has_column(conn, "candidates", "filter_reason"):
            conn.execute("ALTER TABLE candidates ADD COLUMN filter_reason TEXT")
        if not _has_column(conn, "evaluations", "evaluation_stage"):
            conn.execute("ALTER TABLE evaluations ADD COLUMN evaluation_stage TEXT")
        if not _has_column(conn, "evaluations", "decision_scope"):
            conn.execute("ALTER TABLE evaluations ADD COLUMN decision_scope TEXT")
        if not _has_column(conn, "evaluations", "decorrelation_grade"):
            conn.execute("ALTER TABLE evaluations ADD COLUMN decorrelation_grade TEXT")
        if not _has_column(conn, "evaluations", "decorrelation_score"):
            conn.execute("ALTER TABLE evaluations ADD COLUMN decorrelation_score REAL")
        if not _has_column(conn, "evaluations", "nearest_decorrelation_target"):
            conn.execute("ALTER TABLE evaluations ADD COLUMN nearest_decorrelation_target TEXT")
        if not _has_column(conn, "evaluations", "corr_to_nearest_decorrelation_target"):
            conn.execute("ALTER TABLE evaluations ADD COLUMN corr_to_nearest_decorrelation_target REAL")
        if not _has_column(conn, "evaluations", "avg_abs_decorrelation_target_corr"):
            conn.execute("ALTER TABLE evaluations ADD COLUMN avg_abs_decorrelation_target_corr REAL")
    return path


def insert_run(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str,
    family: str,
    canonical_seed: str,
    round_id: int,
    parent_candidate_id: str,
    provider: str,
    model: str,
    prompt_hash: str,
    run_dir: str,
    status: str,
    started_at: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, family, canonical_seed, round_id, parent_candidate_id,
                provider, model, prompt_hash, run_dir, status, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT finished_at FROM runs WHERE run_id = ?), NULL))
            """,
            (
                run_id,
                family,
                canonical_seed,
                int(round_id),
                parent_candidate_id or "",
                provider,
                model,
                prompt_hash,
                run_dir,
                status,
                started_at,
                run_id,
            ),
        )


def mark_run_finished(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str,
    status: str,
    finished_at: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, finished_at or utc_now_iso(), run_id),
        )


def insert_candidates(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str,
    candidates: list[dict[str, Any]],
) -> None:
    init_archive_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO candidates (
                candidate_id, run_id, family, round_id, parent_candidate_id,
                factor_name, expression, expression_hash, candidate_role, source_model, source_provider,
                explanation, rationale, validation_warnings, filter_stage, filter_reason, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["candidate_id"],
                    run_id,
                    item["family"],
                    int(item["round_id"]),
                    item.get("parent_candidate_id", ""),
                    item["factor_name"],
                    item["expression"],
                    item["expression_hash"],
                    item.get("candidate_role", ""),
                    item.get("source_model", ""),
                    item.get("source_provider", ""),
                    item.get("explanation", ""),
                    item.get("rationale", ""),
                    json.dumps(item.get("validation_warnings", []), ensure_ascii=False),
                    item.get("filter_stage", ""),
                    item.get("filter_reason", ""),
                    item.get("status", "proposed"),
                    item.get("created_at", utc_now_iso()),
                )
                for item in candidates
            ],
        )


def insert_evaluations(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str,
    rows: list[dict[str, Any]],
) -> None:
    init_archive_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO evaluations (
                candidate_id, run_id, quick_rank_ic_mean, quick_rank_icir,
                net_ann_return, net_excess_ann_return, net_sharpe, mean_turnover,
                decision, decision_reason, evaluation_stage, decision_scope,
                decorrelation_grade, decorrelation_score, nearest_decorrelation_target,
                corr_to_nearest_decorrelation_target, avg_abs_decorrelation_target_corr,
                evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["candidate_id"],
                    run_id,
                    row.get("quick_rank_ic_mean"),
                    row.get("quick_rank_icir"),
                    row.get("net_ann_return"),
                    row.get("net_excess_ann_return"),
                    row.get("net_sharpe"),
                    row.get("mean_turnover"),
                    row.get("decision"),
                    row.get("decision_reason"),
                    row.get("evaluation_stage"),
                    row.get("decision_scope"),
                    row.get("decorrelation_grade"),
                    row.get("decorrelation_score"),
                    row.get("nearest_decorrelation_target"),
                    row.get("corr_to_nearest_decorrelation_target"),
                    row.get("avg_abs_decorrelation_target_corr"),
                    row.get("evaluated_at", utc_now_iso()),
                )
                for row in rows
                if row.get("candidate_id")
            ],
        )


def update_candidate_statuses(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    statuses: list[tuple[str, str]],
) -> None:
    if not statuses:
        return
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE candidates SET status = ? WHERE candidate_id = ?",
            [(status, candidate_id) for candidate_id, status in statuses if candidate_id],
        )


def update_candidate_filter_metadata(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    updates: list[tuple[str, str, str, str]],
) -> None:
    if not updates:
        return
    init_archive_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            UPDATE candidates
            SET status = ?, filter_stage = ?, filter_reason = ?
            WHERE candidate_id = ?
            """,
            [(status, stage, reason, candidate_id) for candidate_id, status, stage, reason in updates if candidate_id],
        )


def insert_lineage(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    round_id: int,
    parent_child_pairs: list[tuple[str, str]],
) -> None:
    if not parent_child_pairs:
        return
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO lineage (family, round_id, parent_candidate_id, child_candidate_id)
            VALUES (?, ?, ?, ?)
            """,
            [(family, int(round_id), parent, child) for parent, child in parent_child_pairs if parent and child],
        )


def load_family_reference_candidates(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
    exclude_run_id: str = "",
    statuses: tuple[str, ...] = ("research_winner", "research_keep", "winner", "keep"),
    limit: int | None = 8,
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in statuses)
    params: list[Any] = [family, *statuses]
    sql = f"""
        SELECT c.candidate_id, c.factor_name, c.expression, c.status, c.run_id,
               c.source_model
        FROM candidates c
        WHERE c.family = ?
          AND c.status IN ({placeholders})
    """
    if exclude_run_id:
        sql += " AND c.run_id != ?"
        params.append(exclude_run_id)
    sql += " ORDER BY c.created_at DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "candidate_id": row[0],
            "factor_name": row[1],
            "expression": row[2],
            "status": row[3],
            "run_id": row[4],
            "source_model": row[5],
        }
        for row in rows
    ]


def get_latest_family_round(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(round_id) FROM runs WHERE family = ?",
            (family,),
        ).fetchone()
    value = row[0] if row else None
    return int(value) if value is not None else 0


def get_run_record(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str,
) -> dict[str, Any] | None:
    if not run_id:
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT run_id, family, canonical_seed, round_id, parent_candidate_id,
                   provider, model, prompt_hash, run_dir, status, started_at, finished_at
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return None
    keys = (
        "run_id",
        "family",
        "canonical_seed",
        "round_id",
        "parent_candidate_id",
        "provider",
        "model",
        "prompt_hash",
        "run_dir",
        "status",
        "started_at",
        "finished_at",
    )
    return dict(zip(keys, row))


def get_run_record_by_dir(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_dir: str | Path,
) -> dict[str, Any] | None:
    path = str(Path(run_dir).expanduser().resolve())
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT run_id, family, canonical_seed, round_id, parent_candidate_id,
                   provider, model, prompt_hash, run_dir, status, started_at, finished_at
            FROM runs
            WHERE run_dir = ?
            ORDER BY started_at DESC, run_id DESC
            LIMIT 1
            """,
            (path,),
        ).fetchone()
    if row is None:
        return None
    keys = (
        "run_id",
        "family",
        "canonical_seed",
        "round_id",
        "parent_candidate_id",
        "provider",
        "model",
        "prompt_hash",
        "run_dir",
        "status",
        "started_at",
        "finished_at",
    )
    return dict(zip(keys, row))


def get_latest_family_run(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT run_id, family, canonical_seed, round_id, parent_candidate_id,
                   provider, model, prompt_hash, run_dir, status, started_at, finished_at
            FROM runs
            WHERE family = ?
            ORDER BY started_at DESC, run_id DESC
            LIMIT 1
            """,
            (family,),
        ).fetchone()
    if row is None:
        return None
    keys = (
        "run_id",
        "family",
        "canonical_seed",
        "round_id",
        "parent_candidate_id",
        "provider",
        "model",
        "prompt_hash",
        "run_dir",
        "status",
        "started_at",
        "finished_at",
    )
    return dict(zip(keys, row))


def load_run_candidate_records(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    run_id: str,
    statuses: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    init_archive_db(db_path)
    if not run_id:
        return []
    params: list[Any] = [run_id]
    status_sql = ""
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        status_sql = f" AND c.status IN ({placeholders})"
        params.extend(statuses)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                c.candidate_id,
                c.run_id,
                c.family,
                c.round_id,
                c.parent_candidate_id,
                c.factor_name,
                c.expression,
                c.candidate_role,
                c.source_model,
                c.source_provider,
                c.status,
                e.quick_rank_ic_mean,
                e.quick_rank_icir,
                e.net_ann_return,
                e.net_excess_ann_return,
                e.net_sharpe,
                e.mean_turnover,
                e.evaluation_stage,
                e.decision_scope,
                e.decorrelation_grade,
                e.decorrelation_score,
                e.nearest_decorrelation_target,
                e.corr_to_nearest_decorrelation_target,
                e.avg_abs_decorrelation_target_corr,
                e.evaluated_at
            FROM candidates c
            LEFT JOIN evaluations e
              ON c.candidate_id = e.candidate_id
            WHERE c.run_id = ?
            {status_sql}
            ORDER BY COALESCE(e.evaluated_at, c.created_at) DESC, c.candidate_id DESC
            """,
            params,
        ).fetchall()
    keys = (
        "candidate_id",
        "run_id",
        "family",
        "round_id",
        "parent_candidate_id",
        "factor_name",
        "expression",
        "candidate_role",
        "source_model",
        "source_provider",
        "status",
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "net_excess_ann_return",
        "net_sharpe",
        "mean_turnover",
        "evaluation_stage",
        "decision_scope",
        "decorrelation_grade",
        "decorrelation_score",
        "nearest_decorrelation_target",
        "corr_to_nearest_decorrelation_target",
        "avg_abs_decorrelation_target_corr",
        "evaluated_at",
    )
    return [dict(zip(keys, row)) for row in rows]


def get_candidate_record(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    candidate_id: str,
) -> dict[str, Any] | None:
    init_archive_db(db_path)
    if not candidate_id:
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                c.candidate_id,
                c.run_id,
                c.family,
                c.round_id,
                c.parent_candidate_id,
                c.factor_name,
                c.expression,
                c.candidate_role,
                c.source_model,
                c.source_provider,
                c.status,
                e.quick_rank_ic_mean,
                e.quick_rank_icir,
                e.net_ann_return,
                e.net_excess_ann_return,
                e.net_sharpe,
                e.mean_turnover,
                e.evaluation_stage,
                e.decision_scope,
                e.decorrelation_grade,
                e.decorrelation_score,
                e.nearest_decorrelation_target,
                e.corr_to_nearest_decorrelation_target,
                e.avg_abs_decorrelation_target_corr
            FROM candidates c
            LEFT JOIN evaluations e
              ON c.candidate_id = e.candidate_id
            WHERE c.candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()
    if row is None:
        return None
    keys = (
        "candidate_id",
        "run_id",
        "family",
        "round_id",
        "parent_candidate_id",
        "factor_name",
        "expression",
        "candidate_role",
        "source_model",
        "source_provider",
        "status",
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "net_excess_ann_return",
        "net_sharpe",
        "mean_turnover",
        "evaluation_stage",
        "decision_scope",
        "decorrelation_grade",
        "decorrelation_score",
        "nearest_decorrelation_target",
        "corr_to_nearest_decorrelation_target",
        "avg_abs_decorrelation_target_corr",
    )
    return dict(zip(keys, row))


def get_latest_family_winner(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
) -> dict[str, Any] | None:
    init_archive_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                c.candidate_id,
                c.run_id,
                c.family,
                c.round_id,
                c.parent_candidate_id,
                c.factor_name,
                c.expression,
                c.candidate_role,
                c.source_model,
                c.source_provider,
                c.status,
                e.quick_rank_ic_mean,
                e.quick_rank_icir,
                e.net_ann_return,
                e.net_excess_ann_return,
                e.net_sharpe,
                e.mean_turnover,
                e.evaluation_stage,
                e.decision_scope,
                e.decorrelation_grade,
                e.decorrelation_score,
                e.nearest_decorrelation_target,
                e.corr_to_nearest_decorrelation_target,
                e.avg_abs_decorrelation_target_corr,
                e.evaluated_at
            FROM candidates c
            LEFT JOIN evaluations e
              ON c.candidate_id = e.candidate_id
            WHERE c.family = ?
              AND c.status IN ('research_winner', 'winner')
            ORDER BY c.round_id DESC, e.evaluated_at DESC, c.candidate_id DESC
            LIMIT 1
            """,
            (family,),
        ).fetchone()
    if row is None:
        return None
    keys = (
        "candidate_id",
        "run_id",
        "family",
        "round_id",
        "parent_candidate_id",
        "factor_name",
        "expression",
        "candidate_role",
        "source_model",
        "source_provider",
        "status",
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "net_excess_ann_return",
        "net_sharpe",
        "mean_turnover",
        "evaluation_stage",
        "decision_scope",
        "decorrelation_grade",
        "decorrelation_score",
        "nearest_decorrelation_target",
        "corr_to_nearest_decorrelation_target",
        "avg_abs_decorrelation_target_corr",
        "evaluated_at",
    )
    return dict(zip(keys, row))


def get_best_family_winner(
    *,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    family: str,
) -> dict[str, Any] | None:
    init_archive_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                c.candidate_id,
                c.run_id,
                c.family,
                c.round_id,
                c.parent_candidate_id,
                c.factor_name,
                c.expression,
                c.candidate_role,
                c.source_model,
                c.source_provider,
                c.status,
                e.quick_rank_ic_mean,
                e.quick_rank_icir,
                e.net_ann_return,
                e.net_excess_ann_return,
                e.net_sharpe,
                e.mean_turnover,
                e.evaluation_stage,
                e.decision_scope,
                e.decorrelation_grade,
                e.decorrelation_score,
                e.nearest_decorrelation_target,
                e.corr_to_nearest_decorrelation_target,
                e.avg_abs_decorrelation_target_corr,
                e.evaluated_at
            FROM candidates c
            LEFT JOIN evaluations e
              ON c.candidate_id = e.candidate_id
            WHERE c.family = ?
              AND c.status IN ('research_winner', 'winner')
            ORDER BY
                COALESCE(e.quick_rank_ic_mean, -1e9) DESC,
                COALESCE(e.quick_rank_icir, -1e9) DESC,
                COALESCE(e.net_excess_ann_return, -1e9) DESC,
                COALESCE(e.net_sharpe, -1e9) DESC,
                COALESCE(e.net_ann_return, -1e9) DESC,
                COALESCE(e.mean_turnover, 1e9) ASC,
                COALESCE(e.evaluated_at, c.created_at) DESC,
                c.candidate_id DESC
            LIMIT 1
            """,
            (family,),
        ).fetchone()
    if row is None:
        return None
    keys = (
        "candidate_id",
        "run_id",
        "family",
        "round_id",
        "parent_candidate_id",
        "factor_name",
        "expression",
        "candidate_role",
        "source_model",
        "source_provider",
        "status",
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "net_excess_ann_return",
        "net_sharpe",
        "mean_turnover",
        "evaluation_stage",
        "decision_scope",
        "decorrelation_grade",
        "decorrelation_score",
        "nearest_decorrelation_target",
        "corr_to_nearest_decorrelation_target",
        "avg_abs_decorrelation_target_corr",
        "evaluated_at",
    )
    return dict(zip(keys, row))


def _proposal_markdown(family: SeedFamily, proposal: LLMProposal) -> str:
    lines = [
        "# LLM Refine Run",
        "",
        f"- family: `{family.family}`",
        f"- canonical_seed: `{family.canonical_seed}`",
        f"- parent_factor: `{proposal.parent_factor}`",
        "",
        "## Diagnosed Weaknesses",
    ]
    lines.extend(f"- {item}" for item in proposal.diagnosed_weaknesses)
    lines.extend(["", "## Refinement Rationale", proposal.refinement_rationale or "(empty)", "", "## Candidates"])
    for candidate in proposal.candidates:
        lines.extend(
            [
                f"### {candidate.name}",
                f"- candidate_role: {candidate.candidate_role or '(empty)'}",
                f"- expression: `{candidate.expression}`",
                f"- explanation: {candidate.explanation}",
                f"- rationale: {candidate.rationale or '(empty)'}",
                f"- expected_improvement: {candidate.expected_improvement or '(empty)'}",
                f"- risk: {candidate.risk or '(empty)'}",
            ]
        )
        if candidate.validation_warnings:
            lines.append(f"- validation_warnings: {list(candidate.validation_warnings)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_run_artifacts(
    *,
    run_dir: Path,
    family: SeedFamily,
    prompt: PromptBundle,
    proposal: LLMProposal,
    provider_payload: dict[str, Any],
    name_prefix: str = "llmgen",
) -> dict[str, Path]:
    prompts_dir = ensure_run_subdir(run_dir, "prompts")
    proposal_dir = ensure_run_subdir(run_dir, "proposal")
    metadata_dir = ensure_run_subdir(run_dir, "metadata")

    system_path = prompts_dir / "system_prompt.txt"
    user_path = prompts_dir / "user_prompt.txt"
    raw_path = proposal_dir / "raw_response.txt"
    parsed_path = proposal_dir / "parsed_proposal.json"
    library_path = proposal_dir / "candidate_library.yaml"
    report_path = proposal_dir / "proposal_report.md"
    meta_path = metadata_dir / "run_meta.json"

    system_path.write_text(prompt.system_prompt, encoding="utf-8")
    user_path.write_text(prompt.user_prompt, encoding="utf-8")
    raw_path.write_text(proposal.raw_response, encoding="utf-8")
    parsed_path.write_text(json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    yaml.safe_dump(
        {
            "metadata": {
                "family": family.family,
                "canonical_seed": family.canonical_seed,
                "provider": provider_payload,
            },
            "factors": [candidate.to_library_item(name_prefix=name_prefix) for candidate in proposal.candidates],
        },
        library_path.open("w", encoding="utf-8"),
        allow_unicode=True,
        sort_keys=False,
    )
    report_path.write_text(_proposal_markdown(family, proposal), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "family": family.to_dict(),
                "provider": provider_payload,
                "name_prefix": name_prefix,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "run_dir": run_dir,
        "prompts_dir": prompts_dir,
        "proposal_dir": proposal_dir,
        "metadata_dir": metadata_dir,
        "system_prompt": system_path,
        "user_prompt": user_path,
        "raw_response": raw_path,
        "parsed_proposal": parsed_path,
        "candidate_library": library_path,
        "proposal_report": report_path,
        "run_meta": meta_path,
    }
