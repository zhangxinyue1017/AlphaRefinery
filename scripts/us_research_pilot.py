#!/usr/bin/env python3
'''Run one fixed-candidate US research cycle without an LLM call.'''

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

from factors_store.datahub.panel import file_sha256
from factors_store.llm_refine.core.archive import (
    create_run_dir,
    expression_hash,
    init_archive_db,
    insert_candidates,
    insert_lineage,
    insert_run,
    load_run_candidate_records,
    make_candidate_id,
    make_run_id,
    make_seed_candidate_id,
    mark_run_finished,
    utc_now_iso,
    write_run_artifacts,
)
from factors_store.llm_refine.core.models import LLMProposal, PromptBundle, RefinementCandidate
from factors_store.llm_refine.core.seed_loader import load_seed_pool, resolve_family_formula
from factors_store.llm_refine.evaluation.evaluator import evaluate_refinement_run
from factors_store.llm_refine.search.core.engine import SearchEngine
from factors_store.llm_refine.search.core.policy import SearchPolicy
from factors_store.llm_refine.search.core.state import SearchBudget


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "us_research_pilot.yaml"
KEEP_STATUSES = ("research_winner", "winner", "research_keep", "keep", "research_keep_exploratory")


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _archive_counts(db_path: Path, run_id: str) -> dict[str, int]:
    with sqlite3.connect(db_path) as connection:
        return {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)
                ).fetchone()[0]
            )
            for table in ("runs", "candidates", "evaluations")
        } | {
            "lineage": int(
                connection.execute(
                    "SELECT COUNT(*) FROM lineage WHERE child_candidate_id IN "
                    "(SELECT candidate_id FROM candidates WHERE run_id = ?)",
                    (run_id,),
                ).fetchone()[0]
            )
        }


def run(config_path: Path) -> Path:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pilot = dict(raw_config["us_pilot"])
    defaults = dict(raw_config["evaluation_defaults"])
    candidate_config = dict(pilot["fixed_candidate"])

    panel_path = _resolve_project_path(str(defaults["panel_path"]))
    expected_checksum = str(pilot["dataset_sha256"])
    actual_checksum = file_sha256(panel_path)
    if actual_checksum != expected_checksum:
        raise RuntimeError(
            f"US panel checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
        )
    if bool(pilot.get("automatic_promotion", False)):
        raise ValueError("US engineering pilot requires automatic_promotion=false")

    archive_db = init_archive_db(_resolve_project_path(str(pilot["archive_db"])))
    runs_dir = _resolve_project_path(str(pilot["runs_dir"]))
    seed_pool = load_seed_pool(config_path)
    family = seed_pool.get_family("open_volume_correlation")

    round_id = 1
    parent_name = family.canonical_seed
    parent_expression = resolve_family_formula(family, parent_name)
    parent_candidate_id = make_seed_candidate_id(parent_name)
    candidate_name = str(candidate_config["name"])
    candidate_expression = str(candidate_config["expression"])
    candidate_id = make_candidate_id(
        family=family.family,
        round_id=round_id,
        expression=candidate_expression,
        name=candidate_name,
    )
    candidate = RefinementCandidate(
        name=candidate_name,
        expression=candidate_expression,
        explanation="Fixed offline candidate for US pipeline verification.",
        candidate_role=str(candidate_config.get("role", "conservative")),
        rationale="Normalize open by the previous close while preserving the open-volume correlation motif.",
        expected_improvement="Unknown; this run verifies evaluation and decision plumbing.",
        risk="Ten-stock cross sections are small and noisy.",
        source_model="fixed_offline",
        source_provider="none",
        parent_factor=parent_name,
        family=family.family,
        candidate_id=candidate_id,
        round_id=round_id,
        parent_candidate_id=parent_candidate_id,
    )
    proposal = LLMProposal(
        parent_factor=parent_name,
        diagnosed_weaknesses=("Scale sensitivity of raw opening prices.",),
        refinement_rationale="Pre-specified engineering candidate; no model generated this proposal.",
        candidates=(candidate,),
        expected_behavior_change="Opening gap normalization may alter parent correlation and turnover.",
        risk_notes="Engineering pilot only.",
        raw_response="",
    )

    started_at = utc_now_iso()
    run_id = make_run_id(family=family.family, round_id=round_id, started_at=started_at)
    run_dir = create_run_dir(family=family.family, runs_dir=runs_dir)
    prompt = PromptBundle(
        system_prompt="Offline fixed-candidate pipeline verification. No LLM call is permitted.",
        user_prompt=f"Evaluate the pre-specified expression: {candidate_expression}",
    )
    prompt_hash = hashlib.sha1(
        f"{prompt.system_prompt}\n---\n{prompt.user_prompt}".encode("utf-8")
    ).hexdigest()
    insert_run(
        db_path=archive_db,
        run_id=run_id,
        family=family.family,
        canonical_seed=parent_name,
        round_id=round_id,
        parent_candidate_id=parent_candidate_id,
        provider="none",
        model="fixed_offline",
        prompt_hash=prompt_hash,
        run_dir=str(run_dir),
        status="running",
        started_at=started_at,
    )
    insert_candidates(
        db_path=archive_db,
        run_id=run_id,
        candidates=[
            {
                "candidate_id": candidate_id,
                "family": family.family,
                "round_id": round_id,
                "parent_candidate_id": parent_candidate_id,
                "factor_name": f"us_pilot.{candidate_name}",
                "expression": candidate_expression,
                "expression_hash": expression_hash(candidate_expression),
                "candidate_role": candidate.candidate_role,
                "source_model": candidate.source_model,
                "source_provider": candidate.source_provider,
                "explanation": candidate.explanation,
                "rationale": candidate.rationale,
                "validation_warnings": [],
                "status": "proposed",
                "created_at": started_at,
            }
        ],
    )
    insert_lineage(
        db_path=archive_db,
        family=family.family,
        round_id=round_id,
        parent_child_pairs=[(parent_candidate_id, candidate_id)],
    )
    write_run_artifacts(
        run_dir=run_dir,
        family=family,
        prompt=prompt,
        proposal=proposal,
        provider_payload={"provider": "none", "model": "fixed_offline", "llm_called": False},
        name_prefix="us_pilot",
    )

    search_engine = SearchEngine(
        family=family.family,
        budget=SearchBudget(
            max_rounds=1,
            family_budget=1,
            branch_budget=1,
            max_frontier_size=4,
            max_depth=2,
            stop_if_no_improve=0,
        ),
        policy=SearchPolicy.local_best_first(preset="balanced").with_target_profile("raw_alpha"),
    )
    seed_node = search_engine.register_seed(
        factor_name=parent_name,
        expression=parent_expression,
        candidate_id=parent_candidate_id,
        node_kind="canonical_seed",
        status="seed",
        round_id=0,
    )
    selected_parent = search_engine.select_next() or seed_node

    try:
        evaluation_outputs = evaluate_refinement_run(
            run_dir=run_dir,
            seed_pool=seed_pool,
            family=family,
            proposal=proposal,
            name_prefix="us_pilot",
            panel_path=str(panel_path),
            run_id=run_id,
            round_id=round_id,
            parent_candidate_id=parent_candidate_id,
            parent_factor_name=parent_name,
            parent_expression=parent_expression,
            archive_db=archive_db,
            auto_apply_promotion=False,
            stage_mode="focused_refine",
            target_profile="raw_alpha",
        )
        all_candidate_records = load_run_candidate_records(db_path=archive_db, run_id=run_id)
        kept_records = [
            record for record in all_candidate_records if str(record.get("status", "")) in KEEP_STATUSES
        ]
        search_engine.register_expansion(
            parent_node_id=selected_parent.node_id,
            child_records=kept_records,
            success=True,
            source_run_id=run_id,
            note="offline fixed-candidate evaluation completed",
        )
        search_state_path = run_dir / "metadata" / "search_state.json"
        _write_json(search_state_path, search_engine.summary())

        selection_full = pd.read_csv(evaluation_outputs["selection_summary_full"])
        selection_rows = json.loads(selection_full.to_json(orient="records"))
        summary = {
            "pipeline": "Seed -> Fixed Candidate -> Full Evaluation -> Research Decision -> Archive",
            "llm_called": False,
            "trading_or_execution": False,
            "baseline_commit": str(pilot["baseline_commit"]),
            "dataset_sha256": actual_checksum,
            "config": str(config_path),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "archive_db": str(archive_db),
            "automatic_promotion": False,
            "parent": {"factor_name": parent_name, "expression": parent_expression},
            "candidate": candidate.to_dict(),
            "evaluation_protocol": seed_pool.evaluation_protocol.to_dict()
            if seed_pool.evaluation_protocol is not None
            else None,
            "evaluation_settings": seed_pool.evaluation_defaults,
            "selection_rows": selection_rows,
            "candidate_archive_records": all_candidate_records,
            "kept_candidate_count_for_search": len(kept_records),
            "search_state": str(search_state_path),
            "archive_counts": _archive_counts(Path(archive_db), run_id),
            "generated_outputs": {key: str(value) for key, value in evaluation_outputs.items()},
        }
        summary_path = run_dir / "us_pipeline_summary.json"
        _write_json(summary_path, summary)
        mark_run_finished(db_path=archive_db, run_id=run_id, status="completed")
        return summary_path
    except Exception:
        mark_run_finished(db_path=archive_db, run_id=run_id, status="failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_path = run(args.config.resolve())
    print(f"saved_us_pipeline_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
