'''Family-level candidate evaluation and keep/drop decision logic.

Builds factor series, applies redundancy checks, runs raw and neutralized backtests, and writes reports.
The implementation is split across preparation, window, decision, and reporting modules; this file remains the public entry point.
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.archive import (
    DEFAULT_ARCHIVE_DB,
    ensure_run_subdir,
    insert_evaluations,
    update_candidate_filter_metadata,
    update_candidate_statuses,
)
from ..core.models import LLMProposal, SeedFamily, SeedPool
from .decisions import (
    CORE_FULL_METRICS,
    EXPLORATION_BONUS_THRESHOLD,
    EXPLORATION_CROSS_MODEL_WEIGHT,
    EXPLORATION_DECORRELATION_WEIGHT,
    EXPLORATION_MOTIF_NOVELTY_WEIGHT,
    EXPLORATION_MUTATION_NOVELTY_WEIGHT,
    EXPLORATION_QUALITY_FLOOR_ICIR,
    EXPLORATION_QUALITY_FLOOR_IC_MEAN,
    MAX_EXPLORATORY_KEEPS_PER_ROUND,
    NEW_FAMILY_BROAD_MATERIAL_THRESHOLDS,
    WINNER_SCORE_WEIGHTS,
    _assign_winners,
    _candidate_decision,
    _has_reference_metrics,
    _metrics_completeness,
    _parent_reference,
)
from .preparation import register_proposal_candidates
from .promotion import write_pending_curated_manifest
from .reporting import _write_window_outputs
from .window import (
    FAMILY_CORR_THRESHOLD,
    FAMILY_CORR_THRESHOLD_HARD,
    FAMILY_CORR_THRESHOLD_SOFT,
    PARENT_CORR_THRESHOLD,
    PARENT_CORR_THRESHOLD_HARD,
    PARENT_CORR_THRESHOLD_SOFT,
    _evaluate_one_window,
)

def evaluate_refinement_run(
    *,
    run_dir: str | Path,
    seed_pool: SeedPool,
    family: SeedFamily,
    proposal: LLMProposal,
    name_prefix: str = "llmgen",
    panel_path: str | None = None,
    benchmark_path: str | None = None,
    start: str | None = None,
    end: str | None = None,
    run_id: str = "",
    round_id: int = 1,
    parent_candidate_id: str = "",
    parent_factor_name: str = "",
    parent_expression: str = "",
    archive_db: str | Path = DEFAULT_ARCHIVE_DB,
    auto_apply_promotion: bool = False,
    stage_mode: str = "auto",
    target_profile: str = "raw_alpha",
    decorrelation_targets: tuple[str, ...] = (),
) -> dict[str, Path]:
    run_path = Path(run_dir)
    use_protocol = seed_pool.evaluation_protocol is not None and not (start or end)
    metadata_dir = ensure_run_subdir(run_path, "metadata")
    meta_path = metadata_dir / "evaluation_meta.json"
    outputs: dict[str, Path] = {}

    if use_protocol:
        protocol = seed_pool.evaluation_protocol
        assert protocol is not None
        stage_windows: dict[str, Any] = {
            "search": protocol.search,
            "selection": protocol.selection,
        }
        if protocol.final_oos is not None:
            stage_windows["final_oos"] = protocol.final_oos
        decision_stage = protocol.keep_decision_stage
        stage_details: dict[str, Any] = {}
        decision_summary_df: pd.DataFrame | None = None
        decision_data: dict[str, pd.Series] | None = None
        decision_settings: dict[str, Any] | None = None
        decision_data_meta: dict[str, Any] | None = None

        for stage_name in tuple(stage_windows.keys()):
            window = stage_windows[stage_name]
            summary_df, data, settings, data_meta = _evaluate_one_window(
                seed_pool=seed_pool,
                family=family,
                proposal=proposal,
                name_prefix=name_prefix,
                panel_path=panel_path,
                benchmark_path=benchmark_path,
                start=window.start or None,
                end=window.end or None,
                run_id=run_id,
                archive_db=archive_db,
                stage_mode=stage_mode,
                target_profile=target_profile,
                decorrelation_targets=decorrelation_targets,
                parent_factor_name_override=parent_factor_name,
                parent_expression=parent_expression,
                parent_candidate_id=parent_candidate_id,
            )
            stage_outputs = _write_window_outputs(
                run_path=run_path,
                family=family,
                stage_name=stage_name,
                summary_df=summary_df,
                settings=settings,
                data_meta=data_meta,
                make_canonical_alias=stage_name == decision_stage,
            )
            outputs.update(stage_outputs)
            stage_details[stage_name] = {
                "settings": settings,
                "data_meta": data_meta,
            }
            if stage_name == decision_stage:
                decision_summary_df = summary_df
                decision_data = data
                decision_settings = settings
                decision_data_meta = data_meta

        if (
            decision_summary_df is None
            or decision_data is None
            or decision_settings is None
            or decision_data_meta is None
        ):
            raise ValueError(f"decision stage `{decision_stage}` did not produce evaluation output")

        meta_payload = {
            "family": family.family,
            "canonical_seed": family.canonical_seed,
            "name_prefix": name_prefix,
            "evaluation_mode": "protocol",
            "protocol": protocol.to_dict(),
            "decision_stage": decision_stage,
            "stage_mode": str(stage_mode or "auto"),
            "decision_policy": (
                "system-internal research decisions are based on selection. "
                "If final_oos is configured, it is review-only and excluded from optimization and auto-parenting."
            ),
            "decorrelation_targets": list(decorrelation_targets),
            "stage_details": stage_details,
            "stage_files": {key: str(value) for key, value in outputs.items()},
        }
        summary_df_for_archive = decision_summary_df
    else:
        summary_df, data, settings, data_meta = _evaluate_one_window(
            seed_pool=seed_pool,
            family=family,
            proposal=proposal,
            name_prefix=name_prefix,
            panel_path=panel_path,
            benchmark_path=benchmark_path,
            start=start,
            end=end,
            run_id=run_id,
            archive_db=archive_db,
            stage_mode=stage_mode,
            target_profile=target_profile,
            decorrelation_targets=decorrelation_targets,
            parent_factor_name_override=parent_factor_name,
            parent_expression=parent_expression,
            parent_candidate_id=parent_candidate_id,
        )
        outputs.update(
            _write_window_outputs(
                run_path=run_path,
                family=family,
                stage_name="selection",
                summary_df=summary_df,
                settings=settings,
                data_meta=data_meta,
                make_canonical_alias=True,
            )
        )
        meta_payload = {
            "family": family.family,
            "canonical_seed": family.canonical_seed,
            "name_prefix": name_prefix,
            "evaluation_mode": "single_window",
            "stage_mode": str(stage_mode or "auto"),
            "decorrelation_targets": list(decorrelation_targets),
            "settings": settings,
            "data_meta": data_meta,
            "stage_files": {key: str(value) for key, value in outputs.items()},
        }
        summary_df_for_archive = summary_df

    archive_decision_stage = str(meta_payload.get("decision_stage", "selection"))
    if not summary_df_for_archive.empty:
        summary_df_for_archive = summary_df_for_archive.copy()
        summary_df_for_archive["evaluation_stage"] = archive_decision_stage
        summary_df_for_archive["decision_scope"] = "formal_decision"

    meta_path.write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    insert_evaluations(
        db_path=archive_db,
        run_id=run_id,
        rows=summary_df_for_archive.to_dict(orient="records"),
    )
    update_candidate_filter_metadata(
        db_path=archive_db,
        updates=[
            (
                str(row.get("candidate_id", "")),
                str(row.get("decision", "")),
                "evaluation" if str(row.get("decision", "")).startswith("drop_redundant") else "",
                str(row.get("decision_reason", "")),
            )
            for row in summary_df_for_archive.to_dict(orient="records")
            if str(row.get("role", "")) == "candidate" and str(row.get("decision", "")).startswith("drop_redundant")
        ],
    )
    update_candidate_statuses(
        db_path=archive_db,
        statuses=[
            (str(row.get("candidate_id", "")), str(row.get("decision", "")))
            for row in summary_df_for_archive.to_dict(orient="records")
            if str(row.get("role", "")) == "candidate"
        ],
    )
    promote_outputs = write_pending_curated_manifest(
        family=family,
        summary_df=summary_df_for_archive,
        run_id=run_id,
        round_id=round_id,
        run_dir=run_path,
        decision_stage=str(meta_payload.get("decision_stage", "selection")),
        name_prefix=name_prefix,
        metadata_dir=metadata_dir,
        auto_apply=auto_apply_promotion,
        data=decision_data if use_protocol else data,
    )
    if promote_outputs:
        outputs.update(promote_outputs)
    outputs["evaluation_meta"] = meta_path
    return outputs
