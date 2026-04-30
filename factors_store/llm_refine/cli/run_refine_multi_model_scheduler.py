'''Best-first multi-round scheduler for multi-model refinement.

Maintains a search frontier, selects parents, dispatches rounds, and writes stage-transition artifacts.
'''

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_AUTO_APPLY_PROMOTION,
    DEFAULT_MULTI_SCHEDULER_MAX_ROUNDS,
    DEFAULT_MULTI_SCHEDULER_RUNS_DIR,
    DEFAULT_MULTI_SCHEDULER_SLEEP_BETWEEN_ROUNDS,
    DEFAULT_MULTI_SCHEDULER_STOP_IF_NO_NEW_WINNER,
    PROJECT_ROOT,
)
from ..core.archive import (
    DEFAULT_ARCHIVE_DB,
    get_best_family_winner,
    get_latest_family_winner,
    make_seed_candidate_id,
    utc_now_iso,
)
from ..core.seed_loader import load_seed_pool, resolve_family_formula, resolve_preferred_refine_seed
from ..knowledge.round1 import build_bootstrap_frontier, is_seed_stage_node_kind, select_bootstrap_parent
from ..search import (
    DEFAULT_POLICY_CONFIG,
    EvaluationFeedback,
    FamilyState,
    RefinementAction,
    SaturationAnalyzer,
    SearchBudget,
    SearchEngine,
    SearchPolicy,
    SignalExtractor,
    build_stage_transition_evidence,
    build_search_normalizer,
    compare_stage_transition_decisions,
    resolve_round_transition_plan,
    resolve_stage_table_policy,
)
from ..search.transition.context_resolver import ContextEvidence, OrchestrationProfile, resolve_context_profile
from ..search.core.scoring import pairwise_similarity
from ..search.io.run_ingest import load_multi_run_candidate_records, resolve_materialized_multi_run_dir
from ..search.transition.stage_transition import resolve_stage_transition_from_state
from .run_refine_multi_model import build_arg_parser as build_multi_model_arg_parser

_STAGE_MODE_CHOICES = (
    "auto",
    "new_family_broad",
    "broad_followup",
    "focused_refine",
    "confirmation",
    "donor_validation",
)


def _normalize_models(values: list[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if item and item not in out:
                out.append(item)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    multi = build_multi_model_arg_parser()
    parser = argparse.ArgumentParser(
        description="Run multiple llm_refine multi-model rounds with a unified best-first search frontier."
    )
    parser.add_argument("--family", required=True, help="seed family name from the seed pool")
    parser.add_argument("--models", action="append", required=True, help="comma-separated model list; may be repeated")
    parser.add_argument(
        "--bootstrap-parent-name",
        action="append",
        default=[],
        help="extra parent name to seed into the search frontier; may be repeated",
    )
    parser.add_argument(
        "--bootstrap-parent-expression",
        action="append",
        default=[],
        help="expression for each --bootstrap-parent-name; order must match",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MULTI_SCHEDULER_MAX_ROUNDS,
        help="number of multi-model rounds to execute",
    )
    parser.add_argument(
        "--stop-if-no-new-winner",
        type=int,
        default=DEFAULT_MULTI_SCHEDULER_STOP_IF_NO_NEW_WINNER,
        help="stop after this many consecutive successful rounds without a search-level improvement",
    )
    parser.add_argument(
        "--scheduler-runs-dir",
        default=str(DEFAULT_MULTI_SCHEDULER_RUNS_DIR),
        help="artifact root for multi-model scheduler logs",
    )
    parser.add_argument(
        "--sleep-between-rounds",
        type=float,
        default=DEFAULT_MULTI_SCHEDULER_SLEEP_BETWEEN_ROUNDS,
        help="optional sleep in seconds between completed rounds",
    )

    passthrough_defaults = {
        "--seed-pool": str(multi.get_default("seed_pool")),
        "--n-candidates": multi.get_default("n_candidates"),
        "--runs-dir": str(multi.get_default("runs_dir")),
        "--archive-db": str(multi.get_default("archive_db")),
        "--name-prefix": str(multi.get_default("name_prefix")),
        "--provider-name": str(multi.get_default("provider_name")),
        "--base-url": str(multi.get_default("base_url")),
        "--api-key": str(multi.get_default("api_key")),
        "--temperature": multi.get_default("temperature"),
        "--max-tokens": multi.get_default("max_tokens"),
        "--timeout": multi.get_default("timeout"),
        "--additional-notes": str(multi.get_default("additional_notes")),
        "--policy-preset": str(multi.get_default("policy_preset")),
        "--target-profile": str(multi.get_default("target_profile")),
        "--current-parent-name": str(multi.get_default("current_parent_name")),
        "--current-parent-expression": str(multi.get_default("current_parent_expression")),
        "--panel-path": str(multi.get_default("panel_path")),
        "--benchmark-path": str(multi.get_default("benchmark_path")),
        "--start": str(multi.get_default("start")),
        "--end": str(multi.get_default("end")),
        "--max-parallel": multi.get_default("max_parallel"),
        "--multi-runs-dir": str(multi.get_default("multi_runs_dir")),
        "--prompt-template-version": str(multi.get_default("prompt_template_version")),
        "--primary-objective": "",
        "--secondary-objective": "",
    }
    for arg, default in passthrough_defaults.items():
        parser.add_argument(arg, default=default)
    parser.add_argument(
        "--decorrelation-target",
        action="append",
        default=list(multi.get_default("decorrelation_target") or []),
        help="explicit factor(s) to decorrelate from; may be repeated or passed as comma-separated names",
    )

    parser.add_argument("--skip-eval", action="store_true", help="skip backtest stage in each round")
    parser.add_argument("--dry-run", action="store_true", help="dry-run each round without provider calls")
    parser.add_argument(
        "--auto-apply-promotion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_APPLY_PROMOTION,
        help="automatically apply pending curated promotion patches in each child run after evaluation",
    )
    parser.add_argument(
        "--disable-mmr-rerank",
        action="store_true",
        help="disable MMR rerank and use raw frontier score ordering only",
    )
    parser.add_argument(
        "--enable-dual-parent-round",
        action="store_true",
        help="conditionally expand two complementary parents in the same round",
    )
    parser.add_argument(
        "--dual-parent-quality-models",
        default="",
        help="optional comma-separated model list for quality_parent when dual-parent round is triggered",
    )
    parser.add_argument(
        "--dual-parent-expandability-models",
        default="",
        help="optional comma-separated model list for expandability_parent when dual-parent round is triggered",
    )
    parser.add_argument(
        "--stage-mode",
        default="auto",
        choices=_STAGE_MODE_CHOICES,
        help="explicit orchestration stage label for this scheduler run",
    )
    parser.add_argument(
        "--transition-authority",
        default=DEFAULT_POLICY_CONFIG.round_transition.default_authority,
        choices=DEFAULT_POLICY_CONFIG.round_transition.allowed_authorities,
        help="how much authority round transition planning has over scheduler execution",
    )
    parser.add_argument(
        "--max-policy-extensions",
        type=int,
        default=DEFAULT_POLICY_CONFIG.round_transition.max_policy_extensions,
        help="max extra rounds guarded_control may grant after --max-rounds is reached",
    )
    parser.add_argument(
        "--max-total-rounds",
        type=int,
        default=DEFAULT_POLICY_CONFIG.round_transition.default_max_total_rounds,
        help="hard total round cap when guarded_control is enabled; never below --max-rounds",
    )
    return parser


def _build_round_cmd(
    args: argparse.Namespace,
    *,
    parent: dict[str, Any],
    round_multi_root: Path,
    child_stage_mode: str,
    target_profile: str = "",
    force_round1_seed_stage: bool = False,
    models_override: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "factors_store.llm_refine.cli.run_refine_multi_model",
        "--family",
        args.family,
        "--seed-pool",
        str(args.seed_pool),
        "--n-candidates",
        str(args.n_candidates),
        "--runs-dir",
        str(args.runs_dir),
        "--archive-db",
        str(args.archive_db),
        "--name-prefix",
        str(args.name_prefix),
        "--provider-name",
        str(args.provider_name),
        "--base-url",
        str(args.base_url),
        "--api-key",
        str(args.api_key),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--max-parallel",
        str(min(int(args.max_parallel), len(models_override or _normalize_models(args.models)))),
        "--multi-runs-dir",
        str(round_multi_root),
    ]
    for model in (models_override or _normalize_models(args.models)):
        cmd.extend(["--models", model])
    if str(args.additional_notes).strip():
        cmd.extend(["--additional-notes", str(args.additional_notes)])
    for target in list(args.decorrelation_target or []):
        if str(target).strip():
            cmd.extend(["--decorrelation-target", str(target)])
    cmd.extend(["--current-parent-name", str(parent.get("factor_name", ""))])
    cmd.extend(["--current-parent-expression", str(parent.get("expression", ""))])
    if str(parent.get("candidate_id", "")).strip():
        cmd.extend(["--parent-candidate-id", str(parent.get("candidate_id", ""))])
    if str(args.panel_path).strip():
        cmd.extend(["--panel-path", str(args.panel_path)])
    if str(args.benchmark_path).strip():
        cmd.extend(["--benchmark-path", str(args.benchmark_path)])
    if str(args.start).strip():
        cmd.extend(["--start", str(args.start)])
    if str(args.end).strip():
        cmd.extend(["--end", str(args.end)])
    if args.skip_eval:
        cmd.append("--skip-eval")
    if args.dry_run:
        cmd.append("--dry-run")
    if str(args.policy_preset).strip():
        cmd.extend(["--policy-preset", str(args.policy_preset)])
    effective_target_profile = str(target_profile or args.target_profile or "").strip()
    if effective_target_profile:
        cmd.extend(["--target-profile", effective_target_profile])
    if str(child_stage_mode).strip():
        cmd.extend(["--stage-mode", str(child_stage_mode)])
    if str(args.prompt_template_version).strip():
        cmd.extend(["--prompt-template-version", str(args.prompt_template_version)])
    if str(args.primary_objective or "").strip():
        cmd.extend(["--primary-objective", str(args.primary_objective).strip()])
    if str(args.secondary_objective or "").strip():
        cmd.extend(["--secondary-objective", str(args.secondary_objective).strip()])
    if args.disable_mmr_rerank:
        cmd.append("--disable-mmr-rerank")
    cmd.append("--auto-apply-promotion" if args.auto_apply_promotion else "--no-auto-apply-promotion")
    if force_round1_seed_stage or is_seed_stage_node_kind(str(parent.get("node_kind", ""))):
        cmd.append("--round1-seed-stage")
    return cmd


def _resolve_child_stage_mode(stage_mode: str, *, round_idx: int, last_round: dict[str, Any] | None = None) -> str:
    round_plan = dict((last_round or {}).get("round_transition_plan") or {})
    planned = str(round_plan.get("next_stage_mode", "") or "").strip()
    if bool(round_plan.get("control_effective")) and planned in _STAGE_MODE_CHOICES and planned != "auto":
        return planned

    stage = str(stage_mode or "auto").strip() or "auto"
    if stage == "auto":
        if int(round_idx) == 1:
            return "new_family_broad"
        if planned in _STAGE_MODE_CHOICES and planned != "auto":
            return planned
        # Advisory StageTransitionDecision is now the sole source of truth.
        stage_transition = dict((last_round or {}).get("stage_transition") or {})
        recommended = str(stage_transition.get("next_stage", "") or "").strip()
        if recommended in _STAGE_MODE_CHOICES and recommended != "auto":
            return recommended
        return "broad_followup"
    if stage == "new_family_broad":
        return "new_family_broad" if int(round_idx) == 1 else "broad_followup"
    return stage


def _pick_round_prompt_trace(sub_runs: list[dict[str, Any]]) -> dict[str, Any]:
    for item in sub_runs:
        trace = dict(item.get("prompt_trace") or {})
        if trace:
            return trace
    return {}


def _build_orchestration_trace(
    *,
    family: str,
    stage_mode: str,
    target_profile: str,
    policy_preset: str,
    parent_kind: str,
    requested_candidate_count: int,
    final_candidate_target: int,
    has_decorrelation_targets: bool,
    round_status: str,
    search_improved: bool,
    winner: dict[str, Any],
    keep: dict[str, Any],
    recommended_stage_mode_hint: str = "",
    consecutive_no_improve: int = 0,
    high_corr_count: int = 0,
    high_turnover_count: int = 0,
    validation_fail_count: int = 0,
    children_collected: int = 0,
    children_added_to_search: int = 0,
    budget_exhausted: bool = False,
    frontier_exhausted: bool = False,
    round_idx: int = 0,
    base_max_rounds: int = 0,
    max_total_rounds: int = 0,
    transition_authority: str = "",
    policy_extension_count: int = 0,
    max_policy_extensions: int = 0,
) -> dict[str, Any]:
    evidence = ContextEvidence.from_runtime(
        family=family,
        stage_mode=stage_mode,
        target_profile=target_profile,
        policy_preset=policy_preset,
        is_seed_stage=stage_mode == "new_family_broad",
        has_bootstrap_frontier=stage_mode == "new_family_broad",
        has_donor_motifs=False,
        has_decorrelation_targets=has_decorrelation_targets,
        selected_parent_kind=parent_kind,
        requested_candidate_count=requested_candidate_count,
        final_candidate_target=final_candidate_target,
    )
    context_profile = resolve_context_profile(evidence)
    family_state = FamilyState(
        family_id=family,
        stage=stage_mode,
        target_profile=target_profile,
        failure_state={
            "high_corr_count": int(high_corr_count),
            "high_turnover_count": int(high_turnover_count),
            "validation_fail_count": int(validation_fail_count),
        },
        budget_state={
            "consecutive_no_improve": int(consecutive_no_improve),
            "children_collected": int(children_collected),
            "children_added_to_search": int(children_added_to_search),
            "budget_exhausted": bool(budget_exhausted),
            "frontier_exhausted": bool(frontier_exhausted),
        },
        redundancy_state={"has_decorrelation_targets": bool(has_decorrelation_targets)},
    )
    refinement_action = RefinementAction(
        stage_mode=stage_mode,
        target_profile=target_profile,
        policy_preset=policy_preset,
        decorrelation_targets=("__present__",) if has_decorrelation_targets else (),
        n_candidates=int(requested_candidate_count or 0),
    )
    evaluation_feedback = EvaluationFeedback(
        status=round_status,
        search_improved=search_improved,
        winner=dict(winner or {}),
        keep=dict(keep or {}),
        consecutive_no_improve=int(consecutive_no_improve),
        children_collected=int(children_collected),
        children_added_to_search=int(children_added_to_search),
        high_corr_count=int(high_corr_count),
        high_turnover_count=int(high_turnover_count),
        validation_fail_count=int(validation_fail_count),
        budget_exhausted=bool(budget_exhausted),
        frontier_exhausted=bool(frontier_exhausted),
    )
    stage_transition_evidence = build_stage_transition_evidence(
        family_state,
        refinement_action,
        evaluation_feedback,
    )
    legacy_stage_transition = resolve_stage_transition_from_state(
        family_state,
        refinement_action,
        evaluation_feedback,
    )
    stage_transition_signals = SignalExtractor.from_evidence(stage_transition_evidence)
    saturation_assessment = SaturationAnalyzer.from_evidence(
        stage_transition_evidence,
        stage_transition_signals,
    )
    stage_transition = resolve_stage_table_policy(
        stage_transition_evidence,
        stage_transition_signals,
    )
    stage_transition_legacy_compare = compare_stage_transition_decisions(
        legacy_decision=legacy_stage_transition,
        shadow_decision=stage_transition,
    )
    round_transition_plan = resolve_round_transition_plan(
        stage_transition=stage_transition,
        signals=stage_transition_signals,
        saturation_assessment=saturation_assessment,
        current_stage=stage_mode,
        target_profile=target_profile,
        rounds_completed=int(round_idx or 0),
        base_max_rounds=int(base_max_rounds or 0),
        max_total_rounds=int(max_total_rounds or 0),
        policy_extension_count=int(policy_extension_count or 0),
        max_policy_extensions=int(max_policy_extensions or 0),
        transition_authority=transition_authority,
    )
    orchestration_profile = OrchestrationProfile(
        recommended_stage_mode=stage_transition.next_stage,
        round_strategy=stage_transition.action,
        promotion_bias="normal",
        parent_selection_bias=stage_transition.parent_selection_bias,
        termination_bias=stage_transition.termination_bias,
        confidence=stage_transition.confidence,
        rationale_tags=stage_transition.rationale_tags,
    )
    return {
        "context_evidence": evidence.to_dict(),
        "context_profile": context_profile.to_dict(),
        "orchestration_profile": orchestration_profile.to_dict(),
        "family_state": family_state.to_dict(),
        "refinement_action": refinement_action.to_dict(),
        "evaluation_feedback": evaluation_feedback.to_dict(),
        "stage_transition_evidence": stage_transition_evidence.to_dict(),
        "stage_transition": stage_transition.to_dict(),
        "stage_transition_policy": stage_transition.to_dict(),
        "stage_transition_policy_source": "table_policy",
        "stage_transition_legacy_audit": legacy_stage_transition.to_dict(),
        "stage_transition_legacy_compare": stage_transition_legacy_compare,
        "stage_transition_signals": stage_transition_signals.to_dict(),
        "saturation_assessment": saturation_assessment.to_dict(),
        "round_transition_plan": round_transition_plan.to_dict(),
        "transition_authority": round_transition_plan.transition_authority,
        "budget_gate_decision": round_transition_plan.budget_gate,
        "policy_extension_count": int(round_transition_plan.policy_extension_count),
        "stage_transition_legacy": legacy_stage_transition.to_dict(),
        "stage_transition_shadow_table": stage_transition.to_dict(),
        "stage_transition_shadow_compare": stage_transition_legacy_compare,
        "family_state_decision": stage_transition.to_dict(),
        "table_orchestration_decision": orchestration_profile.to_dict(),
        "legacy_orchestration_decision": legacy_stage_transition.to_dict(),
        "stage_transition_shadow": stage_transition_legacy_compare,
    }


def _resolve_dual_parent_model_allocations(args: argparse.Namespace, models: list[str]) -> dict[str, list[str]]:
    quality = _normalize_models([str(args.dual_parent_quality_models or "")])
    expandability = _normalize_models([str(args.dual_parent_expandability_models or "")])
    if quality or expandability:
        quality_final = quality or models
        expandability_final = expandability or models
        return {
            "quality_parent": quality_final,
            "expandability_parent": expandability_final,
        }

    if len(models) <= 1:
        return {
            "quality_parent": list(models),
            "expandability_parent": list(models),
        }

    shared = models[:1]
    quality_tail = models[1:3]
    expandability_tail = models[3:]
    quality_final = list(dict.fromkeys([*shared, *quality_tail]))
    expandability_final = list(dict.fromkeys([*shared, *expandability_tail]))

    target_width = min(3, len(models))
    for model in models:
        if len(quality_final) < target_width and model not in quality_final:
            quality_final.append(model)
        if len(expandability_final) < target_width and model not in expandability_final:
            expandability_final.append(model)
    return {
        "quality_parent": quality_final,
        "expandability_parent": expandability_final,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _metric_line(payload: dict[str, Any]) -> str:
    if not payload:
        return "(empty)"

    def _float(name: str, default: float = 0.0) -> float:
        value = payload.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return (
        f"IC={_float('quick_rank_ic_mean'):.4f}, "
        f"ICIR={_float('quick_rank_icir'):.4f}, "
        f"Ann={_float('net_ann_return'):.4f}, "
        f"Excess={_float('net_excess_ann_return'):.4f}, "
        f"Sharpe={_float('net_sharpe'):.4f}, "
        f"TO={_float('mean_turnover'):.4f}"
    )


def render_scheduler_markdown(summary: dict[str, Any]) -> str:
    last_round_winner = dict(summary.get("last_round_winner") or {})
    last_best_candidate = dict(summary.get("last_round_best_candidate") or {})
    last_best_keep = dict(summary.get("last_round_best_keep") or {})
    best_node = dict(summary.get("best_node") or {})
    prompt_trace = dict(summary.get("prompt_trace") or {})
    context_evidence = dict(summary.get("orchestration_context_evidence") or {})
    context_profile = dict(summary.get("orchestration_context_profile") or {})
    orchestration_profile = dict(summary.get("orchestration_profile") or {})
    stage_transition = dict(summary.get("stage_transition") or {})
    stage_transition_legacy_audit = dict(summary.get("stage_transition_legacy_audit") or summary.get("stage_transition_legacy") or {})
    stage_transition_legacy_compare = dict(
        summary.get("stage_transition_legacy_compare") or summary.get("stage_transition_shadow_compare") or {}
    )
    stage_transition_signals = dict(summary.get("stage_transition_signals") or {})
    saturation_assessment = dict(summary.get("saturation_assessment") or {})
    saturation_components = dict(saturation_assessment.get("components") or {})
    round_transition_plan = dict(summary.get("round_transition_plan") or {})
    budget_gate = dict(round_transition_plan.get("budget_gate") or summary.get("budget_gate_decision") or {})
    rounds = list(summary.get("rounds") or [])
    last_round = dict(rounds[-1] or {}) if rounds else {}
    rationale_tags = list(orchestration_profile.get("rationale_tags") or [])
    stage_transition_tags = list(stage_transition.get("rationale_tags") or [])
    legacy_tags = list(stage_transition_legacy_audit.get("rationale_tags") or [])

    lines = [
        f"# Multi-Model Scheduler Summary: {summary.get('family', '')}",
        "",
        f"- stage_mode: `{summary.get('stage_mode', '')}`",
        f"- target_profile: `{summary.get('target_profile', '')}`",
        f"- scheduler_dir: `{summary.get('scheduler_dir', '')}`",
        f"- rounds_completed: `{summary.get('rounds_completed', 0)}`",
        f"- stop_reason: `{summary.get('stop_reason', '')}`",
        f"- last_round_status: `{summary.get('last_round_status', '')}`",
        f"- last_selected_parent_name: `{summary.get('last_selected_parent_name', '')}`",
        "",
        "## Prompt Trace",
        f"- stage_mode: `{prompt_trace.get('stage_mode', '')}`",
        f"- prompt_template_version: `{prompt_trace.get('prompt_template_version', '')}`",
        f"- seed_stage_active: `{prompt_trace.get('seed_stage_active', '')}`",
        f"- selected_parent_kind: `{prompt_trace.get('selected_parent_kind', '')}`",
        f"- requested_candidate_count: `{prompt_trace.get('requested_candidate_count', '')}`",
        f"- bootstrap_frontier_count: `{prompt_trace.get('bootstrap_frontier_count', '')}`",
        f"- donor_motifs_count: `{prompt_trace.get('donor_motifs_count', '')}`",
        "",
        "## Shared Context",
        f"- search_phase: `{context_profile.get('search_phase', '')}`",
        f"- exploration_pressure: `{context_profile.get('exploration_pressure', '')}`",
        f"- redundancy_pressure: `{context_profile.get('redundancy_pressure', '')}`",
        f"- prompt_constraint_style: `{context_profile.get('prompt_constraint_style', '')}`",
        f"- memory_mode: `{context_profile.get('memory_mode', '')}`",
        f"- examples_mode: `{context_profile.get('examples_mode', '')}`",
        f"- branching_bias: `{context_profile.get('branching_bias', '')}`",
        f"- next_action_bias: `{context_profile.get('next_action_bias', '')}`",
        "",
        "## Orchestration Trace (Advisory-driven)",
        f"- recommended_stage_mode: `{orchestration_profile.get('recommended_stage_mode', '')}`",
        f"- round_strategy: `{orchestration_profile.get('round_strategy', '')}`",
        f"- promotion_bias: `{orchestration_profile.get('promotion_bias', '')}`",
        f"- parent_selection_bias: `{orchestration_profile.get('parent_selection_bias', '')}`",
        f"- termination_bias: `{orchestration_profile.get('termination_bias', '')}`",
        f"- confidence: `{orchestration_profile.get('confidence', '')}`",
        f"- rationale_tags: `{', '.join(str(item) for item in rationale_tags)}`",
        "",
        "*Note: Stage transition is now table-policy driven. Legacy if/else output is retained only as an audit artifact.*",
        "",
        "## Stage Transition Table Policy",
        f"- current_stage: `{stage_transition.get('current_stage', '')}`",
        f"- next_stage: `{stage_transition.get('next_stage', '')}`",
        f"- action: `{stage_transition.get('action', '')}`",
        f"- confidence: `{stage_transition.get('confidence', '')}`",
        f"- termination_bias: `{stage_transition.get('termination_bias', '')}`",
        f"- parent_selection_bias: `{stage_transition.get('parent_selection_bias', '')}`",
        f"- target_profile_bias: `{stage_transition.get('target_profile_bias', '')}`",
        f"- rationale_tags: `{', '.join(str(item) for item in stage_transition_tags)}`",
        f"- reason: {stage_transition.get('reason', '')}",
        "",
        "## Legacy Transition Audit",
        f"- legacy_next_stage: `{stage_transition_legacy_audit.get('next_stage', '')}`",
        f"- legacy_action: `{stage_transition_legacy_audit.get('action', '')}`",
        f"- legacy_confidence: `{stage_transition_legacy_audit.get('confidence', '')}`",
        f"- legacy_rationale_tags: `{', '.join(str(item) for item in legacy_tags)}`",
        f"- legacy_reason: {stage_transition_legacy_audit.get('reason', '')}",
        f"- table_vs_legacy_stage_agrees: `{stage_transition_legacy_compare.get('stage_agrees', '')}`",
        f"- table_vs_legacy_action_agrees: `{stage_transition_legacy_compare.get('action_agrees', '')}`",
        "",
        "## Stage Transition Signals",
        f"- anchor_strength: `{stage_transition_signals.get('anchor_strength', '')}`",
        f"- winner_quality: `{stage_transition_signals.get('winner_quality', '')}`",
        f"- material_gain: `{stage_transition_signals.get('material_gain', '')}`",
        f"- material_gain_score: `{stage_transition_signals.get('material_gain_score', '')}`",
        f"- corr_pressure: `{stage_transition_signals.get('corr_pressure', '')}`",
        f"- turnover_pressure: `{stage_transition_signals.get('turnover_pressure', '')}`",
        f"- frontier_health: `{stage_transition_signals.get('frontier_health', '')}`",
        f"- no_improve_count: `{stage_transition_signals.get('no_improve_count', '')}`",
        f"- validation_fail_count: `{stage_transition_signals.get('validation_fail_count', '')}`",
        "",
        "## Saturation Assessment",
        f"- grade: `{saturation_assessment.get('grade', '')}`",
        f"- score: `{saturation_assessment.get('score', '')}`",
        f"- recommended_escape_mode: `{saturation_assessment.get('recommended_escape_mode', '')}`",
        f"- components: `corr={saturation_components.get('corr', '')}, "
        f"motif={saturation_components.get('motif', '')}, "
        f"turnover={saturation_components.get('turnover', '')}, "
        f"plateau={saturation_components.get('plateau', '')}, "
        f"frontier={saturation_components.get('frontier', '')}, "
        f"anchor_reuse={saturation_components.get('anchor_reuse', '')}`",
        f"- advisory_only: `{dict(saturation_assessment.get('diagnostics') or {}).get('advisory_only', '')}`",
        "",
        "## Round Transition Plan",
        f"- transition_authority: `{round_transition_plan.get('transition_authority', '')}`",
        f"- control_effective: `{round_transition_plan.get('control_effective', '')}`",
        f"- execute_next_round: `{round_transition_plan.get('execute_next_round', '')}`",
        f"- next_stage_mode: `{round_transition_plan.get('next_stage_mode', '')}`",
        f"- next_target_profile: `{round_transition_plan.get('next_target_profile', '')}`",
        f"- policy_extension_granted: `{round_transition_plan.get('policy_extension_granted', '')}`",
        f"- policy_extension_count: `{round_transition_plan.get('policy_extension_count', '')}`",
        f"- stop_reason: `{round_transition_plan.get('stop_reason', '')}`",
        f"- reason: {round_transition_plan.get('reason', '')}",
        f"- budget_gate: `base_remaining={budget_gate.get('base_budget_remaining', '')}, "
        f"total_remaining={budget_gate.get('total_budget_remaining', '')}, "
        f"needs_extension={budget_gate.get('needs_policy_extension', '')}`",
        "",
        "## Runtime Evidence",
        f"- selected_parent_kind: `{context_evidence.get('selected_parent_kind', '')}`",
        f"- requested_candidate_count: `{context_evidence.get('requested_candidate_count', '')}`",
        f"- final_candidate_target: `{context_evidence.get('final_candidate_target', '')}`",
        f"- has_decorrelation_targets: `{context_evidence.get('has_decorrelation_targets', '')}`",
        "",
        "## Last Round Winner",
        f"- factor: `{last_round_winner.get('factor_name', '')}`",
        f"- status: `{last_round_winner.get('status', '')}`",
        f"- metrics: {_metric_line(last_round_winner)}",
        "",
        "## Last Round Best Candidate",
        f"- factor: `{last_best_candidate.get('factor_name', '')}`",
        f"- status: `{last_best_candidate.get('status', '')}`",
        f"- metrics: {_metric_line(last_best_candidate)}",
        "",
        "## Last Round Best Keep",
        f"- factor: `{last_best_keep.get('factor_name', '')}`",
        f"- status: `{last_best_keep.get('status', '')}`",
        f"- metrics: {_metric_line(last_best_keep)}",
        "",
        "## Search Best Node",
        f"- factor: `{best_node.get('candidate_name', '') or best_node.get('factor_name', '')}`",
        f"- status: `{best_node.get('status', '')}`",
        f"- metrics: {_metric_line(best_node)}",
        "",
        "## Last Round Rollup",
        f"- child_stage_mode: `{last_round.get('child_stage_mode', '')}`",
        f"- search_improved: `{last_round.get('search_improved', '')}`",
        f"- children_collected: `{last_round.get('children_collected', 0)}`",
        f"- children_added_to_search: `{last_round.get('children_added_to_search', 0)}`",
        f"- successful_model_count: `{last_round.get('successful_model_count', 0)}`",
        f"- failed_model_count: `{last_round.get('failed_model_count', 0)}`",
    ]
    return "\n".join(lines) + "\n"


def _write_scheduler_summary_artifacts(summary_json_path: Path, payload: dict[str, Any]) -> None:
    _write_json(summary_json_path, payload)
    summary_md_path = summary_json_path.with_suffix(".md")
    summary_md_path.write_text(render_scheduler_markdown(payload), encoding="utf-8")


def _write_scheduler_error(
    path: Path,
    *,
    family: str,
    scheduler_dir: Path,
    error: BaseException,
    context: dict[str, Any] | None = None,
) -> None:
    _write_json(
        path,
        {
            "family": family,
            "scheduler_dir": str(scheduler_dir),
            "at": utc_now_iso(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": dict(context or {}),
        },
    )


def _read_multi_run_summary(multi_run_dir: Path | None) -> dict[str, Any]:
    if multi_run_dir is None:
        return {}
    summary_path = multi_run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_initial_parent(args: argparse.Namespace) -> dict[str, Any]:
    seed_pool = load_seed_pool(args.seed_pool)
    family = seed_pool.get_family(args.family)
    explicit_name = str(args.current_parent_name or "").strip()
    explicit_expression = str(args.current_parent_expression or "").strip()
    explicit_candidate_id = str(getattr(args, "parent_candidate_id", "") or "").strip()

    if explicit_name:
        return {
            "factor_name": explicit_name,
            "expression": explicit_expression or resolve_family_formula(family, explicit_name),
            "candidate_id": explicit_candidate_id or make_seed_candidate_id(explicit_name),
            "status": "explicit_seed",
            "node_kind": "explicit_parent",
        }

    latest_winner = get_latest_family_winner(db_path=args.archive_db, family=args.family)
    if latest_winner is not None:
        payload = dict(latest_winner)
        payload["node_kind"] = "archive_winner"
        return payload

    bootstrap_parent = select_bootstrap_parent(build_bootstrap_frontier(seed_pool=seed_pool, family=family))
    if bootstrap_parent:
        payload = dict(bootstrap_parent)
        payload["status"] = str(payload.get("status", "") or "seed")
        payload["node_kind"] = str(
            payload.get(
                "node_kind",
                "preferred_seed" if str(payload.get("factor_name", "")) != family.canonical_seed else "canonical_seed",
            )
        )
        return payload

    default_parent_name = resolve_preferred_refine_seed(family)
    return {
        "factor_name": default_parent_name,
        "expression": resolve_family_formula(family, default_parent_name),
        "candidate_id": make_seed_candidate_id(default_parent_name),
        "status": "seed",
        "node_kind": "preferred_seed" if default_parent_name != family.canonical_seed else "canonical_seed",
    }


def _resolve_bootstrap_parents(args: argparse.Namespace) -> list[dict[str, Any]]:
    names = [str(item or "").strip() for item in list(getattr(args, "bootstrap_parent_name", []) or [])]
    exprs = [str(item or "").strip() for item in list(getattr(args, "bootstrap_parent_expression", []) or [])]
    names = [item for item in names if item]
    if not names:
        seed_pool = load_seed_pool(args.seed_pool)
        family = seed_pool.get_family(args.family)
        initial_parent = _resolve_initial_parent(args)
        if not is_seed_stage_node_kind(str(initial_parent.get("node_kind", ""))):
            return []
        frontier = build_bootstrap_frontier(seed_pool=seed_pool, family=family)
        return [
            {
                **item,
                "status": str(item.get("status", "") or "bootstrap_seed"),
                "node_kind": "bootstrap_parent",
            }
            for item in frontier
            if str(item.get("factor_name", "")).strip()
            and str(item.get("factor_name", "")).strip() != str(initial_parent.get("factor_name", "")).strip()
        ]
    if len(names) != len(exprs):
        raise ValueError("--bootstrap-parent-name and --bootstrap-parent-expression must have the same length")
    parents: list[dict[str, Any]] = []
    for name, expr in zip(names, exprs):
        if not expr:
            raise ValueError(f"bootstrap parent {name} is missing expression")
        parents.append(
            {
                "factor_name": name,
                "expression": expr,
                "candidate_id": make_seed_candidate_id(name),
                "status": "bootstrap_seed",
                "node_kind": "bootstrap_parent",
            }
        )
    return parents


def _selection_pool_payload(nodes: list[Any], *, limit: int) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in nodes[: max(int(limit), 1)]:
        payload.append(
            {
                "node_id": item.node_id,
                "factor_name": item.factor_name,
                "branch_key": item.branch_key,
                "motif_signature": item.motif_signature,
                "frontier_score": item.frontier_score,
                "base_score": item.base_score,
                "score_breakdown": dict(item.score_breakdown),
            }
        )
    return payload


def _winner_sort_key(payload: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    status_rank = {
        "research_winner": 4.0,
        "winner": 3.0,
        "research_keep": 2.0,
        "keep": 1.0,
    }

    def _float(name: str, default: float) -> float:
        value = payload.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return (
        float(status_rank.get(str(payload.get("status", "")).strip().lower(), 0.0)),
        _float("quick_rank_ic_mean", float("-inf")),
        _float("quick_rank_icir", float("-inf")),
        _float("net_excess_ann_return", float("-inf")),
        _float("net_sharpe", float("-inf")),
        -_float("mean_turnover", float("inf")),
    )


def _pick_best_winner_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [dict(item) for item in payloads if dict(item or {})]
    if not candidates:
        return {}
    return max(candidates, key=_winner_sort_key)


def _pick_best_keep_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    keep_candidates = [
        dict(item)
        for item in payloads
        if dict(item or {}) and str(dict(item).get("status", "")).strip().lower() in {"research_keep", "keep", "research_keep_exploratory"}
    ]
    if not keep_candidates:
        return {}
    return max(keep_candidates, key=_winner_sort_key)


def _build_scheduler_summary_payload(
    *,
    family: str,
    target_profile: str,
    stage_mode: str,
    scheduler_dir: Path,
    archive_db: str,
    round_records: list[dict[str, Any]],
    current_search: dict[str, Any],
    stop_reason: str | None = None,
) -> dict[str, Any]:
    latest_archive_winner = get_latest_family_winner(db_path=archive_db, family=family)
    best_archive_winner = get_best_family_winner(db_path=archive_db, family=family)
    final_best = current_search.get("best_node") or {}
    last_round = round_records[-1] if round_records else {}
    last_round_winner = dict(last_round.get("winner") or {})
    last_round_best_candidate = dict(last_round.get("global_best_candidate") or last_round_winner)
    last_round_best_keep = dict(last_round.get("global_best_keep") or {})
    return {
        "family": family,
        "stage_mode": str(stage_mode or "auto"),
        "target_profile": target_profile,
        "scheduler_dir": str(scheduler_dir),
        "rounds_completed": len(round_records),
        "latest_winner": latest_archive_winner,
        "latest_archive_winner": latest_archive_winner,
        "best_archive_winner": best_archive_winner,
        "last_round_status": last_round.get("status"),
        "last_selected_parent_name": last_round.get("selected_parent_name", ""),
        "last_selected_parent_expression": last_round.get("selected_parent_expression", ""),
        "last_selected_parent_score_breakdown": last_round.get("selected_parent_score_breakdown", {}),
        "last_selected_parent_reason_tags": last_round.get("selected_parent_reason_tags", []),
        "last_selected_parents": last_round.get("selected_parents", []),
        "last_winner_name": str(last_round_winner.get("factor_name", "") or ""),
        "last_winner_expression": str(last_round_winner.get("expression", "") or ""),
        "last_round_winner": last_round_winner,
        "last_best_candidate_name": str(last_round_best_candidate.get("factor_name", "") or ""),
        "last_best_candidate_expression": str(last_round_best_candidate.get("expression", "") or ""),
        "last_round_best_candidate": last_round_best_candidate,
        "last_best_keep_name": str(last_round_best_keep.get("factor_name", "") or ""),
        "last_best_keep_expression": str(last_round_best_keep.get("expression", "") or ""),
        "last_round_best_keep": last_round_best_keep,
        "prompt_trace": dict(last_round.get("prompt_trace") or {}),
        "orchestration_context_evidence": dict(last_round.get("context_evidence") or {}),
        "orchestration_context_profile": dict(last_round.get("context_profile") or {}),
        "orchestration_profile": dict(last_round.get("orchestration_profile") or {}),
        "family_state": dict(last_round.get("family_state") or {}),
        "refinement_action": dict(last_round.get("refinement_action") or {}),
        "evaluation_feedback": dict(last_round.get("evaluation_feedback") or {}),
        "stage_transition_evidence": dict(last_round.get("stage_transition_evidence") or {}),
        "stage_transition": dict(last_round.get("stage_transition") or {}),
        "stage_transition_policy": dict(last_round.get("stage_transition_policy") or {}),
        "stage_transition_policy_source": str(last_round.get("stage_transition_policy_source") or ""),
        "stage_transition_legacy_audit": dict(last_round.get("stage_transition_legacy_audit") or {}),
        "stage_transition_legacy_compare": dict(last_round.get("stage_transition_legacy_compare") or {}),
        "stage_transition_signals": dict(last_round.get("stage_transition_signals") or {}),
        "saturation_assessment": dict(last_round.get("saturation_assessment") or {}),
        "round_transition_plan": dict(last_round.get("round_transition_plan") or {}),
        "transition_authority": str(last_round.get("transition_authority") or ""),
        "budget_gate_decision": dict(last_round.get("budget_gate_decision") or {}),
        "policy_extension_count": int(last_round.get("policy_extension_count") or 0),
        "family_state_decision": dict(last_round.get("family_state_decision") or {}),
        "legacy_orchestration_decision": dict(last_round.get("legacy_orchestration_decision") or {}),
        "stage_transition_shadow": dict(last_round.get("stage_transition_shadow") or {}),
        "best_node": final_best,
        "best_node_name": str(final_best.get("candidate_name", "") or final_best.get("factor_name", "") or ""),
        "best_node_expression": str(final_best.get("expression", "") or ""),
        "rounds": round_records,
        "search": current_search,
        "stop_reason": stop_reason,
    }


def _bootstrap_round1_selection(
    *,
    engine: SearchEngine,
    seed_node: Any,
    bootstrap_nodes: list[Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_parents: list[dict[str, Any]] = []
    selection_pool_nodes = (
        [seed_node, *bootstrap_nodes]
        if is_seed_stage_node_kind(getattr(seed_node, "node_kind", ""))
        else list(bootstrap_nodes)
    )
    if not selection_pool_nodes:
        return selected_parents, {}

    quality_parent = selection_pool_nodes[0]
    quality_parent.visits += 1
    selected_parents.append(
        {
            "node": quality_parent,
            "role": "quality_parent",
            "reason_tags": ["bootstrap_parent", "forced_round1"],
            "similarity_to_primary": None,
        }
    )

    if bool(engine.policy.dual_parent_enabled) and len(selection_pool_nodes) > 1:
        expandability_parent = selection_pool_nodes[1]
        expandability_parent.visits += 1
        selected_parents.append(
            {
                "node": expandability_parent,
                "role": "expandability_parent",
                "reason_tags": ["bootstrap_parent", "forced_round1", "dual_parent_expandability"],
                "similarity_to_primary": pairwise_similarity(quality_parent, expandability_parent, engine.policy),
            }
        )

    latest_selection = {
        "event": "select",
        "at": utc_now_iso(),
        "selected_parent_id": quality_parent.node_id,
        "selected_parent_factor_name": quality_parent.factor_name,
        "selected_parent_score_breakdown": dict(quality_parent.score_breakdown),
        "selected_parent_reason_tags": ["bootstrap_parent", "forced_round1"],
        "runner_up": None,
        "selected_vs_runnerup_delta": None,
        "selection_pool": _selection_pool_payload(
            selection_pool_nodes,
            limit=max(int(engine.policy.selection_pool_size), len(selection_pool_nodes)),
        ),
        "selected_parents": [
            {
                "node_id": item["node"].node_id,
                "factor_name": item["node"].factor_name,
                "role": item["role"],
                "reason_tags": list(item.get("reason_tags") or []),
                "similarity_to_primary": item.get("similarity_to_primary"),
                "score_breakdown": dict(item["node"].score_breakdown),
            }
            for item in selected_parents
        ],
        "dual_parent_triggered": len(selected_parents) > 1,
    }
    engine.event_log.append(latest_selection)
    return selected_parents, latest_selection

def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    args = build_arg_parser().parse_args()
    models = _normalize_models(args.models)
    transition_authority = str(args.transition_authority or DEFAULT_POLICY_CONFIG.round_transition.default_authority)
    base_max_rounds = max(int(args.max_rounds), 0)
    max_policy_extensions = max(int(args.max_policy_extensions), 0)
    requested_max_total_rounds = int(args.max_total_rounds or 0)
    max_total_rounds = max(
        requested_max_total_rounds,
        base_max_rounds,
        base_max_rounds + (max_policy_extensions if transition_authority == "guarded_control" else 0),
    )
    effective_scheduler_rounds = max_total_rounds if transition_authority == "guarded_control" else base_max_rounds
    active_target_profile = str(args.target_profile)
    policy_extension_count = 0

    scheduler_dir = Path(args.scheduler_runs_dir).expanduser().resolve() / (
        time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_{args.family}"
    )
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    summary_path = scheduler_dir / "summary.json"
    plan_path = scheduler_dir / "plan.json"

    policy = (
        SearchPolicy.multi_model_best_first(preset=args.policy_preset)
        .with_target_profile(active_target_profile)
        .with_mmr_rerank(not bool(args.disable_mmr_rerank))
    )
    policy = policy.with_dual_parent(bool(args.enable_dual_parent_round))
    budget = SearchBudget(
        max_rounds=int(effective_scheduler_rounds),
        family_budget=int(effective_scheduler_rounds),
        branch_budget=2,
        max_frontier_size=max(len(models) * max(int(args.n_candidates), 1), 6),
        max_depth=max(int(effective_scheduler_rounds), 2) + 1,
        stop_if_no_improve=int(args.stop_if_no_new_winner),
    )
    engine = SearchEngine(
        family=args.family,
        budget=budget,
        policy=policy,
        normalizer=build_search_normalizer(db_path=args.archive_db, family=args.family),
    )
    initial_parent = _resolve_initial_parent(args)
    seed_node = engine.register_seed(
        factor_name=str(initial_parent.get("factor_name", "")),
        expression=str(initial_parent.get("expression", "")),
        candidate_id=str(initial_parent.get("candidate_id", "")),
        node_kind=str(initial_parent.get("node_kind", "seed")),
        status=str(initial_parent.get("status", "seed")),
        source_run_id=str(initial_parent.get("run_id", "")),
        source_model=str(initial_parent.get("source_model", "")),
        source_provider=str(initial_parent.get("source_provider", "")),
        round_id=int(initial_parent.get("round_id") or 0),
        metrics=initial_parent,
    )
    bootstrap_nodes = []
    for bootstrap_parent in _resolve_bootstrap_parents(args):
        if str(bootstrap_parent.get("factor_name", "")).strip() == str(seed_node.factor_name).strip():
            continue
        bootstrap_nodes.append(
            engine.register_seed(
                factor_name=str(bootstrap_parent.get("factor_name", "")),
                expression=str(bootstrap_parent.get("expression", "")),
                candidate_id=str(bootstrap_parent.get("candidate_id", "")),
                node_kind=str(bootstrap_parent.get("node_kind", "bootstrap_parent")),
                status=str(bootstrap_parent.get("status", "bootstrap_seed")),
                source_run_id=str(bootstrap_parent.get("run_id", "")),
                source_model=str(bootstrap_parent.get("source_model", "")),
                source_provider=str(bootstrap_parent.get("source_provider", "")),
                round_id=int(bootstrap_parent.get("round_id") or 0),
                metrics=bootstrap_parent,
            )
        )

    plan = {
        "family": args.family,
        "stage_mode": str(args.stage_mode or "auto"),
        "target_profile": str(args.target_profile),
        "started_at": utc_now_iso(),
        "base_max_rounds": int(base_max_rounds),
        "effective_scheduler_rounds": int(effective_scheduler_rounds),
        "max_total_rounds": int(max_total_rounds),
        "transition_authority": transition_authority,
        "max_policy_extensions": int(max_policy_extensions),
        "stop_if_no_new_winner": int(args.stop_if_no_new_winner),
        "models": models,
        "n_candidates": int(args.n_candidates),
        "max_parallel": int(args.max_parallel),
        "selection_mode": "best_first_ucb_lite",
        "search_policy": policy.to_dict(),
        "search_budget": budget.to_dict(),
        "initial_parent": seed_node.to_dict(),
        "bootstrap_parents": [node.to_dict() for node in bootstrap_nodes],
        "auto_apply_promotion": bool(args.auto_apply_promotion),
        "dual_parent_enabled": bool(args.enable_dual_parent_round),
        "dual_parent_model_allocations": _resolve_dual_parent_model_allocations(args, models),
    }
    _write_json(plan_path, plan)

    round_records: list[dict[str, Any]] = []
    stop_reason = "max_rounds"
    error_path = scheduler_dir / "scheduler_error.json"
    error_context: dict[str, Any] = {
        "family": args.family,
        "target_profile": str(args.target_profile),
        "scheduler_dir": str(scheduler_dir),
    }

    try:
        while engine.can_continue():
            round_idx = len(round_records) + 1
            latest_selection = None
            error_context["round_idx"] = round_idx
            if round_idx == 1 and bootstrap_nodes:
                selected_parents, latest_selection = _bootstrap_round1_selection(
                    engine=engine,
                    seed_node=seed_node,
                    bootstrap_nodes=bootstrap_nodes,
                )
            else:
                selected_parents = engine.select_next_parents()
                if not selected_parents:
                    stop_reason = "frontier_exhausted"
                    break
                for event in reversed(engine.event_log):
                    if event.get("event") == "select":
                        latest_selection = event
                        break
            log_path = scheduler_dir / f"round_{round_idx:02d}.log"
            round_multi_root = scheduler_dir / "multi_runs" / f"round_{round_idx:02d}"
            round_multi_root.mkdir(parents=True, exist_ok=True)
            parent_model_allocations = _resolve_dual_parent_model_allocations(args, models)
            sub_runs: list[dict[str, Any]] = []
            expansion_payloads: list[dict[str, Any]] = []
            error_context.update(
                {
                    "log_path": str(log_path),
                    "round_multi_root": str(round_multi_root),
                    "selected_parent_names": [
                        str(item["node"].factor_name) for item in selected_parents
                    ],
                }
            )
            print(f"[launch] family={args.family} round={round_idx} log={log_path}")
            with log_path.open("w", encoding="utf-8") as log_fp:
                launched_runs: list[dict[str, Any]] = []
                for idx, selected_entry in enumerate(selected_parents, start=1):
                    selected_parent = selected_entry["node"]
                    parent_role = str(selected_entry["role"])
                    selected_models = models if len(selected_parents) == 1 else parent_model_allocations.get(parent_role, models)
                    parent_multi_root = round_multi_root / f"{idx:02d}_{parent_role}"
                    parent_multi_root.mkdir(parents=True, exist_ok=True)
                    child_stage_mode = _resolve_child_stage_mode(
                        str(args.stage_mode or "auto"),
                        round_idx=round_idx,
                        last_round=round_records[-1] if round_records else None,
                    )
                    cmd = _build_round_cmd(
                        args,
                        parent={
                            "factor_name": selected_parent.factor_name,
                            "expression": selected_parent.expression,
                            "candidate_id": selected_parent.candidate_id,
                            "node_kind": selected_parent.node_kind,
                        },
                        round_multi_root=parent_multi_root,
                        child_stage_mode=child_stage_mode,
                        target_profile=active_target_profile,
                        force_round1_seed_stage=child_stage_mode == "new_family_broad",
                        models_override=selected_models,
                    )
                    log_fp.write(
                        f"[parent] role={parent_role} factor={selected_parent.factor_name} "
                        f"stage_mode={child_stage_mode} models={','.join(selected_models)}\n"
                    )
                    log_fp.flush()
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(PROJECT_ROOT),
                        env=os.environ.copy(),
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    launched_runs.append(
                        {
                            "proc": proc,
                            "selected_parent": selected_parent,
                            "parent_role": parent_role,
                            "selected_models": list(selected_models),
                            "parent_multi_root": parent_multi_root,
                            "child_stage_mode": child_stage_mode,
                        }
                    )

                for launched in launched_runs:
                    proc = launched["proc"]
                    selected_parent = launched["selected_parent"]
                    parent_role = str(launched["parent_role"])
                    selected_models = list(launched["selected_models"])
                    parent_multi_root = launched["parent_multi_root"]
                    returncode = int(proc.wait())
                    multi_run_dir = resolve_materialized_multi_run_dir(parent_multi_root)
                    multi_run_summary = _read_multi_run_summary(multi_run_dir)
                    parent_status = str(
                        multi_run_summary.get("status") or ("ok" if returncode == 0 else "failed")
                    )
                    child_records = (
                        load_multi_run_candidate_records(
                            archive_db=str(args.archive_db),
                            multi_run_dir=multi_run_dir,
                            family=args.family,
                        )
                        if multi_run_dir is not None
                        else []
                    )
                    expansion_payloads.append(
                        {
                            "parent_node_id": selected_parent.node_id,
                            "child_records": child_records,
                            "success": parent_status != "failed",
                            "source_run_id": str(multi_run_dir or ""),
                            "note": (
                                f"role={parent_role};multi_run_dir={multi_run_dir}"
                                if multi_run_dir is not None
                                else f"role={parent_role};multi_run_dir_missing"
                            ),
                        }
                    )
                    sub_runs.append(
                        {
                            "role": parent_role,
                            "stage_mode": str(multi_run_summary.get("stage_mode") or launched.get("child_stage_mode") or ""),
                            "selected_parent": selected_parent.to_dict(),
                            "selected_parent_name": selected_parent.factor_name,
                            "selected_parent_expression": selected_parent.expression,
                            "selected_models": list(selected_models),
                            "status": parent_status,
                            "returncode": returncode,
                            "multi_run_dir": str(multi_run_dir) if multi_run_dir is not None else "",
                            "winner": multi_run_summary.get("winner") or {},
                            "global_best_candidate": multi_run_summary.get("global_best_candidate")
                            or multi_run_summary.get("winner")
                            or {},
                            "global_best_keep": multi_run_summary.get("global_best_keep") or {},
                            "winner_selection_mode": str(multi_run_summary.get("winner_selection_mode") or ""),
                            "global_rerank_preview": list(multi_run_summary.get("global_rerank_preview") or []),
                            "successful_model_count": int(multi_run_summary.get("successful_model_count") or 0),
                            "failed_model_count": int(multi_run_summary.get("failed_model_count") or 0),
                            "successful_models": list(multi_run_summary.get("successful_models") or []),
                            "failed_models": list(multi_run_summary.get("failed_models") or []),
                            "children_collected": len(child_records),
                            "prompt_trace": dict(multi_run_summary.get("prompt_trace") or {}),
                        }
                    )

            success_count = sum(1 for item in sub_runs if str(item.get("status")) != "failed")
            if success_count <= 0:
                round_status = "failed"
            elif success_count == len(sub_runs):
                round_status = "ok"
            else:
                round_status = "partial_success"
            expansion = engine.register_round_expansions(
                expansions=expansion_payloads,
                note=f"round={round_idx};dual_parent={len(selected_parents) > 1}",
            )
            latest_archive_winner = get_latest_family_winner(db_path=args.archive_db, family=args.family)
            best_archive_winner = get_best_family_winner(db_path=args.archive_db, family=args.family)
            round_winner = _pick_best_winner_payload(
                [dict(item.get("winner") or {}) for item in sub_runs]
            )
            round_best_candidate = _pick_best_winner_payload(
                [dict(item.get("global_best_candidate") or {}) for item in sub_runs]
            )
            round_best_keep = _pick_best_keep_payload(
                [dict(item.get("global_best_keep") or {}) for item in sub_runs]
            )
            primary_parent = selected_parents[0]["node"]
            current_search = engine.summary()
            current_best = current_search.get("best_node") or {}
            resolved_child_stage_mode = _resolve_child_stage_mode(
                str(args.stage_mode or "auto"),
                round_idx=round_idx,
                last_round=round_records[-1] if round_records else None,
            )
            record = {
                "round": round_idx,
                "child_stage_mode": resolved_child_stage_mode,
                "status": round_status,
                "returncode": 0 if round_status != "failed" else 1,
                "log_path": str(log_path),
                "multi_run_dir": str(round_multi_root),
                "dual_parent_triggered": len(selected_parents) > 1,
                "selected_parent": primary_parent.to_dict(),
                "selected_parent_name": primary_parent.factor_name,
                "selected_parent_expression": primary_parent.expression,
                "selected_parent_score_breakdown": dict((latest_selection or {}).get("selected_parent_score_breakdown") or {}),
                "selected_parent_reason_tags": list((latest_selection or {}).get("selected_parent_reason_tags") or []),
                "runner_up": (latest_selection or {}).get("runner_up"),
                "selected_vs_runnerup_delta": (latest_selection or {}).get("selected_vs_runnerup_delta"),
                "selection_pool": list((latest_selection or {}).get("selection_pool") or []),
                "selected_parents": list((latest_selection or {}).get("selected_parents") or []),
                "parent_model_allocations": parent_model_allocations,
                "children_collected": sum(int(item.get("children_collected") or 0) for item in sub_runs),
                "children_added_to_search": sum(len(item.get("children", [])) for item in expansion.get("results", [])),
                "successful_model_count": sum(int(item.get("successful_model_count") or 0) for item in sub_runs),
                "failed_model_count": sum(int(item.get("failed_model_count") or 0) for item in sub_runs),
                "successful_models": sorted(
                    {name for item in sub_runs for name in list(item.get("successful_models") or [])}
                ),
                "failed_models": sorted(
                    {name for item in sub_runs for name in list(item.get("failed_models") or [])}
                ),
                "search_improved": bool(expansion.get("improved")),
                "winner": round_winner,
                "winner_name": str(round_winner.get("factor_name", "") or ""),
                "winner_expression": str(round_winner.get("expression", "") or ""),
                "global_best_candidate": round_best_candidate,
                "global_best_candidate_name": str(round_best_candidate.get("factor_name", "") or ""),
                "global_best_candidate_expression": str(round_best_candidate.get("expression", "") or ""),
                "global_best_keep": round_best_keep,
                "global_best_keep_name": str(round_best_keep.get("factor_name", "") or ""),
                "global_best_keep_expression": str(round_best_keep.get("expression", "") or ""),
                "latest_winner": latest_archive_winner,
                "latest_archive_winner": latest_archive_winner,
                "best_archive_winner": best_archive_winner,
                "prompt_trace": _pick_round_prompt_trace(sub_runs)
                or {
                    "stage_mode": resolved_child_stage_mode,
                    "seed_stage_active": resolved_child_stage_mode == "new_family_broad",
                    "selected_parent_kind": str(primary_parent.node_kind),
                    "selected_parent_factor_name": str(primary_parent.factor_name),
                },
                "best_node": current_best,
                "best_node_name": str(current_best.get("candidate_name", "") or current_best.get("factor_name", "") or ""),
                "best_node_expression": str(current_best.get("expression", "") or ""),
                "sub_runs": sub_runs,
                "search_summary": current_search,
            }
            record.update(
                _build_orchestration_trace(
                    family=args.family,
                    stage_mode=resolved_child_stage_mode,
                    target_profile=str(active_target_profile),
                    policy_preset=str(args.policy_preset),
                    parent_kind=str(primary_parent.node_kind),
                    requested_candidate_count=int(getattr(args, "n_candidates", 0) or 0),
                    final_candidate_target=int(getattr(args, "n_candidates", 0) or 0),
                    has_decorrelation_targets=bool(args.decorrelation_target),
                    round_status=round_status,
                    search_improved=bool(expansion.get("improved")),
                    winner=dict(round_best_candidate or round_winner or {}),
                    keep=dict(round_best_keep or {}),
                    consecutive_no_improve=int(engine.consecutive_no_improve),
                    children_collected=int(record.get("children_collected") or 0),
                    children_added_to_search=int(record.get("children_added_to_search") or 0),
                    round_idx=int(round_idx),
                    base_max_rounds=int(base_max_rounds),
                    max_total_rounds=int(max_total_rounds),
                    transition_authority=transition_authority,
                    policy_extension_count=int(policy_extension_count),
                    max_policy_extensions=int(max_policy_extensions),
                )
            )
            round_transition_plan = dict(record.get("round_transition_plan") or {})
            if bool(round_transition_plan.get("policy_extension_granted")):
                policy_extension_count = int(round_transition_plan.get("policy_extension_count") or policy_extension_count)
            round_records.append(record)
            _write_scheduler_summary_artifacts(
                summary_path,
                _build_scheduler_summary_payload(
                    family=args.family,
                    target_profile=str(active_target_profile),
                    stage_mode=str(args.stage_mode or "auto"),
                    scheduler_dir=scheduler_dir,
                    archive_db=str(args.archive_db),
                    round_records=round_records,
                    current_search=current_search,
                    stop_reason=None,
                ),
            )

            round_returncode = max((int(item.get("returncode") or 0) for item in sub_runs), default=0)
            if round_status == "failed":
                print(f"[stop] round={round_idx} failed with returncode={round_returncode}")
                stop_reason = "round_failed"
                break
            if round_status == "partial_success":
                print(
                    f"[warn] round={round_idx} partial_success: "
                    f"{record['successful_model_count']} succeeded, {record['failed_model_count']} failed"
                )
            if int(args.stop_if_no_new_winner) > 0 and engine.consecutive_no_improve >= int(args.stop_if_no_new_winner):
                print(f"[stop] reached stop_if_no_new_winner threshold: {engine.consecutive_no_improve}")
                stop_reason = "no_new_search_improvement"
                break
            if transition_authority == "guarded_control":
                if not bool(round_transition_plan.get("execute_next_round")):
                    stop_reason = str(round_transition_plan.get("stop_reason") or "round_transition_stop")
                    print(f"[stop] round transition denied next round: {stop_reason}")
                    break
                next_target_profile = str(round_transition_plan.get("next_target_profile") or active_target_profile)
                if next_target_profile and next_target_profile != active_target_profile:
                    active_target_profile = next_target_profile
                    engine.policy = (
                        SearchPolicy.multi_model_best_first(preset=args.policy_preset)
                        .with_target_profile(active_target_profile)
                        .with_mmr_rerank(not bool(args.disable_mmr_rerank))
                        .with_dual_parent(bool(args.enable_dual_parent_round))
                    )
            if float(args.sleep_between_rounds) > 0 and engine.can_continue():
                time.sleep(float(args.sleep_between_rounds))
    except KeyboardInterrupt as exc:
        stop_reason = "interrupted"
        _write_scheduler_error(
            error_path,
            family=args.family,
            scheduler_dir=scheduler_dir,
            error=exc,
            context={**error_context, "rounds_completed": len(round_records), "stop_reason": stop_reason},
        )
        _write_scheduler_summary_artifacts(
            summary_path,
            _build_scheduler_summary_payload(
                family=args.family,
                target_profile=str(active_target_profile),
                stage_mode=str(args.stage_mode or "auto"),
                scheduler_dir=scheduler_dir,
                archive_db=str(args.archive_db),
                round_records=round_records,
                current_search=engine.summary(),
                stop_reason=stop_reason,
            ),
        )
        return 130
    except Exception as exc:
        stop_reason = "scheduler_exception"
        _write_scheduler_error(
            error_path,
            family=args.family,
            scheduler_dir=scheduler_dir,
            error=exc,
            context={**error_context, "rounds_completed": len(round_records), "stop_reason": stop_reason},
        )
        _write_scheduler_summary_artifacts(
            summary_path,
            _build_scheduler_summary_payload(
                family=args.family,
                target_profile=str(active_target_profile),
                stage_mode=str(args.stage_mode or "auto"),
                scheduler_dir=scheduler_dir,
                archive_db=str(args.archive_db),
                round_records=round_records,
                current_search=engine.summary(),
                stop_reason=stop_reason,
            ),
        )
        return 1

    _write_scheduler_summary_artifacts(
        summary_path,
        _build_scheduler_summary_payload(
            family=args.family,
            target_profile=str(active_target_profile),
            stage_mode=str(args.stage_mode or "auto"),
            scheduler_dir=scheduler_dir,
            archive_db=str(args.archive_db),
            round_records=round_records,
            current_search=engine.summary(),
            stop_reason=stop_reason,
        ),
    )
    print(f"[scheduler_dir] {scheduler_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
