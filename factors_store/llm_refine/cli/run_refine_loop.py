from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

from ..config import (
    DEFAULT_AUTO_APPLY_PROMOTION,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_N_CANDIDATES,
    DEFAULT_NAME_PREFIX,
    DEFAULT_POLICY_PRESET,
    DEFAULT_PROVIDER_NAME,
    DEFAULT_RETRY_ON_PARSE_FAIL,
    DEFAULT_ROUND1_DONOR_FACTOR_LIMIT,
    DEFAULT_ROUND1_DONOR_FAMILY_LIMIT,
    DEFAULT_TARGET_PROFILE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)
from ..core.archive import (
    DEFAULT_ARCHIVE_DB,
    DEFAULT_RUNS_DIR,
    create_run_dir,
    ensure_run_subdir,
    expression_hash,
    get_candidate_record,
    get_latest_family_round,
    get_latest_family_winner,
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
from ..evaluation.evaluator import evaluate_refinement_run
from ..core.models import LLMProposal, LLMProviderConfig, RefinementCandidate
from ..parsing.parser import deduplicate_candidates, parse_refinement_response
from ..prompting.prompt_builder import PROMPT_TEMPLATE_VERSIONS, build_refinement_prompt, load_prompt_history_row
from ..prompting.prompt_plan import build_prompt_plan
from ..core.providers import OpenAICompatProvider
from ..evaluation.redundancy import filter_structurally_redundant, structure_filter_markdown
from ..knowledge.next_experiments import retrieve_runtime_donor_motifs
from ..knowledge.round1 import (
    build_bootstrap_frontier,
    light_rerank_candidates,
    light_rerank_markdown,
    resolve_requested_candidate_count,
    resolve_round1_role_slots,
    select_bootstrap_parent,
)
from ..knowledge.reflection import build_reflection_card, reflection_markdown
from ..core.seed_loader import (
    DEFAULT_SEED_POOL,
    apply_direction_rule,
    load_seed_pool,
    resolve_factor_direction,
    resolve_family_formula,
    resolve_preferred_refine_seed,
)
from ..parsing.validator import validate_expression
from ..search import SearchBudget, SearchEngine, SearchPolicy, build_search_normalizer
from ..search.context_resolver import ContextEvidence, resolve_context_profile
from ..search.scoring import safe_float

_STAGE_MODE_CHOICES = (
    "auto",
    "new_family_broad",
    "broad_followup",
    "focused_refine",
    "confirmation",
    "donor_validation",
)


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _warn_if_provider_env_missing(args: argparse.Namespace) -> None:
    env_provider = os.getenv("LLM_PROVIDER_NAME", "").strip()
    env_base_url = os.getenv("LLM_BASE_URL", "").strip()
    env_api_key = os.getenv("LLM_API_KEY", "").strip()
    if env_provider or env_base_url or env_api_key:
        return

    provider_name = str(getattr(args, "provider_name", "") or "").strip()
    base_url = str(getattr(args, "base_url", "") or "").strip()
    api_key = str(getattr(args, "api_key", "") or "").strip()
    using_cli_fallback = (
        provider_name == DEFAULT_PROVIDER_NAME
        and base_url == DEFAULT_BASE_URL
        and api_key == DEFAULT_API_KEY
    )
    if not using_cli_fallback:
        return

    print(
        "[warn] provider env is not loaded; llm_refine is falling back to CLI defaults "
        f"(provider={DEFAULT_PROVIDER_NAME}, base_url={DEFAULT_BASE_URL})."
    )
    print(
        "[warn] if this is not intentional, create `./llm_refine_provider_env.sh` "
        "from `./llm_refine_provider_env.example.sh`, then source it before invoking "
        "run_refine_* commands."
    )


def _provider_env_loaded() -> bool:
    return any(os.getenv(name, "").strip() for name in ("LLM_PROVIDER_NAME", "LLM_BASE_URL", "LLM_API_KEY"))


def _using_cli_fallback_defaults(args: argparse.Namespace) -> bool:
    provider_name = str(getattr(args, "provider_name", "") or "").strip()
    base_url = str(getattr(args, "base_url", "") or "").strip()
    api_key = str(getattr(args, "api_key", "") or "").strip()
    return (
        provider_name == DEFAULT_PROVIDER_NAME
        and base_url == DEFAULT_BASE_URL
        and api_key == DEFAULT_API_KEY
    )


def _api_key_required(args: argparse.Namespace) -> bool:
    provider_name = str(getattr(args, "provider_name", "") or "").strip().lower()
    base_url = str(getattr(args, "base_url", "") or "").strip().lower()
    if provider_name in {"qwen", "local"}:
        return False
    local_prefixes = (
        "http://127.0.0.1",
        "http://localhost",
        "http://10.",
        "http://172.",
        "http://192.168.",
    )
    return not any(base_url.startswith(prefix) for prefix in local_prefixes)


def _run_preflight_checks(args: argparse.Namespace) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    runtime_mode = "dry_run" if bool(args.dry_run) else "live"
    env_loaded = _provider_env_loaded()
    fallback_defaults = _using_cli_fallback_defaults(args)
    api_key = str(getattr(args, "api_key", "") or "").strip()
    openai_spec = importlib.util.find_spec("openai")

    record(
        "provider_env_or_explicit_provider",
        env_loaded or not fallback_defaults,
        (
            "provider env loaded"
            if env_loaded
            else "using explicit CLI provider settings"
            if not fallback_defaults
            else "provider env missing while CLI stayed on fallback defaults"
        ),
    )
    record(
        "api_key_present",
        (not _api_key_required(args)) or bool(args.dry_run) or (bool(api_key) and api_key != DEFAULT_API_KEY),
        (
            "api key not required for local-compatible provider"
            if not _api_key_required(args)
            else "api key is configured"
            if bool(api_key) and api_key != DEFAULT_API_KEY
            else "api key is empty or fallback placeholder"
        ),
    )
    record(
        "openai_dependency",
        bool(args.dry_run) or openai_spec is not None,
        (
            f"openai importable via {sys.executable}"
            if openai_spec is not None
            else f"openai package not found for {sys.executable}"
        ),
    )
    for field_name, label in (("panel_path", "panel_path"), ("benchmark_path", "benchmark_path")):
        raw_value = str(getattr(args, field_name, "") or "").strip()
        if not raw_value:
            continue
        path = os.path.expanduser(raw_value)
        record(label, os.path.exists(path), f"{label} exists: {path}" if os.path.exists(path) else f"{label} missing: {path}")

    failed = [item for item in checks if not bool(item["passed"])]
    payload = {
        "runtime_mode": runtime_mode,
        "python_executable": sys.executable,
        "provider_name": str(getattr(args, "provider_name", "") or ""),
        "base_url": str(getattr(args, "base_url", "") or ""),
        "model": str(getattr(args, "model", "") or ""),
        "provider_env_loaded": env_loaded,
        "using_cli_fallback_defaults": fallback_defaults,
        "checks": checks,
        "ok": not failed,
    }
    if failed:
        details = "; ".join(f"{item['name']}: {item['detail']}" for item in failed)
        raise RuntimeError(f"preflight failed: {details}")
    return payload


def _make_runtime_status_writer(
    *,
    status_path: str | os.PathLike[str],
    family: str,
    run_id: str,
    model: str,
) -> callable:
    started_monotonic = time.monotonic()

    def _write(phase: str, message: str = "", **extra: object) -> None:
        elapsed_seconds = round(time.monotonic() - started_monotonic, 3)
        payload: dict[str, object] = {
            "family": family,
            "run_id": run_id,
            "model": model,
            "phase": str(phase),
            "message": str(message),
            "elapsed_seconds": elapsed_seconds,
            "updated_at": utc_now_iso(),
        }
        if extra:
            payload["extra"] = extra
        _write_json(status_path, payload)
        suffix = f" message={message}" if str(message).strip() else ""
        print(f"[status] phase={phase} elapsed={elapsed_seconds:.1f}s{suffix}", flush=True)

    return _write


def _with_validation(candidate: RefinementCandidate) -> RefinementCandidate:
    merged_warnings = tuple(
        dict.fromkeys((*candidate.validation_warnings, *validate_expression(candidate.expression)))
    )
    return RefinementCandidate(
        name=candidate.name,
        expression=candidate.expression,
        explanation=candidate.explanation,
        candidate_role=candidate.candidate_role,
        rationale=candidate.rationale,
        expected_improvement=candidate.expected_improvement,
        risk=candidate.risk,
        source_model=candidate.source_model,
        source_provider=candidate.source_provider,
        parent_factor=candidate.parent_factor,
        family=candidate.family,
        candidate_id=candidate.candidate_id,
        round_id=candidate.round_id,
        parent_candidate_id=candidate.parent_candidate_id,
        status=candidate.status,
        validation_warnings=merged_warnings,
    )


def _rebuild_proposal(proposal: LLMProposal, candidates: tuple[RefinementCandidate, ...]) -> LLMProposal:
    return LLMProposal(
        parent_factor=proposal.parent_factor,
        diagnosed_weaknesses=proposal.diagnosed_weaknesses,
        refinement_rationale=proposal.refinement_rationale,
        candidates=candidates,
        expected_behavior_change=proposal.expected_behavior_change,
        risk_notes=proposal.risk_notes,
        raw_response=proposal.raw_response,
    )


def _write_json(path: str | os.PathLike[str], payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)


def _build_prompt_trace(
    *,
    family: str,
    stage_mode: str,
    target_profile: str,
    policy_preset: str,
    seed_stage_active: bool,
    selected_parent: object,
    requested_candidate_count: int,
    final_candidate_target: int,
    role_slots: list[str],
    bootstrap_frontier: list[dict[str, object]],
    donor_motifs: list[dict[str, object]],
    decorrelation_targets: list[str],
    prompt_template_version: str,
) -> dict[str, object]:
    context_evidence = ContextEvidence.from_runtime(
        family=family,
        stage_mode=stage_mode,
        target_profile=target_profile,
        policy_preset=policy_preset,
        is_seed_stage=seed_stage_active,
        has_bootstrap_frontier=bool(bootstrap_frontier),
        has_donor_motifs=bool(donor_motifs),
        has_decorrelation_targets=bool(decorrelation_targets),
        selected_parent_kind=str(getattr(selected_parent, "node_kind", "") or ""),
        requested_candidate_count=int(requested_candidate_count),
        final_candidate_target=int(final_candidate_target),
    )
    context_profile = resolve_context_profile(context_evidence)
    prompt_plan = build_prompt_plan(
        stage_mode=stage_mode,
        target_profile=target_profile,
        policy_preset=policy_preset,
        is_seed_stage=seed_stage_active,
        has_donor_motifs=bool(donor_motifs),
        has_bootstrap_frontier=bool(bootstrap_frontier),
        has_decorrelation_targets=bool(decorrelation_targets),
        context_profile=context_profile,
    )
    return {
        "stage_mode": str(stage_mode or "auto"),
        "target_profile": str(target_profile or "raw_alpha"),
        "policy_preset": str(policy_preset or "balanced"),
        "prompt_template_version": str(prompt_template_version or "current_compact"),
        "seed_stage_active": bool(seed_stage_active),
        "selected_parent_kind": str(getattr(selected_parent, "node_kind", "") or ""),
        "selected_parent_factor_name": str(getattr(selected_parent, "factor_name", "") or ""),
        "requested_candidate_count": int(requested_candidate_count),
        "final_candidate_target": int(final_candidate_target),
        "role_slots": list(role_slots),
        "bootstrap_frontier_count": len(list(bootstrap_frontier or [])),
        "donor_motifs_count": len(list(donor_motifs or [])),
        "decorrelation_target_count": len(list(decorrelation_targets or [])),
        "decorrelation_targets": list(decorrelation_targets or []),
        "context_evidence": context_evidence.to_dict(),
        "context_profile": context_profile.to_dict(),
        "prompt_plan": prompt_plan.to_dict(),
    }


def _normalize_multi_text(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    out: list[str] = []
    for raw in values or ():
        for part in str(raw).split(","):
            text = part.strip()
            if text and text not in out:
                out.append(text)
    return tuple(out)


def _hard_validation_filter_reason(candidate: RefinementCandidate) -> str:
    warnings = tuple(str(item) for item in candidate.validation_warnings)
    if any(item.startswith("contains dotted factor references:") for item in warnings):
        return "contains dotted factor reference(s) unsupported by expression engine"
    if any(item.startswith("contains unsupported keyword arguments:") for item in warnings):
        return "contains unsupported keyword argument(s) for current expression engine"
    if any(item.startswith("contains tokens outside current whitelist:") for item in warnings):
        return "contains tokens outside current expression whitelist"
    return ""


def _build_parse_retry_user_prompt(*, original_user_prompt: str, parse_error: str) -> str:
    return (
        f"{original_user_prompt}\n\n"
        "Your previous reply could not be parsed as valid JSON.\n"
        f"Parse error: {parse_error}\n\n"
        "Re-emit the full response from scratch as exactly one complete JSON object.\n"
        "Requirements:\n"
        "- Output JSON only, no markdown fences, no commentary.\n"
        "- Make `explanation` very short (<= 40 Chinese chars).\n"
        "- Make `rationale` very short (<= 24 Chinese chars).\n"
        "- Keep `diagnosed_weaknesses` <= 4 short bullets.\n"
        "- Keep `refinement_rationale`, `expected_behavior_change`, and `risk_notes` to one short sentence each.\n"
        "- Keep `expected_improvement` and `risk` concise enum values only.\n"
        "- Ensure all quotes, commas, brackets, and braces are closed.\n"
        "- Keep the same schema with `parent_factor` and `candidate_formulas`.\n"
        "- Do not truncate the object.\n"
    )


def _pick_best_child_record(records: list[dict[str, object]]) -> dict[str, object] | None:
    if not records:
        return None

    status_rank = {
        "research_winner": 4,
        "winner": 3,
        "research_keep": 2,
        "keep": 1,
    }

    def _score(item: dict[str, object]) -> tuple[float, float, float, float, float, float]:
        status = str(item.get("status", "")).strip().lower()
        return (
            float(status_rank.get(status, 0)),
            safe_float(item.get("quick_rank_ic_mean"), default=float("-inf")),
            safe_float(item.get("quick_rank_icir"), default=float("-inf")),
            safe_float(item.get("net_sharpe"), default=float("-inf")),
            safe_float(item.get("net_ann_return"), default=float("-inf")),
            -safe_float(item.get("mean_turnover"), default=float("inf")),
        )

    return max(records, key=_score)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one LLM-guided refinement round for a seed family.")
    parser.add_argument("--seed-pool", default=str(DEFAULT_SEED_POOL), help="seed pool yaml path")
    parser.add_argument("--family", required=True, help="seed family name from the seed pool")
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=DEFAULT_N_CANDIDATES,
        help="number of candidates requested from the LLM",
    )
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="artifact output directory")
    parser.add_argument("--archive-db", default=str(DEFAULT_ARCHIVE_DB), help="sqlite archive path")
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX, help="prefix used when writing candidate YAML")
    parser.add_argument("--round-id", type=int, default=None, help="refinement round id; round 0 is reserved for seed baseline")
    parser.add_argument("--parent-candidate-id", default="", help="parent candidate id; defaults to seed::<parent_factor>")
    parser.add_argument("--auto-parent", action="store_true", help="use latest family research winner from archive as current parent")
    parser.add_argument("--current-parent-name", default="", help="explicit parent factor name override")
    parser.add_argument("--current-parent-expression", default="", help="explicit parent expression override")
    parser.add_argument(
        "--provider-name",
        default=_env_or_default("LLM_PROVIDER_NAME", DEFAULT_PROVIDER_NAME),
        help="provider label stored in artifacts (env: LLM_PROVIDER_NAME)",
    )
    parser.add_argument(
        "--base-url",
        default=_env_or_default("LLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL (env: LLM_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=_env_or_default("LLM_API_KEY", DEFAULT_API_KEY),
        help="OpenAI-compatible api key (env: LLM_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=_env_or_default("LLM_MODEL", DEFAULT_MODEL),
        help="model name (env: LLM_MODEL)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(_env_or_default("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
        help="generation temperature (env: LLM_TEMPERATURE)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(_env_or_default("LLM_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
        help="max completion tokens (env: LLM_MAX_TOKENS)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(_env_or_default("LLM_TIMEOUT", str(DEFAULT_TIMEOUT))),
        help="request timeout (env: LLM_TIMEOUT)",
    )
    parser.add_argument(
        "--retry-on-parse-fail",
        type=int,
        default=DEFAULT_RETRY_ON_PARSE_FAIL,
        help="retry provider generation this many times when the response cannot be parsed as valid JSON",
    )
    parser.add_argument("--additional-notes", default="", help="extra run-specific notes appended to the prompt")
    parser.add_argument(
        "--decorrelation-target",
        action="append",
        default=[],
        help="explicit factor(s) to decorrelate from; may be repeated or passed as comma-separated names",
    )
    parser.add_argument("--dry-run", action="store_true", help="only write prompts, do not call the provider")
    parser.add_argument("--print-only", action="store_true", help="print the prompt bundle and exit")
    parser.add_argument("--skip-eval", action="store_true", help="skip automatic family backtest for generated candidates")
    parser.add_argument(
        "--policy-preset",
        default=DEFAULT_POLICY_PRESET,
        choices=SearchPolicy.available_presets(),
        help="search scoring preset for SearchEngine",
    )
    parser.add_argument(
        "--target-profile",
        default=DEFAULT_TARGET_PROFILE,
        choices=SearchPolicy.available_target_profiles(),
        help="target-conditioned search profile",
    )
    parser.add_argument(
        "--disable-mmr-rerank",
        action="store_true",
        help="disable MMR rerank and use raw frontier score ordering only",
    )
    parser.add_argument(
        "--auto-apply-promotion",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUTO_APPLY_PROMOTION,
        help="automatically apply pending curated promotion patches into formal llm_refined family modules",
    )
    parser.add_argument("--panel-path", default="", help="optional override for evaluation panel path")
    parser.add_argument("--benchmark-path", default="", help="optional override for evaluation benchmark path")
    parser.add_argument("--start", default="", help="optional override for evaluation start date")
    parser.add_argument("--end", default="", help="optional override for evaluation end date")
    parser.add_argument(
        "--round1-seed-stage",
        action="store_true",
        help="force round1 seed-stage prompt augmentation even when current parent is passed explicitly",
    )
    parser.add_argument(
        "--stage-mode",
        default="auto",
        choices=_STAGE_MODE_CHOICES,
        help="explicit orchestration stage label used for prompt routing and trace auditing",
    )
    parser.add_argument(
        "--prompt-template-version",
        default="current_compact",
        choices=PROMPT_TEMPLATE_VERSIONS,
        help="prompt template variant used for this run",
    )
    return parser


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass

    args = build_arg_parser().parse_args()
    _warn_if_provider_env_missing(args)

    seed_pool = load_seed_pool(args.seed_pool)
    family = seed_pool.get_family(args.family)
    archive_db = init_archive_db(args.archive_db)
    latest_winner = get_latest_family_winner(db_path=archive_db, family=family.family) if args.auto_parent else None
    effective_round_id = (
        int(args.round_id)
        if args.round_id is not None
        else (get_latest_family_round(db_path=archive_db, family=family.family) + 1 if args.auto_parent else 1)
    )
    stage_mode = str(args.stage_mode or "auto").strip() or "auto"
    explicit_parent_name = args.current_parent_name.strip()
    explicit_parent_expression = args.current_parent_expression.strip()
    explicit_parent_candidate_id = args.parent_candidate_id.strip()
    explicit_parent_record = (
        get_candidate_record(db_path=archive_db, candidate_id=explicit_parent_candidate_id)
        if explicit_parent_name and explicit_parent_candidate_id
        else None
    )
    bootstrap_frontier = build_bootstrap_frontier(seed_pool=seed_pool, family=family)
    bootstrap_parent = select_bootstrap_parent(bootstrap_frontier)
    forced_seed_stage = stage_mode == "new_family_broad"
    auto_seed_stage = stage_mode == "auto" and not explicit_parent_name and latest_winner is None
    seed_stage_active = bool(args.round1_seed_stage or forced_seed_stage or auto_seed_stage)
    effective_parent_name = (
        explicit_parent_name
        or (str(latest_winner.get("factor_name", "")) if latest_winner else str(bootstrap_parent.get("factor_name", "")))
        or resolve_preferred_refine_seed(family)
    )
    effective_parent_expression = (
        explicit_parent_expression
        or (str((explicit_parent_record or {}).get("expression", "")) if explicit_parent_name else "")
        or (str(latest_winner.get("expression", "")) if latest_winner else str(bootstrap_parent.get("expression", "")))
        or resolve_family_formula(family, effective_parent_name)
    )
    if explicit_parent_expression:
        effective_parent_expression = apply_direction_rule(
            effective_parent_expression,
            resolve_factor_direction(family, effective_parent_name),
        )
    effective_parent_candidate_id = (
        explicit_parent_candidate_id
        or (str((explicit_parent_record or {}).get("candidate_id", "")) if explicit_parent_name else "")
        or (str(latest_winner.get("candidate_id", "")) if latest_winner and not explicit_parent_name else "")
        or make_seed_candidate_id(effective_parent_name)
    )
    search_policy = (
        SearchPolicy.local_best_first(preset=args.policy_preset)
        .with_target_profile(args.target_profile)
        .with_mmr_rerank(not bool(args.disable_mmr_rerank))
    )
    role_slots = resolve_round1_role_slots(
        family=family,
        final_candidate_target=int(args.n_candidates),
        seed_stage_active=seed_stage_active,
    )
    requested_candidate_count = resolve_requested_candidate_count(
        final_candidate_target=int(args.n_candidates),
        role_slots=role_slots,
        seed_stage_active=seed_stage_active,
    )
    decorrelation_targets = _normalize_multi_text(args.decorrelation_target)
    search_budget = SearchBudget(
        max_rounds=1,
        family_budget=1,
        branch_budget=max(int(requested_candidate_count), 1),
        max_frontier_size=max(int(requested_candidate_count), 4),
        max_depth=2,
        stop_if_no_improve=0,
    )
    engine = SearchEngine(
        family=family.family,
        budget=search_budget,
        policy=search_policy,
        normalizer=build_search_normalizer(db_path=args.archive_db, family=family.family),
    )
    seed_node = engine.register_seed(
        factor_name=effective_parent_name,
        expression=effective_parent_expression,
        candidate_id=effective_parent_candidate_id,
        node_kind=(
            "archive_winner"
            if latest_winner and not explicit_parent_name
            else (
                "explicit_parent"
                if explicit_parent_name
                else str(
                    bootstrap_parent.get(
                        "node_kind",
                        "preferred_seed" if effective_parent_name != family.canonical_seed else "canonical_seed",
                    )
                )
            )
        ),
        status=str((latest_winner or {}).get("status", "") or ("explicit_seed" if explicit_parent_name else "seed")),
        source_run_id=str((explicit_parent_record or latest_winner or {}).get("run_id", "")),
        source_model=str((explicit_parent_record or latest_winner or {}).get("source_model", "")),
        source_provider=str((explicit_parent_record or latest_winner or {}).get("source_provider", "")),
        round_id=int((explicit_parent_record or latest_winner or {}).get("round_id") or 0),
        metrics=explicit_parent_record or latest_winner or bootstrap_parent or {},
    )
    selected_parent = engine.select_next() or seed_node
    donor_motifs = (
        retrieve_runtime_donor_motifs(
            seed_pool=seed_pool,
            target_family=family.family,
            db_path=archive_db,
            max_donor_families=DEFAULT_ROUND1_DONOR_FAMILY_LIMIT,
            max_donor_factors=DEFAULT_ROUND1_DONOR_FACTOR_LIMIT,
        )
        if seed_stage_active
        else []
    )
    prompt_trace = _build_prompt_trace(
        family=family.family,
        stage_mode=stage_mode,
        target_profile=str(args.target_profile),
        policy_preset=str(args.policy_preset),
        seed_stage_active=seed_stage_active,
        selected_parent=selected_parent,
        requested_candidate_count=int(requested_candidate_count),
        final_candidate_target=int(args.n_candidates),
        role_slots=list(role_slots),
        bootstrap_frontier=bootstrap_frontier,
        donor_motifs=donor_motifs,
        decorrelation_targets=list(decorrelation_targets),
        prompt_template_version=str(args.prompt_template_version),
    )
    prompt_parent_row = load_prompt_history_row(seed_pool, selected_parent.factor_name, family=family)
    prompt = build_refinement_prompt(
        seed_pool=seed_pool,
        family=family,
        n_candidates=args.n_candidates,
        additional_notes=args.additional_notes,
        current_parent_name=selected_parent.factor_name,
        current_parent_expression=selected_parent.expression,
        current_parent_row=prompt_parent_row,
        archive_db=archive_db,
        current_model_name=args.model,
        current_parent_candidate_id=selected_parent.candidate_id or effective_parent_candidate_id,
        requested_candidate_count=requested_candidate_count,
        final_candidate_target=int(args.n_candidates),
        role_slots=role_slots,
        donor_motifs=donor_motifs,
        bootstrap_frontier=bootstrap_frontier,
        is_seed_stage=seed_stage_active,
        decorrelation_targets=decorrelation_targets,
        stage_mode=stage_mode,
        target_profile=str(args.target_profile),
        policy_preset=str(args.policy_preset),
        prompt_template_version=str(args.prompt_template_version),
        context_profile=resolve_context_profile(
            ContextEvidence.from_runtime(
                family=family.family,
                stage_mode=stage_mode,
                target_profile=str(args.target_profile),
                policy_preset=str(args.policy_preset),
                is_seed_stage=seed_stage_active,
                has_bootstrap_frontier=bool(bootstrap_frontier),
                has_donor_motifs=bool(donor_motifs),
                has_decorrelation_targets=bool(decorrelation_targets),
                selected_parent_kind=str(getattr(selected_parent, "node_kind", "") or ""),
                requested_candidate_count=int(requested_candidate_count),
                final_candidate_target=int(args.n_candidates),
            )
        ),
    )
    run_dir = create_run_dir(family=family.family, runs_dir=args.runs_dir)

    if args.print_only:
        print("=== SYSTEM PROMPT ===")
        print(prompt.system_prompt)
        print("\n=== USER PROMPT ===")
        print(prompt.user_prompt)
        print(f"\n[info] prompt preview only; run_dir={run_dir}")
        return 0

    provider_config = LLMProviderConfig(
        name=args.provider_name,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    started_at = utc_now_iso()
    prompt_hash = hashlib.sha1(f"{prompt.system_prompt}\n---\n{prompt.user_prompt}".encode("utf-8")).hexdigest()
    run_id = make_run_id(family=family.family, round_id=int(effective_round_id), started_at=started_at)
    insert_run(
        db_path=archive_db,
        run_id=run_id,
        family=family.family,
        canonical_seed=family.canonical_seed,
        round_id=int(effective_round_id),
        parent_candidate_id=effective_parent_candidate_id,
        provider=provider_config.name,
        model=provider_config.model,
        prompt_hash=prompt_hash,
        run_dir=str(run_dir),
        status="running",
        started_at=started_at,
    )
    metadata_dir = ensure_run_subdir(run_dir, "metadata")
    _write_json(
        metadata_dir / "search_plan.json",
        {
            "family": family.family,
            "stage_mode": stage_mode,
            "selection_mode": "single_round_local_best_first",
            "search_policy": search_policy.to_dict(),
            "search_budget": search_budget.to_dict(),
            "initial_parent": seed_node.to_dict(),
            "selected_parent": selected_parent.to_dict(),
            "prompt_trace": prompt_trace,
            "seed_stage_active": seed_stage_active,
            "requested_candidate_count": int(requested_candidate_count),
            "final_candidate_target": int(args.n_candidates),
            "role_slots": list(role_slots),
            "bootstrap_frontier": bootstrap_frontier,
            "donor_motifs": donor_motifs,
            "decorrelation_targets": list(decorrelation_targets),
        },
    )
    runtime_status = _make_runtime_status_writer(
        status_path=metadata_dir / "runtime_status.json",
        family=family.family,
        run_id=run_id,
        model=provider_config.model,
    )
    try:
        runtime_status(
            "preflight",
            "checking runtime environment",
            parent_factor=selected_parent.factor_name,
            stage_mode=stage_mode,
        )
        _write_json(metadata_dir / "preflight.json", _run_preflight_checks(args))
        runtime_status(
            "ready",
            "runtime checks passed",
            provider=provider_config.name,
            prompt_template_version=str(args.prompt_template_version),
        )
        engine.record_attempt(parent_node_id=selected_parent.node_id, note="run_refine_loop_single_attempt")
        if args.dry_run:
            runtime_status("dry_run", "building placeholder response")
            raw_response = json.dumps(
                {
                    "parent_factor": effective_parent_name,
                    "diagnosed_weaknesses": [],
                    "refinement_rationale": "dry-run placeholder",
                    "expected_behavior_change": "",
                    "risk_notes": "",
                    "candidate_formulas": [
                        {
                            "name": "placeholder_candidate",
                            "candidate_role": "conservative",
                            "expression": effective_parent_expression,
                            "explanation": "dry-run placeholder",
                            "rationale": "",
                            "expected_improvement": "",
                            "risk": "",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        else:
            provider = OpenAICompatProvider(provider_config)
            runtime_status("llm_request", f"requesting proposals from {provider_config.model}")
            raw_response = provider.generate(
                [
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt},
                ]
            )
            runtime_status("llm_response", "provider response received")

        parse_retry_records: list[dict[str, object]] = []
        last_parse_error: Exception | None = None
        proposal = None
        candidate_raw_response = raw_response
        retry_budget = max(int(args.retry_on_parse_fail), 0)
        while True:
            try:
                runtime_status("parse", "parsing provider response", retry_count=len(parse_retry_records))
                proposal = parse_refinement_response(
                    candidate_raw_response,
                    family=family,
                    provider_name=provider_config.name,
                    model_name=provider_config.model,
                    allowed_candidate_roles=role_slots,
                    default_candidate_roles=role_slots,
                )
                raw_response = candidate_raw_response
                break
            except Exception as exc:
                last_parse_error = exc
                retry_index = len(parse_retry_records) + 1
                if args.dry_run or retry_index > retry_budget:
                    break
                retry_prompt = _build_parse_retry_user_prompt(
                    original_user_prompt=prompt.user_prompt,
                    parse_error=str(exc),
                )
                runtime_status("parse_retry", f"retrying parse via provider call #{retry_index}")
                retry_raw_response = provider.generate(
                    [
                        {"role": "system", "content": prompt.system_prompt},
                        {"role": "user", "content": retry_prompt},
                    ]
                )
                parse_retry_records.append(
                    {
                        "retry_index": retry_index,
                        "error": str(exc),
                        "raw_response": retry_raw_response,
                    }
                )
                candidate_raw_response = retry_raw_response

        if proposal is None:
            exc = last_parse_error or ValueError("provider response JSON parse failed")
            parse_failure_dir = ensure_run_subdir(run_dir, "parse_failure")
            (parse_failure_dir / "raw_response.txt").write_text(candidate_raw_response, encoding="utf-8")
            (parse_failure_dir / "system_prompt.txt").write_text(prompt.system_prompt, encoding="utf-8")
            (parse_failure_dir / "user_prompt.txt").write_text(prompt.user_prompt, encoding="utf-8")
            if parse_retry_records:
                for item in parse_retry_records:
                    retry_index = int(item["retry_index"])
                    (parse_failure_dir / f"retry_{retry_index:02d}_raw_response.txt").write_text(
                        str(item["raw_response"]),
                        encoding="utf-8",
                    )
            _write_json(
                parse_failure_dir / "parse_error.json",
                {
                    "family": family.family,
                    "run_id": run_id,
                    "provider_name": provider_config.name,
                    "model_name": provider_config.model,
                    "parent_factor": effective_parent_name,
                    "parent_candidate_id": effective_parent_candidate_id,
                    "error": str(exc),
                    "retry_on_parse_fail": retry_budget,
                    "retry_count_used": len(parse_retry_records),
                },
            )
            raise exc
        runtime_status("candidate_processing", "running dedup and validation")
        unique_candidates, dropped_duplicates = deduplicate_candidates(proposal.candidates)
        if dropped_duplicates:
            proposal = _rebuild_proposal(proposal, unique_candidates)
            print(
                "[dedup] dropped "
                f"{len(dropped_duplicates)} duplicate candidate(s) before archive/write/backtest"
            )
            for item in dropped_duplicates:
                print(
                    "[dedup] "
                    f"{item['name']} -> {item['duplicate_of_name']}"
                )
        if not proposal.candidates:
            raise ValueError("all parsed candidates were removed by expression deduplication")
        rebuilt_candidates: list[RefinementCandidate] = []
        for candidate in proposal.candidates:
            candidate_id = make_candidate_id(
                family=family.family,
                round_id=int(effective_round_id),
                expression=candidate.expression,
                name=candidate.name,
            )
            rebuilt_candidates.append(
                _with_validation(
                    RefinementCandidate(
                        name=candidate.name,
                        expression=candidate.expression,
                        explanation=candidate.explanation,
                        candidate_role=candidate.candidate_role,
                        rationale=candidate.rationale,
                        expected_improvement=candidate.expected_improvement,
                        risk=candidate.risk,
                        source_model=candidate.source_model,
                        source_provider=candidate.source_provider,
                        parent_factor=candidate.parent_factor,
                        family=candidate.family,
                        candidate_id=candidate_id,
                        round_id=int(effective_round_id),
                        parent_candidate_id=effective_parent_candidate_id,
                        status="proposed",
                        validation_warnings=candidate.validation_warnings,
                    )
                )
            )
        valid_rebuilt_candidates: list[RefinementCandidate] = []
        dropped_validation: list[dict[str, object]] = []
        for candidate in rebuilt_candidates:
            filter_reason = _hard_validation_filter_reason(candidate)
            if not filter_reason:
                valid_rebuilt_candidates.append(candidate)
                continue
            dropped_validation.append(
                {
                    "candidate": candidate,
                    "filter_stage": "validation",
                    "filter_reason": filter_reason,
                }
            )
        if dropped_validation:
            print(
                "[validation-filter] dropped "
                f"{len(dropped_validation)} candidate(s) before structure filter / backtest"
            )
            for item in dropped_validation:
                candidate = item["candidate"]
                print(
                    "[validation-filter] "
                    f"{candidate.name}: {item['filter_reason']}"
                )
        if not valid_rebuilt_candidates:
            raise ValueError("all rebuilt candidates were removed by hard validation filter")
        kept_candidates, dropped_structure = filter_structurally_redundant(tuple(valid_rebuilt_candidates))
        if dropped_structure:
            print(
                "[structure-filter] dropped "
                f"{len(dropped_structure)} candidate(s) before backtest"
            )
            for item in dropped_structure:
                candidate = item["candidate"]
                print(
                    "[structure-filter] "
                    f"{candidate.name} -> {item['matched_name']}: {item['filter_reason']}"
                )
        if not kept_candidates:
            raise ValueError("all rebuilt candidates were removed by structure redundancy filter")
        light_rerank_target = int(args.n_candidates) if seed_stage_active else len(kept_candidates)
        kept_candidates, light_rerank_dropped, light_rerank_report = light_rerank_candidates(
            candidates=tuple(kept_candidates),
            family=family,
            parent_factor_name=selected_parent.factor_name,
            parent_expression=selected_parent.expression,
            policy=search_policy,
            final_candidate_target=light_rerank_target,
            role_slots=role_slots,
        )
        if light_rerank_dropped:
            print(
                "[light-rerank] kept "
                f"{len(kept_candidates)}/{len(kept_candidates) + len(light_rerank_dropped)} "
                "candidate(s) after role-coverage selection"
            )
        proposal = _rebuild_proposal(proposal, kept_candidates)
        insert_candidates(
            db_path=archive_db,
            run_id=run_id,
            candidates=[
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "round_id": candidate.round_id,
                    "parent_candidate_id": candidate.parent_candidate_id,
                    "factor_name": f"{args.name_prefix}.{candidate.name}",
                    "expression": candidate.expression,
                    "expression_hash": expression_hash(candidate.expression),
                    "candidate_role": candidate.candidate_role,
                    "source_model": candidate.source_model,
                    "source_provider": candidate.source_provider,
                    "explanation": candidate.explanation,
                    "rationale": candidate.rationale,
                    "validation_warnings": list(candidate.validation_warnings),
                    "filter_stage": "",
                    "filter_reason": "",
                    "status": candidate.status,
                    "created_at": started_at,
                }
                for candidate in proposal.candidates
            ]
            + [
                {
                    "candidate_id": item["candidate"].candidate_id,
                    "family": item["candidate"].family,
                    "round_id": item["candidate"].round_id,
                    "parent_candidate_id": item["candidate"].parent_candidate_id,
                    "factor_name": f"{args.name_prefix}.{item['candidate'].name}",
                    "expression": item["candidate"].expression,
                    "expression_hash": expression_hash(item["candidate"].expression),
                    "candidate_role": item["candidate"].candidate_role,
                    "source_model": item["candidate"].source_model,
                    "source_provider": item["candidate"].source_provider,
                    "explanation": item["candidate"].explanation,
                    "rationale": item["candidate"].rationale,
                    "validation_warnings": list(item["candidate"].validation_warnings),
                    "filter_stage": str(item["filter_stage"]),
                    "filter_reason": str(item["filter_reason"]),
                    "status": "drop_invalid_expression",
                    "created_at": started_at,
                }
                for item in dropped_validation
            ]
            + [
                {
                    "candidate_id": item["candidate"].candidate_id,
                    "family": item["candidate"].family,
                    "round_id": item["candidate"].round_id,
                    "parent_candidate_id": item["candidate"].parent_candidate_id,
                    "factor_name": f"{args.name_prefix}.{item['candidate'].name}",
                    "expression": item["candidate"].expression,
                    "expression_hash": expression_hash(item["candidate"].expression),
                    "candidate_role": item["candidate"].candidate_role,
                    "source_model": item["candidate"].source_model,
                    "source_provider": item["candidate"].source_provider,
                    "explanation": item["candidate"].explanation,
                    "rationale": item["candidate"].rationale,
                    "validation_warnings": list(item["candidate"].validation_warnings),
                    "filter_stage": item["filter_stage"],
                    "filter_reason": item["filter_reason"],
                    "status": "drop_redundant_structure",
                    "created_at": started_at,
                }
                for item in dropped_structure
            ]
            + [
                {
                    "candidate_id": item["candidate"].candidate_id,
                    "family": item["candidate"].family,
                    "round_id": item["candidate"].round_id,
                    "parent_candidate_id": item["candidate"].parent_candidate_id,
                    "factor_name": f"{args.name_prefix}.{item['candidate'].name}",
                    "expression": item["candidate"].expression,
                    "expression_hash": expression_hash(item["candidate"].expression),
                    "candidate_role": item["candidate"].candidate_role,
                    "source_model": item["candidate"].source_model,
                    "source_provider": item["candidate"].source_provider,
                    "explanation": item["candidate"].explanation,
                    "rationale": item["candidate"].rationale,
                    "validation_warnings": list(item["candidate"].validation_warnings),
                    "filter_stage": str(item["filter_stage"]),
                    "filter_reason": str(item["filter_reason"]),
                    "status": "drop_light_rerank",
                    "created_at": started_at,
                }
                for item in light_rerank_dropped
            ],
        )
        insert_lineage(
            db_path=archive_db,
            family=family.family,
            round_id=int(effective_round_id),
            parent_child_pairs=(
                [(effective_parent_candidate_id, candidate.candidate_id) for candidate in proposal.candidates]
                + [(effective_parent_candidate_id, item["candidate"].candidate_id) for item in dropped_validation]
                + [(effective_parent_candidate_id, item["candidate"].candidate_id) for item in dropped_structure]
                + [(effective_parent_candidate_id, item["candidate"].candidate_id) for item in light_rerank_dropped]
            ),
        )

        runtime_status("artifact_write", "writing run artifacts", candidate_count=len(proposal.candidates))
        paths = write_run_artifacts(
            run_dir=run_dir,
            family=family,
            prompt=prompt,
            proposal=proposal,
            provider_payload={
                "name": provider_config.name,
                "base_url": provider_config.base_url,
                "model": provider_config.model,
                "temperature": provider_config.temperature,
                "max_tokens": provider_config.max_tokens,
                "timeout": provider_config.timeout,
            },
            name_prefix=args.name_prefix,
        )
        print(f"[saved] run_dir={paths['run_dir']}")
        print(f"[saved] candidate_library={paths['candidate_library']}")
        print(f"[saved] proposal_report={paths['proposal_report']}")
        filters_dir = ensure_run_subdir(run_dir, "filters")
        structure_report_path = filters_dir / "structure_filter_report.md"
        structure_report_path.write_text(
            structure_filter_markdown(
                family=family,
                kept_candidates=proposal.candidates,
                dropped_candidates=dropped_structure,
            ),
            encoding="utf-8",
        )
        print(f"[saved] structure_filter_report={structure_report_path}")
        light_rerank_report_path = filters_dir / "light_rerank_report.md"
        light_rerank_report_path.write_text(
            light_rerank_markdown(
                family=family,
                parent_factor_name=selected_parent.factor_name,
                parent_expression=selected_parent.expression,
                report=light_rerank_report,
            ),
            encoding="utf-8",
        )
        _write_json(filters_dir / "light_rerank_report.json", light_rerank_report)
        print(f"[saved] light_rerank_report={light_rerank_report_path}")
        if any(candidate.validation_warnings for candidate in proposal.candidates):
            print("[warn] some candidates contain validation warnings; inspect proposal_report.md")
        if not args.skip_eval:
            runtime_status("evaluation", "running family evaluation")
            eval_paths = evaluate_refinement_run(
                run_dir=run_dir,
                seed_pool=seed_pool,
                family=family,
                proposal=proposal,
                name_prefix=args.name_prefix,
                panel_path=args.panel_path or None,
                benchmark_path=args.benchmark_path or None,
                start=args.start or None,
                end=args.end or None,
                run_id=run_id,
                round_id=int(effective_round_id),
                parent_candidate_id=effective_parent_candidate_id,
                archive_db=archive_db,
                auto_apply_promotion=args.auto_apply_promotion,
                stage_mode=stage_mode,
                decorrelation_targets=decorrelation_targets,
            )
            print(f"[saved] family_backtest_summary={eval_paths['family_backtest_summary']}")
            print(f"[saved] family_backtest_ranked={eval_paths['family_backtest_ranked']}")
            print(f"[saved] keep_drop_report={eval_paths['keep_drop_report']}")
            if "research_gate_report" in eval_paths:
                print(f"[saved] research_gate_report={eval_paths['research_gate_report']}")
            child_records = load_run_candidate_records(
                db_path=archive_db,
                run_id=run_id,
                statuses=("research_winner", "winner", "research_keep", "keep"),
            )
        else:
            runtime_status("evaluation_skipped", "skip_eval enabled")
            child_records = [
                {
                    "candidate_id": candidate.candidate_id,
                    "run_id": run_id,
                    "family": candidate.family,
                    "round_id": candidate.round_id,
                    "parent_candidate_id": candidate.parent_candidate_id,
                    "factor_name": f"{args.name_prefix}.{candidate.name}",
                    "expression": candidate.expression,
                    "candidate_role": candidate.candidate_role,
                    "source_model": candidate.source_model,
                    "source_provider": candidate.source_provider,
                    "status": candidate.status,
                    "quick_rank_ic_mean": None,
                    "quick_rank_icir": None,
                    "net_ann_return": None,
                    "net_excess_ann_return": None,
                    "net_sharpe": None,
                    "mean_turnover": None,
                    "evaluated_at": "",
                }
                for candidate in proposal.candidates
            ]
        engine.register_expansion(
            parent_node_id=selected_parent.node_id,
            child_records=child_records,
            success=True,
            source_run_id=run_id,
            note=f"run_dir={run_dir}",
            count_attempt=False,
        )
        reflection_card = build_reflection_card(
            db_path=archive_db,
            family=family,
            run_id=run_id,
            selected_parent=selected_parent.to_dict(),
        )
        _write_json(metadata_dir / "reflection_card.json", reflection_card)
        (metadata_dir / "reflection_card.md").write_text(
            reflection_markdown(reflection_card),
            encoding="utf-8",
        )
        top_level_summary = {
            "family": family.family,
            "target_profile": str(args.target_profile),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "round_id": int(effective_round_id),
            "stage_mode": stage_mode,
            "prompt_trace": prompt_trace,
            "seed_stage_active": seed_stage_active,
            "requested_candidate_count": int(requested_candidate_count),
            "final_candidate_target": int(args.n_candidates),
            "role_slots": list(role_slots),
            "selected_parent": selected_parent.to_dict(),
            "winner": _pick_best_child_record(child_records),
            "children": child_records,
            "search_policy": search_policy.to_dict(),
            "search_budget": search_budget.to_dict(),
            "search": engine.summary(),
            "bootstrap_frontier": bootstrap_frontier,
            "donor_motifs": donor_motifs,
            "decorrelation_targets": list(decorrelation_targets),
            "light_rerank": light_rerank_report,
            "reflection": reflection_card,
        }
        _write_json(run_dir / "summary.json", top_level_summary)
        _write_json(metadata_dir / "search_summary.json", engine.summary())
        runtime_status("completed", "run finished", child_record_count=len(child_records))
        mark_run_finished(db_path=archive_db, run_id=run_id, status="completed")
        return 0
    except Exception as exc:
        runtime_status("failed", str(exc))
        mark_run_finished(db_path=archive_db, run_id=run_id, status="failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
