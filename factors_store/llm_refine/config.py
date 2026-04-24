'''Configuration constants for LLM refinement workflows.

Defines artifact paths, default models, stage presets, gating thresholds, and promotion settings.
'''

from __future__ import annotations

from pathlib import Path


"""
llm_refine shared defaults.

This file is intentionally scoped to:
- path defaults
- provider/runtime defaults
- CLI/run budget defaults

This file intentionally does not contain the full SearchPolicy weight table.
Those weights live in `search/policy.py`, because they are search semantics
rather than generic runtime defaults.
"""


# ---------------------------------------------------------------------------
# Project paths
# These are the canonical filesystem roots used by llm_refine.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Repository root.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # Shared artifact root.
RUNS_DIR = ARTIFACTS_DIR / "runs"  # All run outputs live here.
CONFIG_DIR = PROJECT_ROOT / "config"  # Tracked seed pool and other yaml/json configs.
REPORTS_DIR = ARTIFACTS_DIR / "reports"  # Generated markdown/json/csv reports.
PROMOTIONS_DIR = ARTIFACTS_DIR / "llm_refine_promotions"  # Pending promotion payloads.
FACTORS_DIR = PROJECT_ROOT / "factors_store" / "factors"  # Registered factor implementations.
LLM_REFINED_DIR = FACTORS_DIR / "llm_refined"  # Formalized llm_refined family modules.


# ---------------------------------------------------------------------------
# Shared artifact/config files
# These are the canonical file defaults referenced by multiple modules.
# ---------------------------------------------------------------------------

DEFAULT_ARCHIVE_DB_PATH = ARTIFACTS_DIR / "llm_refine_archive.db"  # SQLite archive for runs/candidates/lineage.
DEFAULT_SEED_POOL_PATH = CONFIG_DIR / "refinement_seed_pool.yaml"  # Main seed family pool yaml.
DEFAULT_LLM_REFINED_INIT_PATH = LLM_REFINED_DIR / "__init__.py"  # llm_refined package init file.


# ---------------------------------------------------------------------------
# Run output directories
# One default root per orchestration layer.
# ---------------------------------------------------------------------------

DEFAULT_SINGLE_RUNS_DIR = RUNS_DIR / "llm_refine_single"  # Single-model, single-round runs.
DEFAULT_MULTI_RUNS_DIR = RUNS_DIR / "llm_refine_multi"  # Multi-model, single-round orchestrator runs.
DEFAULT_MULTI_SCHEDULER_RUNS_DIR = RUNS_DIR / "llm_refine_multi_scheduler"  # Multi-round scheduler runs.
DEFAULT_FAMILY_EXPLORE_RUNS_DIR = RUNS_DIR / "llm_refine_family_explore"  # Breadth-first family explore runs.
DEFAULT_FAMILY_LOOP_RUNS_DIR = RUNS_DIR / "llm_refine_family_loop"  # Broad -> anchor -> focused family-loop runs.
DEFAULT_AUTOFACTORSET_RUNS_DIR = RUNS_DIR / "autofactorset_ingest"  # Library admission / autofactorset runs.
DEFAULT_AUTOFACTORSET_MANIFESTS_DIR = ARTIFACTS_DIR / "autofactorset_ingest" / "manifests"  # Bridge manifest outputs.


# ---------------------------------------------------------------------------
# Report / promotion directories
# These are outputs produced by planning/evaluation/promotion utilities.
# ---------------------------------------------------------------------------

DEFAULT_EVALUATOR_OUTPUT_DIR = REPORTS_DIR / "evaluator"  # research_funnel outputs.
DEFAULT_NEXT_EXPERIMENTS_OUTPUT_DIR = REPORTS_DIR / "next_experiments"  # next_experiments outputs.
DEFAULT_PENDING_CURATED_DIR = PROMOTIONS_DIR / "pending"  # Pending curated promotion manifests/patches.


# ---------------------------------------------------------------------------
# Provider defaults
# These are only fallback defaults.
# Normal workflow should still source local `llm_refine_provider_env.sh` first
# after copying `llm_refine_provider_env.example.sh`.
# ---------------------------------------------------------------------------

DEFAULT_PROVIDER_NAME = "kuai"  # Fallback provider label stored in artifacts.
DEFAULT_BASE_URL = "https://api.kuai.host/v1"  # Fallback OpenAI-compatible endpoint.
DEFAULT_API_KEY = "EMPTY"  # Fallback API key placeholder.
DEFAULT_MODEL = "qwen3.5-plus"  # Fallback default model when env is absent.
DEFAULT_TEMPERATURE = 0.4  # Default generation temperature.
DEFAULT_MAX_TOKENS = 3000  # Default completion token budget.
DEFAULT_TIMEOUT = 600.0  # Default provider timeout in seconds.
DEFAULT_RETRY_ON_PARSE_FAIL = 1  # Retry count when raw LLM output cannot be parsed.


# ---------------------------------------------------------------------------
# Generic refine defaults
# These are the cross-entry defaults most frequently reused by CLI commands.
# ---------------------------------------------------------------------------

DEFAULT_NAME_PREFIX = "llmgen"  # Default prefix for generated factor names.
DEFAULT_N_CANDIDATES = 3  # Default number of candidates requested per model call.
DEFAULT_POLICY_PRESET = "balanced"  # Default high-level search preset.
DEFAULT_TARGET_PROFILE = "raw_alpha"  # Default target-conditioned profile.
DEFAULT_MAX_PARALLEL = 0  # 0 means "use all available child slots/models".
DEFAULT_AUTO_APPLY_PROMOTION = True  # Default to auto-applying pending curated promotions into llm_refined modules.


# ---------------------------------------------------------------------------
# Multi-model scheduler defaults
# These control the standard multi-round search loop.
# ---------------------------------------------------------------------------

DEFAULT_MULTI_SCHEDULER_MAX_ROUNDS = 2  # Default max scheduler rounds.
DEFAULT_MULTI_SCHEDULER_STOP_IF_NO_NEW_WINNER = 2  # Stop after this many no-improve rounds.
DEFAULT_MULTI_SCHEDULER_SLEEP_BETWEEN_ROUNDS = 0.0  # Optional pause between rounds.


# ---------------------------------------------------------------------------
# Family explore defaults
# These control breadth-first, multi-seed family probing.
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_EXPLORE_MAX_SEEDS = 4  # Max seeds/aliases/recent refs to dispatch.
DEFAULT_FAMILY_EXPLORE_ALIAS_LIMIT = 8  # Max family aliases considered before truncation.
DEFAULT_FAMILY_EXPLORE_RECENT_REFERENCE_LIMIT = 2  # Max recent archive refs to include.
DEFAULT_FAMILY_EXPLORE_SLEEP_BETWEEN_SEEDS = 0.0  # Optional pause between seed dispatches.


# ---------------------------------------------------------------------------
# Next-experiments defaults
# These control the lightweight planning report.
# ---------------------------------------------------------------------------

DEFAULT_NEXT_EXPERIMENTS_MAX_SUGGESTIONS_PER_FAMILY = 2  # Max motif-transfer suggestions kept per family.


# ---------------------------------------------------------------------------
# Round1 uplift defaults
# These control the seed-stage boost path: bootstrap frontier, donor hints,
# and generate-then-rank candidate expansion.
# ---------------------------------------------------------------------------

DEFAULT_ROUND1_BOOTSTRAP_ALIAS_LIMIT = 2  # Max aliases considered when choosing a better seed-stage parent.
DEFAULT_ROUND1_DONOR_FAMILY_LIMIT = 2  # Max donor families injected into round1 prompt.
DEFAULT_ROUND1_DONOR_FACTOR_LIMIT = 4  # Max donor factors injected into round1 prompt.
DEFAULT_ROUND1_EXTRA_CANDIDATES = 2  # Extra proposals requested in round1 before light rerank.
DEFAULT_ROUND1_LIGHT_RERANK_MAX_SELECTED_SIMILARITY = 0.9  # Max peer similarity allowed when keeping multiple round1 proposals.
DEFAULT_ROUND1_CANDIDATE_ROLES = (
    "conservative",
    "donor_transfer",
    "decorrelating",
    "confirmation",
    "simplifying",
    "stretch",
)  # Preferred role slots for seed-stage generation.


# ---------------------------------------------------------------------------
# Family-loop defaults
# These control the v1 broad -> anchor -> focused family controller.
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_LOOP_BROAD_STAGE_PRESET = "new_family_broad"  # Default stage preset for the broad stage.
DEFAULT_FAMILY_LOOP_FOCUSED_STAGE_PRESET = "focused_refine"  # Default stage preset for the focused stage.
DEFAULT_FAMILY_LOOP_BROAD_POLICY_PRESET = "exploratory"  # Broad stage policy preset.
DEFAULT_FAMILY_LOOP_FOCUSED_POLICY_PRESET = "balanced"  # Focused stage policy preset.
DEFAULT_FAMILY_LOOP_FOCUSED_N_CANDIDATES = 6  # Focused stage candidate budget per model.
DEFAULT_FAMILY_LOOP_BROAD_MAX_ROUNDS = 2  # Broad stage max scheduler rounds.
DEFAULT_FAMILY_LOOP_FOCUSED_MAX_ROUNDS = 2  # Focused stage max scheduler rounds.
DEFAULT_FAMILY_LOOP_BROAD_STOP_IF_NO_NEW_WINNER = 2  # Broad stage early-stop patience.
DEFAULT_FAMILY_LOOP_FOCUSED_STOP_IF_NO_NEW_WINNER = 2  # Focused stage early-stop patience.
FAMILY_LOOP_STAGE_PRESETS = {
    "new_family_broad": {
        "policy_preset": DEFAULT_FAMILY_LOOP_BROAD_POLICY_PRESET,
        "n_candidates": 8,
        "max_rounds": DEFAULT_FAMILY_LOOP_BROAD_MAX_ROUNDS,
        "stop_if_no_new_winner": DEFAULT_FAMILY_LOOP_BROAD_STOP_IF_NO_NEW_WINNER,
    },
    "focused_refine": {
        "policy_preset": DEFAULT_FAMILY_LOOP_FOCUSED_POLICY_PRESET,
        "n_candidates": DEFAULT_FAMILY_LOOP_FOCUSED_N_CANDIDATES,
        "max_rounds": DEFAULT_FAMILY_LOOP_FOCUSED_MAX_ROUNDS,
        "stop_if_no_new_winner": DEFAULT_FAMILY_LOOP_FOCUSED_STOP_IF_NO_NEW_WINNER,
    },
    "confirmation": {
        "policy_preset": "conservative",
        "n_candidates": 4,
        "max_rounds": 1,
        "stop_if_no_new_winner": 1,
    },
    "donor_validation": {
        "policy_preset": "balanced",
        "n_candidates": 3,
        "max_rounds": 1,
        "stop_if_no_new_winner": 1,
    },
}  # Stage-aware protocol presets used by family_loop v1.5.


# ---------------------------------------------------------------------------
# Family-loop anchor graduation gate
# These are deterministic thresholds used when selecting the best anchor.
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_LOOP_ANCHOR_MIN_ICIR = 0.45  # Minimum ICIR required for anchor graduation.
DEFAULT_FAMILY_LOOP_ANCHOR_MIN_SHARPE = 3.0  # Minimum Sharpe required for anchor graduation.
DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TURNOVER = 0.45  # Maximum turnover allowed for anchor graduation.
DEFAULT_FAMILY_LOOP_ANCHOR_MIN_METRICS_COMPLETENESS = 0.75  # Minimum metrics completeness ratio.
DEFAULT_FAMILY_LOOP_ANCHOR_MAX_PARENT_SIMILARITY = 0.80  # Corr-like similarity guard against near-parent clones.
DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_PARENT_CORR = 0.995  # True series corr guard against parent clones.
DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_SIBLING_CORR = 0.98  # True series corr guard against stronger sibling clones.
DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_EXCESS_GAIN = 0.02  # Minimum excess gain to justify a near-parent variant.
DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_ICIR_GAIN = 0.05  # Minimum ICIR gain to justify a near-parent variant.


__all__ = (
    "PROJECT_ROOT",
    "ARTIFACTS_DIR",
    "RUNS_DIR",
    "CONFIG_DIR",
    "REPORTS_DIR",
    "PROMOTIONS_DIR",
    "FACTORS_DIR",
    "LLM_REFINED_DIR",
    "DEFAULT_ARCHIVE_DB_PATH",
    "DEFAULT_SEED_POOL_PATH",
    "DEFAULT_LLM_REFINED_INIT_PATH",
    "DEFAULT_SINGLE_RUNS_DIR",
    "DEFAULT_MULTI_RUNS_DIR",
    "DEFAULT_MULTI_SCHEDULER_RUNS_DIR",
    "DEFAULT_FAMILY_EXPLORE_RUNS_DIR",
    "DEFAULT_FAMILY_LOOP_RUNS_DIR",
    "DEFAULT_AUTOFACTORSET_RUNS_DIR",
    "DEFAULT_AUTOFACTORSET_MANIFESTS_DIR",
    "DEFAULT_EVALUATOR_OUTPUT_DIR",
    "DEFAULT_NEXT_EXPERIMENTS_OUTPUT_DIR",
    "DEFAULT_PENDING_CURATED_DIR",
    "DEFAULT_PROVIDER_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_API_KEY",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_RETRY_ON_PARSE_FAIL",
    "DEFAULT_NAME_PREFIX",
    "DEFAULT_N_CANDIDATES",
    "DEFAULT_POLICY_PRESET",
    "DEFAULT_TARGET_PROFILE",
    "DEFAULT_MAX_PARALLEL",
    "DEFAULT_AUTO_APPLY_PROMOTION",
    "DEFAULT_MULTI_SCHEDULER_MAX_ROUNDS",
    "DEFAULT_MULTI_SCHEDULER_STOP_IF_NO_NEW_WINNER",
    "DEFAULT_MULTI_SCHEDULER_SLEEP_BETWEEN_ROUNDS",
    "DEFAULT_FAMILY_EXPLORE_MAX_SEEDS",
    "DEFAULT_FAMILY_EXPLORE_ALIAS_LIMIT",
    "DEFAULT_FAMILY_EXPLORE_RECENT_REFERENCE_LIMIT",
    "DEFAULT_FAMILY_EXPLORE_SLEEP_BETWEEN_SEEDS",
    "DEFAULT_NEXT_EXPERIMENTS_MAX_SUGGESTIONS_PER_FAMILY",
    "DEFAULT_ROUND1_BOOTSTRAP_ALIAS_LIMIT",
    "DEFAULT_ROUND1_DONOR_FAMILY_LIMIT",
    "DEFAULT_ROUND1_DONOR_FACTOR_LIMIT",
    "DEFAULT_ROUND1_EXTRA_CANDIDATES",
    "DEFAULT_ROUND1_LIGHT_RERANK_MAX_SELECTED_SIMILARITY",
    "DEFAULT_ROUND1_CANDIDATE_ROLES",
    "DEFAULT_FAMILY_LOOP_BROAD_STAGE_PRESET",
    "DEFAULT_FAMILY_LOOP_FOCUSED_STAGE_PRESET",
    "DEFAULT_FAMILY_LOOP_BROAD_POLICY_PRESET",
    "DEFAULT_FAMILY_LOOP_FOCUSED_POLICY_PRESET",
    "DEFAULT_FAMILY_LOOP_FOCUSED_N_CANDIDATES",
    "DEFAULT_FAMILY_LOOP_BROAD_MAX_ROUNDS",
    "DEFAULT_FAMILY_LOOP_FOCUSED_MAX_ROUNDS",
    "DEFAULT_FAMILY_LOOP_BROAD_STOP_IF_NO_NEW_WINNER",
    "DEFAULT_FAMILY_LOOP_FOCUSED_STOP_IF_NO_NEW_WINNER",
    "FAMILY_LOOP_STAGE_PRESETS",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MIN_ICIR",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MIN_SHARPE",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TURNOVER",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MIN_METRICS_COMPLETENESS",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MAX_PARENT_SIMILARITY",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_PARENT_CORR",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MAX_TRUE_SIBLING_CORR",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_EXCESS_GAIN",
    "DEFAULT_FAMILY_LOOP_ANCHOR_MIN_MATERIAL_ICIR_GAIN",
)
