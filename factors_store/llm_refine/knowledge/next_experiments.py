from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import DEFAULT_NEXT_EXPERIMENTS_OUTPUT_DIR, LLM_REFINED_DIR
from ..core.archive import DEFAULT_ARCHIVE_DB
from ..core.models import SeedFamily, SeedPool
from ..core.seed_loader import DEFAULT_SEED_POOL, load_seed_pool
from .archive_queries import extract_expression_tags, load_recent_keeps, load_recent_winners

_LLM_REFINED_DIR = LLM_REFINED_DIR
_DEFAULT_OUTPUT_DIR = DEFAULT_NEXT_EXPERIMENTS_OUTPUT_DIR
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "than",
    "more",
    "less",
    "keep",
    "main",
    "idea",
    "family",
    "factor",
    "factors",
    "logic",
    "structure",
    "structures",
    "version",
    "versions",
    "window",
    "windows",
    "mean",
    "sum",
    "use",
    "using",
    "current",
    "recent",
    "positive",
    "negative",
    "only",
    "rather",
    "explicit",
    "closely",
    "related",
    "still",
    "holds",
    "preserve",
    "retain",
    "replace",
    "compare",
    "test",
    "blend",
    "avoid",
    "simple",
    "plain",
    "generic",
    "core",
    "theme",
    "themes",
    "signal",
    "signals",
    "lookback",
    "bucket",
    "buckets",
    "can",
    "these",
    "their",
    "through",
    "around",
    "while",
    "where",
    "should",
    "under",
    "like",
    "family",
}
_THEME_KEYWORDS = {
    "pressure_distribution": {
        "pressure",
        "distribution",
        "accumulation",
        "buy_pressure",
        "sell_pressure",
        "net_pressure",
        "bias",
        "share",
        "high_price",
        "low_price",
        "crowding",
    },
    "volume_liquidity": {
        "volume",
        "amount",
        "turnover",
        "liquidity",
        "participation",
        "rel_volume",
        "rel_amount",
        "dry",
        "flow",
    },
    "amplitude_volatility": {
        "amplitude",
        "volatility",
        "range",
        "shadow",
        "panic",
        "salience",
        "std",
        "heated",
        "spike",
        "unstable",
    },
    "downside_position": {
        "downside",
        "low",
        "high",
        "lower",
        "tail",
        "position",
        "exhaustion",
        "prev_close",
    },
    "anchor_centroid": {
        "anchor",
        "centroid",
        "vwap",
        "quantile",
        "median",
        "historical",
        "distance",
        "lower_tail",
    },
    "open_marketfit": {
        "open",
        "marketfit",
        "market_return",
        "returns",
        "beta",
        "covariance",
        "cov",
        "regression",
        "fit",
        "rsq",
    },
    "quality_momentum": {
        "momentum",
        "trend",
        "reversal",
        "stable",
        "quality",
        "follow",
    },
    "neutralization": {
        "neutral",
        "neutralization",
        "industry",
        "residual",
        "size",
        "cap",
    },
}
_THEME_COMPATIBILITY = {
    "pressure_distribution": {
        "pressure_distribution": 1.0,
        "volume_liquidity": 0.8,
        "anchor_centroid": 0.5,
        "downside_position": 0.35,
    },
    "volume_liquidity": {
        "volume_liquidity": 1.0,
        "pressure_distribution": 0.8,
        "open_marketfit": 0.45,
        "neutralization": 0.35,
    },
    "amplitude_volatility": {
        "amplitude_volatility": 1.0,
        "downside_position": 0.55,
        "quality_momentum": 0.45,
        "volume_liquidity": 0.35,
    },
    "downside_position": {
        "downside_position": 1.0,
        "anchor_centroid": 0.7,
        "amplitude_volatility": 0.55,
        "pressure_distribution": 0.35,
    },
    "anchor_centroid": {
        "anchor_centroid": 1.0,
        "downside_position": 0.7,
        "pressure_distribution": 0.5,
        "volume_liquidity": 0.35,
    },
    "open_marketfit": {
        "open_marketfit": 1.0,
        "volume_liquidity": 0.45,
        "neutralization": 0.45,
        "quality_momentum": 0.25,
    },
    "quality_momentum": {
        "quality_momentum": 1.0,
        "amplitude_volatility": 0.45,
        "pressure_distribution": 0.25,
        "volume_liquidity": 0.2,
    },
    "neutralization": {
        "neutralization": 1.0,
        "open_marketfit": 0.45,
        "volume_liquidity": 0.35,
        "pressure_distribution": 0.2,
    },
}
_FAMILY_STATUS_ORDER = {
    "seed_only": 0,
    "legacy_refresh": 1,
    "new_framework_started": 2,
    "new_framework_done": 3,
}


@dataclass(frozen=True)
class FamilyState:
    family: str
    priority: str
    canonical_seed: str
    canonical_formula: str
    status: str
    archive_run_count: int
    has_llm_refined_module: bool
    module_paths: tuple[str, ...]
    recommended_action: str
    recommended_policy: str
    keywords: tuple[str, ...]
    themes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NextExperiment:
    target_family: str
    target_status: str
    target_priority: str
    experiment_type: str
    recommended_policy: str
    canonical_seed: str
    canonical_formula: str
    source_family: str
    source_factor_name: str
    source_model: str
    source_status: str
    source_tags: tuple[str, ...]
    source_themes: tuple[str, ...]
    theme_overlap: tuple[str, ...]
    overlap_keywords: tuple[str, ...]
    motif_score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokenize_text(text: str) -> set[str]:
    tokens = {token.lower() for token in _WORD_RE.findall(str(text or ""))}
    return {token for token in tokens if len(token) >= 3 and token not in _STOPWORDS}


def _family_keywords(family: SeedFamily) -> tuple[str, ...]:
    tokens: set[str] = set()
    tokens.update(_tokenize_text(family.family))
    tokens.update(_tokenize_text(family.canonical_seed))
    tokens.update(_tokenize_text(family.interpretation))
    tokens.update(_tokenize_text(family.primary_objective))
    tokens.update(_tokenize_text(family.secondary_objective))
    for value in family.formulas.values():
        tokens.update(_tokenize_text(value))
        tokens.update(extract_expression_tags(value))
    for text in family.refinement_axes:
        tokens.update(_tokenize_text(text))
    for text in family.likely_weaknesses:
        tokens.update(_tokenize_text(text))
    tokens.discard("family")
    return tuple(sorted(tokens))


def _special_theme_tokens(family: SeedFamily) -> set[str]:
    text = " ".join(
        [
            family.family,
            family.canonical_seed,
            family.interpretation,
            family.primary_objective,
            family.secondary_objective,
            " ".join(family.refinement_axes),
        ]
    ).lower()
    extra: set[str] = set()
    if "qp_pressure" in text or "apb" in text or "centroid" in text:
        extra.update({"pressure", "vwap", "centroid"})
    if "amplitude" in text or "panic" in text or "shadow" in text:
        extra.update({"amplitude", "volatility", "shadow"})
    if "low_over_high" in text or "downside" in text or "lower tail" in text:
        extra.update({"downside", "low", "high"})
    if "open" in text or "marketfit" in text or "beta" in text or "rsq" in text:
        extra.update({"open", "marketfit", "beta", "regression"})
    if "anchor" in text or "quantile" in text or "historical-anchor" in text:
        extra.update({"anchor", "historical", "quantile"})
    if "lower-tail" in text or ("lower" in text and "tail" in text):
        extra.update({"downside", "lower", "tail"})
    if "amount" in text or "volume" in text or "turnover" in text:
        extra.update({"amount", "volume", "turnover"})
    return extra


def _infer_themes(*, family: SeedFamily, keywords: tuple[str, ...]) -> tuple[str, ...]:
    tokens = set(keywords) | _special_theme_tokens(family)
    scored: list[tuple[float, str]] = []
    for theme, theme_tokens in _THEME_KEYWORDS.items():
        overlap = tokens & theme_tokens
        score = float(len(overlap))
        if theme == "pressure_distribution" and family.family.startswith("qp_") and "pressure" in family.family:
            score += 1.5
        if theme == "amplitude_volatility" and "amplitude" in family.family:
            score += 1.5
        if theme == "anchor_centroid" and ("anchor" in family.family or "centroid" in family.family):
            score += 1.5
        if theme == "open_marketfit" and ("open" in family.family or "marketfit" in family.family):
            score += 1.5
        if theme == "downside_position" and ("downside" in family.family or "low" in family.family):
            score += 1.0
        if theme == "downside_position" and ("anchor" in family.family or "lower_tail" in family.interpretation.lower()):
            score += 1.0
        if score > 0:
            scored.append((score, theme))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [theme for score, theme in scored if score >= 2.0][:3]
    if not chosen and scored:
        chosen = [scored[0][1]]
    return tuple(chosen)


def _infer_module_paths(family: SeedFamily) -> tuple[str, ...]:
    candidates: list[Path] = []
    direct = _LLM_REFINED_DIR / f"{family.family}_family.py"
    candidates.append(direct)

    canonical_tail = str(family.canonical_seed).split(".")[-1].strip()
    if canonical_tail:
        candidates.append(_LLM_REFINED_DIR / f"{canonical_tail}_family.py")

    if family.family.endswith("_score"):
        candidates.append(_LLM_REFINED_DIR / f"{family.family[:-6]}_family.py")

    seen: set[str] = set()
    existing: list[str] = []
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            existing.append(resolved)
    return tuple(existing)


def _load_archive_run_counts(db_path: str | Path = DEFAULT_ARCHIVE_DB) -> dict[str, int]:
    sql = "SELECT family, COUNT(*) FROM runs GROUP BY family"
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(sql).fetchall()
    return {str(family): int(count) for family, count in rows}


def _recommended_action_and_policy(*, status: str, run_count: int) -> tuple[str, str]:
    if status == "seed_only":
        return "new_family_initial_run", "exploratory"
    if status == "legacy_refresh":
        return "legacy_family_refresh", "balanced"
    if status == "new_framework_started":
        return "continue_under_new_framework", "balanced"
    if run_count >= 8:
        return "hold_expand_adjacent_families", "balanced"
    return "policy_confirmation", "conservative"


def build_family_inventory(
    *,
    seed_pool: SeedPool,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
) -> list[FamilyState]:
    run_counts = _load_archive_run_counts(db_path)
    states: list[FamilyState] = []
    for family in seed_pool.families:
        module_paths = _infer_module_paths(family)
        run_count = int(run_counts.get(family.family, 0))
        if run_count > 0:
            status = "new_framework_done" if run_count >= 3 else "new_framework_started"
        elif module_paths:
            status = "legacy_refresh"
        else:
            status = "seed_only"
        action, policy = _recommended_action_and_policy(status=status, run_count=run_count)
        canonical_formula = next(iter(family.formulas.values()), "")
        keywords = _family_keywords(family)
        states.append(
            FamilyState(
                family=family.family,
                priority=family.priority,
                canonical_seed=family.canonical_seed,
                canonical_formula=canonical_formula,
                status=status,
                archive_run_count=run_count,
                has_llm_refined_module=bool(module_paths),
                module_paths=module_paths,
                recommended_action=action,
                recommended_policy=policy,
                keywords=keywords,
                themes=_infer_themes(family=family, keywords=keywords),
            )
        )
    states.sort(
        key=lambda item: (
            _FAMILY_STATUS_ORDER.get(item.status, 99),
            {"high": 0, "medium": 1, "low": 2}.get(item.priority, 9),
            item.family,
        )
    )
    return states


def _source_records_for_family(
    *,
    db_path: str | Path,
    family: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    records = load_recent_winners(db_path=db_path, family=family, limit=limit)
    if len(records) < int(limit):
        records.extend(
            load_recent_keeps(
                db_path=db_path,
                family=family,
                limit=max(0, int(limit) - len(records)),
            )
        )
    return records


def _source_keywords(record: dict[str, Any], family_keywords: tuple[str, ...]) -> tuple[str, ...]:
    tokens: set[str] = set(family_keywords)
    tokens.update(record.get("expression_tags") or ())
    tokens.update(_tokenize_text(record.get("factor_name", "")))
    tokens.update(_tokenize_text(record.get("expression", "")))
    return tuple(sorted(tokens))


def _infer_source_themes(*, source_family_themes: tuple[str, ...], record: dict[str, Any]) -> tuple[str, ...]:
    tokens = set(source_family_themes)
    tags = set(record.get("expression_tags") or ())
    if {"amount", "volume", "turnover", "rel_amount", "rel_volume"} & tags:
        tokens.add("volume_liquidity")
    if {"amount_std_normalization", "turnover_std_normalization"} & tags:
        tokens.add("volume_liquidity")
    if {"ema", "smoothing", "decay_linear"} & tags:
        tokens.add("quality_momentum")
    if {"where", "bucket_sum"} & tags:
        tokens.add("pressure_distribution")
    if {"corr", "ts_corr", "cov", "ts_cov"} & tags:
        tokens.add("open_marketfit")
    if {"ts_rank", "cs_rank"} & tags:
        tokens.add("quality_momentum")
    return tuple(sorted(tokens))


def _theme_pair_score(target_themes: tuple[str, ...], source_themes: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    overlap = tuple(sorted(set(target_themes) & set(source_themes)))
    score = float(len(overlap)) * 10.0
    for target_theme in target_themes:
        compat = _THEME_COMPATIBILITY.get(target_theme, {})
        for source_theme in source_themes:
            score += float(compat.get(source_theme, 0.0)) * 6.0
    return score, overlap


def _score_motif_overlap(
    *,
    target_themes: tuple[str, ...],
    target_keywords: tuple[str, ...],
    source_themes: tuple[str, ...],
    source_keywords: tuple[str, ...],
    record: dict[str, Any],
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    theme_score, theme_overlap = _theme_pair_score(target_themes, source_themes)
    if theme_score <= 0.0:
        return 0.0, (), ()

    overlap_keywords = tuple(sorted(set(target_keywords) & set(source_keywords)))
    keyword_score = float(len(overlap_keywords[:8])) * 1.5
    quality_score = max(float(record.get("net_sharpe") or 0.0), 0.0) * 0.18
    quality_score += max(float(record.get("net_excess_ann_return") or 0.0), 0.0) * 0.25
    status_bonus = 1.2 if str(record.get("status", "")) in {"research_winner", "winner"} else 0.5
    score = theme_score + keyword_score + quality_score + status_bonus
    return score, overlap_keywords, theme_overlap


def _build_rationale(
    *,
    family_state: FamilyState,
    source_record: dict[str, Any],
    source_themes: tuple[str, ...],
    theme_overlap: tuple[str, ...],
    overlap_keywords: tuple[str, ...],
) -> str:
    theme_text = ", ".join(theme_overlap[:3]) if theme_overlap else "adjacent structural themes"
    overlap_text = ", ".join(overlap_keywords[:5]) if overlap_keywords else theme_text
    source_theme_text = ", ".join(source_themes[:3]) if source_themes else "related structural motifs"
    return (
        f"Start from the target family's own canonical seed, but guide the prompt using the successful "
        f"`{source_record.get('factor_name', '')}` motif from `{source_record.get('family', '')}`. "
        f"The transfer is mainly theme-level ({theme_text}) and is further supported by keywords such as: {overlap_text}. "
        f"Treat the source side as a {source_theme_text} hint rather than a formula to copy."
    )


def _fallback_rationale(*, family_state: FamilyState) -> str:
    theme_text = ", ".join(family_state.themes[:3]) or "the family's own canonical structure"
    return (
        "This family has no strong external motif match yet; start from the canonical seed and use the first run "
        f"to establish a baseline branch around {theme_text}. Prefer the recommended policy and keep the transfer budget low."
    )


def build_next_experiments(
    *,
    seed_pool: SeedPool,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    max_suggestions_per_family: int = 2,
) -> tuple[list[FamilyState], list[NextExperiment]]:
    states = build_family_inventory(seed_pool=seed_pool, db_path=db_path)

    source_records: list[dict[str, Any]] = []
    for state in states:
        if state.archive_run_count <= 0:
            continue
        for record in _source_records_for_family(db_path=db_path, family=state.family, limit=5):
            merged = dict(record)
            merged["source_keywords"] = _source_keywords(record, state.keywords)
            merged["source_themes"] = _infer_source_themes(source_family_themes=state.themes, record=record)
            source_records.append(merged)

    experiments: list[NextExperiment] = []
    for state in states:
        if state.recommended_action == "hold_expand_adjacent_families":
            experiments.append(
                NextExperiment(
                    target_family=state.family,
                    target_status=state.status,
                    target_priority=state.priority,
                    experiment_type=state.recommended_action,
                    recommended_policy=state.recommended_policy,
                    canonical_seed=state.canonical_seed,
                    canonical_formula=state.canonical_formula,
                    source_family="",
                    source_factor_name="",
                    source_model="",
                    source_status="",
                    source_tags=(),
                    source_themes=(),
                    theme_overlap=(),
                    overlap_keywords=(),
                    motif_score=0.0,
                    rationale=(
                        "This family already has substantial new-framework coverage. "
                        "Do not prioritize another immediate rerun; focus on adjacent families or refresh under-covered legacy families."
                    ),
                )
            )
            continue

        scored: list[tuple[float, tuple[str, ...], tuple[str, ...], dict[str, Any]]] = []
        for record in source_records:
            if str(record.get("family", "")) == state.family:
                continue
            score, overlap_keywords, theme_overlap = _score_motif_overlap(
                target_themes=state.themes,
                target_keywords=state.keywords,
                source_themes=tuple(record.get("source_themes") or ()),
                source_keywords=tuple(record.get("source_keywords") or ()),
                record=record,
            )
            if score <= 0.0:
                continue
            scored.append((score, overlap_keywords, theme_overlap, record))

        scored.sort(
            key=lambda item: (
                item[0],
                float(item[3].get("net_sharpe") or 0.0),
                float(item[3].get("net_excess_ann_return") or 0.0),
            ),
            reverse=True,
        )

        if not scored:
            experiments.append(
                NextExperiment(
                    target_family=state.family,
                    target_status=state.status,
                    target_priority=state.priority,
                    experiment_type=state.recommended_action,
                    recommended_policy=state.recommended_policy,
                    canonical_seed=state.canonical_seed,
                    canonical_formula=state.canonical_formula,
                    source_family="",
                    source_factor_name="",
                    source_model="",
                    source_status="",
                    source_tags=(),
                    source_themes=(),
                    theme_overlap=(),
                    overlap_keywords=(),
                    motif_score=0.0,
                    rationale=_fallback_rationale(family_state=state),
                )
            )
            continue

        used_sources: set[tuple[str, str]] = set()
        target_count = 0
        for score, overlap_keywords, theme_overlap, record in scored:
            source_key = (str(record.get("family", "")), str(record.get("factor_name", "")))
            if source_key in used_sources:
                continue
            used_sources.add(source_key)
            experiments.append(
                NextExperiment(
                    target_family=state.family,
                    target_status=state.status,
                    target_priority=state.priority,
                    experiment_type=state.recommended_action,
                    recommended_policy=state.recommended_policy,
                    canonical_seed=state.canonical_seed,
                    canonical_formula=state.canonical_formula,
                    source_family=str(record.get("family", "")),
                    source_factor_name=str(record.get("factor_name", "")),
                    source_model=str(record.get("source_model", "")),
                    source_status=str(record.get("status", "")),
                    source_tags=tuple(record.get("expression_tags") or ()),
                    source_themes=tuple(record.get("source_themes") or ()),
                    theme_overlap=theme_overlap,
                    overlap_keywords=overlap_keywords,
                    motif_score=round(float(score), 4),
                    rationale=_build_rationale(
                        family_state=state,
                        source_record=record,
                        source_themes=tuple(record.get("source_themes") or ()),
                        theme_overlap=theme_overlap,
                        overlap_keywords=overlap_keywords,
                    ),
                )
            )
            target_count += 1
            if target_count >= int(max_suggestions_per_family):
                break

    experiments.sort(
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(item.target_priority, 9),
            _FAMILY_STATUS_ORDER.get(item.target_status, 99),
            item.target_family,
            -float(item.motif_score),
        )
    )
    return states, experiments


def retrieve_runtime_donor_motifs(
    *,
    seed_pool: SeedPool,
    target_family: str,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    max_donor_families: int = 2,
    max_donor_factors: int = 4,
) -> list[dict[str, Any]]:
    def _choose(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chosen_local: list[dict[str, Any]] = []
        chosen_families_local: list[str] = []
        per_family_counts_local: dict[str, int] = {}
        seen_pairs_local: set[tuple[str, str]] = set()
        for item_local in items:
            family_name = str(item_local.get("source_family", "")).strip()
            factor_name = str(item_local.get("source_factor_name", "")).strip()
            if not family_name or not factor_name:
                continue
            pair = (family_name, factor_name)
            if pair in seen_pairs_local:
                continue
            if family_name not in chosen_families_local and len(chosen_families_local) >= int(max_donor_families):
                continue
            if int(per_family_counts_local.get(family_name, 0)) >= 2:
                continue
            seen_pairs_local.add(pair)
            if family_name not in chosen_families_local:
                chosen_families_local.append(family_name)
            per_family_counts_local[family_name] = int(per_family_counts_local.get(family_name, 0)) + 1
            chosen_local.append(item_local)
            if len(chosen_local) >= int(max_donor_factors):
                break
        return chosen_local

    family_states = build_family_inventory(seed_pool=seed_pool, db_path=db_path)
    state_by_family = {item.family: item for item in family_states}
    target_state = state_by_family.get(str(target_family).strip())
    if target_state is None:
        return []

    scored: list[dict[str, Any]] = []
    for source_state in family_states:
        if source_state.family == target_state.family or source_state.archive_run_count <= 0:
            continue
        for record in _source_records_for_family(db_path=db_path, family=source_state.family, limit=5):
            source_keywords = _source_keywords(record, source_state.keywords)
            source_themes = _infer_source_themes(source_family_themes=source_state.themes, record=record)
            score, overlap_keywords, theme_overlap = _score_motif_overlap(
                target_themes=target_state.themes,
                target_keywords=target_state.keywords,
                source_themes=source_themes,
                source_keywords=source_keywords,
                record=record,
            )
            if score <= 0.0:
                continue
            scored.append(
                {
                    "source_family": source_state.family,
                    "source_factor_name": str(record.get("factor_name", "")),
                    "source_expression": str(record.get("expression", "")),
                    "source_model": str(record.get("source_model", "")),
                    "source_status": str(record.get("status", "")),
                    "source_tags": tuple(record.get("expression_tags") or ()),
                    "source_themes": tuple(source_themes),
                    "theme_overlap": tuple(theme_overlap),
                    "overlap_keywords": tuple(overlap_keywords),
                    "motif_score": round(float(score), 4),
                    "quick_rank_ic_mean": record.get("quick_rank_ic_mean"),
                    "quick_rank_icir": record.get("quick_rank_icir"),
                    "net_ann_return": record.get("net_ann_return"),
                    "net_excess_ann_return": record.get("net_excess_ann_return"),
                    "net_sharpe": record.get("net_sharpe"),
                    "mean_turnover": record.get("mean_turnover"),
                    "rationale": _build_rationale(
                        family_state=target_state,
                        source_record=record,
                        source_themes=tuple(source_themes),
                        theme_overlap=tuple(theme_overlap),
                        overlap_keywords=tuple(overlap_keywords),
                    ),
                }
            )

    scored.sort(
        key=lambda item: (
            float(item.get("motif_score") or 0.0),
            1.0 if str(item.get("source_status", "")) in {"research_winner", "winner"} else 0.0,
            float(item.get("net_excess_ann_return") or 0.0),
            float(item.get("quick_rank_icir") or 0.0),
            float(item.get("net_sharpe") or 0.0),
        ),
        reverse=True,
    )
    chosen = _choose(scored)
    if chosen:
        return chosen

    fallback_scored: list[dict[str, Any]] = []
    for source_state in family_states:
        if source_state.family == target_state.family or source_state.archive_run_count <= 0:
            continue
        for record in _source_records_for_family(db_path=db_path, family=source_state.family, limit=5):
            source_keywords = _source_keywords(record, source_state.keywords)
            source_themes = _infer_source_themes(source_family_themes=source_state.themes, record=record)
            theme_score, theme_overlap = _theme_pair_score(target_state.themes, source_themes)
            if theme_score <= 0.0:
                continue
            quality_score = max(float(record.get("net_sharpe") or 0.0), 0.0) * 0.18
            quality_score += max(float(record.get("net_excess_ann_return") or 0.0), 0.0) * 0.25
            fallback_score = theme_score + quality_score
            overlap_keywords = tuple(sorted(set(target_state.keywords) & set(source_keywords)))
            fallback_scored.append(
                {
                    "source_family": source_state.family,
                    "source_factor_name": str(record.get("factor_name", "")),
                    "source_expression": str(record.get("expression", "")),
                    "source_model": str(record.get("source_model", "")),
                    "source_status": str(record.get("status", "")),
                    "source_tags": tuple(record.get("expression_tags") or ()),
                    "source_themes": tuple(source_themes),
                    "theme_overlap": tuple(theme_overlap),
                    "overlap_keywords": tuple(overlap_keywords),
                    "motif_score": round(float(fallback_score), 4),
                    "quick_rank_ic_mean": record.get("quick_rank_ic_mean"),
                    "quick_rank_icir": record.get("quick_rank_icir"),
                    "net_ann_return": record.get("net_ann_return"),
                    "net_excess_ann_return": record.get("net_excess_ann_return"),
                    "net_sharpe": record.get("net_sharpe"),
                    "mean_turnover": record.get("mean_turnover"),
                    "retrieval_mode": "theme_fallback",
                    "rationale": (
                        _build_rationale(
                            family_state=target_state,
                            source_record={**record, "family": source_state.family},
                            source_themes=tuple(source_themes),
                            theme_overlap=tuple(theme_overlap),
                            overlap_keywords=tuple(overlap_keywords),
                        )
                        + " Fallback donor selection was used because strict motif-overlap retrieval returned no candidates."
                    ),
                }
            )

    fallback_scored.sort(
        key=lambda item: (
            float(item.get("motif_score") or 0.0),
            1.0 if str(item.get("source_status", "")) in {"research_winner", "winner"} else 0.0,
            float(item.get("net_excess_ann_return") or 0.0),
            float(item.get("quick_rank_icir") or 0.0),
            float(item.get("net_sharpe") or 0.0),
        ),
        reverse=True,
    )
    return _choose(fallback_scored)


def render_next_experiments_markdown(
    *,
    family_states: list[FamilyState],
    experiments: list[NextExperiment],
) -> str:
    lines = [
        "# Next Experiments",
        "",
        "## Family Inventory",
        "",
        "| Family | Status | Runs | Module | Action | Policy | Themes | Canonical Seed |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for state in family_states:
        themes = ", ".join(state.themes[:3]) or "-"
        lines.append(
            f"| `{state.family}` | `{state.status}` | {state.archive_run_count} | "
            f"`{'yes' if state.has_llm_refined_module else 'no'}` | "
            f"`{state.recommended_action}` | `{state.recommended_policy}` | {themes} | `{state.canonical_seed}` |"
        )

    lines.extend(
        [
            "",
            "## Recommended Next Experiments",
            "",
            "These suggestions keep the target family's own canonical seed as the starting point. "
            "The source family/factor only acts as a transferable motif hint.",
            "",
            "| Target Family | Type | Policy | Source Family | Source Factor | Motif Score | Theme Overlap | Overlap Keywords |",
            "|---|---|---|---|---|---:|---|---|",
        ]
    )
    for item in experiments:
        overlap = ", ".join(item.overlap_keywords[:5]) or "-"
        theme_overlap = ", ".join(item.theme_overlap[:3]) or "-"
        lines.append(
            f"| `{item.target_family}` | `{item.experiment_type}` | `{item.recommended_policy}` | "
            f"`{item.source_family or '-'}` | `{item.source_factor_name or '-'}` | {item.motif_score:.2f} | {theme_overlap} | {overlap} |"
        )

    lines.extend(["", "## Notes", ""])
    for item in experiments:
        lines.append(f"- `{item.target_family}`: {item.rationale}")
    return "\n".join(lines).strip() + "\n"


def write_next_experiments_report(
    *,
    seed_pool_path: str | Path = DEFAULT_SEED_POOL,
    db_path: str | Path = DEFAULT_ARCHIVE_DB,
    out_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    max_suggestions_per_family: int = 2,
) -> dict[str, str]:
    seed_pool = load_seed_pool(seed_pool_path)
    family_states, experiments = build_next_experiments(
        seed_pool=seed_pool,
        db_path=db_path,
        max_suggestions_per_family=max_suggestions_per_family,
    )

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{ts}_next_experiments"

    payload = {
        "generated_at": ts,
        "seed_pool_path": str(Path(seed_pool_path).expanduser().resolve()),
        "archive_db": str(Path(db_path).expanduser().resolve()),
        "family_states": [item.to_dict() for item in family_states],
        "experiments": [item.to_dict() for item in experiments],
    }

    json_path = root / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = root / f"{stem}.md"
    md_path.write_text(
        render_next_experiments_markdown(family_states=family_states, experiments=experiments),
        encoding="utf-8",
    )
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }
