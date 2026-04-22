from __future__ import annotations

import ast
import difflib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import DEFAULT_LLM_REFINED_INIT_PATH, DEFAULT_PENDING_CURATED_DIR, LLM_REFINED_DIR
from ..core.archive import utc_now_iso
from .redundancy import factor_series_correlations
from ..parsing.expression_engine import guess_required_fields
from ..parsing.parser import expression_dedup_key
from ..core.models import SeedFamily

DEFAULT_LLM_REFINED_INIT = DEFAULT_LLM_REFINED_INIT_PATH
MAX_PENDING_RESEARCH_KEEP = 2
RESEARCH_KEEP_WINNER_SCORE_THRESHOLD = 0.55
RESEARCH_KEEP_MAX_TURNOVER = 0.25
RESEARCH_WINNER_MAX_EXISTING_FAMILY_CORR = 0.90
RESEARCH_KEEP_MAX_EXISTING_FAMILY_CORR = 0.85


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "none", "nan"}:
        return False
    return True


def _suggest_registry_name(factor_name: str) -> str:
    if "." not in factor_name:
        return f"llm_refined.{factor_name}"
    _, suffix = factor_name.split(".", 1)
    return f"llm_refined.{suffix}"


def _corr_threshold_for_decision(decision: str) -> float:
    return (
        float(RESEARCH_WINNER_MAX_EXISTING_FAMILY_CORR)
        if str(decision or "").strip() == "research_winner"
        else float(RESEARCH_KEEP_MAX_EXISTING_FAMILY_CORR)
    )


def _promotion_gate(
    row: dict[str, Any],
    *,
    family_existing_refs: list[tuple[str, pd.Series]] | None = None,
    data: dict[str, pd.Series] | None = None,
) -> tuple[bool, str]:
    if str(row.get("role", "")) != "candidate":
        return False, "not candidate role"
    decision = str(row.get("decision", "")).strip()
    corr_threshold = _corr_threshold_for_decision(decision)
    if decision == "research_winner":
        expression = str(row.get("expression", "")).strip()
        if not expression:
            return False, "empty expression"
    elif decision != "research_keep":
        return False, "decision is not research_winner/research_keep"
    else:
        expression = str(row.get("expression", "")).strip()
        if not expression:
            return False, "empty expression"

        completeness = _safe_float(row.get("metrics_completeness"))
        missing_core = _safe_int(row.get("missing_core_metrics_count"), default=99)
        turnover = _safe_float(row.get("mean_turnover"))
        winner_score = _safe_float(row.get("winner_score"))
        net_ann = _safe_float(row.get("net_ann_return"))
        net_sharpe = _safe_float(row.get("net_sharpe"))
        rank_icir = _safe_float(row.get("quick_rank_icir"))

        if completeness is not None and completeness < 0.95:
            return False, "research_keep incomplete metrics"
        if missing_core > 0:
            return False, "research_keep missing core metrics"
        if turnover is not None and turnover > RESEARCH_KEEP_MAX_TURNOVER:
            return False, "research_keep turnover too high"

        quality_ok = False
        if winner_score is not None and winner_score >= RESEARCH_KEEP_WINNER_SCORE_THRESHOLD:
            quality_ok = True
        elif (
            (net_ann is not None and net_ann > 0.0)
            and (net_sharpe is not None and net_sharpe > 0.0)
            and (rank_icir is not None and rank_icir > 0.0)
        ):
            quality_ok = True
        if not quality_ok:
            return False, "research_keep quality bar not met"

    nearest_target_corr = _safe_float(row.get("corr_to_nearest_decorrelation_target"))
    nearest_target_name = str(row.get("nearest_decorrelation_target", "") or "").strip()
    if nearest_target_corr is not None and math.isfinite(nearest_target_corr):
        if nearest_target_corr >= corr_threshold:
            target_label = nearest_target_name or "decorrelation_target"
            return (
                False,
                f"decorrelation target corr with {target_label} = {nearest_target_corr:.4f} >= {corr_threshold:.2f}",
            )

    expression = str(row.get("expression", "")).strip()
    if family_existing_refs and data and expression:
        try:
            from ...factors.llm_refined.common import evaluate_expression_factor

            factor_name = str(row.get("factor_name", "")).strip() or "promotion_candidate"
            candidate_series = evaluate_expression_factor(
                data,
                expression=expression,
                factor_name=factor_name,
            )
            corr_map = factor_series_correlations(candidate_series, family_existing_refs)
            best_name = ""
            best_corr = float("nan")
            for ref_name, corr_value in corr_map.items():
                if corr_value is None or not math.isfinite(float(corr_value)):
                    continue
                abs_corr = abs(float(corr_value))
                if not math.isfinite(best_corr) or abs_corr > best_corr:
                    best_corr = abs_corr
                    best_name = str(ref_name or "")
            if math.isfinite(best_corr) and best_corr >= corr_threshold:
                matched = best_name or "existing_family_factor"
                return (
                    False,
                    f"existing family corr with {matched} = {best_corr:.4f} >= {corr_threshold:.2f}",
                )
        except Exception:
            pass

    if decision == "research_winner":
        return True, "research_winner passed pending-curation gate"
    return True, "high-quality research_keep passed pending-curation gate"


def _promotion_priority(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    decision = str(row.get("decision", "")).strip()
    if decision == "research_winner":
        return (1.0, 0.0, 0.0, 0.0, 0.0)

    winner_score = _safe_float(row.get("winner_score")) or 0.0
    net_excess = _safe_float(row.get("net_excess_ann_return")) or 0.0
    net_ann = _safe_float(row.get("net_ann_return")) or 0.0
    net_sharpe = _safe_float(row.get("net_sharpe")) or 0.0
    turnover = _safe_float(row.get("mean_turnover")) or 0.0
    return (0.0, winner_score, net_excess, net_ann + 0.25 * net_sharpe, -turnover)


def _entry_from_row(
    *,
    row: dict[str, Any],
    family: SeedFamily,
    run_id: str,
    round_id: int,
    run_dir: str | Path,
    decision_stage: str,
    name_prefix: str,
) -> dict[str, Any]:
    return {
        "candidate_id": str(row.get("candidate_id", "")),
        "factor_name": str(row.get("factor_name", "")),
        "suggested_registry_name": _suggest_registry_name(str(row.get("factor_name", ""))),
        "suggested_function_name": _function_name(_suggest_registry_name(str(row.get("factor_name", "")))),
        "expression": str(row.get("expression", "")),
        "candidate_role": str(row.get("candidate_role", "")),
        "source_model": str(row.get("model", "")),
        "source_provider": str(row.get("provider", "")),
        "decision": str(row.get("decision", "")),
        "decision_reason": str(row.get("decision_reason", "")),
        "selection_metrics": {
            "quick_rank_ic_mean": _safe_float(row.get("quick_rank_ic_mean"))
            if _safe_float(row.get("quick_rank_ic_mean")) is not None
            else _safe_float(row.get("quick_rank_ic")),
            "quick_rank_icir": _safe_float(row.get("quick_rank_icir")),
            "net_ann_return": _safe_float(row.get("net_ann_return")),
            "net_excess_ann_return": _safe_float(row.get("net_excess_ann_return")),
            "net_sharpe": _safe_float(row.get("net_sharpe")),
            "mean_turnover": _safe_float(row.get("mean_turnover")),
        },
        "promotion_context": {
            "family": family.family,
            "canonical_seed": family.canonical_seed,
            "run_id": run_id,
            "round_id": int(round_id),
            "run_dir": str(run_dir),
            "decision_stage": decision_stage,
            "name_prefix": name_prefix,
            "target_family_module": str(LLM_REFINED_DIR / f"{family.family}_family.py"),
        },
    }


def _function_name(registry_name: str) -> str:
    return registry_name.replace(".", "_").replace("-", "_").replace("/", "_")


def _string_literal(text: str) -> str:
    return json.dumps(str(text), ensure_ascii=False)


def _tuple_literal(items: tuple[str, ...]) -> str:
    if not items:
        return "()"
    if len(items) == 1:
        return f"({_string_literal(items[0])},)"
    return "(" + ", ".join(_string_literal(item) for item in items) + ")"


def _notes_literal(entry: dict[str, Any]) -> str:
    metric = entry.get("selection_metrics", {}) or {}
    parts = [
        f"Auto-promoted pending candidate for {entry['promotion_context']['family']}; ",
        f"run_id={entry['promotion_context']['run_id']}; ",
        f"round_id={entry['promotion_context']['round_id']}; ",
        f"source_model={entry.get('source_model', '') or 'LLM'}; ",
        f"decision={entry.get('decision', '')}. ",
        (
            "Selection summary: "
            f"RankIC={metric.get('quick_rank_ic_mean')}, "
            f"NetAnn={metric.get('net_ann_return')}, "
            f"Turnover={metric.get('mean_turnover')}."
        ),
    ]
    return _string_literal("".join(parts))


def _docstring_lines(entry: dict[str, Any]) -> str:
    context = entry["promotion_context"]
    return (
        f'    """parent factor: {context["canonical_seed"]}\n'
        f"    round: llm_refine_round_{context['round_id']}\n"
        f"    source model: {entry.get('source_model', '') or 'LLM'}\n"
        f"    keep-drop status: {entry.get('decision', '')}\n"
        '    """'
    )


def _render_function_block(entry: dict[str, Any]) -> str:
    function_name = entry["suggested_function_name"]
    registry_name = entry["suggested_registry_name"]
    expression = _string_literal(entry["expression"])
    return (
        f"def {function_name}(data: dict[str, pd.Series]) -> pd.Series:\n"
        f"{_docstring_lines(entry)}\n"
        "    return evaluate_expression_factor(\n"
        "        data,\n"
        f"        expression={expression},\n"
        f"        factor_name={_string_literal(registry_name)},\n"
        "    )\n"
    )


def _render_factor_spec_block(entry: dict[str, Any]) -> str:
    function_name = entry["suggested_function_name"]
    registry_name = entry["suggested_registry_name"]
    required_fields = guess_required_fields(entry["expression"])
    return (
        "    FactorSpec(\n"
        f"        name={_string_literal(registry_name)},\n"
        f"        func={function_name},\n"
        f"        required_fields={_tuple_literal(required_fields)},\n"
        f"        notes={_notes_literal(entry)},\n"
        "    ),\n"
    )


def _ensure_common_import(module_text: str) -> str:
    pattern = r"from \.common import ([^\n]+)"
    match = re.search(pattern, module_text)
    if not match:
        return module_text
    imports = [item.strip() for item in match.group(1).split(",")]
    if "evaluate_expression_factor" in imports:
        return module_text
    imports.append("evaluate_expression_factor")
    replacement = "from .common import " + ", ".join(imports)
    return re.sub(pattern, replacement, module_text, count=1)


def _extract_existing_module_index(module_text: str) -> dict[str, Any]:
    registry_names: set[str] = set()
    expression_index: dict[str, dict[str, str]] = {}
    expression_entries: list[dict[str, str]] = []
    text = str(module_text or "").strip()
    if not text:
        return {
            "registry_names": registry_names,
            "expression_index": expression_index,
            "expression_entries": expression_entries,
        }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "registry_names": registry_names,
            "expression_index": expression_index,
            "expression_entries": expression_entries,
        }

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        factor_name = ""
        expression = ""
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call) or getattr(sub.func, "id", None) != "evaluate_expression_factor":
                continue
            for kw in sub.keywords:
                if kw.arg == "factor_name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    factor_name = str(kw.value.value)
                if kw.arg == "expression" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    expression = str(kw.value.value)
            if factor_name or expression:
                break
        if factor_name:
            registry_names.add(factor_name)
        if expression:
            dedup_key = expression_dedup_key(expression) or expression.strip()
            if dedup_key and dedup_key not in expression_index:
                expression_index[dedup_key] = {
                    "factor_name": factor_name or node.name,
                    "expression": expression,
                }
            expression_entries.append(
                {
                    "factor_name": factor_name or node.name,
                    "expression": expression,
                }
            )
    return {
        "registry_names": registry_names,
        "expression_index": expression_index,
        "expression_entries": expression_entries,
    }


def _build_existing_family_refs(
    *,
    module_text: str,
    data: dict[str, pd.Series] | None,
) -> list[tuple[str, pd.Series]]:
    if not module_text or not data:
        return []
    try:
        from ...factors.llm_refined.common import evaluate_expression_factor
    except Exception:
        return []

    existing = _extract_existing_module_index(module_text)
    refs: list[tuple[str, pd.Series]] = []
    seen_names: set[str] = set()
    for item in list(existing.get("expression_entries") or []):
        factor_name = str(dict(item).get("factor_name", "") or "").strip()
        expression = str(dict(item).get("expression", "") or "").strip()
        if not factor_name or not expression or factor_name in seen_names:
            continue
        try:
            series = evaluate_expression_factor(
                data,
                expression=expression,
                factor_name=factor_name,
            )
        except Exception:
            continue
        if not isinstance(series, pd.Series):
            continue
        refs.append((factor_name, series))
        seen_names.add(factor_name)
    return refs


def _filter_entries_against_module(
    *,
    module_text: str,
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    existing = _extract_existing_module_index(module_text)
    seen_names = set(existing["registry_names"])
    seen_expr_keys = set(existing["expression_index"])
    batch_expr_sources: dict[str, str] = {}
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for entry in entries:
        registry_name = str(entry.get("suggested_registry_name", "")).strip()
        expression = str(entry.get("expression", "")).strip()
        dedup_key = expression_dedup_key(expression) or expression
        if registry_name and registry_name in seen_names:
            skipped.append(
                {
                    "suggested_registry_name": registry_name,
                    "reason": "duplicate_registry_name",
                    "matched_registry_name": registry_name,
                }
            )
            continue
        if dedup_key and dedup_key in seen_expr_keys:
            matched = existing["expression_index"].get(dedup_key, {})
            skipped.append(
                {
                    "suggested_registry_name": registry_name,
                    "reason": "duplicate_expression_existing_module",
                    "matched_registry_name": str(matched.get("factor_name", "")),
                }
            )
            continue
        if dedup_key and dedup_key in batch_expr_sources:
            skipped.append(
                {
                    "suggested_registry_name": registry_name,
                    "reason": "duplicate_expression_same_batch",
                    "matched_registry_name": batch_expr_sources[dedup_key],
                }
            )
            continue
        kept.append(entry)
        if registry_name:
            seen_names.add(registry_name)
        if dedup_key:
            seen_expr_keys.add(dedup_key)
            batch_expr_sources[dedup_key] = registry_name
    return kept, skipped


def _merge_existing_module(
    *,
    module_text: str,
    entries: list[dict[str, Any]],
) -> str:
    text = module_text
    fresh_entries, _ = _filter_entries_against_module(module_text=text, entries=entries)
    if not fresh_entries:
        return text
    text = _ensure_common_import(text)

    function_blocks = "\n\n".join(_render_function_block(entry) for entry in fresh_entries) + "\n\n"
    factor_match = re.search(
        r"^FACTOR_SPECS(?:\s*:\s*tuple\[FactorSpec,\s*\.\.\.\])?\s*=\s*\(\n",
        text,
        flags=re.MULTILINE,
    )
    if factor_match is None:
        return text
    factor_idx = factor_match.start()
    text = text[:factor_idx] + function_blocks + text[factor_idx:]

    all_idx = text.find("__all__ = [")
    if all_idx == -1:
        return text
    spec_end = text.rfind("\n)", factor_idx, all_idx)
    if spec_end != -1:
        spec_blocks = "".join(_render_factor_spec_block(entry) for entry in fresh_entries)
        text = text[: spec_end + 1] + spec_blocks + text[spec_end + 1 :]

    export_end = text.find("\n]", all_idx)
    if export_end != -1:
        export_block = "".join(
            f"    {_string_literal(entry['suggested_function_name'])},\n" for entry in fresh_entries
        )
        text = text[: export_end + 1] + export_block + text[export_end + 1 :]
    return text


def _ensure_line_token(text: str, *, prefix: str, token: str) -> str:
    pattern = rf"^{re.escape(prefix)}(.+)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return text
    items = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if token not in items:
        items.append(token)
    replacement = prefix + ", ".join(items)
    return re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)


def _ensure_star_import(text: str, *, module_stem: str) -> str:
    line = f"from .{module_stem} import *  # noqa: F401,F403"
    if line in text:
        return text
    marker = "from .common import LLM_REFINED_SOURCE"
    if marker in text:
        return text.replace(marker, f"{line}\n{marker}", 1)
    return text + ("\n" if not text.endswith("\n") else "") + line + "\n"


def _ensure_family_module_tuple(text: str, *, module_stem: str) -> str:
    pattern = r"FAMILY_MODULES = \(\n(?P<body>.*?)\n\)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return text
    body = match.group("body")
    if re.search(rf"^\s*{re.escape(module_stem)},\s*$", body, flags=re.MULTILINE):
        return text
    updated_body = body + f"\n    {module_stem},"
    return text[: match.start('body')] + updated_body + text[match.end('body') :]


def _ensure_llm_refined_init_registered(module_path: Path) -> dict[str, str]:
    init_path = DEFAULT_LLM_REFINED_INIT
    old_text = init_path.read_text(encoding="utf-8")
    module_stem = module_path.stem
    new_text = old_text
    new_text = _ensure_line_token(new_text, prefix="from . import ", token=module_stem)
    new_text = _ensure_star_import(new_text, module_stem=module_stem)
    new_text = _ensure_family_module_tuple(new_text, module_stem=module_stem)
    changed = new_text != old_text
    if changed:
        init_path.write_text(new_text, encoding="utf-8")
    return {
        "llm_refined_init": str(init_path),
        "llm_refined_init_changed": "true" if changed else "false",
    }


def _apply_payload_to_store(
    *,
    payload: dict[str, Any],
    metadata_dir: Path,
) -> dict[str, str]:
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in payload.get("entries", []):
        target = str(entry.get("promotion_context", {}).get("target_family_module", "")).strip()
        if target:
            by_module[target].append(entry)

    applied: dict[str, str] = {}
    summary: list[dict[str, Any]] = []
    for module_path_str, entries in by_module.items():
        module_path = Path(module_path_str)
        module_stem = module_path.stem
        family = str(entries[0]["promotion_context"]["family"])
        canonical_seed = str(entries[0]["promotion_context"]["canonical_seed"])

        if module_path.exists():
            old_text = module_path.read_text(encoding="utf-8")
            new_text = _merge_existing_module(module_text=old_text, entries=entries)
        else:
            old_text = ""
            new_text = _render_new_module(family=family, canonical_seed=canonical_seed, entries=entries)

        changed = old_text != new_text
        if changed:
            module_path.parent.mkdir(parents=True, exist_ok=True)
            module_path.write_text(new_text, encoding="utf-8")
        init_update = _ensure_llm_refined_init_registered(module_path)
        summary.append(
            {
                "module_path": str(module_path),
                "changed": changed,
                "entries": [entry["suggested_registry_name"] for entry in entries],
                **init_update,
            }
        )
        applied[f"{module_stem}_applied"] = str(module_path)

    applied_summary_path = metadata_dir / "auto_applied_promotion.json"
    applied_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    applied["auto_applied_promotion"] = str(applied_summary_path)
    return applied


def _render_new_module(
    *,
    family: str,
    canonical_seed: str,
    entries: list[dict[str, Any]],
) -> str:
    function_blocks = "\n\n".join(_render_function_block(entry) for entry in entries)
    spec_blocks = "".join(_render_factor_spec_block(entry) for entry in entries)
    export_block = "".join(
        f"    {_string_literal(entry['suggested_function_name'])},\n" for entry in entries
    )
    return (
        "from __future__ import annotations\n\n"
        f'"""Auto-promoted LLM refined candidates for the {canonical_seed} family."""\n\n'
        "import pandas as pd\n\n"
        "from .common import FactorSpec, evaluate_expression_factor\n\n"
        f'PARENT_FACTOR = {_string_literal(canonical_seed)}\n'
        f'FAMILY_KEY = {_string_literal(f"{family}_family")}\n'
        f'SEED_FAMILY = {_string_literal(family)}\n'
        f'SUMMARY_GLOB = {_string_literal(f"llm_refined_{family}_family_summary_*.csv")}\n\n\n'
        f"{function_blocks}\n\n\n"
        "FACTOR_SPECS: tuple[FactorSpec, ...] = (\n"
        f"{spec_blocks}"
        ")\n\n\n"
        "__all__ = [\n"
        '    "FACTOR_SPECS",\n'
        '    "FAMILY_KEY",\n'
        '    "PARENT_FACTOR",\n'
        '    "SEED_FAMILY",\n'
        '    "SUMMARY_GLOB",\n'
        f"{export_block}"
        "]\n"
    )


def _write_patch_bundle(
    *,
    payload: dict[str, Any],
    metadata_dir: Path,
    pending_dir: Path,
) -> dict[str, str]:
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in payload.get("entries", []):
        target = str(entry.get("promotion_context", {}).get("target_family_module", "")).strip()
        if target:
            by_module[target].append(entry)

    created: dict[str, str] = {}
    central_patch_dir = pending_dir / "patches"
    central_preview_dir = pending_dir / "preview_modules"
    central_patch_dir.mkdir(parents=True, exist_ok=True)
    central_preview_dir.mkdir(parents=True, exist_ok=True)
    run_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("run_id", "pending")))

    for module_path_str, entries in by_module.items():
        module_path = Path(module_path_str)
        module_stem = module_path.stem
        family = str(entries[0]["promotion_context"]["family"])
        canonical_seed = str(entries[0]["promotion_context"]["canonical_seed"])

        if module_path.exists():
            old_text = module_path.read_text(encoding="utf-8")
            new_text = _merge_existing_module(module_text=old_text, entries=entries)
        else:
            old_text = ""
            new_text = _render_new_module(family=family, canonical_seed=canonical_seed, entries=entries)

        preview_path = metadata_dir / f"{module_stem}.pending_preview.py"
        preview_path.write_text(new_text, encoding="utf-8")

        patch_lines = difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(module_path) if module_path.exists() else "/dev/null",
            tofile=str(module_path),
        )
        patch_text = "".join(patch_lines)
        patch_path = metadata_dir / f"{module_stem}.pending.patch"
        patch_path.write_text(patch_text, encoding="utf-8")

        central_preview_path = central_preview_dir / f"{run_slug}__{preview_path.name}"
        central_preview_path.write_text(new_text, encoding="utf-8")
        central_patch_path = central_patch_dir / f"{run_slug}__{patch_path.name}"
        central_patch_path.write_text(patch_text, encoding="utf-8")

        created[f"{module_stem}_preview"] = str(preview_path)
        created[f"{module_stem}_patch"] = str(patch_path)
        created[f"{module_stem}_preview_central"] = str(central_preview_path)
        created[f"{module_stem}_patch_central"] = str(central_patch_path)
    return created


def write_pending_curated_manifest(
    *,
    family: SeedFamily,
    summary_df: pd.DataFrame,
    run_id: str,
    round_id: int,
    run_dir: str | Path,
    decision_stage: str,
    name_prefix: str,
    metadata_dir: str | Path,
    pending_dir: str | Path = DEFAULT_PENDING_CURATED_DIR,
    auto_apply: bool = False,
    data: dict[str, pd.Series] | None = None,
) -> dict[str, Path] | None:
    records = summary_df.to_dict(orient="records")
    selected_winners: list[dict[str, Any]] = []
    selected_keeps: list[tuple[tuple[float, float, float, float, float], dict[str, Any]]] = []
    module_path = LLM_REFINED_DIR / f"{family.family}_family.py"
    module_text = module_path.read_text(encoding="utf-8") if module_path.exists() else ""
    family_existing_refs = _build_existing_family_refs(module_text=module_text, data=data)
    for row in records:
        ok, reason = _promotion_gate(
            row,
            family_existing_refs=family_existing_refs,
            data=data,
        )
        if not ok:
            continue
        entry = _entry_from_row(
            row=row,
            family=family,
            run_id=run_id,
            round_id=round_id,
            run_dir=run_dir,
            decision_stage=decision_stage,
            name_prefix=name_prefix,
        )
        entry["promotion_gate_reason"] = reason
        decision = str(row.get("decision", "")).strip()
        if decision == "research_keep":
            selected_keeps.append((_promotion_priority(row), entry))
        else:
            selected_winners.append(entry)

    selected_keeps.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = list(selected_winners)
    selected.extend(entry for _, entry in selected_keeps[:MAX_PENDING_RESEARCH_KEEP])

    dedup_skipped: list[dict[str, str]] = []
    filtered_selected: list[dict[str, Any]] = []
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in selected:
        target = str(entry.get("promotion_context", {}).get("target_family_module", "")).strip()
        if target:
            by_module[target].append(entry)
    for module_path_str, module_entries in by_module.items():
        module_path = Path(module_path_str)
        module_text = module_path.read_text(encoding="utf-8") if module_path.exists() else ""
        kept_entries, skipped_entries = _filter_entries_against_module(
            module_text=module_text,
            entries=module_entries,
        )
        filtered_selected.extend(kept_entries)
        for item in skipped_entries:
            dedup_skipped.append(
                {
                    "target_family_module": module_path_str,
                    **item,
                }
            )

    payload = {
        "version": 1,
        "created_at": utc_now_iso(),
        "family": family.family,
        "canonical_seed": family.canonical_seed,
        "run_id": run_id,
        "round_id": int(round_id),
        "decision_stage": decision_stage,
        "name_prefix": name_prefix,
        "promotion_policy": {
            "required_decision": "research_winner",
            "broad_mode": True,
            "require_nonempty_expression": True,
        },
        "entries": filtered_selected,
        "dedup_skipped": dedup_skipped,
    }

    metadata_path = Path(metadata_dir)
    metadata_path.mkdir(parents=True, exist_ok=True)
    local_path = metadata_path / "pending_curated_manifest.yaml"
    local_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    central_dir = Path(pending_dir)
    central_dir.mkdir(parents=True, exist_ok=True)
    run_slug = Path(run_dir).name
    central_path = central_dir / f"{run_slug}_pending_curated.yaml"
    central_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    summary_path = metadata_path / "pending_curated_manifest.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    patch_artifacts = _write_patch_bundle(
        payload=payload,
        metadata_dir=metadata_path,
        pending_dir=central_dir,
    )

    out = {
        "pending_curated_manifest": local_path,
        "pending_curated_manifest_central": central_path,
        "pending_curated_manifest_json": summary_path,
    }
    out.update({key: Path(value) for key, value in patch_artifacts.items()})
    if auto_apply and filtered_selected:
        applied = _apply_payload_to_store(payload=payload, metadata_dir=metadata_path)
        out.update({key: Path(value) for key, value in applied.items() if value})
    return out
