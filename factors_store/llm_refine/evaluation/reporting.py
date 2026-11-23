'''CSV and Markdown output helpers for evaluator runs.

Turns full evaluation frames into compact user-facing reports and canonical stage files.
'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.archive import ensure_run_subdir
from ..core.models import SeedFamily
from .decisions import _parent_reference, _sorted_summary


def _slim_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    work = summary_df.copy()
    rename_map = {
        "quick_rank_ic_mean": "quick_rank_ic",
        "quick_ic_mean": "quick_ic",
    }
    work = work.rename(columns=rename_map)
    desired = [
        "factor_name",
        "role",
        "evaluation_stage",
        "decision_scope",
        "model",
        "expression",
        "winner_score",
        "quick_rank_ic",
        "quick_rank_icir",
        "neutral_quick_rank_icir",
        "quick_ic",
        "quick_icir",
        "net_ann_return",
        "neutral_net_ann_return",
        "net_excess_ann_return",
        "neutral_net_excess_ann_return",
        "long_only_net_ann_return",
        "net_sharpe",
        "neutral_net_sharpe",
        "mean_turnover",
        "neutral_mean_turnover",
        "avg_abs_style_corr",
        "max_abs_style_corr",
        "top_style_exposure",
        "top_style_corr",
        "nearest_decorrelation_target",
        "corr_to_nearest_decorrelation_target",
        "avg_abs_decorrelation_target_corr",
        "decorrelation_grade",
        "decorrelation_score",
        "decorrelation_gate_action",
        "decorrelation_gate_reason",
        "decorrelation_winner_allowed",
        "decorrelation_keep_allowed",
        "decorrelation_reference_allowed",
        "decorrelation_arbitration_reason",
        "decorrelation_quality_gate_passed",
        "decorrelation_strong_quality_passed",
        "raw_neutral_rank_icir_gap",
        "raw_neutral_net_sharpe_gap",
        "neutral_winner_guard_passed",
        "neutralization_status",
        "stage_winner_guard_passed",
        "stage_winner_guard_reason",
        "metrics_completeness",
        "missing_core_metrics_count",
        "eligible_for_best_node",
        "decision",
        "decision_reason",
    ]
    existing = [col for col in desired if col in work.columns]
    return work[existing].copy()


def _markdown_report(
    *,
    family: SeedFamily,
    summary_df: pd.DataFrame,
    settings: dict[str, Any],
    data_meta: dict[str, object],
) -> str:
    parent = _parent_reference(summary_df, family)
    lines = [
        "# LLM Refine Evaluation Report",
        "",
        f"- family: `{family.family}`",
        f"- canonical_seed: `{family.canonical_seed}`",
        f"- panel_path: `{settings.get('panel_path')}`",
        f"- benchmark_path: `{settings.get('benchmark_path') or 'none'}`",
        f"- start: `{settings.get('start')}`",
        f"- end: `{settings.get('end') or 'latest'}`",
        f"- horizon: `{settings.get('horizon')}`",
        f"- n_groups: `{settings.get('n_groups')}`",
        f"- cost_bps: `{settings.get('cost_bps')}`",
        f"- rows_after_filter: `{data_meta.get('rows_after_filter')}`",
        f"- instruments_after_filter: `{data_meta.get('instruments_after_filter')}`",
        "",
        "## Results",
        "",
        "| Factor | Role | Model | WinnerScore | RankICIR | NRankICIR | Net Sharpe | NSharpe | StyleCorr | Decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('factor_name', '')}`",
                    str(row.get("role", "")),
                    str(row.get("model", "")),
                    _fmt_md_num(row.get("winner_score")),
                    _fmt_md_num(row.get("quick_rank_icir")),
                    _fmt_md_num(row.get("neutral_quick_rank_icir")),
                    _fmt_md_num(row.get("net_sharpe")),
                    _fmt_md_num(row.get("neutral_net_sharpe")),
                    _fmt_md_num(row.get("avg_abs_style_corr")),
                    str(row.get("decision", "")),
                ]
            )
            + " |"
        )
    if parent is not None:
        lines.extend(["", "## Parent Reference", ""])
        lines.append(f"- expression: `{parent.get('expression', '')}`")
        lines.append(f"- RankIC: `{_fmt_md_num(parent.get('quick_rank_ic_mean'))}`")
        lines.append(f"- RankICIR: `{_fmt_md_num(parent.get('quick_rank_icir'))}`")
        lines.append(f"- Neutral RankICIR: `{_fmt_md_num(parent.get('neutral_quick_rank_icir'))}`")
        lines.append(f"- Net Ann Return: `{_fmt_md_num(parent.get('net_ann_return'))}`")
        lines.append(f"- Net Excess Ann Return: `{_fmt_md_num(parent.get('net_excess_ann_return'))}`")
        lines.append(f"- Neutral Net Sharpe: `{_fmt_md_num(parent.get('neutral_net_sharpe'))}`")
        lines.append(f"- Mean Turnover: `{_fmt_md_num(parent.get('mean_turnover'))}`")
    lines.extend(["", "## Research Gate Notes", ""])
    for _, row in summary_df[summary_df["role"] == "candidate"].iterrows():
        lines.append(f"### {row.get('factor_name')}")
        lines.append(f"- model: {row.get('model')}")
        if str(row.get("evaluation_stage", "")).strip():
            lines.append(f"- evaluation_stage: {row.get('evaluation_stage')}")
        if str(row.get("decision_scope", "")).strip():
            lines.append(f"- decision_scope: {row.get('decision_scope')}")
        lines.append(f"- expression: `{row.get('expression', '')}`")
        lines.append(f"- research_decision: {row.get('decision', '')}")
        lines.append(f"- reason: {row.get('decision_reason', '')}")
        lines.append(f"- neutral RankICIR: {_fmt_md_num(row.get('neutral_quick_rank_icir'))}")
        lines.append(f"- neutral NetSharpe: {_fmt_md_num(row.get('neutral_net_sharpe'))}")
        lines.append(f"- avg_abs_style_corr: {_fmt_md_num(row.get('avg_abs_style_corr'))}")
        lines.append(f"- nearest_decorrelation_target: {row.get('nearest_decorrelation_target', '')}")
        lines.append(
            f"- corr_to_nearest_decorrelation_target: "
            f"{_fmt_md_num(row.get('corr_to_nearest_decorrelation_target'))}"
        )
        lines.append(
            f"- avg_abs_decorrelation_target_corr: "
            f"{_fmt_md_num(row.get('avg_abs_decorrelation_target_corr'))}"
        )
        if str(row.get("decorrelation_gate_action", "")).strip():
            lines.append(f"- decorrelation_gate_action: {row.get('decorrelation_gate_action')}")
            lines.append(f"- decorrelation_gate_reason: {row.get('decorrelation_gate_reason', '')}")
            lines.append(f"- decorrelation_keep_allowed: {row.get('decorrelation_keep_allowed')}")
            lines.append(f"- decorrelation_winner_allowed: {row.get('decorrelation_winner_allowed')}")
            lines.append(f"- decorrelation_reference_allowed: {row.get('decorrelation_reference_allowed')}")
            if str(row.get("decorrelation_arbitration_reason", "")).strip():
                lines.append(f"- decorrelation_arbitration_reason: {row.get('decorrelation_arbitration_reason')}")
        if str(row.get("top_style_exposure", "")).strip():
            lines.append(
                f"- top_style_exposure: {row.get('top_style_exposure')} "
                f"({_fmt_md_num(row.get('top_style_corr'))})"
            )
        lines.append(f"- raw_neutral_icir_gap: {_fmt_md_num(row.get('raw_neutral_rank_icir_gap'))}")
        lines.append(f"- raw_neutral_sharpe_gap: {_fmt_md_num(row.get('raw_neutral_net_sharpe_gap'))}")
        lines.append(f"- neutral_winner_guard_passed: {row.get('neutral_winner_guard_passed')}")
        if str(row.get("stage_winner_guard_reason", "")).strip():
            lines.append(f"- stage_winner_guard: {row.get('stage_winner_guard_reason')}")
        if row.get("validation_warnings"):
            lines.append(f"- validation_warnings: {row.get('validation_warnings')}")
        if row.get("error"):
            lines.append(f"- error: {row.get('error')}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _fmt_md_num(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isnan(value):
            return "NA"
        return f"{float(value):.4f}"
    return str(value)


def _write_window_outputs(
    *,
    run_path: Path,
    family: SeedFamily,
    stage_name: str,
    summary_df: pd.DataFrame,
    settings: dict[str, Any],
    data_meta: dict[str, object],
    make_canonical_alias: bool = False,
) -> dict[str, Path]:
    evaluation_dir = ensure_run_subdir(run_path, "evaluation")
    base_name = f"family_backtest_{stage_name}"
    summary_path = evaluation_dir / f"{base_name}_summary.csv"
    ranked_path = evaluation_dir / f"{base_name}_ranked.csv"
    summary_full_path = evaluation_dir / f"{base_name}_summary_full.csv"
    ranked_full_path = evaluation_dir / f"{base_name}_ranked_full.csv"

    decision_scope = "formal_decision" if make_canonical_alias else "diagnostic_only"
    summary_full_df = summary_df.copy()
    summary_full_df["evaluation_stage"] = str(stage_name)
    summary_full_df["decision_scope"] = decision_scope
    ranked_full_df = _sorted_summary(summary_full_df)
    slim_summary_df = _slim_summary(summary_full_df)
    slim_ranked_df = _slim_summary(ranked_full_df)

    summary_full_df.to_csv(summary_full_path, index=False)
    ranked_full_df.to_csv(ranked_full_path, index=False)
    slim_summary_df.to_csv(summary_path, index=False)
    slim_ranked_df.to_csv(ranked_path, index=False)

    outputs = {
        "evaluation_dir": evaluation_dir,
        f"{stage_name}_summary": summary_path,
        f"{stage_name}_ranked": ranked_path,
        f"{stage_name}_summary_full": summary_full_path,
        f"{stage_name}_ranked_full": ranked_full_path,
    }

    if make_canonical_alias:
        canonical_summary = evaluation_dir / "family_backtest_summary.csv"
        canonical_ranked = evaluation_dir / "family_backtest_ranked.csv"
        canonical_summary_full = evaluation_dir / "family_backtest_summary_full.csv"
        canonical_ranked_full = evaluation_dir / "family_backtest_ranked_full.csv"
        canonical_report = evaluation_dir / "keep_drop_report.md"
        research_gate_report = evaluation_dir / "research_gate_report.md"

        report_text = _markdown_report(
            family=family,
            summary_df=summary_full_df,
            settings=settings,
            data_meta=data_meta,
        )

        slim_summary_df.to_csv(canonical_summary, index=False)
        slim_ranked_df.to_csv(canonical_ranked, index=False)
        summary_full_df.to_csv(canonical_summary_full, index=False)
        ranked_full_df.to_csv(canonical_ranked_full, index=False)
        canonical_report.write_text(report_text, encoding="utf-8")
        research_gate_report.write_text(report_text, encoding="utf-8")
        outputs.update(
            {
                "family_backtest_summary": canonical_summary,
                "family_backtest_ranked": canonical_ranked,
                "family_backtest_summary_full": canonical_summary_full,
                "family_backtest_ranked_full": canonical_ranked_full,
                "keep_drop_report": canonical_report,
                "research_gate_report": research_gate_report,
            }
        )
    return outputs


