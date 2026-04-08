from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def append_library_manifest(path: Path, record: dict[str, Any]) -> None:
    row = pd.DataFrame([record])
    if path.exists():
        existing = pd.read_csv(path)
        if "factor_name" in existing.columns and record["factor_name"] in set(existing["factor_name"]):
            existing = existing[existing["factor_name"] != record["factor_name"]]
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(path, index=False)


def build_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    quick = metrics.get("quick_ic", {})
    quick_rank = metrics.get("quick_rank_ic", {})
    pure = metrics.get("pure_ic", {})
    pure_rank = metrics.get("pure_rank_ic", {})
    group = metrics.get("group_backtest", {})
    row = {
        "factor_name": metrics.get("factor_name"),
        "candidate_rank": metrics.get("candidate_rank"),
        "expr": metrics.get("expr"),
        "direction": metrics.get("direction"),
        "quick_rank_ic_mean": quick_rank.get("mean_ic"),
        "quick_rank_icir": quick_rank.get("icir"),
        "quick_rank_ic_win_rate": quick_rank.get("win_rate", quick_rank.get("ic_positive_rate")),
        "quick_rank_ic_monthly_win_rate": quick_rank.get("monthly_positive_rate"),
        "quick_rank_ic_yearly_win_rate": quick_rank.get("yearly_positive_rate"),
        "quick_ic_mean": quick.get("mean_ic"),
        "quick_icir": quick.get("icir"),
        "quick_ic_win_rate": quick.get("win_rate", quick.get("ic_positive_rate")),
        "quick_ic_monthly_win_rate": quick.get("monthly_positive_rate"),
        "quick_ic_yearly_win_rate": quick.get("yearly_positive_rate"),
        "pure_rank_ic_mean": pure_rank.get("mean_ic", np.nan),
        "pure_rank_icir": pure_rank.get("icir", np.nan),
        "pure_rank_ic_win_rate": pure_rank.get("win_rate", pure_rank.get("ic_positive_rate", np.nan)),
        "pure_ic_mean": pure.get("mean_ic", np.nan),
        "pure_icir": pure.get("icir", np.nan),
        "pure_ic_win_rate": pure.get("win_rate", pure.get("ic_positive_rate", np.nan)),
        "net_ann_return": group.get("net_ann_return"),
        "net_excess_ann_return": group.get("net_excess_ann_return"),
        "net_sharpe": group.get("net_sharpe"),
        "net_win_rate": group.get("net_win_rate"),
        "net_monthly_win_rate": group.get("net_monthly_win_rate"),
        "net_yearly_win_rate": group.get("net_yearly_win_rate"),
        "long_only_group": group.get("long_only_group"),
        "long_only_net_ann_return": group.get("long_only_net_ann_return"),
        "long_only_net_excess_ann_return": group.get("long_only_net_excess_ann_return"),
        "long_only_net_sharpe": group.get("long_only_net_sharpe"),
        "long_only_win_rate": group.get("long_only_win_rate"),
        "long_only_monthly_win_rate": group.get("long_only_monthly_win_rate"),
        "long_only_yearly_win_rate": group.get("long_only_yearly_win_rate"),
        "gross_ann_return": group.get("gross_ann_return"),
        "gross_excess_ann_return": group.get("gross_excess_ann_return"),
        "gross_win_rate": group.get("gross_win_rate"),
        "long_only_gross_ann_return": group.get("long_only_gross_ann_return"),
        "long_only_gross_excess_ann_return": group.get("long_only_gross_excess_ann_return"),
        "benchmark_ann_return": group.get("benchmark_ann_return"),
        "benchmark_mean_daily_return": group.get("benchmark_mean_daily_return"),
        "mean_turnover": group.get("mean_turnover"),
        "mean_cost_bps": group.get("mean_cost_bps"),
        "long_only_mean_turnover": group.get("long_only_mean_turnover"),
        "long_only_mean_cost_bps": group.get("long_only_mean_cost_bps"),
        "long_only_net_excess_mean_daily_return": group.get("long_only_net_excess_mean_daily_return"),
        "top_group_mean_return": group.get("top_group_mean_return"),
        "bottom_group_mean_return": group.get("bottom_group_mean_return"),
        "best_group": group.get("best_group"),
        "worst_group": group.get("worst_group"),
        "fitness": metrics.get("fitness"),
        "train_fitness": metrics.get("train_fitness"),
        "train_raw_ic": metrics.get("train_raw_ic"),
        "train_raw_ic_signed": metrics.get("train_raw_ic_signed"),
        "train_direction": metrics.get("train_direction"),
        "valid_fitness": metrics.get("valid_fitness"),
        "valid_raw_ic": metrics.get("valid_raw_ic"),
        "valid_raw_ic_signed": metrics.get("valid_raw_ic_signed"),
        "valid_direction": metrics.get("valid_direction"),
        "raw_ic_signed": metrics.get("raw_ic_signed"),
        # Backward-compatible aliases: keep old quick/pure columns pointing to RankIC metrics.
        "quick_mean_ic": quick_rank.get("mean_ic"),
        "quick_ic_positive_rate": quick_rank.get("win_rate", quick_rank.get("ic_positive_rate")),
        "quick_monthly_positive_rate": quick_rank.get("monthly_positive_rate"),
        "quick_yearly_positive_rate": quick_rank.get("yearly_positive_rate"),
        "pure_mean_ic": pure_rank.get("mean_ic", np.nan),
        "alphalens_status": metrics.get("alphalens", {}).get("status", "disabled"),
        "source_run_name": metrics.get("source_run_name"),
        "source_candidates_csv": metrics.get("source_candidates_csv"),
    }
    alpha = metrics.get("alphalens", {}) or {}
    if isinstance(alpha, dict):
        if "n_rows_clean_factor" in alpha:
            row["alphalens_n_rows_clean_factor"] = alpha.get("n_rows_clean_factor")
        ic_mean = alpha.get("ic_mean", {})
        if isinstance(ic_mean, dict):
            for period, value in ic_mean.items():
                row[f"alphalens_ic_mean_{period}"] = value
        ic_ir = alpha.get("ic_ir", {})
        if isinstance(ic_ir, dict):
            for period, value in ic_ir.items():
                row[f"alphalens_ic_ir_{period}"] = value
    return row


def _save_frame(obj: pd.Series | pd.DataFrame, path: Path) -> None:
    if isinstance(obj, pd.Series):
        obj.to_frame(obj.name or "value").to_csv(path)
    else:
        obj.to_csv(path)


def _plot_ic_panel(result: dict[str, Any], out_dir: Path, prefix: str) -> None:
    quick_rank = result["tables"].get("quick_rank_ic_series")
    quick = result["tables"].get("quick_ic_series")
    main = quick_rank if isinstance(quick_rank, pd.DataFrame) and not quick_rank.empty else quick
    if not isinstance(main, pd.DataFrame) or main.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    title = "Cumulative Rank IC" if main is quick_rank else "Cumulative IC"
    main.iloc[:, 0].cumsum().plot(ax=axes[0], title=title)
    rolling = result["tables"].get("quick_rank_ic_rolling")
    if not (isinstance(rolling, pd.DataFrame) and "rolling_mean_ic" in rolling):
        rolling = result["tables"].get("quick_ic_rolling")
    if isinstance(rolling, pd.DataFrame) and "rolling_mean_ic" in rolling:
        rolling["rolling_mean_ic"].plot(ax=axes[1], title="Rolling Mean IC")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "ic_overview.png", dpi=150)
    plt.close(fig)


def _plot_nav_panel(result: dict[str, Any], out_dir: Path, prefix: str) -> None:
    long_short = result["tables"].get("long_short_daily")
    long_only = result["tables"].get("long_only_daily")
    group_nav = result["tables"].get("group_nav")
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    if isinstance(long_short, pd.DataFrame) and {"gross_nav", "net_nav"}.issubset(long_short.columns):
        long_short[["gross_nav", "net_nav"]].plot(ax=axes[0], title="Long-Short NAV")
    if isinstance(long_only, pd.DataFrame) and {"gross_nav", "net_nav"}.issubset(long_only.columns):
        long_only[["gross_nav", "net_nav"]].plot(ax=axes[1], title="Long-Only NAV")
    if isinstance(group_nav, pd.DataFrame) and not group_nav.empty:
        group_nav.plot(ax=axes[2], title="Group NAV")
    fig.tight_layout()
    fig.savefig(out_dir / "nav_overview.png", dpi=150)
    plt.close(fig)


def _table_output_name(name: str) -> str:
    mapping = {
        "factor": "factor.parquet",
        "quick_ic_series": "quick_ic_daily.csv",
        "quick_ic_monthly": "quick_ic_monthly.csv",
        "quick_ic_yearly": "quick_ic_yearly.csv",
        "quick_ic_rolling": "quick_ic_rolling.csv",
        "quick_rank_ic_series": "quick_rank_ic_daily.csv",
        "quick_rank_ic_monthly": "quick_rank_ic_monthly.csv",
        "quick_rank_ic_yearly": "quick_rank_ic_yearly.csv",
        "quick_rank_ic_rolling": "quick_rank_ic_rolling.csv",
        "pure_ic_series": "pure_ic_daily.csv",
        "pure_ic_monthly": "pure_ic_monthly.csv",
        "pure_ic_yearly": "pure_ic_yearly.csv",
        "pure_ic_rolling": "pure_ic_rolling.csv",
        "pure_rank_ic_series": "pure_rank_ic_daily.csv",
        "pure_rank_ic_monthly": "pure_rank_ic_monthly.csv",
        "pure_rank_ic_yearly": "pure_rank_ic_yearly.csv",
        "pure_rank_ic_rolling": "pure_rank_ic_rolling.csv",
        "group_returns": "group_returns_daily.csv",
        "group_nav": "group_nav.csv",
        "long_short_daily": "long_short_daily.csv",
        "long_only_daily": "long_only_daily.csv",
        "group_labels": "group_labels.csv",
        "alphalens_ic": "alphalens_ic.csv",
        "alphalens_clean_factor": "alphalens_clean_factor.csv",
    }
    if name in mapping:
        return mapping[name]
    return f"{name}.csv"


def write_backtest_report(
    *,
    result: dict[str, Any],
    out_dir: Path,
    prefix: str,
    save_to_store: Path | None = None,
    library_manifest: Path | None = None,
) -> Path:
    report_dir = out_dir / prefix
    tables_dir = report_dir / "tables"
    plots_dir = report_dir / "plots"
    meta_dir = report_dir / "meta"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in result["tables"].items():
        if not isinstance(obj, (pd.Series, pd.DataFrame)):
            continue
        if obj.empty:
            continue
        filename = _table_output_name(name)
        path = tables_dir / filename
        if path.suffix == ".parquet":
            if isinstance(obj, pd.Series):
                obj.to_frame(obj.name or prefix).to_parquet(path)
            else:
                obj.to_parquet(path)
        else:
            _save_frame(obj, path)

    _plot_ic_panel(result, plots_dir, prefix)
    _plot_nav_panel(result, plots_dir, prefix)

    summary_row = build_summary_row(result)
    pd.DataFrame([summary_row]).to_csv(meta_dir / "summary.csv", index=False)
    with (meta_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=2, allow_nan=True)

    if save_to_store:
        try:
            from ...risk.factors.factor_store import FactorStore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "save_to_store requires FactorStore support, which is not vendored in the "
                "AlphaRefinery runtime bundle"
            ) from exc
        store = FactorStore(save_to_store)
        factor = result["tables"].get("factor")
        if isinstance(factor, pd.Series):
            factor_name = result["metrics"].get("factor_name") or prefix
            store.save(
                factor=factor,
                name=factor_name,
                description=str(result["metrics"].get("expr") or ""),
                overwrite=True,
            )
            manifest_path = library_manifest or save_to_store / "library_manifest.csv"
            append_library_manifest(
                manifest_path,
                {
                    **summary_row,
                    "expr_json": result["metrics"].get("expr_json"),
                    "expr_qlib": result["metrics"].get("expr_qlib"),
                    "data_source": result["metrics"].get("data_source"),
                    "data_dir": result["metrics"].get("data_dir"),
                    "eval_backend": result["metrics"].get("eval_backend"),
                    "factor_store_dir": str(save_to_store),
                    "factor_store_file": f"{factor_name}.parquet",
                },
            )
    return report_dir
