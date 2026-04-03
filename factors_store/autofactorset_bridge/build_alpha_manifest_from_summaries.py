from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_ALPHA101_SUMMARY = Path(
    "/root/workspace/zxy_workspace/AlphaRefinery/artifacts/backtests/alpha/alpha101_full_from2018_summary_20260319_061948.csv"
)
DEFAULT_ALPHA191_SUMMARY = Path(
    "/root/workspace/zxy_workspace/AlphaRefinery/artifacts/backtests/alpha/alpha191_full_from2018_summary_20260319_094549.csv"
)
DEFAULT_MANIFEST_OUT = Path(
    "/root/workspace/zxy_workspace/AlphaRefinery/artifacts/autofactorset_ingest/manifests/alpha101_alpha191_unified_full_from2018_20260401.yaml"
)
DEFAULT_FACTOR_LIST_OUT = Path(
    "/root/workspace/zxy_workspace/AlphaRefinery/artifacts/autofactorset_ingest/factor_lists/alpha101_alpha191_unified_full_from2018_20260401.txt"
)


def _normalize_rows(
    df: pd.DataFrame,
    *,
    family_tag: str,
    source_report: str,
    min_nonnull: int,
) -> list[dict[str, Any]]:
    work = df.copy()
    if "factor_name" not in work.columns:
        raise ValueError(f"{source_report} missing factor_name column")

    if "nonnull" in work.columns:
        work = work[work["nonnull"].fillna(0) >= int(min_nonnull)]

    if "net_sharpe" in work.columns:
        work = work.sort_values(["net_sharpe", "net_ann_return"], ascending=[False, False], na_position="last")

    rows: list[dict[str, Any]] = []
    for _, item in work.iterrows():
        factor_name = str(item["factor_name"]).strip()
        if not factor_name:
            continue
        rows.append(
            {
                "factor_name": factor_name,
                "bucket": "alpha_baseline",
                "family": family_tag,
                "role": "registry_baseline",
                "source_model": "registry_builtin",
                "source_report": source_report,
                "historical_nonnull": int(float(item["nonnull"])) if pd.notna(item.get("nonnull")) else None,
                "historical_quick_icir": float(item["quick_icir"]) if pd.notna(item.get("quick_icir")) else None,
                "historical_net_ann_return": float(item["net_ann_return"]) if pd.notna(item.get("net_ann_return")) else None,
                "historical_net_sharpe": float(item["net_sharpe"]) if pd.notna(item.get("net_sharpe")) else None,
                "historical_mean_turnover": float(item["mean_turnover"]) if pd.notna(item.get("mean_turnover")) else None,
            }
        )
    return rows


def build_manifest(
    *,
    alpha101_summary: Path,
    alpha191_summary: Path,
    manifest_out: Path,
    factor_list_out: Path | None,
    min_nonnull: int,
) -> tuple[Path, int]:
    alpha101_df = pd.read_csv(alpha101_summary)
    alpha191_df = pd.read_csv(alpha191_summary)

    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _normalize_rows(
            alpha101_df,
            family_tag="alpha101",
            source_report=str(alpha101_summary),
            min_nonnull=min_nonnull,
        )
    )
    candidates.extend(
        _normalize_rows(
            alpha191_df,
            family_tag="alpha191",
            source_report=str(alpha191_summary),
            min_nonnull=min_nonnull,
        )
    )

    manifest = {
        "manifest_name": manifest_out.stem,
        "created_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "source_reports": [
            str(alpha101_summary),
            str(alpha191_summary),
        ],
        "notes": [
            "Unified alpha101 + alpha191 admission manifest built from local full-from-2018 backtest summaries.",
            f"Candidates with nonnull < {int(min_nonnull)} are excluded.",
            "This manifest is intended for autofactorset_bridge/evaluate_registry_manifest.py.",
            "alpha191 factors require benchmark fields; run the bridge with a valid --benchmark-path.",
        ],
        "eval_defaults": {
            "data_source": "baostock_parquet",
            "data_dir": "/root/dmd/BaoStock/panel.parquet",
            "eval_options": {
                "data_begin": "2023-07-01",
                "data_end": "2026-03-01",
                "n_days": 0,
                "n_sh_stocks": 0,
                "n_sz_stocks": 0,
                "sampling_mode": "latest_mcap_262",
                "sampling_seed": 42,
                "eval_backend": "python",
            },
        },
        "candidates": candidates,
    }

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    if factor_list_out is not None:
        factor_list_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [str(item["factor_name"]) for item in candidates]
        factor_list_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return manifest_out, len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified alpha101/alpha191 autofactorset ingest manifest from local summary CSVs.")
    parser.add_argument("--alpha101-summary", default=str(DEFAULT_ALPHA101_SUMMARY))
    parser.add_argument("--alpha191-summary", default=str(DEFAULT_ALPHA191_SUMMARY))
    parser.add_argument("--out-manifest", default=str(DEFAULT_MANIFEST_OUT))
    parser.add_argument("--out-factor-list", default=str(DEFAULT_FACTOR_LIST_OUT))
    parser.add_argument("--min-nonnull", type=int, default=1)
    args = parser.parse_args()

    manifest_out, count = build_manifest(
        alpha101_summary=Path(args.alpha101_summary).expanduser().resolve(),
        alpha191_summary=Path(args.alpha191_summary).expanduser().resolve(),
        manifest_out=Path(args.out_manifest).expanduser().resolve(),
        factor_list_out=Path(args.out_factor_list).expanduser().resolve() if args.out_factor_list else None,
        min_nonnull=int(args.min_nonnull),
    )
    print(f"[manifest] {manifest_out}")
    print(f"[count] {count}")


if __name__ == "__main__":
    main()
