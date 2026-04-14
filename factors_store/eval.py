from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._bootstrap import ensure_project_roots

ensure_project_roots()

from ._vendor.gpqlib_runtime.evaluation.quick_ic.evaluation import evaluate_ic  # noqa: E402
from ._vendor.gpqlib_runtime.evaluation.quick_ic.orthogonalize import gram_schmidt_orthogonalize  # noqa: E402
from ._vendor.gpqlib_runtime.evaluation.single_factor_eval import (  # noqa: E402
    BacktestConfig,
    run_single_factor_backtest,
    write_backtest_report,
)
from ._vendor.gpqlib_runtime.evaluation.single_factor_eval.report import build_summary_row  # noqa: E402
from ._vendor.gpqlib_runtime.risk.factors.cne5_core import compute_cne5  # noqa: E402

EPS = 1e-12


def _coerce_datetime_instrument_index(
    frame: pd.DataFrame,
    *,
    reference_index: pd.Index,
) -> pd.DataFrame:
    out = frame.copy()
    idx = out.index
    if isinstance(idx, pd.MultiIndex):
        if idx.nlevels == 2:
            out.index = idx.set_names(["datetime", "instrument"])
            return out
        if idx.nlevels > 2:
            out.index = pd.MultiIndex.from_arrays(
                [idx.get_level_values(-2), idx.get_level_values(-1)],
                names=["datetime", "instrument"],
            )
            return out
    if len(idx) == len(reference_index):
        if isinstance(reference_index, pd.MultiIndex) and reference_index.nlevels == 2:
            out.index = reference_index.set_names(["datetime", "instrument"])
            return out
    if len(idx) > 0 and all(isinstance(item, tuple) and len(item) == 2 for item in idx):
        out.index = pd.MultiIndex.from_tuples(idx, names=["datetime", "instrument"])
        return out
    raise ValueError(
        "unable to coerce exposures index into (datetime, instrument) MultiIndex; "
        f"got {type(idx).__name__} with nlevels={getattr(idx, 'nlevels', 1)}"
    )


def make_forward_return(price: pd.Series, horizon: int = 5) -> pd.Series:
    future = price.groupby(level="instrument", group_keys=False).shift(-int(horizon))
    return (future / price - 1.0).rename(f"fwd_ret_{horizon}")


def evaluate_factor(
    factor: pd.Series,
    *,
    label: pd.Series | None = None,
    close: pd.Series | None = None,
    horizon: int = 5,
    n_groups: int = 5,
    min_stocks: int = 5,
) -> dict[str, object]:
    if label is None:
        if close is None:
            raise ValueError("provide either label or close")
        label = make_forward_return(close, horizon=horizon)
    return evaluate_ic(factor, label, n_groups=n_groups, min_stocks=min_stocks)


def build_price_panel(data: dict[str, pd.Series], fields: tuple[str, ...] = ("close", "open")) -> pd.DataFrame:
    parts: dict[str, pd.Series] = {}
    for field in fields:
        value = data.get(field)
        if isinstance(value, pd.Series):
            parts[field] = value.astype(float)
    if not parts:
        raise ValueError(f"no usable price fields found for {fields}")
    panel = pd.concat(parts, axis=1).sort_index()
    panel.index = panel.index.set_names(["datetime", "instrument"])
    return panel


def build_proxy_exposures(data: dict[str, pd.Series]) -> pd.DataFrame:
    close = data["close"].astype(float).rename("close")
    returns = data.get("returns")
    if returns is None:
        returns = close.groupby(level="instrument", group_keys=False).pct_change(fill_method=None).rename("returns")
    else:
        returns = returns.astype(float).rename("returns")

    market_return = data.get("market_return")
    if market_return is not None:
        market_ret_daily = market_return.groupby(level="datetime").first().astype(float).rename("market_return")
    else:
        market_ret_daily = (
            close.unstack("instrument").pct_change(fill_method=None).mean(axis=1).dropna().rename("market_return")
        )

    amount = data.get("amount")
    volume = data.get("volume")
    float_market_cap = data.get("float_market_cap")
    turnover_rate = data.get("turnover_rate")

    if amount is not None:
        liquidity_base = amount.astype(float).rename("amount")
    elif volume is not None:
        liquidity_base = volume.astype(float).abs().rename("volume")
    else:
        liquidity_base = close.abs().rename("close_abs")

    market_cap_proxy = (
        liquidity_base.replace(0.0, np.nan)
        .groupby(level="instrument", group_keys=False)
        .rolling(252, min_periods=21)
        .mean()
        .droplevel(0)
        .rename("market_cap_proxy")
    )

    if float_market_cap is not None:
        market_cap_for_risk = float_market_cap.astype(float).combine_first(market_cap_proxy).rename("market_cap")
    else:
        market_cap_for_risk = market_cap_proxy.rename("market_cap")

    turnover_proxy = (liquidity_base / (market_cap_proxy + EPS)).rename("turnover_proxy")
    if turnover_rate is not None:
        turnover_for_risk = turnover_rate.astype(float).combine_first(turnover_proxy).rename("turnover")
    else:
        turnover_for_risk = turnover_proxy.rename("turnover")

    exposures = compute_cne5(
        {
            "market_cap": market_cap_for_risk,
            "stock_ret": returns,
            "market_ret": market_ret_daily,
            "turnover": turnover_for_risk,
        }
    )
    exposures = (
        exposures.groupby(level=0, group_keys=False)
        .apply(lambda d: d.fillna(d.median(numeric_only=True)))
        .fillna(0.0)
    )
    std = exposures.std(axis=0, skipna=True)
    keep_cols = std[std > EPS].index.tolist()
    exposures = exposures[keep_cols]
    exposures = _coerce_datetime_instrument_index(exposures, reference_index=market_cap_for_risk.index)
    return exposures


def build_style_exposures(
    data: dict[str, pd.Series],
    *,
    extra_exposures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    exposures = build_proxy_exposures(data)
    if extra_exposures is None or extra_exposures.empty:
        return exposures

    extra = _coerce_datetime_instrument_index(extra_exposures, reference_index=exposures.index)
    extra = extra.reindex(exposures.index)
    extra = (
        extra.groupby(level=0, group_keys=False)
        .apply(lambda d: d.fillna(d.median(numeric_only=True)))
        .fillna(0.0)
    )
    merged = pd.concat([exposures, extra], axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]
    std = merged.std(axis=0, skipna=True)
    keep_cols = std[std > EPS].index.tolist()
    return merged[keep_cols]


def neutralize_factor(
    factor: pd.Series,
    *,
    exposures: pd.DataFrame,
    min_stocks: int = 10,
) -> pd.Series:
    if not isinstance(factor, pd.Series):
        raise TypeError("factor must be a pandas Series")
    if exposures.empty:
        return factor.astype(float).rename(factor.name)

    aligned_factor = factor.astype(float)
    aligned_exposures = exposures.reindex(aligned_factor.index)
    aligned_exposures = aligned_exposures.select_dtypes(include=[np.number]).copy()
    if aligned_exposures.empty:
        return aligned_factor.rename(factor.name)

    std = aligned_exposures.std(axis=0, skipna=True)
    keep_cols = std[std > EPS].index.tolist()
    aligned_exposures = aligned_exposures[keep_cols]
    if aligned_exposures.empty:
        return aligned_factor.rename(factor.name)

    aligned_exposures = (
        aligned_exposures.groupby(level=0, group_keys=False)
        .apply(lambda d: d.fillna(d.median(numeric_only=True)))
        .fillna(0.0)
    )
    residual = gram_schmidt_orthogonalize(
        aligned_factor,
        aligned_exposures,
        min_stocks=int(min_stocks),
    )
    return residual.rename(factor.name)


def style_exposure_diagnostics(
    factor: pd.Series,
    *,
    exposures: pd.DataFrame,
) -> dict[str, object]:
    if exposures.empty:
        return {
            "style_exposure_count": 0,
            "avg_abs_style_corr": np.nan,
            "max_abs_style_corr": np.nan,
            "top_style_exposure": "",
            "top_style_corr": np.nan,
        }

    aligned_factor = factor.astype(float)
    aligned_exposures = exposures.reindex(aligned_factor.index)
    aligned_exposures = aligned_exposures.select_dtypes(include=[np.number]).copy()
    if aligned_exposures.empty:
        return {
            "style_exposure_count": 0,
            "avg_abs_style_corr": np.nan,
            "max_abs_style_corr": np.nan,
            "top_style_exposure": "",
            "top_style_corr": np.nan,
        }

    std = aligned_exposures.std(axis=0, skipna=True)
    keep_cols = std[std > EPS].index.tolist()
    aligned_exposures = aligned_exposures[keep_cols]
    if aligned_exposures.empty:
        return {
            "style_exposure_count": 0,
            "avg_abs_style_corr": np.nan,
            "max_abs_style_corr": np.nan,
            "top_style_exposure": "",
            "top_style_corr": np.nan,
        }

    corr_series = aligned_exposures.corrwith(aligned_factor).dropna()
    if corr_series.empty:
        return {
            "style_exposure_count": int(aligned_exposures.shape[1]),
            "avg_abs_style_corr": np.nan,
            "max_abs_style_corr": np.nan,
            "top_style_exposure": "",
            "top_style_corr": np.nan,
        }
    corr_series = corr_series.astype(float)
    top_name = str(corr_series.abs().idxmax())
    top_corr = float(corr_series.loc[top_name])
    return {
        "style_exposure_count": int(aligned_exposures.shape[1]),
        "avg_abs_style_corr": float(corr_series.abs().mean()),
        "max_abs_style_corr": float(abs(top_corr)),
        "top_style_exposure": top_name,
        "top_style_corr": float(top_corr),
    }


def prepare_backtest_inputs(
    data: dict[str, pd.Series],
    *,
    horizon: int = 5,
    extra_style_exposures: pd.DataFrame | None = None,
) -> dict[str, pd.Series | pd.DataFrame]:
    label = make_forward_return(data["close"], horizon=horizon).rename("label")
    price_panel = build_price_panel(data)
    exposures = build_style_exposures(data, extra_exposures=extra_style_exposures)
    return {
        "label": label,
        "price_panel": price_panel,
        "exposures": exposures,
    }


def _align_backtest_inputs(
    factor: pd.Series,
    *,
    label: pd.Series,
    price_panel: pd.DataFrame,
    exposures: pd.DataFrame,
    factor_name: str,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    idx = factor.index.intersection(label.index)
    idx = idx.intersection(price_panel.index)
    idx = idx.intersection(exposures.index)
    aligned_factor = factor.reindex(idx).astype(float).rename(factor_name)
    aligned_label = label.reindex(idx).astype(float)
    aligned_price_panel = price_panel.reindex(idx)
    aligned_exposures = exposures.reindex(idx)
    return aligned_factor, aligned_label, aligned_price_panel, aligned_exposures


def _run_factor_backtest_on_aligned(
    factor: pd.Series,
    *,
    label: pd.Series,
    price_panel: pd.DataFrame,
    exposures: pd.DataFrame,
    factor_name: str,
    horizon: int,
    min_stocks: int,
    winsorize: bool,
    zscore: bool,
    pure_mode: str,
    rolling_window: int,
    n_groups: int,
    long_group: int | None,
    short_group: int | None,
    cost_bps: float,
    enable_alphalens: bool,
    alphalens_periods: tuple[int, ...],
    alphalens_quantiles: int,
    alphalens_max_loss: float,
) -> dict[str, object]:
    config = BacktestConfig(
        min_stocks=min_stocks,
        winsorize=winsorize,
        zscore=zscore,
        pure_mode=pure_mode,
        rolling_window=rolling_window,
        n_groups=n_groups,
        long_group=long_group,
        short_group=short_group,
        cost_bps=cost_bps,
        enable_alphalens=enable_alphalens,
        alphalens_periods=alphalens_periods,
        alphalens_quantiles=alphalens_quantiles,
        alphalens_max_loss=alphalens_max_loss,
    )
    metadata = {
        "factor_name": factor_name,
        "data_source": "factors_store",
        "eval_backend": "factors_store_bridge",
        "label_horizon": int(horizon),
    }
    return run_single_factor_backtest(
        factor=factor,
        label=label,
        exposures=exposures,
        price_panel=price_panel,
        metadata=metadata,
        config=config,
    )


def run_factor_backtest(
    factor: pd.Series,
    *,
    data: dict[str, pd.Series] | None = None,
    prepared: dict[str, pd.Series | pd.DataFrame] | None = None,
    factor_name: str,
    horizon: int = 5,
    min_stocks: int = 10,
    winsorize: bool = False,
    zscore: bool = False,
    pure_mode: str = "none",
    rolling_window: int = 60,
    n_groups: int = 5,
    long_group: int | None = None,
    short_group: int | None = None,
    cost_bps: float = 10.0,
    enable_alphalens: bool = False,
    alphalens_periods: tuple[int, ...] = (1, 5, 10),
    alphalens_quantiles: int = 5,
    alphalens_max_loss: float = 0.35,
) -> dict[str, object]:
    if prepared is None:
        if data is None:
            raise ValueError("provide either data or prepared backtest inputs")
        prepared = prepare_backtest_inputs(data, horizon=horizon)

    label = prepared["label"]
    price_panel = prepared["price_panel"]
    exposures = prepared["exposures"]
    if not isinstance(label, pd.Series) or not isinstance(price_panel, pd.DataFrame) or not isinstance(exposures, pd.DataFrame):
        raise TypeError("prepared backtest inputs have unexpected types")

    (
        aligned_factor,
        aligned_label,
        aligned_price_panel,
        aligned_exposures,
    ) = _align_backtest_inputs(
        factor,
        label=label,
        price_panel=price_panel,
        exposures=exposures,
        factor_name=factor_name,
    )
    return _run_factor_backtest_on_aligned(
        aligned_factor,
        label=aligned_label,
        price_panel=aligned_price_panel,
        exposures=aligned_exposures,
        factor_name=factor_name,
        horizon=horizon,
        min_stocks=min_stocks,
        winsorize=winsorize,
        zscore=zscore,
        pure_mode=pure_mode,
        rolling_window=rolling_window,
        n_groups=n_groups,
        long_group=long_group,
        short_group=short_group,
        cost_bps=cost_bps,
        enable_alphalens=enable_alphalens,
        alphalens_periods=alphalens_periods,
        alphalens_quantiles=alphalens_quantiles,
        alphalens_max_loss=alphalens_max_loss,
    )


def run_factor_backtest_dual(
    factor: pd.Series,
    *,
    data: dict[str, pd.Series] | None = None,
    prepared: dict[str, pd.Series | pd.DataFrame] | None = None,
    factor_name: str,
    horizon: int = 5,
    min_stocks: int = 10,
    winsorize: bool = False,
    zscore: bool = False,
    pure_mode: str = "none",
    rolling_window: int = 60,
    n_groups: int = 5,
    long_group: int | None = None,
    short_group: int | None = None,
    cost_bps: float = 10.0,
    enable_alphalens: bool = False,
    alphalens_periods: tuple[int, ...] = (1, 5, 10),
    alphalens_quantiles: int = 5,
    alphalens_max_loss: float = 0.35,
) -> dict[str, object]:
    if prepared is None:
        if data is None:
            raise ValueError("provide either data or prepared backtest inputs")
        prepared = prepare_backtest_inputs(data, horizon=horizon)

    exposures = prepared["exposures"]
    if not isinstance(exposures, pd.DataFrame):
        raise TypeError("prepared backtest inputs have unexpected exposure type")

    label = prepared["label"]
    price_panel = prepared["price_panel"]
    if not isinstance(label, pd.Series) or not isinstance(price_panel, pd.DataFrame):
        raise TypeError("prepared backtest inputs have unexpected types")

    (
        aligned_factor,
        aligned_label,
        aligned_price_panel,
        aligned_exposures,
    ) = _align_backtest_inputs(
        factor,
        label=label,
        price_panel=price_panel,
        exposures=exposures,
        factor_name=factor_name,
    )

    raw_result = _run_factor_backtest_on_aligned(
        aligned_factor,
        label=aligned_label,
        price_panel=aligned_price_panel,
        exposures=aligned_exposures,
        factor_name=factor_name,
        horizon=horizon,
        min_stocks=min_stocks,
        winsorize=winsorize,
        zscore=zscore,
        pure_mode=pure_mode,
        rolling_window=rolling_window,
        n_groups=n_groups,
        long_group=long_group,
        short_group=short_group,
        cost_bps=cost_bps,
        enable_alphalens=enable_alphalens,
        alphalens_periods=alphalens_periods,
        alphalens_quantiles=alphalens_quantiles,
        alphalens_max_loss=alphalens_max_loss,
    )

    diagnostics = style_exposure_diagnostics(aligned_factor, exposures=aligned_exposures)
    neutralization_status = "ok"
    neutralized_result: dict[str, object] | None = None
    neutralized_factor_nonnull = 0
    neutralized_factor_std = np.nan

    try:
        neutralized_factor = neutralize_factor(
            aligned_factor,
            exposures=aligned_exposures,
            min_stocks=min_stocks,
        )
        neutralized_factor = neutralized_factor.reindex(aligned_factor.index).astype(float).rename(factor_name)
        neutralized_factor_nonnull = int(neutralized_factor.notna().sum())
        neutralized_factor_std = float(neutralized_factor.std(skipna=True))
        if neutralized_factor_nonnull <= 0 or not np.isfinite(neutralized_factor_std) or neutralized_factor_std <= EPS:
            neutralization_status = "degenerate_residual"
        else:
            neutralized_result = _run_factor_backtest_on_aligned(
                neutralized_factor,
                label=aligned_label,
                price_panel=aligned_price_panel,
                exposures=aligned_exposures,
                factor_name=factor_name,
                horizon=horizon,
                min_stocks=min_stocks,
                winsorize=winsorize,
                zscore=zscore,
                pure_mode=pure_mode,
                rolling_window=rolling_window,
                n_groups=n_groups,
                long_group=long_group,
                short_group=short_group,
                cost_bps=cost_bps,
                enable_alphalens=enable_alphalens,
                alphalens_periods=alphalens_periods,
                alphalens_quantiles=alphalens_quantiles,
                alphalens_max_loss=alphalens_max_loss,
            )
    except Exception as exc:
        neutralization_status = f"failed: {exc}"

    return {
        "raw_result": raw_result,
        "neutralized_result": neutralized_result,
        "style_diagnostics": diagnostics,
        "neutralization_status": neutralization_status,
        "neutralized_factor_nonnull": neutralized_factor_nonnull,
        "neutralized_factor_std": neutralized_factor_std,
    }


def summarize_backtest_result(result: dict[str, object]) -> dict[str, object]:
    row = build_summary_row(result)
    tables = result.get("tables", {})
    if isinstance(tables, dict):
        factor = tables.get("factor")
        if isinstance(factor, pd.Series):
            row["nonnull"] = int(factor.notna().sum())
    return row


def run_factor_backtest_report(
    factor: pd.Series,
    *,
    data: dict[str, pd.Series] | None = None,
    prepared: dict[str, pd.Series | pd.DataFrame] | None = None,
    factor_name: str,
    out_dir: str | Path,
    out_prefix: str | None = None,
    horizon: int = 5,
    min_stocks: int = 10,
    winsorize: bool = False,
    zscore: bool = False,
    pure_mode: str = "none",
    rolling_window: int = 60,
    n_groups: int = 5,
    long_group: int | None = None,
    short_group: int | None = None,
    cost_bps: float = 10.0,
    enable_alphalens: bool = False,
    alphalens_periods: tuple[int, ...] = (1, 5, 10),
    alphalens_quantiles: int = 5,
    alphalens_max_loss: float = 0.35,
) -> dict[str, object]:
    result = run_factor_backtest(
        factor,
        data=data,
        prepared=prepared,
        factor_name=factor_name,
        horizon=horizon,
        min_stocks=min_stocks,
        winsorize=winsorize,
        zscore=zscore,
        pure_mode=pure_mode,
        rolling_window=rolling_window,
        n_groups=n_groups,
        long_group=long_group,
        short_group=short_group,
        cost_bps=cost_bps,
        enable_alphalens=enable_alphalens,
        alphalens_periods=alphalens_periods,
        alphalens_quantiles=alphalens_quantiles,
        alphalens_max_loss=alphalens_max_loss,
    )

    report_dir = write_backtest_report(
        result=result,
        out_dir=Path(out_dir),
        prefix=out_prefix or factor_name,
    )
    result["report_dir"] = str(report_dir)
    return result
