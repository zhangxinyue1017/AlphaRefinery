from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..quick_ic.evaluation import calc_grouped_returns, calc_ic, calc_icir, calc_rank_ic
from ..quick_ic.orthogonalize import gram_schmidt_orthogonalize


@dataclass
class BacktestConfig:
    min_stocks: int = 10
    winsorize: bool = False
    zscore: bool = False
    pure_mode: str = "partial"
    rolling_window: int = 60
    n_groups: int = 5
    long_group: int | None = None
    short_group: int | None = None
    cost_bps: float = 10.0
    enable_alphalens: bool = False
    alphalens_periods: Sequence[int] = field(default_factory=lambda: (1, 5, 10))
    alphalens_quantiles: int = 5
    alphalens_max_loss: float = 0.35


def winsorize_mad(
    factor: pd.Series,
    n_sigma: float = 3.0,
    eps: float = 1e-12,
) -> pd.Series:
    def _one_day(sub: pd.Series) -> pd.Series:
        sub = sub.replace([np.inf, -np.inf], np.nan)
        med = sub.median()
        mad = (sub - med).abs().median()
        if (not np.isfinite(mad)) or mad < eps:
            return sub
        half = n_sigma * 1.4826 * mad
        return sub.clip(lower=med - half, upper=med + half)

    return factor.groupby(level=0).transform(_one_day)


def zscore_cs(factor: pd.Series, eps: float = 1e-12) -> pd.Series:
    def _one_day(sub: pd.Series) -> pd.Series:
        sub = sub.replace([np.inf, -np.inf], np.nan)
        mu = sub.mean()
        sd = sub.std()
        if (not np.isfinite(sd)) or sd < eps:
            return sub - mu
        return (sub - mu) / sd

    return factor.groupby(level=0).transform(_one_day)


def period_summary(ic: pd.Series, freq: str) -> pd.DataFrame:
    clean = ic.dropna()
    if clean.empty:
        return pd.DataFrame(
            columns=["mean_ic", "std_ic", "icir", "positive_rate", "abs_mean_ic", "n_days"]
        )
    grouped = clean.groupby(clean.index.to_period(freq))
    out = pd.DataFrame(
        {
            "mean_ic": grouped.mean(),
            "std_ic": grouped.std(ddof=1),
            "positive_rate": grouped.apply(lambda x: float((x > 0).mean())),
            "abs_mean_ic": grouped.apply(lambda x: float(np.abs(x).mean())),
            "n_days": grouped.size().astype(int),
        }
    )
    out["icir"] = out["mean_ic"] / out["std_ic"]
    out.index = out.index.astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_summary(ic: pd.Series, window: int) -> pd.DataFrame:
    clean = ic.dropna().sort_index()
    if clean.empty:
        return pd.DataFrame(columns=["rolling_mean_ic", "rolling_icir", "rolling_positive_rate"])
    roll_mean = clean.rolling(window).mean()
    roll_std = clean.rolling(window).std(ddof=1)
    roll_pos = clean.gt(0).astype(float).rolling(window).mean()
    return pd.DataFrame(
        {
            "rolling_mean_ic": roll_mean,
            "rolling_icir": roll_mean / roll_std,
            "rolling_positive_rate": roll_pos,
        }
    ).replace([np.inf, -np.inf], np.nan)


def summarize_ic_profile(
    ic: pd.Series,
    rolling_window: int = 60,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean = ic.dropna()
    monthly = period_summary(clean, "M")
    yearly = period_summary(clean, "Y")
    rolling = rolling_summary(clean, rolling_window)

    mean_ic = float(clean.mean()) if len(clean) else float("nan")
    std_ic = float(clean.std(ddof=1)) if len(clean) >= 2 else float("nan")
    out = {
        "mean_ic": mean_ic,
        "abs_mean_ic": float(np.abs(clean).mean()) if len(clean) else float("nan"),
        "std_ic": std_ic,
        "icir": float(calc_icir(clean)),
        "icir_ann": float(calc_icir(clean, annualize=True)),
        "win_rate": float((clean > 0).mean()) if len(clean) else float("nan"),
        "ic_positive_rate": float((clean > 0).mean()) if len(clean) else float("nan"),
        "t_stat": float(mean_ic / (std_ic / np.sqrt(len(clean))))
        if len(clean) >= 2 and np.isfinite(std_ic) and std_ic > 1e-12
        else float("nan"),
        "n_days_total": float(ic.shape[0]),
        "n_days_valid": float(clean.shape[0]),
        "monthly_positive_rate": float((monthly["mean_ic"] > 0).mean()) if len(monthly) else float("nan"),
        "yearly_positive_rate": float((yearly["mean_ic"] > 0).mean()) if len(yearly) else float("nan"),
        "best_month_mean_ic": float(monthly["mean_ic"].max()) if len(monthly) else float("nan"),
        "worst_month_mean_ic": float(monthly["mean_ic"].min()) if len(monthly) else float("nan"),
    }
    return out, monthly, yearly, rolling


def summarize_return_series(ret: pd.Series) -> dict[str, float]:
    clean = ret.dropna()
    if clean.empty:
        return {
            "mean_daily_return": float("nan"),
            "vol_daily": float("nan"),
            "ann_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "n_days": 0.0,
        }
    nav = (1.0 + clean).cumprod()
    vol = float(clean.std(ddof=1)) if len(clean) >= 2 else float("nan")
    ann_return = float(nav.iloc[-1] ** (252.0 / len(clean)) - 1.0) if len(clean) else float("nan")
    sharpe = float(clean.mean() / vol * np.sqrt(252.0)) if np.isfinite(vol) and vol > 1e-12 else float("nan")
    drawdown = nav / nav.cummax() - 1.0
    return {
        "mean_daily_return": float(clean.mean()),
        "vol_daily": vol,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "n_days": float(len(clean)),
    }


def assign_quantile_groups(
    factor: pd.Series,
    n_groups: int = 5,
    min_stocks: int = 10,
) -> pd.Series:
    df = factor.to_frame("factor").dropna()

    def _one_day(sub: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=sub.index, dtype=float)
        if len(sub) < max(min_stocks, n_groups):
            return out
        try:
            groups = pd.qcut(sub["factor"], q=n_groups, labels=False, duplicates="drop")
        except ValueError:
            return out
        out.loc[sub.index] = groups.astype(float) + 1.0
        return out

    return df.groupby(level=0, group_keys=False).apply(_one_day).rename("group")


def compute_long_short_backtest(
    factor: pd.Series,
    forward_returns: pd.Series,
    *,
    n_groups: int = 5,
    min_stocks: int = 10,
    long_group: int | None = None,
    short_group: int | None = None,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    group_labels = [f"G{i}" for i in range(1, n_groups + 1)]
    long_group = long_group or n_groups
    short_group = short_group or 1
    long_label = f"G{long_group}"
    short_label = f"G{short_group}"

    df = pd.DataFrame({"factor": factor, "ret": forward_returns}).dropna()
    if df.empty:
        empty_group_returns = pd.DataFrame(columns=group_labels, dtype=float)
        empty_group_returns.index.name = "datetime"
        empty_long_short = pd.DataFrame(
            columns=["long_short_gross", "turnover", "cost", "long_short_net", "gross_nav", "net_nav"],
            dtype=float,
        )
        empty_long_short.index.name = "datetime"
        empty_long_only = pd.DataFrame(
            columns=["long_only_gross", "turnover", "cost", "long_only_net", "gross_nav", "net_nav"],
            dtype=float,
        )
        empty_long_only.index.name = "datetime"
        empty_labels = pd.Series(dtype=object, name="group")
        empty_labels.index = pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"])
        return empty_group_returns, empty_long_short, empty_long_only, empty_labels

    daily_group_returns: list[pd.Series] = []
    daily_stats: list[dict[str, float | str]] = []
    long_only_daily_stats: list[dict[str, float | str]] = []
    prev_weights = pd.Series(dtype=float)
    prev_long_only_weights = pd.Series(dtype=float)

    for dt, sub in df.groupby(level=0):
        row = {"datetime": dt}
        long_only_row = {"datetime": dt}
        group_ret = pd.Series(np.nan, index=group_labels, dtype=float)
        turnover = np.nan
        gross = np.nan
        long_only_turnover = np.nan
        long_only_gross = np.nan

        if len(sub) >= max(min_stocks, n_groups):
            try:
                group_ids = pd.qcut(sub["factor"], q=n_groups, labels=group_labels, duplicates="drop")
            except ValueError:
                group_ids = None
            if group_ids is not None:
                grouped = sub["ret"].groupby(group_ids, observed=True).mean()
                group_ret.loc[grouped.index.astype(str)] = grouped.astype(float)

                instrument_index = sub.index.get_level_values("instrument")
                weights = pd.Series(0.0, index=instrument_index, dtype=float)
                long_idx = group_ids[group_ids == long_label].index
                short_idx = group_ids[group_ids == short_label].index
                long_only_weights = pd.Series(0.0, index=instrument_index, dtype=float)
                if len(long_idx):
                    long_instruments = long_idx.get_level_values("instrument")
                    weights.loc[long_instruments] = 1.0 / len(long_idx)
                    long_only_weights.loc[long_instruments] = 1.0 / len(long_idx)
                if len(short_idx):
                    short_instruments = short_idx.get_level_values("instrument")
                    weights.loc[short_instruments] = -1.0 / len(short_idx)
                sub_returns = sub["ret"].copy()
                sub_returns.index = instrument_index
                gross = float((weights * sub_returns).sum()) if weights.abs().sum() > 0 else np.nan
                long_only_gross = (
                    float((long_only_weights * sub_returns).sum()) if long_only_weights.abs().sum() > 0 else np.nan
                )
                if prev_weights.empty:
                    turnover = 0.0
                else:
                    union = prev_weights.index.union(weights.index)
                    turnover = 0.5 * float(
                        (prev_weights.reindex(union, fill_value=0.0) - weights.reindex(union, fill_value=0.0))
                        .abs()
                        .sum()
                    )
                prev_weights = weights
                if prev_long_only_weights.empty:
                    long_only_turnover = 0.0
                else:
                    long_only_union = prev_long_only_weights.index.union(long_only_weights.index)
                    long_only_turnover = 0.5 * float(
                        (
                            prev_long_only_weights.reindex(long_only_union, fill_value=0.0)
                            - long_only_weights.reindex(long_only_union, fill_value=0.0)
                        )
                        .abs()
                        .sum()
                    )
                prev_long_only_weights = long_only_weights

        cost = turnover * cost_bps / 10000.0 if np.isfinite(turnover) else np.nan
        net = gross - cost if np.isfinite(gross) and np.isfinite(cost) else gross
        long_only_cost = long_only_turnover * cost_bps / 10000.0 if np.isfinite(long_only_turnover) else np.nan
        long_only_net = (
            long_only_gross - long_only_cost
            if np.isfinite(long_only_gross) and np.isfinite(long_only_cost)
            else long_only_gross
        )
        row.update(
            {
                "long_short_gross": gross,
                "turnover": turnover,
                "cost": cost,
                "long_short_net": net,
            }
        )
        long_only_row.update(
            {
                "long_only_gross": long_only_gross,
                "turnover": long_only_turnover,
                "cost": long_only_cost,
                "long_only_net": long_only_net,
            }
        )
        daily_group_returns.append(group_ret.rename(dt))
        daily_stats.append(row)
        long_only_daily_stats.append(long_only_row)

    group_returns = pd.DataFrame(daily_group_returns)
    group_returns.index.name = "datetime"
    long_short_daily = pd.DataFrame(daily_stats).set_index("datetime").sort_index()
    long_short_daily["gross_nav"] = (1.0 + long_short_daily["long_short_gross"].fillna(0.0)).cumprod()
    long_short_daily["net_nav"] = (1.0 + long_short_daily["long_short_net"].fillna(0.0)).cumprod()
    long_only_daily = pd.DataFrame(long_only_daily_stats).set_index("datetime").sort_index()
    long_only_daily["gross_nav"] = (1.0 + long_only_daily["long_only_gross"].fillna(0.0)).cumprod()
    long_only_daily["net_nav"] = (1.0 + long_only_daily["long_only_net"].fillna(0.0)).cumprod()

    grouped_returns = calc_grouped_returns(
        factor=factor,
        forward_ret=forward_returns,
        n_groups=n_groups,
        min_stocks=min_stocks,
    )
    grouped_returns.columns = group_labels
    return (
        group_returns,
        long_short_daily,
        long_only_daily,
        assign_quantile_groups(factor, n_groups=n_groups, min_stocks=min_stocks),
    )


def summarize_group_backtest(
    group_returns: pd.DataFrame,
    long_short_daily: pd.DataFrame,
    long_only_daily: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    def _positive_rate(series: pd.Series) -> float:
        clean = series.dropna()
        return float((clean > 0).mean()) if len(clean) else float("nan")

    def _period_positive_rate(series: pd.Series, freq: str) -> float:
        clean = series.dropna()
        if clean.empty:
            return float("nan")
        grouped = clean.groupby(clean.index.to_period(freq)).mean()
        return float((grouped > 0).mean()) if len(grouped) else float("nan")

    group_nav = (1.0 + group_returns.fillna(0.0)).cumprod()
    group_mean = group_returns.mean()
    best_group = str(group_mean.idxmax()) if len(group_mean) else ""
    worst_group = str(group_mean.idxmin()) if len(group_mean) else ""

    benchmark_series = None
    if benchmark_returns is not None:
        benchmark_series = benchmark_returns.reindex(long_short_daily.index).astype(float)
        benchmark_series = benchmark_series.replace([np.inf, -np.inf], np.nan).rename("benchmark_return")
        long_short_daily["benchmark_return"] = benchmark_series
        long_short_daily["gross_excess"] = long_short_daily["long_short_gross"] - benchmark_series
        long_short_daily["net_excess"] = long_short_daily["long_short_net"] - benchmark_series
        long_only_daily["benchmark_return"] = benchmark_series
        long_only_daily["gross_excess"] = long_only_daily["long_only_gross"] - benchmark_series
        long_only_daily["net_excess"] = long_only_daily["long_only_net"] - benchmark_series

    gross = summarize_return_series(long_short_daily["long_short_gross"])
    net = summarize_return_series(long_short_daily["long_short_net"])
    long_only_gross = summarize_return_series(long_only_daily["long_only_gross"])
    long_only_net = summarize_return_series(long_only_daily["long_only_net"])
    benchmark = summarize_return_series(benchmark_series) if benchmark_series is not None else {}
    gross_excess = summarize_return_series(long_short_daily["gross_excess"]) if "gross_excess" in long_short_daily else {}
    net_excess = summarize_return_series(long_short_daily["net_excess"]) if "net_excess" in long_short_daily else {}
    long_only_gross_excess = (
        summarize_return_series(long_only_daily["gross_excess"]) if "gross_excess" in long_only_daily else {}
    )
    long_only_net_excess = summarize_return_series(long_only_daily["net_excess"]) if "net_excess" in long_only_daily else {}
    long_group = str(group_returns.columns[-1]) if len(group_returns.columns) else ""
    summary = {
        "top_group_mean_return": float(group_mean.max()) if len(group_mean) else float("nan"),
        "bottom_group_mean_return": float(group_mean.min()) if len(group_mean) else float("nan"),
        "best_group": best_group,
        "worst_group": worst_group,
        "long_only_group": long_group,
        "long_only_win_rate": _positive_rate(long_only_daily["long_only_net"]),
        "long_only_monthly_win_rate": _period_positive_rate(long_only_daily["long_only_net"], "M"),
        "long_only_yearly_win_rate": _period_positive_rate(long_only_daily["long_only_net"], "Y"),
        "long_only_mean_turnover": float(long_only_daily["turnover"].dropna().mean())
        if long_only_daily["turnover"].notna().any()
        else float("nan"),
        "long_only_mean_cost_bps": float(long_only_daily["cost"].dropna().mean() * 10000.0)
        if long_only_daily["cost"].notna().any()
        else float("nan"),
        "long_only_gross_ann_return": long_only_gross["ann_return"],
        "long_only_gross_sharpe": long_only_gross["sharpe"],
        "long_only_gross_max_drawdown": long_only_gross["max_drawdown"],
        "long_only_net_ann_return": long_only_net["ann_return"],
        "long_only_net_sharpe": long_only_net["sharpe"],
        "long_only_net_max_drawdown": long_only_net["max_drawdown"],
        "gross_win_rate": _positive_rate(long_short_daily["long_short_gross"]),
        "net_win_rate": _positive_rate(long_short_daily["long_short_net"]),
        "net_monthly_win_rate": _period_positive_rate(long_short_daily["long_short_net"], "M"),
        "net_yearly_win_rate": _period_positive_rate(long_short_daily["long_short_net"], "Y"),
        "mean_turnover": float(long_short_daily["turnover"].dropna().mean())
        if long_short_daily["turnover"].notna().any()
        else float("nan"),
        "mean_cost_bps": float(long_short_daily["cost"].dropna().mean() * 10000.0)
        if long_short_daily["cost"].notna().any()
        else float("nan"),
        "gross_ann_return": gross["ann_return"],
        "gross_sharpe": gross["sharpe"],
        "gross_max_drawdown": gross["max_drawdown"],
        "net_ann_return": net["ann_return"],
        "net_sharpe": net["sharpe"],
        "net_max_drawdown": net["max_drawdown"],
        "benchmark_ann_return": benchmark.get("ann_return", float("nan")),
        "benchmark_mean_daily_return": benchmark.get("mean_daily_return", float("nan")),
        "long_only_gross_excess_ann_return": long_only_gross_excess.get("ann_return", float("nan")),
        "long_only_net_excess_ann_return": long_only_net_excess.get("ann_return", float("nan")),
        "long_only_gross_excess_mean_daily_return": long_only_gross_excess.get("mean_daily_return", float("nan")),
        "long_only_net_excess_mean_daily_return": long_only_net_excess.get("mean_daily_return", float("nan")),
        "gross_excess_ann_return": gross_excess.get("ann_return", float("nan")),
        "net_excess_ann_return": net_excess.get("ann_return", float("nan")),
        "gross_excess_mean_daily_return": gross_excess.get("mean_daily_return", float("nan")),
        "net_excess_mean_daily_return": net_excess.get("mean_daily_return", float("nan")),
    }
    return summary, group_nav


def run_alphalens_analysis(
    factor: pd.Series,
    price_panel: pd.DataFrame,
    periods: Sequence[int] = (1, 5, 10),
    quantiles: int = 5,
    max_loss: float = 0.35,
    make_tearsheet: bool = False,
) -> dict[str, Any]:
    try:
        import alphalens.utils as alphalens_utils
        from alphalens.performance import factor_information_coefficient
        from alphalens.tears import create_full_tear_sheet
        from alphalens.utils import get_clean_factor_and_forward_returns
    except Exception as exc:
        return {"status": "skipped", "reason": f"alphalens not available: {exc}"}

    class _LegacyModeResult:
        def __init__(self, value: float, count: int = 1) -> None:
            self.mode = np.array([value])
            self.count = np.array([count])

    def _legacy_mode_compatible(
        values: Sequence[int | float],
        axis: int | None = None,
        nan_policy: str = "propagate",
        keepdims: bool = False,
    ) -> _LegacyModeResult:
        del axis, nan_policy, keepdims
        arr = np.asarray(list(values))
        if arr.size == 0:
            return _LegacyModeResult(0)
        mode_values = pd.Series(arr).mode(dropna=False)
        if mode_values.empty:
            return _LegacyModeResult(arr[0], count=1)
        mode_value = mode_values.iloc[0]
        count = int((pd.Series(arr) == mode_value).sum())
        return _LegacyModeResult(mode_value, count=count)

    def _quantize_factor_compatible(
        factor_data: pd.DataFrame,
        quantiles: int | Sequence[float] | None = 5,
        bins: int | Sequence[float] | None = None,
        by_group: bool = False,
        no_raise: bool = False,
        zero_aware: bool = False,
    ) -> pd.Series:
        if not ((quantiles is not None and bins is None) or (quantiles is None and bins is not None)):
            raise ValueError("Either quantiles or bins should be provided")
        if zero_aware and not (isinstance(quantiles, int) or isinstance(bins, int)):
            raise ValueError("zero_aware should only be True when quantiles or bins is an integer")

        def quantile_calc(x, _quantiles, _bins, _zero_aware, _no_raise):
            try:
                if _quantiles is not None and _bins is None and not _zero_aware:
                    return pd.qcut(x, _quantiles, labels=False, duplicates="drop") + 1
                if _quantiles is not None and _bins is None and _zero_aware:
                    pos_quantiles = pd.qcut(
                        x[x >= 0], _quantiles // 2, labels=False, duplicates="drop"
                    ) + _quantiles // 2 + 1
                    neg_quantiles = pd.qcut(
                        x[x < 0], _quantiles // 2, labels=False, duplicates="drop"
                    ) + 1
                    return pd.concat([pos_quantiles, neg_quantiles]).sort_index()
                if _bins is not None and _quantiles is None and not _zero_aware:
                    return pd.cut(x, _bins, labels=False) + 1
                if _bins is not None and _quantiles is None and _zero_aware:
                    pos_bins = pd.cut(x[x >= 0], _bins // 2, labels=False) + _bins // 2 + 1
                    neg_bins = pd.cut(x[x < 0], _bins // 2, labels=False) + 1
                    return pd.concat([pos_bins, neg_bins]).sort_index()
            except Exception:
                if _no_raise:
                    return pd.Series(index=x.index, dtype=float)
                raise
            return pd.Series(index=x.index, dtype=float)

        grouper = [factor_data.index.get_level_values("date")]
        if by_group:
            grouper.append("group")

        factor_quantile = (
            factor_data.groupby(grouper, group_keys=False)["factor"]
            .apply(quantile_calc, quantiles, bins, zero_aware, no_raise)
            .dropna()
        )
        if isinstance(factor_quantile.index, pd.MultiIndex) and factor_quantile.index.nlevels > 2:
            factor_quantile.index = factor_quantile.index.droplevel(list(range(factor_quantile.index.nlevels - 2)))
        factor_quantile.name = "factor_quantile"
        return factor_quantile

    alphalens_utils.mode = _legacy_mode_compatible
    alphalens_utils.quantize_factor = _quantize_factor_compatible

    idx = factor.index.intersection(price_panel.index)
    factor = factor.reindex(idx).replace([np.inf, -np.inf], np.nan)
    close_s = price_panel.reindex(idx)["close"].astype(float).replace([np.inf, -np.inf], np.nan)
    prices_wide = close_s.unstack(level=1).sort_index()

    factor_al = factor.copy()
    factor_al.index = factor_al.index.set_names(["date", "asset"])
    prices_wide.index.name = "date"
    prices_wide.columns.name = "asset"

    clean_factor_data = get_clean_factor_and_forward_returns(
        factor=factor_al,
        prices=prices_wide,
        periods=list(periods),
        quantiles=quantiles,
        max_loss=max_loss,
    )
    ic_frame = factor_information_coefficient(clean_factor_data)
    metrics = {
        "status": "ok",
        "periods": list(periods),
        "quantiles": int(quantiles),
        "max_loss": float(max_loss),
        "n_rows_clean_factor": int(clean_factor_data.shape[0]),
        "ic_mean": {},
        "ic_ir": {},
    }
    for col in ic_frame.columns:
        mu = float(ic_frame[col].mean())
        sd = float(ic_frame[col].std(ddof=1))
        metrics["ic_mean"][str(col)] = mu
        metrics["ic_ir"][str(col)] = float(mu / sd) if np.isfinite(sd) and sd > 1e-12 else float("nan")
    if make_tearsheet:
        create_full_tear_sheet(clean_factor_data)
    return {
        "metrics": metrics,
        "tables": {
            "alphalens_ic": ic_frame,
            "alphalens_clean_factor": clean_factor_data,
        },
    }


def run_single_factor_backtest(
    *,
    factor: pd.Series,
    label: pd.Series,
    exposures: pd.DataFrame,
    price_panel: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BacktestConfig()
    meta = dict(metadata or {})

    idx = factor.index.intersection(label.index)
    factor = factor.reindex(idx).replace([np.inf, -np.inf], np.nan)
    label = label.reindex(idx).replace([np.inf, -np.inf], np.nan)
    exposures = exposures.reindex(idx)
    price_panel = price_panel.reindex(idx)

    if cfg.winsorize:
        factor = winsorize_mad(factor)
    if cfg.zscore:
        factor = zscore_cs(factor)

    valid_per_day = factor.groupby(level=0).count()
    coverage_metrics = {
        "n_rows": float(len(factor)),
        "valid_ratio": float(factor.notna().mean()),
        "median_daily_valid_count": float(valid_per_day.median()) if len(valid_per_day) else float("nan"),
        "min_daily_valid_count": float(valid_per_day.min()) if len(valid_per_day) else float("nan"),
        "max_daily_valid_count": float(valid_per_day.max()) if len(valid_per_day) else float("nan"),
    }

    quick_ic_series = calc_ic(factor, label, min_stocks=cfg.min_stocks)
    quick_metrics, monthly_quick, yearly_quick, rolling_quick = summarize_ic_profile(
        quick_ic_series, rolling_window=cfg.rolling_window
    )
    quick_rank_ic_series = calc_rank_ic(factor, label, min_stocks=cfg.min_stocks)
    quick_rank_metrics, monthly_quick_rank, yearly_quick_rank, rolling_quick_rank = summarize_ic_profile(
        quick_rank_ic_series, rolling_window=cfg.rolling_window
    )

    tables: dict[str, pd.Series | pd.DataFrame] = {
        "factor": factor,
        "quick_ic_series": quick_ic_series.to_frame("ic"),
        "quick_ic_monthly": monthly_quick,
        "quick_ic_yearly": yearly_quick,
        "quick_ic_rolling": rolling_quick,
        "quick_rank_ic_series": quick_rank_ic_series.to_frame("rank_ic"),
        "quick_rank_ic_monthly": monthly_quick_rank,
        "quick_rank_ic_yearly": yearly_quick_rank,
        "quick_rank_ic_rolling": rolling_quick_rank,
    }
    metrics: dict[str, Any] = {
        **meta,
        "coverage": coverage_metrics,
        "quick_ic": quick_metrics,
        "quick_rank_ic": quick_rank_metrics,
    }

    if cfg.pure_mode != "none":
        bases = exposures.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        bases = bases.dropna(axis=1, how="all")
        if bases.shape[1] == 0:
            raise ValueError("pure ic requested but no numeric exposures are available")
        factor_resid = gram_schmidt_orthogonalize(factor, bases, min_stocks=cfg.min_stocks)
        if cfg.pure_mode == "partial":
            label_resid = gram_schmidt_orthogonalize(label, bases, min_stocks=cfg.min_stocks)
            pure_ic_series = calc_ic(factor_resid, label_resid, min_stocks=cfg.min_stocks)
            pure_rank_ic_series = calc_rank_ic(factor_resid, label_resid, min_stocks=cfg.min_stocks)
        else:
            pure_ic_series = calc_ic(factor_resid, label, min_stocks=cfg.min_stocks)
            pure_rank_ic_series = calc_rank_ic(factor_resid, label, min_stocks=cfg.min_stocks)
        pure_metrics, monthly_pure, yearly_pure, rolling_pure = summarize_ic_profile(
            pure_ic_series, rolling_window=cfg.rolling_window
        )
        pure_rank_metrics, monthly_pure_rank, yearly_pure_rank, rolling_pure_rank = summarize_ic_profile(
            pure_rank_ic_series, rolling_window=cfg.rolling_window
        )
        pure_metrics["mode"] = cfg.pure_mode
        pure_metrics["n_bases"] = int(bases.shape[1])
        metrics["pure_ic"] = pure_metrics
        pure_rank_metrics["mode"] = cfg.pure_mode
        pure_rank_metrics["n_bases"] = int(bases.shape[1])
        metrics["pure_rank_ic"] = pure_rank_metrics
        tables.update(
            {
                "pure_ic_series": pure_ic_series.to_frame("pure_ic"),
                "pure_ic_monthly": monthly_pure,
                "pure_ic_yearly": yearly_pure,
                "pure_ic_rolling": rolling_pure,
                "pure_rank_ic_series": pure_rank_ic_series.to_frame("pure_rank_ic"),
                "pure_rank_ic_monthly": monthly_pure_rank,
                "pure_rank_ic_yearly": yearly_pure_rank,
                "pure_rank_ic_rolling": rolling_pure_rank,
            }
        )

    group_returns, long_short_daily, long_only_daily, group_labels = compute_long_short_backtest(
        factor=factor,
        forward_returns=label,
        n_groups=cfg.n_groups,
        min_stocks=cfg.min_stocks,
        long_group=cfg.long_group,
        short_group=cfg.short_group,
        cost_bps=cfg.cost_bps,
    )
    benchmark_returns = label.groupby(level=0).mean().rename("benchmark_return")
    group_metrics, group_nav = summarize_group_backtest(
        group_returns,
        long_short_daily,
        long_only_daily,
        benchmark_returns=benchmark_returns,
    )
    metrics["group_backtest"] = group_metrics
    tables.update(
        {
            "group_returns": group_returns,
            "group_nav": group_nav,
            "long_short_daily": long_short_daily,
            "long_only_daily": long_only_daily,
            "group_labels": group_labels.to_frame("group"),
        }
    )

    if cfg.enable_alphalens:
        alphalens_result = run_alphalens_analysis(
            factor=factor,
            price_panel=price_panel,
            periods=cfg.alphalens_periods,
            quantiles=cfg.alphalens_quantiles,
            max_loss=cfg.alphalens_max_loss,
            make_tearsheet=False,
        )
        if "metrics" in alphalens_result:
            metrics["alphalens"] = alphalens_result["metrics"]
            tables.update(alphalens_result["tables"])
        else:
            metrics["alphalens"] = alphalens_result
    else:
        metrics["alphalens"] = {"status": "disabled"}

    return {"metrics": metrics, "tables": tables}
