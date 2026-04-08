"""
B3 评估框架 — IC 计算核心模块

提供：
    calc_ic              每日截面 Pearson IC 序列
    calc_rank_ic         每日截面 Spearman RankIC 序列
    calc_icir            ICIR()IC均值 / IC标准差)
    calc_grouped_returns 分组平均收益()IC分组回测)

数据约定：
    所有 factor / label / forward_ret 均为
    pd.Series,MultiIndex = (datetime, instrument)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_STOCKS_DEFAULT = 5


def calc_ic(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS_DEFAULT,
) -> pd.Series:
    """每日截面 Pearson IC。"""
    if not isinstance(factor.index, pd.MultiIndex):
        raise TypeError("factor 须为 MultiIndex (datetime, instrument)")
    if not isinstance(label.index, pd.MultiIndex):
        raise TypeError("label 须为 MultiIndex (datetime, instrument)")

    df = pd.DataFrame({"f": factor, "y": label}).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="ic")

    def _one_day(sub: pd.DataFrame) -> float:
        if len(sub) < min_stocks:
            return np.nan
        if sub["f"].nunique(dropna=True) < 2 or sub["y"].nunique(dropna=True) < 2:
            return np.nan
        return float(sub["f"].corr(sub["y"], method="pearson"))

    return df.groupby(level=0).apply(_one_day).rename("ic")


def calc_rank_ic(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS_DEFAULT,
) -> pd.Series:
    """每日截面 Spearman Rank IC。

    Parameters
    ----------
    factor : pd.Series
        MultiIndex (datetime, instrument),因子值。
    label : pd.Series
        MultiIndex (datetime, instrument),前瞻收益标签。
    min_stocks : int
        每日有效股票数阈值,低于此值返回 NaN。

    Returns
    -------
    pd.Series
        Index = datetime,值为当日 RankIC (float),数据不足时为 NaN。
    """
    if not isinstance(factor.index, pd.MultiIndex):
        raise TypeError("factor 须为 MultiIndex (datetime, instrument)")
    if not isinstance(label.index, pd.MultiIndex):
        raise TypeError("label 须为 MultiIndex (datetime, instrument)")

    df = pd.DataFrame({"f": factor, "y": label}).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="rank_ic")

    def _one_day(sub: pd.DataFrame) -> float:
        if len(sub) < min_stocks:
            return np.nan
        if sub["f"].nunique(dropna=True) < 2 or sub["y"].nunique(dropna=True) < 2:
            return np.nan
        return float(sub["f"].corr(sub["y"], method="spearman"))

    return df.groupby(level=0).apply(_one_day).rename("rank_ic")


def calc_icir(
    ic_series: pd.Series,
    annualize: bool = False,
    trading_days: int = 252,
) -> float:
    """IC 信息比率 ICIR = mean(IC) / std(IC)。

    Parameters
    ----------
    ic_series : pd.Series
        calc_rank_ic 的输出序列。
    annualize : bool
        是否年化()乘以 sqrt(trading_days)),默认 False。
    trading_days : int
        年化基准交易日数,仅在 annualize=True 时生效。

    Returns
    -------
    float
        ICIR,数据不足或方差为零时返回 NaN。
    """
    clean = ic_series.dropna()
    if len(clean) < 2:
        return float("nan")
    mu = float(clean.mean())
    sd = float(clean.std(ddof=1))
    if sd < 1e-10:
        return float("nan")
    icir = mu / sd
    if annualize:
        icir *= np.sqrt(trading_days)
    return icir


def calc_grouped_returns(
    factor: pd.Series,
    forward_ret: pd.Series,
    n_groups: int = 5,
    min_stocks: int = _MIN_STOCKS_DEFAULT,
) -> pd.DataFrame:
    """每日分组平均收益()分位数分组)。

    Parameters
    ----------
    factor : pd.Series
        MultiIndex (datetime, instrument),因子值()用于分组)。
    forward_ret : pd.Series
        MultiIndex (datetime, instrument),前瞻收益。
    n_groups : int
        分组数,默认 5()G1 最低,G{n} 最高)。
    min_stocks : int
        每日最低有效股票数,低于此值跳过当天。

    Returns
    -------
    pd.DataFrame
        Index = datetime,columns = ['G1', ..., 'G{n_groups}'],
        值为对应组的等权平均前瞻收益。
    """
    group_labels = [f"G{i + 1}" for i in range(n_groups)]
    df = pd.DataFrame({"f": factor, "r": forward_ret}).dropna()

    def _one_day(sub: pd.DataFrame) -> pd.Series:
        if len(sub) < n_groups:
            return pd.Series({lb: np.nan for lb in group_labels})
        try:
            q = pd.qcut(sub["f"], q=n_groups, labels=group_labels, duplicates="drop")
        except ValueError:
            return pd.Series({lb: np.nan for lb in group_labels})
        return sub["r"].groupby(q, observed=True).mean().reindex(group_labels)

    return df.groupby(level=0).apply(_one_day)


def _summarize_ic_series(ic_series: pd.Series) -> dict[str, float]:
    clean = ic_series.dropna()
    mean_ic = float(clean.mean()) if len(clean) else float("nan")
    std_ic = float(clean.std(ddof=1)) if len(clean) >= 2 else float("nan")
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": calc_icir(ic_series),
        "icir_ann": calc_icir(ic_series, annualize=True),
        "win_rate": float((clean > 0).mean()) if len(clean) else float("nan"),
        "ic_positive_rate": float((clean > 0).mean()) if len(clean) else float("nan"),
    }


def summarize_ic(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS_DEFAULT,
) -> dict[str, float]:
    """一次性计算 Pearson IC 与 RankIC 汇总指标。

    Returns
    -------
    dict with keys:
        ic_mean, ic_std, icir, icir_ann, ic_win_rate,
        rank_ic_mean, rank_ic_std, rank_icir, rank_icir_ann, rank_ic_win_rate
    """
    ic = calc_ic(factor, label, min_stocks=min_stocks)
    rank_ic = calc_rank_ic(factor, label, min_stocks=min_stocks)
    ic_summary = _summarize_ic_series(ic)
    rank_summary = _summarize_ic_series(rank_ic)
    return {
        "ic_mean": ic_summary["mean_ic"],
        "ic_std": ic_summary["std_ic"],
        "icir": ic_summary["icir"],
        "icir_ann": ic_summary["icir_ann"],
        "ic_win_rate": ic_summary["win_rate"],
        "rank_ic_mean": rank_summary["mean_ic"],
        "rank_ic_std": rank_summary["std_ic"],
        "rank_icir": rank_summary["icir"],
        "rank_icir_ann": rank_summary["icir_ann"],
        "rank_ic_win_rate": rank_summary["win_rate"],
    }


def evaluate_ic(
    factor: pd.Series,
    label: pd.Series,
    *,
    n_groups: int = 5,
    min_stocks: int = _MIN_STOCKS_DEFAULT,
) -> dict[str, pd.Series | pd.DataFrame | float]:
    """一次性计算 IC / RankIC 汇总、序列和分组收益。"""
    ic_series = calc_ic(factor, label, min_stocks=min_stocks)
    rank_ic_series = calc_rank_ic(factor, label, min_stocks=min_stocks)
    grouped = calc_grouped_returns(
        factor,
        label,
        n_groups=n_groups,
        min_stocks=min_stocks,
    )
    ic_summary = _summarize_ic_series(ic_series)
    rank_summary = _summarize_ic_series(rank_ic_series)
    return {
        "ic_mean": ic_summary["mean_ic"],
        "ic_std": ic_summary["std_ic"],
        "icir": ic_summary["icir"],
        "icir_ann": ic_summary["icir_ann"],
        "ic_win_rate": ic_summary["win_rate"],
        "rank_ic_mean": rank_summary["mean_ic"],
        "rank_ic_std": rank_summary["std_ic"],
        "rank_icir": rank_summary["icir"],
        "rank_icir_ann": rank_summary["icir_ann"],
        "rank_ic_win_rate": rank_summary["win_rate"],
        "ic_series": ic_series,
        "rank_ic_series": rank_ic_series,
        "grouped_returns": grouped,
        "cumulative_grouped_returns": (1 + grouped.fillna(0)).cumprod() - 1,
    }
