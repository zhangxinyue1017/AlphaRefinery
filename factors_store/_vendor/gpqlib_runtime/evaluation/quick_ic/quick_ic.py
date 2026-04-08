"""
B3 quick_ic 主接口模块

这是 A 组遗传规划适应度函数的核心调用入口。

使用方式：
    from quick_ic import quick_ic
    score = quick_ic(factor_series, label_series)

提供：
    quick_ic      均值 RankIC,GP 适应度函数的基础评估指标
    quick_icir    非年化 ICIR,对因子稳定性有更高要求时使用
    quick_icir_ann 年化 ICIR,用于需要可比年化尺度时使用
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import calc_icir, calc_rank_ic

_MIN_STOCKS = 5
_MIN_DAYS = 3


def quick_ic(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS,
    min_days: int = _MIN_DAYS,
) -> float:
    """计算因子与标签之间的均值 RankIC。

    设计原则：
    - 快速、无副作用,适合在遗传规划循环中高频调用
    - 任何异常()NaN、形状不匹配、计算错误)均安全返回 0.0,不抛出异常
    - 有效交易日不足 min_days 时返回 0.0()避免小样本噪声影响进化方向）

    Parameters
    ----------
    factor : pd.Series
        MultiIndex (datetime, instrument),因子值面板。
        可直接传入遗传规划表达式树的求值结果。
    label : pd.Series
        MultiIndex (datetime, instrument),前瞻收益标签。
        推荐使用 label.make_forward_return 生成。
    min_stocks : int
        每日截面最少有效股票数,默认 5。
    min_days : int
        最少有效交易日数,默认 3。

    Returns
    -------
    float
        均值 RankIC,范围约 [-1, 1]。数据不足或异常时返回 0.0。
    """
    try:
        ic_series = calc_rank_ic(factor, label, min_stocks=min_stocks)
        valid = ic_series.dropna()
        if len(valid) < min_days:
            return 0.0
        return float(valid.mean())
    except Exception:
        return 0.0


def quick_icir(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS,
    min_days: int = 10,
    annualize: bool = False,
) -> float:
    """计算因子与标签之间的 ICIR()IC 均值 / IC 标准差）。

    比 quick_ic 对因子稳定性要求更高,适合 Step 3 大规模筛选时使用。

    Parameters
    ----------
    factor : pd.Series
        MultiIndex (datetime, instrument),因子值面板。
    label : pd.Series
        MultiIndex (datetime, instrument),前瞻收益标签。
    min_stocks : int
        每日截面最少有效股票数。
    min_days : int
        最少有效交易日数,默认 10()ICIR 需要足够样本估计方差）。
    annualize : bool
        是否年化()x sqrt(252)),默认 True。

    Returns
    -------
    float
        ICIR()可选年化）。数据不足或异常时返回 0.0。
    """
    try:
        ic_series = calc_rank_ic(factor, label, min_stocks=min_stocks)
        valid = ic_series.dropna()
        if len(valid) < min_days:
            return 0.0
        value = calc_icir(valid, annualize=annualize)
        return float(value) if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def quick_icir_ann(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS,
    min_days: int = 10,
) -> float:
    """计算因子与标签之间的年化 ICIR。"""
    return quick_icir(
        factor,
        label,
        min_stocks=min_stocks,
        min_days=min_days,
        annualize=True,
    )

def quick_mi_fast(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS,
    min_days: int = _MIN_DAYS,
) -> float:
    """使用sklearn的互信息回归估计因子与标签之间的非线性相关性。"""
    try:
        from sklearn.feature_selection import mutual_info_regression
        import warnings
        warnings.filterwarnings('ignore')
        
        common_idx = factor.dropna().index.intersection(label.dropna().index)
        if len(common_idx) == 0:
            return 0.0
            
        # 重塑数据为截面格式
        dates = factor.index.levels[0]
        daily_mi = []
        
        for date in dates:
            try:
                factor_daily = factor.xs(date, level=0).dropna()
                label_daily = label.xs(date, level=0).dropna()
                
                common_stocks = factor_daily.index.intersection(label_daily.index)
                if len(common_stocks) < min_stocks:
                    continue
                    
                x = factor_daily.loc[common_stocks].values.reshape(-1, 1)
                y = label_daily.loc[common_stocks].values
                
                # 使用sklearn的互信息估计
                mi = mutual_info_regression(x, y, random_state=42)[0]
                
                if not np.isnan(mi) and not np.isinf(mi):
                    daily_mi.append(mi)
                    
            except Exception:
                continue
                
        if len(daily_mi) < min_days:
            return 0.0
            
        return float(np.mean(daily_mi))
        
    except Exception:
        return 0.0

def quick_top_quantile_return(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS,
    min_days: int = _MIN_DAYS,
    quantile: float = 0.4,
) -> float:
    """计算因子多头组合的每日平均收益（因子值最高的quantile比例）。

    设计原则：
    - 快速、无副作用，适合在遗传规划循环中高频调用
    - 任何异常(NaN、形状不匹配、计算错误)均安全返回0.0，不抛出异常
    - 有效交易日不足min_days时返回0.0(避免小样本噪声影响进化方向)

    Parameters
    ----------
    factor : pd.Series
        MultiIndex (datetime, instrument)，因子值面板。
        可直接传入遗传规划表达式树的求值结果。
    label : pd.Series
        MultiIndex (datetime, instrument)，前瞻收益标签。
        推荐使用 label.make_forward_return 生成。
    min_stocks : int
        每日截面最少有效股票数，默认5。
    min_days : int
        最少有效交易日数，默认3。
    quantile : float
        头部比例，默认0.2(前20%)。

    Returns
    -------
    float
        多头组合的每日平均收益。数据不足或异常时返回0.0。
    """
    try:
        # 确保索引对齐
        common_idx = factor.dropna().index.intersection(label.dropna().index)
        if len(common_idx) == 0:
            return 0.0
            
        factor_aligned = factor.loc[common_idx]
        label_aligned = label.loc[common_idx]
        
        # 按日期分组计算每日多头收益
        daily_returns = []
        
        for date, group in factor_aligned.groupby(level=0):
            try:
                # 获取当日因子和标签
                factor_daily = group.droplevel(0)
                label_daily = label_aligned.xs(date, level=0)
                
                # 对齐同一日内的股票
                common_stocks = factor_daily.index.intersection(label_daily.index)
                if len(common_stocks) < min_stocks:
                    continue
                    
                x = factor_daily.loc[common_stocks]
                y = label_daily.loc[common_stocks]
                
                # 去除NaN
                valid_mask = ~(pd.isna(x) | pd.isna(y))
                x = x[valid_mask]
                y = y[valid_mask]
                
                if len(x) < min_stocks:
                    continue
                
                # 按因子值排序，取头部quantile比例的股票
                n_top = max(1, int(len(x) * quantile))
                top_indices = x.nlargest(n_top).index
                
                # 计算头部组合平均收益
                top_return = y.loc[top_indices].mean()
                
                if not np.isnan(top_return) and not np.isinf(top_return):
                    daily_returns.append(top_return)
                    
            except Exception:
                continue
        
        # 检查有效天数
        if len(daily_returns) < min_days:
            return 0.0
            
        # 返回平均多头收益
        return float(np.mean(daily_returns))
        
    except Exception:
        return 0.0


def quick_top_quantile_excess_return(
    factor: pd.Series,
    label: pd.Series,
    min_stocks: int = _MIN_STOCKS,
    min_days: int = _MIN_DAYS,
    quantile: float = 0.4,
) -> float:
    """计算头部分组相对当日等权基准的日均超额收益。"""
    try:
        common_idx = factor.dropna().index.intersection(label.dropna().index)
        if len(common_idx) == 0:
            return 0.0

        factor_aligned = factor.loc[common_idx]
        label_aligned = label.loc[common_idx]
        daily_excess_returns: list[float] = []

        for date, group in factor_aligned.groupby(level=0):
            try:
                factor_daily = group.droplevel(0)
                label_daily = label_aligned.xs(date, level=0)

                common_stocks = factor_daily.index.intersection(label_daily.index)
                if len(common_stocks) < min_stocks:
                    continue

                x = factor_daily.loc[common_stocks]
                y = label_daily.loc[common_stocks]

                valid_mask = ~(pd.isna(x) | pd.isna(y))
                x = x[valid_mask]
                y = y[valid_mask]
                if len(x) < min_stocks:
                    continue

                n_top = max(1, int(len(x) * quantile))
                top_indices = x.nlargest(n_top).index
                top_return = float(y.loc[top_indices].mean())
                benchmark_return = float(y.mean())
                excess_return = top_return - benchmark_return

                if np.isfinite(excess_return):
                    daily_excess_returns.append(excess_return)
            except Exception:
                continue

        if len(daily_excess_returns) < min_days:
            return 0.0
        return float(np.mean(daily_excess_returns))
    except Exception:
        return 0.0
