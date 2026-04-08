"""
CNE5 全部核心因子 (B1 + B2)

B1 风格因子:
    LNCAP  - 对数市值()规模因子)
    DASTD  - 日收益率指数加权滚动标准差()波动率因子)
    HBETA  - 历史 Beta()半衰期加权 OLS 斜率,市场敏感度因子)
    HSIGMA - 历史 Sigma()HBETA 回归残差标准差,特质波动率)

B2 流动性/动量因子:
    STOM  - 短期月换手率()对数,21 日)
    STOQ  - 短期季换手率()对数,63 日)
    STOA  - 短期年换手率()对数,252 日)
    RSTR  - 相对强度动量()滞后 21 日、指数加权 252 日均值)
    ATVR  - 年换手率()252 日均值,不取对数)

数据约定:
    所有 factor / stock_ret 均为 pd.Series,
    MultiIndex = (datetime, instrument)。
    market_ret 为普通 pd.Series,index = datetime。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

# ---------------------------------------------------------------------------
# 共用辅助函数
# ---------------------------------------------------------------------------

def _validate_mi(s: pd.Series) -> None:
    if not isinstance(s.index, pd.MultiIndex) or s.index.nlevels < 2:
        raise ValueError("Series index must be MultiIndex [datetime, instrument]")


def _exp_weights(window: int, half_life: int) -> np.ndarray:
    """指数衰减权重()最旧→最新),归一化使 sum=1。"""
    decay = np.log(2) / half_life
    offsets = np.arange(window - 1, -1, -1)
    w = np.exp(-decay * offsets)
    return w / (w.sum() + EPS)


def _rolling_ts_mean(
    series: pd.Series,
    window: int,
    half_life: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """逐证券滚动()可选指数加权)均值。

    有 half_life 时使用 pandas ewm（向量化，比 apply 快 10-100x）。
    ewm 半衰期公式: span = 2*half_life - 1 使 center-of-mass = half_life - 1。
    为模拟固定窗口 (window) 的截断，先将窗口外的值置 NaN 再做 ewm。
    """
    _validate_mi(series)
    if min_periods is None:
        min_periods = window

    # 宽表：date × instrument（unstack 后按列分组更快）
    wide = series.unstack(level="instrument")

    if half_life is None:
        result = wide.rolling(window, min_periods=min_periods).mean()
    else:
        # pandas ewm with halflife 参数：halflife = half_life (rows)
        # min_periods 控制前 min_periods-1 行输出 NaN
        result = wide.ewm(
            halflife=half_life,
            min_periods=min_periods,
            adjust=True,
        ).mean()
        # 前 window-1 行强制 NaN（模拟截断窗口，保持与原实现一致）
        result.iloc[: min_periods - 1] = np.nan

    try:
        out = result.stack(future_stack=True)
    except TypeError:
        out = result.stack()
    out.index.names = ["datetime", "instrument"]
    return out.sort_index()


def _rolling_ts_var(
    series: pd.Series,
    window: int,
    half_life: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """逐证券滚动()可选指数加权)方差 = E[X²] - E[X]²。"""
    m1 = _rolling_ts_mean(series, window, half_life, min_periods)
    # 复用宽表避免重复 unstack
    m2 = _rolling_ts_mean(series ** 2, window, half_life, min_periods)
    return (m2 - m1 ** 2).clip(lower=0)


def _rolling_ts_cov(
    x: pd.Series,
    y: pd.Series,
    window: int,
    half_life: int | None = None,
    min_periods: int | None = None,
) -> pd.Series:
    """逐证券滚动()可选指数加权)协方差 = E[XY] - E[X]E[Y]。"""
    mx = _rolling_ts_mean(x, window, half_life, min_periods)
    my = _rolling_ts_mean(y, window, half_life, min_periods)
    mxy = _rolling_ts_mean(x * y, window, half_life, min_periods)
    return mxy - mx * my


def _broadcast_market_ret(market_ret: pd.Series, stock_ret: pd.Series) -> pd.Series:
    """将 datetime 索引的市场收益广播为与 stock_ret 相同的 MultiIndex。"""
    values = market_ret.reindex(stock_ret.index.get_level_values(0)).values
    return pd.Series(values, index=stock_ret.index, name=market_ret.name)


# ---------------------------------------------------------------------------
# B1 风格因子
# ---------------------------------------------------------------------------

def lncap(market_cap: pd.Series) -> pd.Series:
    """对数市值()SIZE 因子)。

    Parameters
    ----------
    market_cap : pd.Series
        MultiIndex (datetime, instrument),流通市值或总市值()正值)。
    """
    _validate_mi(market_cap)
    return np.log(np.clip(market_cap, EPS, None)).rename("LNCAP")


def dastd(
    stock_ret: pd.Series,
    window: int = 252,
    half_life: int = 42,
) -> pd.Series:
    """日收益率指数加权滚动标准差()波动率因子)。

    CNE5 默认参数: window=252, half_life=42。

    Parameters
    ----------
    stock_ret : pd.Series
        MultiIndex (datetime, instrument),日简单收益率。
    """
    _validate_mi(stock_ret)
    var = _rolling_ts_var(stock_ret, window, half_life)
    return np.sqrt(var).rename("DASTD")


def hbeta(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    window: int = 252,
    half_life: int = 63,
) -> pd.Series:
    """历史 Beta()指数加权 OLS,市场敏感度因子)。

    Beta = cov(r_i, r_m) / var(r_m),协方差和方差均采用指数衰减权重估计。
    CNE5 默认参数: window=252, half_life=63。

    Parameters
    ----------
    stock_ret  : pd.Series  MultiIndex (datetime, instrument),个股日收益率。
    market_ret : pd.Series  index=datetime,市场基准日收益率。
    """
    _validate_mi(stock_ret)
    m_aligned = _broadcast_market_ret(market_ret, stock_ret)
    cov = _rolling_ts_cov(stock_ret, m_aligned, window, half_life)
    var_m = _rolling_ts_var(m_aligned, window, half_life)
    return (cov / (var_m + EPS)).rename("HBETA")


def hsigma(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    window: int = 252,
    half_life: int = 63,
) -> pd.Series:
    """历史 Sigma()HBETA 回归的残差标准差,特质波动率)。

    sigma² = var(r_i) - beta² × var(r_m),均采用指数衰减权重。
    CNE5 默认参数: window=252, half_life=63。

    Parameters
    ----------
    stock_ret  : pd.Series  MultiIndex (datetime, instrument),个股日收益率。
    market_ret : pd.Series  index=datetime,市场基准日收益率。
    """
    _validate_mi(stock_ret)
    m_aligned = _broadcast_market_ret(market_ret, stock_ret)
    cov = _rolling_ts_cov(stock_ret, m_aligned, window, half_life)
    var_m = _rolling_ts_var(m_aligned, window, half_life)
    beta = cov / (var_m + EPS)
    var_s = _rolling_ts_var(stock_ret, window, half_life)
    var_resid = (var_s - beta ** 2 * var_m).clip(lower=0)
    return np.sqrt(var_resid).rename("HSIGMA")


# ---------------------------------------------------------------------------
# B2 流动性/动量因子
# ---------------------------------------------------------------------------

def stom(turnover: pd.Series, window: int = 21) -> pd.Series:
    """短期月换手率()CNE5 流动性因子)。

    log(mean daily turnover) over trailing ``window`` trading days (~1 month)。
    """
    _validate_mi(turnover)
    return np.log(np.clip(_rolling_ts_mean(turnover, window), EPS, None)).rename("STOM")


def stoq(turnover: pd.Series, window: int = 63) -> pd.Series:
    """短期季换手率()CNE5 流动性因子)。

    log(mean daily turnover) over trailing ``window`` trading days (~3 months)。
    """
    _validate_mi(turnover)
    return np.log(np.clip(_rolling_ts_mean(turnover, window), EPS, None)).rename("STOQ")


def stoa(turnover: pd.Series, window: int = 252) -> pd.Series:
    """短期年换手率()CNE5 流动性因子)。

    log(mean daily turnover) over trailing ``window`` trading days (~12 months)。
    """
    _validate_mi(turnover)
    return np.log(np.clip(_rolling_ts_mean(turnover, window), EPS, None)).rename("STOA")


def rstr(
    stock_ret: pd.Series,
    lag: int = 21,
    window: int = 252,
    half_life: int = 126,
) -> pd.Series:
    """相对强度动量因子()CNE5)。

    跳过最近 ``lag`` 个交易日()规避短期反转污染),对过去收益率做指数加权均值。
    CNE5 默认: lag=21, window=252, half_life=126。
    """
    _validate_mi(stock_ret)
    shifted = stock_ret.groupby(level=1, group_keys=False).shift(lag)
    return _rolling_ts_mean(shifted, window=window, half_life=half_life).rename("RSTR")


def atvr(turnover: pd.Series, window: int = 252) -> pd.Series:
    """年换手率()CNE5 流动性因子)。

    Mean daily turnover over trailing ``window`` trading days,不取对数。
    """
    _validate_mi(turnover)
    return _rolling_ts_mean(turnover, window).rename("ATVR")


# ---------------------------------------------------------------------------
# 批量计算入口
# ---------------------------------------------------------------------------

def compute_cne5(inputs: dict[str, pd.Series]) -> pd.DataFrame:
    """计算全部 CNE5 因子()B1 + B2)。

    Required keys
    -------------
    ``"market_cap"``  - 流通市值或总市值,MultiIndex (datetime, instrument)。
    ``"stock_ret"``   - 个股日简单收益率,同 MultiIndex。
    ``"market_ret"``  - 市场基准日收益率,index=datetime()普通 Series)。
    ``"turnover"``    - 日换手率,同 MultiIndex。

    Returns
    -------
    pd.DataFrame
        columns: LNCAP, DASTD, HBETA, HSIGMA, STOM, STOQ, STOA, RSTR, ATVR
        index  : 与输入相同的 MultiIndex (datetime, instrument)。
    """
    mcap = inputs["market_cap"]
    sret = inputs["stock_ret"]
    mret = inputs["market_ret"]
    to = inputs["turnover"]
    return pd.DataFrame(
        {
            "LNCAP": lncap(mcap),
            "DASTD": dastd(sret),
            "HBETA": hbeta(sret, mret),
            "HSIGMA": hsigma(sret, mret),
            "STOM": stom(to),
            "STOQ": stoq(to),
            "STOA": stoa(to),
            "RSTR": rstr(sret),
            "ATVR": atvr(to),
        }
    )
