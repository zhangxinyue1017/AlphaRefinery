"""
B3 正交化工具模块

提供:
    gram_schmidt_orthogonalize   将目标因子对基础因子做截面正交化()逐日回归残差)
    orthogonalize_panel          对一组因子做顺序 Gram-Schmidt 正交化
    avg_abs_corr_to_bases        候选因子与基础因子集的平均绝对相关性

数据约定:
    所有输入 Series / DataFrame 均为
    MultiIndex = (datetime, instrument)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12
_MIN_STOCKS_DEFAULT = 10


def gram_schmidt_orthogonalize(
    target: pd.Series,
    bases: pd.DataFrame,
    min_stocks: int = _MIN_STOCKS_DEFAULT,
) -> pd.Series:
    """将 target 在每个截面日对 bases 做正交化,返回残差序列。

    等价于:对每个交易日,将 target 对 bases()含截距)做 OLS 回归,
    取残差作为正交化后的因子值。残差已去除 bases 方向的线性成分,
    但保留了 target 相对于 bases 的"增量信息"。

    Parameters
    ----------
    target : pd.Series
        MultiIndex (datetime, instrument),待正交化的因子。
    bases : pd.DataFrame
        MultiIndex (datetime, instrument),基础因子暴露矩阵()如 Barra 因子)。
    min_stocks : int
        每个截面日至少需要的有效股票数,不足时该日全为 NaN。

    Returns
    -------
    pd.Series
        与 target 同 index 的残差序列。
    """
    if not isinstance(target.index, pd.MultiIndex):
        raise TypeError("target 须为 MultiIndex (datetime, instrument)")

    bases = bases.reindex(target.index)
    n_bases = bases.shape[1]

    combined = bases.copy()
    combined["__y__"] = target

    def _one_day(sub: pd.DataFrame) -> pd.Series:
        y = sub["__y__"].to_numpy(dtype=float)
        X = sub.drop(columns="__y__").to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        out = np.full(len(y), np.nan, dtype=float)

        # 需要足够的自由度:有效样本 > 截距 + 基础因子数
        if valid.sum() < max(min_stocks, n_bases + 2):
            return pd.Series(out, index=sub.index)

        Xv = np.column_stack([np.ones(valid.sum()), X[valid]])
        yv = y[valid]
        coef, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
        out[np.where(valid)[0]] = yv - Xv @ coef
        return pd.Series(out, index=sub.index)

    result = combined.groupby(level=0, group_keys=False).apply(_one_day)
    result.index = target.index
    return result.rename(target.name)


def orthogonalize_panel(
    factors: pd.DataFrame,
    order: list[str] | None = None,
    min_stocks: int = _MIN_STOCKS_DEFAULT,
) -> pd.DataFrame:
    """对因子面板做顺序 Gram-Schmidt 正交化。

    处理顺序:factors[0] 保持不变,factors[1] 对 factors[0] 正交化,
    factors[2] 对已正交化的 [0,1] 正交化,以此类推。

    Parameters
    ----------
    factors : pd.DataFrame
        MultiIndex (datetime, instrument),列为各因子名。
    order : list[str] | None
        正交化顺序,默认使用 factors.columns 顺序。
        通常将"优先保留"的因子放在前面()如市值、行业)。
    min_stocks : int
        截面最低有效股票数,传入 gram_schmidt_orthogonalize。

    Returns
    -------
    pd.DataFrame
        与输入同形,正交化后的因子面板。
    """
    cols = order if order is not None else list(factors.columns)
    result: dict[str, pd.Series] = {}
    orthogonalized_bases: list[pd.Series] = []

    for col in cols:
        target = factors[col]
        if orthogonalized_bases:
            bases_df = pd.concat(orthogonalized_bases, axis=1)
            target = gram_schmidt_orthogonalize(
                target, bases_df, min_stocks=min_stocks
            )
        result[col] = target
        orthogonalized_bases.append(target.rename(col))

    return pd.DataFrame(result)[cols]


def avg_abs_corr_to_bases(
    candidate: pd.Series,
    bases: pd.DataFrame,
) -> float:
    """候选因子与基础因子集的平均绝对 Pearson 相关系数。

    用于 GP 适应度惩罚项:avg_corr 越高,说明候选因子与已知 Barra 因子
    越相似,应降低其适应度得分。

    Parameters
    ----------
    candidate : pd.Series
        MultiIndex (datetime, instrument),候选因子值。
    bases : pd.DataFrame
        MultiIndex (datetime, instrument),基础因子()如 Barra 暴露矩阵)。

    Returns
    -------
    float
        所有基础因子的平均 |corr|,范围 [0, 1]。
        若无有效相关系数,返回 1.0()最保守估计)。
    """
    bases = bases.reindex(candidate.index)
    corrs = []
    for col in bases.columns:
        c = candidate.corr(bases[col])
        if pd.notna(c):
            corrs.append(abs(float(c)))
    return float(np.mean(corrs)) if corrs else 1.0
