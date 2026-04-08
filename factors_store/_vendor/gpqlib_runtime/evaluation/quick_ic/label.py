"""
B3 标准标签生成模块

提供前瞻收益率标签,作为 IC 计算的基础。

提供:
    make_forward_return          N 日前瞻收盘价收益率(收益 = close[t+n]/close[t] - 1)
    make_open_to_open_return     N 日前瞻开盘价收益率(T+1 开买入,T+n+1 开卖出）
    load_label_from_qlib         从 Qlib 数据目录直接加载前瞻标签

数据约定:
    输入 close/open_ 为 pd.Series,MultiIndex = (datetime, instrument)。
    输出 label 同样为 pd.Series,MultiIndex = (datetime, instrument)。
    最后 n(或 n+1)个交易日标签为 NaN(无未来数据)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_forward_return(
    close: pd.Series,
    n: int = 1,
) -> pd.Series:
    """N 日前瞻收盘价收益率。

    对每个 (日期 t, 股票):
        ret = close(t+n) / close(t) - 1

    Parameters
    ----------
    close : pd.Series
        MultiIndex (datetime, instrument),前复权收盘价。
    n : int
        前瞻交易日数,默认 1。

    Returns
    -------
    pd.Series
        MultiIndex (datetime, instrument),前瞻收益率。
        最后 n 个交易日标签为 NaN。
    """
    if not isinstance(close.index, pd.MultiIndex):
        raise TypeError("close 须为 MultiIndex (datetime, instrument)")
    if n < 1:
        raise ValueError(f"n 须 >= 1,收到 {n}")

    pivot = close.unstack(level=1)          # (datetime, instrument) 宽表
    fwd = pivot.shift(-n) / pivot - 1
    result = fwd.stack(future_stack=True)
    result.index.names = close.index.names
    return result.rename(f"fwd_ret_{n}d")


def make_open_to_open_return(
    open_: pd.Series,
    n: int = 1,
) -> pd.Series:
    """N 日前瞻开盘—开盘收益率（模拟 T+1 开买入、T+n+1 开卖出）。

    对每个 (日期 t, 股票):
        ret = open(t+n+1) / open(t+1) - 1

    Parameters
    ----------
    open_ : pd.Series
        MultiIndex (datetime, instrument),前复权开盘价。
    n : int
        持仓交易日数,默认 1。

    Returns
    -------
    pd.Series
        MultiIndex (datetime, instrument),开盘—开盘前瞻收益率。
        最后 n+1 个交易日标签为 NaN。
    """
    if not isinstance(open_.index, pd.MultiIndex):
        raise TypeError("open_ 须为 MultiIndex (datetime, instrument)")
    if n < 1:
        raise ValueError(f"n 须 >= 1,收到 {n}")

    pivot = open_.unstack(level=1)
    fwd = pivot.shift(-(n + 1)) / pivot.shift(-1) - 1
    result = fwd.stack(future_stack=True)
    result.index.names = open_.index.names
    return result.rename(f"fwd_open_ret_{n}d")


def load_label_from_qlib(
    provider_uri: str = "/root/dmd_factor_arena/qlib_data/sh_daily",
    instruments: str = "stocks",
    start_time: str = "2020-01-01",
    end_time: str = "2026-02-13",
    n: int = 1,
    label_type: str = "close",
) -> pd.Series:
    """从 Qlib 数据目录加载标准前瞻收益标签。

    Parameters
    ----------
    provider_uri : str
        Qlib bin 数据目录（含 calendars/、features/、instruments/ 子目录）。
    instruments : str
        证券池,"stocks" 纯 A 股,"all" 含指数。
    start_time, end_time : str
        数据加载时间范围()YYYY-MM-DD)。注意:实际可用标签的最后 n 天为 NaN。
    n : int
        前瞻交易日数。
    label_type : str
        "close" → 收盘价收益率；"open" → 开盘—开盘收益率。

    Returns
    -------
    pd.Series
        MultiIndex (datetime, instrument),前瞻收益率标签。
    """
    import sys
    import os

    # 防止本地 qlib/ 目录遮蔽已安装包
    _pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for _p in (_pkg_dir, ""):
        if _p in sys.path:
            sys.path.remove(_p)

    try:
        import qlib
        from qlib.constant import REG_CN
        from qlib.data import D
    except ImportError as exc:
        raise ImportError(
            "load_label_from_qlib 需要 qlib,请先安装:pip install qlib"
        ) from exc

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    inst = D.instruments(instruments)

    if label_type == "close":
        df = D.features(inst, ["$close"], start_time=start_time, end_time=end_time)
        df.columns = ["close"]
        return make_forward_return(df["close"], n=n)
    elif label_type == "open":
        df = D.features(inst, ["$open"], start_time=start_time, end_time=end_time)
        df.columns = ["open"]
        return make_open_to_open_return(df["open"], n=n)
    else:
        raise ValueError(f"label_type 须为 'close' 或 'open',收到 {label_type!r}")
