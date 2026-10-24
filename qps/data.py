"""Frozen price panel, month-end calendar, labels and classical features."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICES = os.path.join(ROOT, "data", "prices_2005-01-01_2026-10-17.csv")
PRICES_SHA256 = "8d92ddeca2d56fd319929a4b62543c1674999a145f39d2d2d2d7dd41c14cf028"
UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
            "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ"]
DEV = ("2009-01-01", "2013-12-31")
TEST = ("2014-01-01", "2026-08-31")


def load_prices(path: str = PRICES, check: bool = True) -> pd.DataFrame:
    if check:
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert h == PRICES_SHA256, f"price file hash mismatch: {h}"
    px = pd.read_csv(path, index_col=0, parse_dates=True)[UNIVERSE].sort_index()
    return px.dropna(how="all")


def month_ends(px: pd.DataFrame) -> pd.DatetimeIndex:
    g = px.index.to_series().groupby([px.index.year, px.index.month]).max()
    return pd.DatetimeIndex(g.values)


def panel_labels(px: pd.DataFrame, me: pd.DatetimeIndex) -> pd.DataFrame:
    """Next-month return per (decision date, ETF) and its label END date.

    The last month-end in the file is incomplete when the file stops mid-month, so a
    label only exists if the next month-end is a real month-end (not the file's last row
    of an unfinished month)."""
    last = px.index[-1]
    complete = me[me < last] if (last + pd.offsets.BDay(1)).month == last.month else me
    rows = []
    for t, t_next in zip(complete[:-1], complete[1:]):
        r = px.loc[t_next] / px.loc[t] - 1.0
        for a in px.columns:
            rows.append((t, a, float(r[a]), t_next))
    lab = pd.DataFrame(rows, columns=["date", "asset", "fwd_ret", "label_end"])
    lab["y"] = lab.groupby("date")["fwd_ret"].rank(method="first") > 9
    lab["y"] = lab["y"].astype(int)
    return lab


def classical_features(px: pd.DataFrame, me: pd.DatetimeIndex) -> pd.DataFrame:
    logr = np.log(px).diff()
    rows = []
    for t in me:
        i = px.index.get_loc(t)
        if i < 252:
            continue
        c = px.iloc[i]
        mom_12_1 = px.iloc[i - 21] / px.iloc[i - 252] - 1.0
        mom_1 = c / px.iloc[i - 21] - 1.0
        vol_21 = logr.iloc[i - 20:i + 1].std()
        for a in px.columns:
            rows.append((t, a, float(mom_12_1[a]), float(mom_1[a]), float(vol_21[a])))
    return pd.DataFrame(rows, columns=["date", "asset", "mom_12_1", "mom_1", "vol_21"])


def rank_u(df: pd.DataFrame, cols) -> np.ndarray:
    """Cross-sectional rank per date mapped to (0,1): u = (rank - 0.5)/n."""
    out = []
    for c in cols:
        r = df.groupby("date")[c].rank(method="first")
        n = df.groupby("date")[c].transform("count")
        out.append(((r - 0.5) / n).values)
    return np.column_stack(out)
