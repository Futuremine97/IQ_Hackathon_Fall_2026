"""Prophet feature panel (base config) + the 64-config development table for H5 Stage Q.

results/features_base.csv   : every month-end 2007-01 .. 2026-09 x 18 ETFs, base knobs
results/rsi_dev_table.csv   : 64 knob settings x 20 quarterly dev month-ends x 18 ETFs
"""
import itertools
import os
import time

import pandas as pd

from qps.data import load_prices, month_ends
from qps.prophet_feats import BASE_KNOBS, batch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

if __name__ == "__main__":
    px = load_prices()
    me = month_ends(px)
    first = px.index[0] + pd.Timedelta(days=730)
    dates = [t for t in me if first <= t <= pd.Timestamp("2026-09-30")]
    t0 = time.time()
    base = batch(px, [(t, a, BASE_KNOBS) for t in dates for a in px.columns])
    base.to_csv(os.path.join(OUT, "features_base.csv"), index=False)
    print(f"base: {len(base)} fits, {base['p_ret'].isna().sum()} failed, {time.time()-t0:.0f}s", flush=True)

    dev_q = [t for t in me if pd.Timestamp("2009-01-01") <= t <= pd.Timestamp("2013-12-31")
             and t.month in (3, 6, 9, 12)]
    assert len(dev_q) == 20
    knobs = list(itertools.product((0, 1), repeat=6))
    t0 = time.time()
    tab = batch(px, [(t, a, k) for k in knobs for t in dev_q for a in px.columns])
    tab.to_csv(os.path.join(OUT, "rsi_dev_table.csv"), index=False)
    print(f"dev table: {len(tab)} fits, {tab['p_ret'].isna().sum()} failed, {time.time()-t0:.0f}s", flush=True)
