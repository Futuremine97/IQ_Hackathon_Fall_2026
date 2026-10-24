"""Prophet forecasts as features (same configuration as our earlier forecaster).

BASE = Prophet(daily_seasonality=False, weekly_seasonality=True,
               yearly_seasonality=span>=400d, changepoint_prior_scale=0.05,
               interval_width=0.8), trailing 730 calendar days, 30-day horizon.
`knobs` (6 bits) perturbs BASE for the self-improvement search (PREREG H5)."""
from __future__ import annotations

import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

for _name in ("cmdstanpy", "prophet", "prophet.plot"):
    logging.getLogger(_name).setLevel(logging.ERROR)
    logging.getLogger(_name).disabled = True

HORIZON = 30
BASE_KNOBS = (0, 0, 0, 0, 0, 0)


def knob_config(knobs) -> dict:
    k1, k2, k3, k4, k5, k6 = knobs
    return dict(cps=0.005 if k1 else 0.05,
                mode="multiplicative" if k2 else "additive",
                yearly=not k3, weekly=not k4,
                lookback=365 if k5 else 730,
                growth="flat" if k6 else "linear")


def fit_one(series: pd.Series, t: pd.Timestamp, knobs=BASE_KNOBS, uncertainty: int = 0) -> dict:
    from prophet import Prophet
    cfg = knob_config(knobs)
    hist = series.loc[t - pd.Timedelta(days=cfg["lookback"]):t].dropna()
    df = pd.DataFrame({"ds": hist.index, "y": hist.values})
    span = (df["ds"].max() - df["ds"].min()).days
    m = Prophet(daily_seasonality=False,
                weekly_seasonality=cfg["weekly"],
                yearly_seasonality=cfg["yearly"] and span >= 400,
                changepoint_prior_scale=cfg["cps"],
                seasonality_mode=cfg["mode"],
                growth=cfg["growth"],
                interval_width=0.8,
                uncertainty_samples=uncertainty)
    m.fit(df)
    fut = pd.DataFrame({"ds": [df["ds"].iloc[-1], df["ds"].iloc[-1] + pd.Timedelta(days=HORIZON)]})
    fc = m.predict(fut)
    last = float(df["y"].iloc[-1])
    out = dict(p_ret=float(fc["yhat"].iloc[1]) / last - 1.0,
               p_resid=(last - float(fc["yhat"].iloc[0])) / last,
               p_slope=(float(fc["trend"].iloc[1]) - float(fc["trend"].iloc[0])) / last)
    if uncertainty:
        out["p_lo"] = float(fc["yhat_lower"].iloc[1]) / last - 1.0
        out["p_hi"] = float(fc["yhat_upper"].iloc[1]) / last - 1.0
    return out


_PX = None


def _init(px):
    global _PX
    _PX = px
    for _name in ("cmdstanpy", "prophet"):
        logging.getLogger(_name).disabled = True


def _job(args):
    t, a, knobs = args
    try:
        r = fit_one(_PX[a], t, knobs)
    except Exception as e:  # recorded, never silently dropped
        r = dict(p_ret=np.nan, p_resid=np.nan, p_slope=np.nan, error=repr(e)[:200])
    return dict(date=t, asset=a, knobs="".join(map(str, knobs)), **r)


def batch(px: pd.DataFrame, jobs, workers: int | None = None) -> pd.DataFrame:
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with Pool(workers, initializer=_init, initargs=(px,)) as pool:
        rows = pool.map(_job, jobs, chunksize=8)
    return pd.DataFrame(rows)
