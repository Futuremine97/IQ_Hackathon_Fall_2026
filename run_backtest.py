"""Walk-forward backtest of the classifier arms (PREREG sections 3-5).

A. dev selection (2009-2013)          -> results/DEV_SELECTION.json
B. controls PC3 + NC1 (frozen hparams) -> results/CONTROLS.json   (written before C)
C. test arms M0..M5 (2014-01..2026-08) -> results/test_scores.csv
D. H1 / H2 / secondary statistics      -> results/RESULT.json
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from qps.data import DEV, TEST
from qps.panel import (LogitArm, QsvmArm, RbfArm, VqcArm, build_panel, decision_dates,
                       ic, top9_accuracy, train_mask)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
QSVM_GRID = [(l, C) for l in (0.1, 0.2, 0.3, 0.5, 1.0) for C in (0.1, 1, 10)]
RBF_GRID = [(g, C) for g in (0.1, 0.3, 1, 3, 10) for C in (0.1, 1, 10)]
LOGIT_GRID = [0.01, 0.1, 1, 10]
VQC_GRID = [0.3, 0.5, 1.0]

G = {}


def _init(df, U):
    G["df"], G["U"] = df, U


def make_arm(kind, hp, U):
    if kind == "qsvm":
        return QsvmArm(U, *hp)
    if kind == "rbf":
        return RbfArm(U, *hp)
    if kind == "logit":
        return LogitArm(U, hp)
    if kind == "vqc":
        return VqcArm(U, hp)
    raise ValueError(kind)


def walk(kind, hp, period, y_override=None):
    """Run one arm over a period; returns per-month (date, acc, ic) and scores."""
    df, U = G["df"], G["U"]
    y = df["y"].values.copy() if y_override is None else y_override
    arm = make_arm(kind, hp, U)
    recs, scores = [], []
    for t in decision_dates(df, period):
        tr = np.where(train_mask(df, t))[0]
        te = np.where((df["date"] == t).values)[0]
        s = arm.score(tr, te, y)
        recs.append((t, top9_accuracy(s, y[te]), ic(s, df["fwd_ret"].values[te])))
        scores.append(s)
    return recs, np.concatenate(scores)


def _job(args):
    kind, hp, period, y_override = args
    t0 = time.time()
    recs, scores = walk(kind, hp, period, y_override)
    return dict(kind=kind, hp=hp, acc=float(np.mean([r[1] for r in recs])),
                ic=float(np.mean([r[2] for r in recs])), recs=recs, scores=scores,
                seconds=time.time() - t0)


def sign_flip_p(d, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    obs = abs(d.mean())
    flips = rng.choice([-1, 1], size=(n, d.size))
    null = np.abs((flips * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (n + 1))


if __name__ == "__main__":
    px, df, U = build_panel()
    workers = max(1, (os.cpu_count() or 2) - 2)
    dev_dates, test_dates = decision_dates(df, DEV), decision_dates(df, TEST)
    print(f"panel {len(df)} rows; dev {len(dev_dates)} months; test {len(test_dates)} months", flush=True)
    assert len(test_dates) == 152 and len(dev_dates) == 60

    # A. dev selection ----------------------------------------------------------------
    jobs = ([("qsvm", hp, DEV, None) for hp in QSVM_GRID] + [("rbf", hp, DEV, None) for hp in RBF_GRID]
            + [("logit", hp, DEV, None) for hp in LOGIT_GRID] + [("vqc", hp, DEV, None) for hp in VQC_GRID])
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(df, U)) as pool:
        dev = pool.map(_job, jobs, chunksize=1)
    sel, table = {}, []
    for kind in ("qsvm", "rbf", "logit", "vqc"):
        rows = [r for r in dev if r["kind"] == kind]
        best = max(rows, key=lambda r: r["acc"])  # max() keeps the first on ties = grid order
        sel[kind] = best["hp"]
        table += [dict(kind=r["kind"], hp=r["hp"], dev_acc=r["acc"], dev_ic=r["ic"]) for r in rows]
    json.dump(dict(selected=sel, table=table, seconds=time.time() - t0),
              open(os.path.join(OUT, "DEV_SELECTION.json"), "w"), indent=2, default=str)
    print("dev selection:", sel, f"{time.time()-t0:.0f}s", flush=True)

    # B. controls PC3 + NC1 -------------------------------------------------------------
    rng = np.random.default_rng(2026)
    planted = np.zeros(len(df), int)
    noise = U[:, 0] + U[:, 2] + rng.normal(0, 0.35, len(df))
    for t, idx in df.groupby("date").indices.items():
        planted[idx[np.argsort(-noise[idx])[:9]]] = 1
    shuffles = []
    for s in range(20):
        ys = df["y"].values.copy()
        for t, idx in df.groupby("date").indices.items():
            ys[idx] = rng.permutation(ys[idx])
        shuffles.append(ys)
    cjobs = [("qsvm", sel["qsvm"], TEST, planted), ("rbf", sel["rbf"], TEST, planted)]
    cjobs += [(k, sel[k], TEST, ys) for ys in shuffles for k in ("qsvm", "rbf")]
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(df, U)) as pool:
        cres = pool.map(_job, cjobs, chunksize=1)
    sim = json.load(open(os.path.join(OUT, "CONTROLS_SIM.json")))
    nc1 = {k: [r["acc"] for r in cres[2:] if r["kind"] == k] for k in ("qsvm", "rbf")}
    controls = dict(sim)
    controls.update(
        PC3_planted_acc={"qsvm": cres[0]["acc"], "rbf": cres[1]["acc"]},
        PC3_pass=bool(cres[0]["acc"] >= 0.60 and cres[1]["acc"] >= 0.60),
        NC1_shuffle_acc=nc1,
        NC1_pass=bool(all(abs(a - 0.5) <= 0.03 for v in nc1.values() for a in v)),
        lookahead_assert="train_mask asserts label_end <= t and date < t for every decision",
        seconds=time.time() - t0,
        written=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    controls["VOID"] = not all(controls[k] for k in ("PC1_pass", "PC4_pass", "PC5_pass", "NC1_pass"))
    json.dump(controls, open(os.path.join(OUT, "CONTROLS.json"), "w"), indent=2)
    print("controls:", {k: controls[k] for k in ("PC3_planted_acc", "PC3_pass", "NC1_pass", "VOID")}, flush=True)

    # C. test arms ----------------------------------------------------------------------
    t0 = time.time()
    tjobs = [("qsvm", sel["qsvm"], TEST, None), ("rbf", sel["rbf"], TEST, None),
             ("logit", sel["logit"], TEST, None), ("vqc", sel["vqc"], TEST, None)]
    with Pool(4, initializer=_init, initargs=(df, U)) as pool:
        tres = {r["kind"]: r for r in pool.map(_job, tjobs, chunksize=1)}
    te_mask = df["date"].isin(test_dates).values
    test = df.loc[te_mask, ["date", "asset", "fwd_ret", "y", "p_ret", "mom_12_1"]].copy()
    for k in tres:
        test[k] = tres[k]["scores"]
    test.to_csv(os.path.join(OUT, "test_scores.csv"), index=False)

    # D. statistics ---------------------------------------------------------------------
    per = {}
    for name, col in [("M0_prophet", "p_ret"), ("M5_momentum", "mom_12_1"), ("M1_qsvm", "qsvm"),
                      ("M3_rbf", "rbf"), ("M4_logit", "logit"), ("M2_vqc", "vqc")]:
        g = test.groupby("date")
        acc = g.apply(lambda d: top9_accuracy(d[col].values, d["y"].values))
        icv = g.apply(lambda d: ic(d[col].values, d["fwd_ret"].values))
        per[name] = dict(acc=acc.values, ic=icv.values)

    def summ(name):
        a, i = per[name]["acc"], per[name]["ic"]
        t_ic = stats.ttest_1samp(i, 0.0)
        return dict(acc_pct=100 * a.mean(), acc_se_pct=100 * a.std(ddof=1) / np.sqrt(a.size),
                    mean_ic=float(i.mean()), ic_t=float(t_ic.statistic), ic_p=float(t_ic.pvalue))

    arms = {k: summ(k) for k in per}
    # H1: Prophet channel + shuffle band
    rng = np.random.default_rng(1)
    null = []
    for _ in range(200):
        accs = []
        for t, d in test.groupby("date"):
            accs.append(top9_accuracy(d["p_ret"].values, rng.permutation(d["y"].values)))
        null.append(np.mean(accs))
    h1 = dict(mean_ic=arms["M0_prophet"]["mean_ic"], ic_p=arms["M0_prophet"]["ic_p"],
              acc_pct=arms["M0_prophet"]["acc_pct"],
              shuffle_band_95_pct=[100 * float(np.percentile(null, 2.5)), 100 * float(np.percentile(null, 97.5))],
              verdict="INFORMATIVE" if arms["M0_prophet"]["ic_p"] < 0.05 else "CLOSED-NEG")
    # H2: QSVM vs RBF
    d2 = per["M1_qsvm"]["acc"] - per["M3_rbf"]["acc"]
    p2 = sign_flip_p(d2)
    h2 = dict(gain_pt=100 * float(d2.mean()), p_signflip=p2,
              verdict="POS" if (100 * d2.mean() >= 1.0 and p2 < 0.05) else "CLOSED-NEG")
    dv = per["M2_vqc"]["acc"] - per["M4_logit"]["acc"]
    sec = dict(vqc_minus_logit_pt=100 * float(dv.mean()), p_signflip=sign_flip_p(dv))
    result = dict(prereg_sha256=open(os.path.join(HERE, "PREREG.sha256")).read().split()[0],
                  controls_void=controls["VOID"], selected=sel, arms=arms, H1=h1, H2=h2,
                  secondary_vqc_vs_logit=sec, n_test_months=len(test_dates),
                  seconds_test=time.time() - t0, written=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    json.dump(result, open(os.path.join(OUT, "RESULT.json"), "w"), indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))
