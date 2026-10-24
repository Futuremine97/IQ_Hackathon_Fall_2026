"""AMENDMENT_2: regime prediction (RG1 volatility panel, RG2 bull/bear, RG3 risk-on/off).
-> results/RESULT_REGIME.json, results/regime_scores.csv"""
import hashlib
import json
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from qps.data import DEV, ROOT, TEST, UNIVERSE, classical_features, load_prices, month_ends
from qps.quantum import Vqc, gram, zz_states
from run_backtest import LOGIT_GRID, QSVM_GRID, RBF_GRID, VQC_GRID, sign_flip_p

warnings.filterwarnings("ignore")
OUT = os.path.join(ROOT, "results")
MACRO = os.path.join(ROOT, "data", "macro_2005-01-01_2026-10-17.csv")
MACRO_SHA = "fa019dde64e3c182c724e7e46674036286800638bcb588f7b7fb7ae83e7d44f8"
G = {}


# ------------------------------------------------------------------ data
def build():
    assert hashlib.sha256(open(MACRO, "rb").read()).hexdigest() == MACRO_SHA
    px = load_prices()
    vix = pd.read_csv(MACRO, index_col=0, parse_dates=True)["VIX"].dropna()
    me = month_ends(px)
    last = px.index[-1]
    complete = me[me < last] if (last + pd.offsets.BDay(1)).month == last.month else me
    logr = np.log(px).diff()
    rv = {}                                                # realised vol of the month ending at m
    for a_, b_ in zip(complete[:-1], complete[1:]):
        rv[b_] = logr.loc[(logr.index > a_) & (logr.index <= b_)].std()
    rv = pd.DataFrame(rv).T                                # index month-end, columns ETFs
    mret = px.loc[complete].pct_change()                   # return of the month ending at m
    base = pd.read_csv(os.path.join(OUT, "features_base.csv"), parse_dates=["date"])
    pf = base.pivot(index="date", columns="asset", values="p_ret")
    pr = base.pivot(index="date", columns="asset", values="p_resid")
    cf = classical_features(px, me).pivot(index="date", columns="asset", values="mom_1")
    nxt = dict(zip(complete[:-1], complete[1:]))
    rows1, rows2 = [], []
    for t in pf.index:
        if t not in nxt or t not in rv.index:
            continue
        tn = nxt[t]
        med = rv.loc[:t].median()
        vix_pct = float((vix.loc[:t] <= vix.loc[:t].iloc[-1]).mean())
        for a in UNIVERSE:
            rows1.append(dict(date=t, asset=a, label_end=tn, y=int(rv.loc[tn, a] > med[a]),
                              f1=float(np.log(rv.loc[t, a] / med[a])), f2=vix_pct, f3=float(cf.loc[t, a]),
                              f4=float(abs(pr.loc[t, a]))))
        spy_med = med["SPY"]
        r_spy, r_tlt = mret.loc[tn, "SPY"], mret.loc[tn, "TLT"]
        rows2.append(dict(date=t, label_end=tn, y2=int(r_spy > 0), y3=int(r_spy > r_tlt),
                          f1=float(pf.loc[t, "SPY"]), f2=float(pf.loc[t, "SPY"] - pf.loc[t, "TLT"]), f3=vix_pct,
                          f4=float(np.log(rv.loc[t, "SPY"] / spy_med)),
                          last2=int(mret.loc[t, "SPY"] > 0), last3=int(mret.loc[t, "SPY"] > mret.loc[t, "TLT"])))
    return px, pd.DataFrame(rows1), pd.DataFrame(rows2)


def ecdf_u(Xtr, X):
    """Map columns of X to (0,1) with the training window's empirical CDF."""
    U = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        s = np.sort(Xtr[:, j])
        U[:, j] = (np.searchsorted(s, X[:, j], side="right") + 0.5) / (len(s) + 1)
    return np.clip(U, 1e-3, 1 - 1e-3)


# ------------------------------------------------------------------ walk-forward
def periods(df, period):
    d = sorted(df["date"].unique())
    return [pd.Timestamp(t) for t in d if pd.Timestamp(period[0]) <= t <= pd.Timestamp(period[1])]


def walk(task, kind, hp, period, y_override=None):
    df = G[task]
    ycol = "y" if task == "RG1" else ("y2" if task == "RG2" else "y3")
    y = df[ycol].values if y_override is None else y_override
    X = df[["f1", "f2", "f3", "f4"]].values
    vqc = Vqc(4, reps=3, maxiter=150, seed=7) if kind == "vqc" else None
    out = []
    for t in periods(df, period):
        tr = np.where((df["label_end"] <= t).values)[0]
        te = np.where((df["date"] == t).values)[0]
        assert (df["date"].values[tr] < np.datetime64(t)).all()
        U = ecdf_u(X[tr], X)
        ytr = y[tr]
        if ytr.min() == ytr.max():
            s = np.full(len(te), 1.0 if ytr[0] == 1 else -1.0)
        elif kind == "qsvm":
            lam, C = hp
            ps = zz_states(lam * np.pi * U)
            s = SVC(kernel="precomputed", C=C).fit(gram(ps[tr], ps[tr]), ytr).decision_function(gram(ps[te], ps[tr]))
        elif kind == "rbf":
            g, C = hp
            s = SVC(kernel="rbf", gamma=g, C=C).fit(U[tr], ytr).decision_function(U[te])
        elif kind == "logit":
            s = LogisticRegression(C=hp, max_iter=2000).fit(U[tr], ytr).decision_function(U[te])
        elif kind == "vqc":
            ps = zz_states(hp * np.pi * U)
            vqc.fit(ps[tr], ytr)
            s = vqc.proba1(ps[te]) - 0.5
        else:
            raise ValueError(kind)
        out.append((t, te, np.atleast_1d(s)))
    return out


def _init(g):
    G.update(g)


def _job(a):
    task, kind, hp, period, yo = a
    res = walk(task, kind, hp, period, yo)
    df = G[task]
    ycol = "y" if task == "RG1" else ("y2" if task == "RG2" else "y3")
    y = df[ycol].values if yo is None else yo
    acc = [float(np.mean((s > 0).astype(int) == y[te])) for _, te, s in res]
    return dict(task=task, kind=kind, hp=hp, acc=float(np.mean(acc)), month_acc=acc,
                dates=[t for t, _, _ in res], idx=np.concatenate([te for _, te, _ in res]),
                score=np.concatenate([s for _, _, s in res]))


def ms_job(a):
    """Markov-switching (2 regimes, switching mean + variance) on 756 trailing daily returns."""
    t, series = a
    import statsmodels.api as sm
    x = 100 * series.loc[:t].iloc[-756:].values
    np.random.seed(int(pd.Timestamp(t).value // 10**9) % 2**32)   # statsmodels search_reps uses the global RNG
    try:
        m = sm.tsa.MarkovRegression(x, k_regimes=2, trend="c", switching_variance=True)
        r = m.fit(disp=False, search_reps=5)
        P = m.regime_transition_matrix(r.params)[:, :, 0]          # P[i, j] = P(s_t=i | s_{t-1}=j)
        pi = np.asarray(r.filtered_marginal_probabilities)[-1]
        names = list(m.param_names)                                # transition probs come first
        mu = np.array([r.params[names.index("const[0]")], r.params[names.index("const[1]")]])
        exp, p = 0.0, pi.copy()
        for _ in range(21):
            p = P @ p
            exp += float(p @ mu)
        return t, exp, 0
    except Exception:
        return t, np.nan, 1


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()
    px, rg1, rg2 = build()
    G0 = {"RG1": rg1, "RG2": rg2, "RG3": rg2}
    out = {"amendment_sha256": open(os.path.join(ROOT, "PREREG_AMENDMENT_2_REGIME.sha256")).read().split()[0],
           "base_rates_test": {}}
    workers = max(1, (os.cpu_count() or 2) - 4)
    # dev selection
    jobs = [(task, k, hp, DEV, None) for task in G0 for k, grid in
            (("qsvm", QSVM_GRID), ("rbf", RBF_GRID), ("logit", LOGIT_GRID), ("vqc", VQC_GRID)) for hp in grid]
    with Pool(workers, initializer=_init, initargs=(G0,)) as pool:
        dev = pool.map(_job, jobs, chunksize=1)
        sel = {(r["task"], r["kind"]): None for r in dev}
        for key in sel:
            sel[key] = max([r for r in dev if (r["task"], r["kind"]) == key], key=lambda r: r["acc"])["hp"]
        # test arms + RG1 shuffle control
        tjobs = [(task, k, sel[(task, k)], TEST, None) for task in G0 for k in ("qsvm", "rbf", "logit", "vqc")]
        rng = np.random.default_rng(42)
        shuf = []
        for s in range(10):
            ys = rg1["y"].values.copy()
            for _, idx in rg1.groupby("date").indices.items():
                ys[idx] = rng.permutation(ys[idx])
            shuf.append(ys)
        tjobs += [("RG1", k, sel[("RG1", k)], TEST, ys) for ys in shuf for k in ("qsvm", "rbf")]
        tres = pool.map(_job, tjobs, chunksize=1)
        ms_dates = periods(rg2, (DEV[0], TEST[1]))
        rets = px.pct_change()
        ms2 = pool.map(ms_job, [(t, rets["SPY"].dropna()) for t in ms_dates], chunksize=4)
        ms3 = pool.map(ms_job, [(t, (rets["SPY"] - rets["TLT"]).dropna()) for t in ms_dates], chunksize=4)
    learned = {(r["task"], r["kind"]): r for r in tres[:12]}
    nc = tres[12:]

    tables, scores_rows = {}, []
    for task in ("RG1", "RG2", "RG3"):
        df = G0[task]
        ycol = "y" if task == "RG1" else ("y2" if task == "RG2" else "y3")
        test_dates = periods(df, TEST)
        te_all = np.where(df["date"].isin(test_dates).values)[0]
        y = df[ycol].values
        arms = {}
        for k in ("qsvm", "vqc", "rbf", "logit"):
            r = learned[(task, k)]
            arms[k] = pd.Series(r["score"], index=r["idx"])
        # fixed rules (scores aligned on te_all)
        maj, pers = {}, {}
        for t in test_dates:
            tr = np.where((df["label_end"] <= t).values)[0]
            te = np.where((df["date"] == t).values)[0]
            base = y[tr].mean()
            for i in te:
                maj[i] = 1.0 if base > 0.5 else -1.0
        arms["majority"] = pd.Series(maj)
        if task == "RG1":
            arms["persistence"] = pd.Series(df["f1"].values[te_all], index=te_all)
            arms["vix_rule"] = pd.Series(df["f2"].values[te_all] - 0.5, index=te_all)
        else:
            last = df["last2" if task == "RG2" else "last3"].values
            arms["persistence"] = pd.Series(last[te_all] - 0.5, index=te_all)
            pf = df["f1"].values if task == "RG2" else df["f2"].values
            arms["prophet_rule"] = pd.Series(pf[te_all], index=te_all)
            ms = dict((t, e) for t, e, _ in (ms2 if task == "RG2" else ms3))
            fails = sum(f for _, _, f in (ms2 if task == "RG2" else ms3))
            msv = np.array([ms[pd.Timestamp(df["date"].values[i])] if np.isfinite(ms[pd.Timestamp(df["date"].values[i])])
                            else (last[i] - 0.5) for i in te_all])
            arms["markov_switching"] = pd.Series(msv, index=te_all)
            out.setdefault("markov_switching_failed_fits", {})[task] = int(fails)
        month_of = df["date"].values
        per_month = {}
        summ = {}
        for k, s in arms.items():
            s = s.reindex(te_all)
            correct = ((s.values > 0).astype(int) == y[te_all]).astype(float)
            pm = pd.Series(correct, index=month_of[te_all]).groupby(level=0).mean()
            per_month[k] = pm.values
            yy, ss = y[te_all], s.values
            pos = yy == 1
            summ[k] = dict(acc_pct=100 * float(pm.mean()),
                           balanced_acc_pct=100 * float(0.5 * ((ss[pos] > 0).mean() + (ss[~pos] <= 0).mean())),
                           auc=float(roc_auc_score(yy, ss)) if np.unique(ss).size > 2 else None)
            scores_rows += [dict(task=task, arm=k, date=month_of[i], asset=df["asset"].values[i] if task == "RG1" else "MKT",
                                 score=float(v), y=int(y[i])) for i, v in zip(te_all, ss)]
        def cmp(a, b):
            d = per_month[a] - per_month[b]
            return dict(diff_pt=100 * float(d.mean()), p=sign_flip_p(d))
        comps = {"qsvm-rbf": cmp("qsvm", "rbf"), "vqc-logit": cmp("vqc", "logit")}
        for k in ("qsvm", "vqc", "rbf", "logit"):
            comps[f"{k}-persistence"] = cmp(k, "persistence")
            comps[f"{k}-majority"] = cmp(k, "majority")
        comps["persistence-majority"] = cmp("persistence", "majority")
        if task != "RG1":
            comps["prophet-majority"] = cmp("prophet_rule", "majority")
            comps["markov-persistence"] = cmp("markov_switching", "persistence")
            comps["markov-majority"] = cmp("markov_switching", "majority")
        else:
            comps["vix-persistence"] = cmp("vix_rule", "persistence")
        tables[task] = dict(arms=summ, comparisons=comps, base_rate_test=float(y[te_all].mean()),
                            selected={k: sel[(task, k)] for k in ("qsvm", "rbf", "logit", "vqc")},
                            per_month=dict((k, v.tolist()) for k, v in per_month.items()))
    ps = [tables[t]["comparisons"]["qsvm-rbf"]["p"] for t in ("RG1", "RG2", "RG3")]
    order = np.argsort(ps)
    adj, run = [0.0] * 3, 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (3 - r) * ps[i]))
        adj[i] = run
    gains = [tables[t]["comparisons"]["qsvm-rbf"]["diff_pt"] for t in ("RG1", "RG2", "RG3")]
    out["RG_H1"] = dict(gain_pt=dict(zip(("RG1", "RG2", "RG3"), gains)), p_holm=dict(zip(("RG1", "RG2", "RG3"), adj)),
                        verdict="POS" if any(g >= 1.0 and p < 0.05 for g, p in zip(gains, adj)) else "CLOSED-NEG")
    # controls
    shuf_acc = {k: [r["acc"] for r in nc if r["kind"] == k] for k in ("qsvm", "rbf")}
    maj_shuf = []
    for ys in shuf:
        acc = []
        for t in periods(rg1, TEST):
            tr = np.where((rg1["label_end"] <= t).values)[0]
            te = np.where((rg1["date"] == t).values)[0]
            acc.append(float(np.mean((1 if ys[tr].mean() > 0.5 else 0) == ys[te])))
        maj_shuf.append(np.mean(acc))
    oracle = float(np.mean([max(g.mean(), 1 - g.mean()) for _, g in rg1[rg1["date"].isin(periods(rg1, TEST))].groupby("date")["y"]]))
    out["controls"] = dict(
        RG1_shuffled_acc={k: float(np.mean(v)) for k, v in shuf_acc.items()},
        RG1_shuffled_majority_rule_acc=float(np.mean(maj_shuf)),
        RG1_month_level_oracle_acc=oracle,
        literal_pass=bool(all(abs(np.mean(v) - np.mean(maj_shuf)) <= 0.03 for v in shuf_acc.values())),
        oracle_bound_pass=bool(all(np.mean(v) <= oracle + 1e-9 for v in shuf_acc.values())),
        RG1_persistence_beats_majority=bool(tables["RG1"]["comparisons"]["persistence-majority"]["diff_pt"] > 0
                                            and tables["RG1"]["comparisons"]["persistence-majority"]["p"] < 0.05))
    out["tasks"] = tables
    out["seconds"] = time.time() - t0
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(out, open(os.path.join(OUT, "RESULT_REGIME.json"), "w"), indent=2, default=str)
    pd.DataFrame(scores_rows).to_csv(os.path.join(OUT, "regime_scores.csv"), index=False)
    for task in tables:
        print(task, "base", round(tables[task]["base_rate_test"], 3),
              {k: round(v["acc_pct"], 1) for k, v in tables[task]["arms"].items()})
    print(json.dumps(out["RG_H1"], indent=1)); print(json.dumps(out["controls"], indent=1))
