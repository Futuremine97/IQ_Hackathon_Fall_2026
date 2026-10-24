"""Turnaround-stock study (PREREG.md): events, Prophet feature, classifiers, quantum arms, QAOA.
-> results/RESULT.json, results/events.csv, results/test_scores.csv, results/qaoa_months.csv"""
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qps.prophet_feats import batch  # noqa: E402
from qps.quantum import Vqc, bit_matrix, gram, qaoa_optimize, qaoa_state, ramp_start, zz_states  # noqa: E402

DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
DEV, TEST0 = ("2009-01-01", "2013-12-31"), "2014-01-01"
FEATS = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
QSVM_GRID = [(l, C) for l in (0.1, 0.25, 0.5, 1.0) for C in (0.1, 1, 10)]
RBF_GRID = [(g, C) for g in (0.1, 0.3, 1, 3) for C in (0.1, 1, 10)]
GBM_GRID = [(d, lr) for d in (2, 3) for lr in (0.05, 0.1)]
LOGIT_GRID = [0.01, 0.1, 1, 10]
VQC_GRID = [0.25, 0.5, 1.0]
G = {}


# ------------------------------------------------------------------ data
def check_hashes():
    """Hashes in SHA256SUMS refer to the uncompressed files; the CSVs are shipped gzipped."""
    import gzip
    for line in open(os.path.join(DATA, "SHA256SUMS")):
        h, fn = line.split()
        p = os.path.join(HERE, fn)
        raw = gzip.open(p + ".gz", "rb").read() if fn.endswith(".csv") else open(p, "rb").read()
        assert hashlib.sha256(raw).hexdigest() == h, fn


def build_events():
    px = pd.read_csv(os.path.join(DATA, "close.csv.gz"), index_col=0, parse_dates=True)
    vo = pd.read_csv(os.path.join(DATA, "volume.csv.gz"), index_col=0, parse_dates=True)
    sec = json.load(open(os.path.join(DATA, "universe_sectors.json")))
    spy = px["SPY"]
    st = px.drop(columns="SPY")
    me = pd.DatetimeIndex(spy.index.to_series().groupby([spy.index.year, spy.index.month]).max().values)
    hi, lo = st.rolling(252, min_periods=252).max(), st.rolling(252, min_periods=252).min()
    cnt = st.notna().cumsum()
    logr = np.log(st).diff()
    vol63 = logr.rolling(63, min_periods=50).std()
    vr = vo.drop(columns="SPY", errors="ignore").rolling(21, min_periods=15).mean() / \
        vo.drop(columns="SPY", errors="ignore").rolling(252, min_periods=200).mean()
    spy_dd = spy / spy.rolling(252, min_periods=252).max() - 1
    idx = {d: i for i, d in enumerate(px.index)}
    rows = []
    for t in me:
        i = idx[t]
        if i < 252:
            continue
        c = st.iloc[i]
        cand = (c <= 0.6 * hi.iloc[i]) & (cnt.iloc[i] >= 504) & (c >= 1)
        for a in cand[cand].index:
            j = i + 126
            fwd = ex = np.nan
            lab_end = pd.NaT
            if j < len(px) and pd.notna(st[a].iloc[j]):
                fwd = st[a].iloc[j] / c[a] - 1
                ex = fwd - (spy.iloc[j] / spy.iloc[i] - 1)
                lab_end = px.index[j]
            rows.append(dict(date=t, ticker=a, sector=sec.get(a, "?"), f1=c[a] / hi[a].iloc[i] - 1,
                             f2=c[a] / st[a].iloc[i - 21] - 1, f3=c[a] / st[a].iloc[i - 63] - 1,
                             f4=c[a] / lo[a].iloc[i] - 1, f5=vol63[a].iloc[i], f6=vr[a].iloc[i] if a in vr else np.nan,
                             f8=spy_dd.iloc[i], fwd=fwd, excess=ex, label_end=lab_end,
                             y=int(ex > 0) if pd.notna(ex) else -1))
    ev = pd.DataFrame(rows)
    return px, ev


def ecdf_u(Xtr, X):
    U = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        s = np.sort(Xtr[:, j])
        U[:, j] = (np.searchsorted(s, X[:, j], side="right") + 0.5) / (len(s) + 1)
    return np.clip(U, 1e-3, 1 - 1e-3)


# ------------------------------------------------------------------ walk-forward
def months(ev, start, end):
    d = sorted(ev["date"].unique())
    return [pd.Timestamp(t) for t in d if pd.Timestamp(start) <= t <= pd.Timestamp(end)]


def walk(kind, hp, period, yo=None):
    ev = G["ev"]
    y = ev["y"].values if yo is None else yo
    X = ev[FEATS].values
    labeled = y >= 0
    vqc = Vqc(8, reps=2, maxiter=150, seed=7) if kind == "vqc" else None
    out = []
    for t in months(ev, *period):
        tr = np.flatnonzero((ev["label_end"] <= t).values & labeled)
        te = np.flatnonzero((ev["date"] == t).values)
        assert (ev["date"].values[tr] < np.datetime64(t)).all()
        ytr = y[tr]
        if len(tr) < 50 or ytr.min() == ytr.max():
            continue
        U = ecdf_u(X[tr], X[np.concatenate([tr, te])])
        Utr, Ute = U[:len(tr)], U[len(tr):]
        if kind == "qsvm":
            ps_tr, ps_te = zz_states(hp[0] * np.pi * Utr), zz_states(hp[0] * np.pi * Ute)
            s = SVC(kernel="precomputed", C=hp[1]).fit(gram(ps_tr, ps_tr), ytr).decision_function(gram(ps_te, ps_tr))
        elif kind == "rbf":
            s = SVC(kernel="rbf", gamma=hp[0], C=hp[1]).fit(Utr, ytr).decision_function(Ute)
        elif kind == "logit":
            s = LogisticRegression(C=hp, max_iter=2000).fit(Utr, ytr).decision_function(Ute)
        elif kind == "gbm":
            m = HistGradientBoostingClassifier(max_depth=hp[0], learning_rate=hp[1], max_iter=200, random_state=0)
            s = m.fit(Utr, ytr).predict_proba(Ute)[:, 1]
        elif kind == "vqc":
            vqc.fit(zz_states(hp * np.pi * Utr), ytr)
            s = vqc.proba1(zz_states(hp * np.pi * Ute))
        out.append((t, te, np.atleast_1d(s)))
    return out


def _init(g):
    G.update(g)


def _job(a):
    kind, hp, period, yo = a
    res = walk(kind, hp, period, yo)
    ev = G["ev"]
    y = ev["y"].values if yo is None else yo
    idx = np.concatenate([te for _, te, _ in res]) if res else np.array([], int)
    sc = np.concatenate([s for _, _, s in res]) if res else np.array([])
    m = y[idx] >= 0
    auc = float(roc_auc_score(y[idx][m], sc[m])) if m.sum() > 10 and len(set(y[idx][m])) == 2 else np.nan
    return dict(kind=kind, hp=hp, auc=auc, idx=idx, score=sc)


def block_boot_auc(ev, idx, score, reps=2000, seed=0, other=None):
    """Month-block bootstrap CI of pooled AUC (or of an AUC difference if `other` scores are given)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"m": ev["date"].values[idx], "y": ev["y"].values[idx], "s": score})
    if other is not None:
        df["o"] = other
    df = df[df["y"] >= 0]
    groups = [g for _, g in df.groupby("m")]
    vals = []
    for _ in range(reps):
        smp = pd.concat([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if smp["y"].nunique() < 2:
            continue
        a = roc_auc_score(smp["y"], smp["s"])
        vals.append(a - roc_auc_score(smp["y"], smp["o"]) if other is not None else a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def selection_alpha(ev, idx, score, k=4, min_n=6):
    df = pd.DataFrame({"m": ev["date"].values[idx], "ex": ev["excess"].values[idx], "s": score, "y": ev["y"].values[idx]})
    df = df[df["y"] >= 0]
    out = {}
    for m, g in df.groupby("m"):
        if len(g) >= min_n:
            out[m] = g.nlargest(k, "s")["ex"].mean() - g["ex"].mean()
    return pd.Series(out)


def sign_flip_p(d, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    obs = abs(d.mean())
    null = np.abs((rng.choice([-1, 1], size=(n, d.size)) * d).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (n + 1))


# ------------------------------------------------------------------ QAOA selection
def qaoa_month(args):
    m_idx, tickers, mu, Corr, sectors = args
    n, K = len(tickers), 4
    B = bit_matrix(n).astype(float)
    same = np.array([[1.0 if (i != j and sectors[i] == sectors[j]) else 0.0 for j in range(n)] for i in range(n)])
    A = 1.5 * np.max(np.abs(mu) + (1.0 / K) * np.abs(Corr).sum(axis=1) + 0.5 * (n - 1))
    E = (-(B @ mu) + (1.0 / (2 * K)) * np.einsum("bi,bi->b", B @ Corr, B)
         + 0.5 * 0.5 * np.einsum("bi,bi->b", B @ same, B) + A * (B.sum(axis=1) - K) ** 2)
    b_opt = int(np.argmin(E))
    feas = B.sum(axis=1) == K
    assert feas[b_opt]
    C = (E - E.mean()) / E.std()
    res = qaoa_optimize(C, 3, [ramp_start(3)], maxiter=300)
    prob = np.abs(qaoa_state(C, res.x[:3], res.x[3:])) ** 2
    prob /= prob.sum()
    b_arg = int(np.argmax(prob))
    samp = np.random.default_rng(m_idx).choice(prob.size, size=16, p=prob)
    fs = samp[feas[samp]]
    best16 = int(fs[np.argmin(E[fs])]) if fs.size else -1
    pick = lambda b: [i for i in range(n) if (b >> i) & 1]
    return dict(s1=int(b_arg == b_opt), s3=int(best16 == b_opt), p_opt=float(prob[b_opt]), p_feas=float(prob[feas].sum()),
                argmax_feasible=int(feas[b_arg]), pick_qaoa=pick(b_arg if feas[b_arg] else b_opt), pick_opt=pick(b_opt))


if __name__ == "__main__":
    t_all = time.time()
    check_hashes()
    ev_path = os.path.join(RES, "events.csv")
    px, ev = build_events()
    pj = batch(px.drop(columns="SPY"), [(pd.Timestamp(t), a, (0, 0, 0, 0, 0, 0)) for t, a in zip(ev["date"], ev["ticker"])])
    ev["f7"] = pj["p_ret"].values
    n_before = len(ev)
    ev = ev.dropna(subset=FEATS).reset_index(drop=True)
    dropped = n_before - len(ev)
    ev.to_csv(ev_path, index=False)
    last_label = ev.loc[ev["y"] >= 0, "date"].max()
    TEST = (TEST0, str(last_label.date()))
    print(f"events {len(ev)} (labelled {int((ev.y >= 0).sum())}), test {TEST}, base rate {ev.loc[ev.y >= 0, 'y'].mean():.3f}", flush=True)
    G0 = {"ev": ev}
    workers = max(1, (os.cpu_count() or 2) - 4)
    with Pool(workers, initializer=_init, initargs=(G0,)) as pool:
        jobs = ([("qsvm", hp, DEV, None) for hp in QSVM_GRID] + [("rbf", hp, DEV, None) for hp in RBF_GRID]
                + [("gbm", hp, DEV, None) for hp in GBM_GRID] + [("logit", hp, DEV, None) for hp in LOGIT_GRID]
                + [("vqc", hp, DEV, None) for hp in VQC_GRID])
        dev = pool.map(_job, jobs, chunksize=1)
        sel = {}
        for k in ("qsvm", "rbf", "gbm", "logit", "vqc"):
            rows = [r for r in dev if r["kind"] == k]
            sel[k] = max(rows, key=lambda r: -1 if np.isnan(r["auc"]) else r["auc"])["hp"]
        print("dev selection", sel, flush=True)
        # controls before the test arms
        rng = np.random.default_rng(42)
        shuf, labeled = [], ev["y"].values >= 0
        for _ in range(10):
            ys = ev["y"].values.copy()
            for _, ix in ev[labeled].groupby("date").indices.items():
                ii = np.flatnonzero(labeled)[ix]
                ys[ii] = rng.permutation(ys[ii])
            shuf.append(ys)
        planted = ev["y"].values.copy()
        noise = rng.normal(0, 0.35, len(ev))
        for _, ix in ev.groupby("date").indices.items():
            r2 = ev["f2"].values[ix].argsort().argsort() / max(len(ix) - 1, 1)
            r4 = ev["f4"].values[ix].argsort().argsort() / max(len(ix) - 1, 1)
            z = r2 + r4 + noise[ix]
            planted[ix] = np.where(planted[ix] >= 0, (z > np.median(z)).astype(int), -1)
        cj = [(k, sel[k], TEST, ys) for ys in shuf for k in ("qsvm", "rbf")] + [(k, sel[k], TEST, planted) for k in ("qsvm", "rbf")]
        cres = pool.map(_job, cj, chunksize=1)
        tj = [(k, sel[k], TEST, None) for k in ("qsvm", "rbf", "gbm", "logit", "vqc")]
        tres = {r["kind"]: r for r in pool.map(_job, tj, chunksize=1)}
    # month-level oracle for the shuffle control (within-month shuffles keep each month's base rate)
    lab_te = ev[(ev["y"] >= 0) & ev["date"].isin(months(ev, *TEST))]
    mrate = lab_te.groupby("date")["y"].transform("mean")
    oracle_auc = float(roc_auc_score(lab_te["y"], mrate))
    sh_auc = {k: [r["auc"] for r in cres[:20] if r["kind"] == k] for k in ("qsvm", "rbf")}
    controls = dict(shuffle_auc={k: float(np.mean(v)) for k, v in sh_auc.items()},
                    shuffle_pass_literal=bool(all(abs(np.mean(v) - 0.5) <= 0.03 for v in sh_auc.values())),
                    month_level_oracle_auc=oracle_auc,
                    shuffle_within_oracle=bool(all(np.mean(v) <= oracle_auc for v in sh_auc.values())),
                    planted_auc={r["kind"]: r["auc"] for r in cres[20:]},
                    planted_pass=bool(all(r["auc"] >= 0.65 for r in cres[20:])))
    print("controls", controls, flush=True)
    # test scores table
    idx = tres["qsvm"]["idx"]
    for k in tres:
        assert np.array_equal(tres[k]["idx"], idx), k
    ts = ev.iloc[idx][["date", "ticker", "sector", "excess", "y"] + FEATS].copy()
    for k in tres:
        ts[k] = tres[k]["score"]
    ts["rule_deepest"], ts["rule_reversal"], ts["rule_offlow"], ts["rule_prophet"] = -ts["f1"], ts["f2"], ts["f4"], ts["f7"]
    ts.to_csv(os.path.join(RES, "test_scores.csv"), index=False)
    arms = ["qsvm", "vqc", "rbf", "logit", "gbm", "rule_deepest", "rule_reversal", "rule_offlow", "rule_prophet"]
    lab = ts["y"].values >= 0
    out = {"prereg_sha256": open(os.path.join(HERE, "PREREG.sha256")).read().split()[0], "selected": sel,
           "n_test_events": int(lab.sum()), "n_test_months": int(ts["date"].nunique()), "base_rate_test": float(ts.loc[lab, "y"].mean()),
           "controls": controls, "arms": {}}
    alphas = {}
    for k in arms:
        s = ts[k].values
        alphas[k] = selection_alpha(ev, idx, s)
        out["arms"][k] = dict(auc=float(roc_auc_score(ts.loc[lab, "y"], s[lab])),
                              auc_ci95=block_boot_auc(ev, idx, s),
                              sel_alpha_pct=100 * float(alphas[k].mean()), sel_months=int(len(alphas[k])))
    def cmp(a, b):
        d = (alphas[a] - alphas[b]).dropna()
        return dict(auc_diff=out["arms"][a]["auc"] - out["arms"][b]["auc"],
                    auc_diff_ci95=block_boot_auc(ev, idx, ts[a].values, other=ts[b].values),
                    alpha_diff_pct=100 * float(d.mean()), alpha_p=sign_flip_p(d.values))
    comps = {f"{a}-{b}": cmp(a, b) for a, b in (("qsvm", "rbf"), ("vqc", "logit"), ("gbm", "rbf"),
                                                 ("qsvm", "rule_reversal"), ("gbm", "rule_reversal"), ("qsvm", "gbm"))}
    out["comparisons"] = comps
    best_cl = max(("logit", "rbf", "gbm"), key=lambda k: max(r["auc"] for r in dev if r["kind"] == k))
    out["TA_H1"] = dict(best_classical=best_cl, auc=out["arms"][best_cl]["auc"], ci95=out["arms"][best_cl]["auc_ci95"],
                        verdict="SIGNAL" if out["arms"][best_cl]["auc_ci95"][0] > 0.5 else "NO SIGNAL")
    c = comps["qsvm-rbf"]
    out["TA_H2"] = dict(auc_diff=c["auc_diff"], auc_ci95=c["auc_diff_ci95"], alpha_diff_pct=c["alpha_diff_pct"], alpha_p=c["alpha_p"],
                        verdict="POS" if (c["auc_diff"] >= 0.01 and c["auc_diff_ci95"][0] > 0 and c["alpha_diff_pct"] > 0
                                          and c["alpha_p"] < 0.05) else "CLOSED-NEG")
    # QAOA months
    stl = px.drop(columns="SPY")
    rets = stl.pct_change()
    qjobs, qmeta = [], []
    for mi, (m, g) in enumerate(ts[lab].groupby("date")):
        if len(g) < 12:
            continue
        top = g.nlargest(12, "qsvm")
        mu = ((top["qsvm"] - top["qsvm"].mean()) / top["qsvm"].std()).values
        Corr = rets.loc[:m, top["ticker"]].iloc[-252:].corr().fillna(0).values
        qjobs.append((mi, list(top["ticker"]), mu, Corr, list(top["sector"])))
        qmeta.append((m, top))
    with Pool(workers) as pool:
        qres = pool.map(qaoa_month, qjobs, chunksize=1)
    qrows = []
    for (m, top), r in zip(qmeta, qres):
        allm = ts[(ts["date"] == m) & lab]["excess"].mean()
        ex = top["excess"].values
        greedy = top.nlargest(4, "qsvm")["excess"].mean()
        qrows.append(dict(date=m, **{k: r[k] for k in ("s1", "s3", "p_opt", "p_feas", "argmax_feasible")},
                          alpha_qaoa=ex[r["pick_qaoa"]].mean() - allm, alpha_opt=ex[r["pick_opt"]].mean() - allm,
                          alpha_greedy=greedy - allm))
    qm = pd.DataFrame(qrows)
    qm.to_csv(os.path.join(RES, "qaoa_months.csv"), index=False)
    n = len(qm)
    base16 = 1 - (1 - 1 / 495) ** 16
    s3 = int(qm["s3"].sum()) if n else 0
    out["TA_H3"] = dict(months=n, S1_argmax_rate=float(qm["s1"].mean()) if n else None,
                        S2_mean_p_opt=float(qm["p_opt"].mean()) if n else None, uniform_valid=1 / 495, uniform_all=1 / 4096,
                        S3_best16_rate=s3 / n if n else None, S3_random=base16,
                        S3_binom_p=float(stats.binomtest(s3, n, base16, alternative="greater").pvalue) if n else None,
                        alpha_pct={k: 100 * float(qm[k].mean()) for k in ("alpha_qaoa", "alpha_opt", "alpha_greedy")} if n else None,
                        verdict=("PASS" if n and qm["s1"].mean() >= 0.9 and stats.binomtest(s3, n, base16, alternative="greater").pvalue < 0.05 else "FAIL"))
    out["VOID"] = not controls["shuffle_pass_literal"]          # registered rule; the oracle bound is a diagnostic only
    out["n_events_dropped_missing_features"] = int(dropped)
    out["seconds"] = time.time() - t_all
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(out, open(os.path.join(RES, "RESULT.json"), "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k not in ("arms",)}, indent=1, default=str))
    print(json.dumps({k: (round(v["auc"], 4), v["auc_ci95"], round(v["sel_alpha_pct"], 2)) for k, v in out["arms"].items()}, default=str))
