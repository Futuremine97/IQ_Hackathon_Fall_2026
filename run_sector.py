"""AMENDMENT_3: 9 sector SPDRs -> classifiers within the sectors + 9-qubit QAOA (K=3).
-> results/RESULT_SECTOR.json, results/sector_scores.csv, results/sector_qaoa_months.csv"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from qps.data import DEV, ROOT, TEST, classical_features, load_prices, month_ends, panel_labels, rank_u
from qps.panel import FEATS, LogitArm, QsvmArm, RbfArm, VqcArm, ic
from qps.quantum import bit_matrix, qaoa_optimize, qaoa_state, ramp_start
from run_backtest import LOGIT_GRID, QSVM_GRID, RBF_GRID, VQC_GRID, sign_flip_p
from run_qaoa_portfolio import block_boot_ci, run_portfolio, sharpe

OUT = os.path.join(ROOT, "results")
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB"]
N, K, WIN = 9, 3, 4
G = {}


def build():
    px = load_prices()
    me = month_ends(px)
    pf = pd.read_csv(os.path.join(OUT, "features_base.csv"), parse_dates=["date"])
    cf = classical_features(px, me)
    lab = panel_labels(px, me)
    df = (pf.merge(cf, on=["date", "asset"]).merge(lab[["date", "asset", "fwd_ret", "label_end"]],
                                                   on=["date", "asset"], how="left"))
    df = df[df["asset"].isin(SECTORS)].copy()
    df["asset"] = pd.Categorical(df["asset"], categories=SECTORS, ordered=True)
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    assert (df.groupby("date").size() == N).all()
    med = df.groupby("date")["fwd_ret"].transform("median")
    df["y"] = (df["fwd_ret"] > med).astype(int)
    U = rank_u(df, FEATS)
    return px, df, U


def topk_acc(score, y, k=WIN):
    pred = np.zeros(len(y), int)
    pred[np.argsort(-np.asarray(score), kind="stable")[:k]] = 1
    return float((pred == y).mean())


def dates(df, period):
    d = df.loc[df["fwd_ret"].notna(), "date"].unique()
    return [pd.Timestamp(t) for t in sorted(d) if pd.Timestamp(period[0]) <= t <= pd.Timestamp(period[1])]


def _init(df, U):
    G["df"], G["U"] = df, U


def _job(a):
    kind, hp, period = a
    df, U = G["df"], G["U"]
    y = df["y"].values
    arm = {"qsvm": lambda: QsvmArm(U, *hp), "rbf": lambda: RbfArm(U, *hp), "logit": lambda: LogitArm(U, hp),
           "vqc": lambda: VqcArm(U, hp)}.get(kind, lambda: None)()
    recs, scores = [], []
    for t in dates(df, period):
        tr = np.where((df["label_end"] <= t).values & df["fwd_ret"].notna().values)[0]
        te = np.where((df["date"] == t).values)[0]
        assert (df["date"].values[tr] < np.datetime64(t)).all()
        s = arm.score(tr, te, y)
        recs.append(topk_acc(s, y[te]))
        scores.append(s)
    return dict(kind=kind, hp=hp, acc=float(np.mean(recs)), month_acc=recs, scores=np.concatenate(scores))


def qubo(mu, S):
    B = bit_matrix(N).astype(float)
    A = 1.5 * np.max(np.abs(mu) + (1.0 / K) * np.abs(S).sum(axis=1))
    E = -(B @ mu) + (1.0 / (2 * K)) * np.einsum("bi,bi->b", B @ S, B) + A * (B.sum(axis=1) - K) ** 2
    return E, B


if __name__ == "__main__":
    t0 = time.time()
    px, df, U = build()
    out = {"amendment_sha256": open(os.path.join(ROOT, "PREREG_AMENDMENT_3_SECTOR.sha256")).read().split()[0]}
    jobs = ([("qsvm", hp, DEV) for hp in QSVM_GRID] + [("rbf", hp, DEV) for hp in RBF_GRID]
            + [("logit", hp, DEV) for hp in LOGIT_GRID] + [("vqc", hp, DEV) for hp in VQC_GRID])
    with Pool(max(1, (os.cpu_count() or 2) - 4), initializer=_init, initargs=(df, U)) as pool:
        dev = pool.map(_job, jobs, chunksize=1)
        sel = {k: max([r for r in dev if r["kind"] == k], key=lambda r: r["acc"])["hp"] for k in ("qsvm", "rbf", "logit", "vqc")}
        tres = {r["kind"]: r for r in pool.map(_job, [(k, sel[k], TEST) for k in sel], chunksize=1)}
    test_dates = dates(df, TEST)
    te_mask = df["date"].isin(test_dates).values
    ts = df.loc[te_mask, ["date", "asset", "fwd_ret", "y", "p_ret", "mom_12_1"]].copy()
    for k in tres:
        ts[k] = tres[k]["scores"]
    acc = {c: ts.groupby("date").apply(lambda d: topk_acc(d[c].values, d["y"].values)).values
           for c in ("p_ret", "mom_12_1", "qsvm", "vqc", "rbf", "logit")}
    icv = {c: ts.groupby("date").apply(lambda d: ic(d[c].values, d["fwd_ret"].values)).values for c in acc}
    out["selected"] = sel
    out["classifiers"] = {c: dict(acc_pct=100 * float(acc[c].mean()), mean_ic=float(np.nanmean(icv[c]))) for c in acc}
    out["qsvm_minus_rbf"] = dict(diff_pt=100 * float((acc["qsvm"] - acc["rbf"]).mean()), p=sign_flip_p(acc["qsvm"] - acc["rbf"]))
    out["coin_expectation_pct"] = 100 * 41 / 81
    ts.to_csv(os.path.join(OUT, "sector_scores.csv"), index=False)

    # 9-qubit QAOA, K = 3
    rets = px.pct_change()
    rows, prev_x = [], None
    for m, (t, d) in enumerate(ts.groupby("date")):
        q = d["qsvm"].values
        mu = (q - q.mean()) / q.std()
        S = rets.loc[:t].iloc[-252:][SECTORS].cov().values
        S = S / np.mean(np.diag(S))
        E, B = qubo(mu, S)
        b_opt = int(np.argmin(E))
        feas = B.sum(axis=1) == K
        assert feas[b_opt]
        C = (E - E.mean()) / E.std()
        res = qaoa_optimize(C, 3, [ramp_start(3)] + ([prev_x] if prev_x is not None else []), maxiter=300)
        prev_x = res.x
        prob = np.abs(qaoa_state(C, res.x[:3], res.x[3:])) ** 2
        prob /= prob.sum()
        b_arg = int(np.argmax(prob))
        samp = np.random.default_rng(m).choice(prob.size, size=16, p=prob)
        fs = samp[feas[samp]]
        best16 = int(fs[np.argmin(E[fs])]) if fs.size else -1
        sel_w = lambda b: np.array([(b >> i) & 1 for i in range(N)], float) / K
        rows.append(dict(date=t, s1_argmax_is_opt=int(b_arg == b_opt), p_opt=float(prob[b_opt]),
                         p_feasible=float(prob[feas].sum()), s3_best16_is_opt=int(best16 == b_opt),
                         argmax_feasible=int(feas[b_arg]), nfev=int(res.nfev),
                         w_qaoa=(sel_w(b_arg) if feas[b_arg] else sel_w(b_opt)).tolist(), w_opt=sel_w(b_opt).tolist(),
                         fwd=d["fwd_ret"].values.tolist()))
    qm = pd.DataFrame(rows)
    qm.to_csv(os.path.join(OUT, "sector_qaoa_months.csv"), index=False)
    n = len(qm)
    s3 = int(qm["s3_best16_is_opt"].sum())
    base16 = 1 - (1 - 1 / 84) ** 16
    out["qaoa_9q"] = dict(
        S1_argmax_rate=float(qm["s1_argmax_is_opt"].mean()), argmax_infeasible_months=int((1 - qm["argmax_feasible"]).sum()),
        S2_mean_p_opt=float(qm["p_opt"].mean()), uniform_feasible=1 / 84, uniform_all=1 / 512,
        mean_p_feasible=float(qm["p_feasible"].mean()),
        S3_best16_rate=s3 / n, S3_random_baseline=base16,
        S3_binom_p=float(stats.binomtest(s3, n, base16, alternative="greater").pvalue),
        verdict="PASS" if (qm["s1_argmax_is_opt"].mean() >= 0.90 and
                           stats.binomtest(s3, n, base16, alternative="greater").pvalue < 0.05) else "FAIL")
    R = np.array(qm["fwd"].tolist())
    port = {"qaoa_top3": run_portfolio([np.array(w) for w in qm["w_qaoa"]], R),
            "bruteforce_top3": run_portfolio([np.array(w) for w in qm["w_opt"]], R),
            "equal_weight_9": run_portfolio([np.full(N, 1 / N)] * n, R)}
    out["portfolio"] = {k: dict(sharpe=sharpe(v), ci90_vs_ew=block_boot_ci(v, port["equal_weight_9"])) for k, v in port.items()}
    pd.DataFrame(port, index=qm["date"]).to_csv(os.path.join(OUT, "sector_portfolio_returns.csv"))
    out["per_month_acc"] = {c: acc[c].tolist() for c in acc}
    out["seconds"] = time.time() - t0
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(out, open(os.path.join(OUT, "RESULT_SECTOR.json"), "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "per_month_acc"}, indent=1, default=str))
