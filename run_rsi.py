"""H5: recursive self-improvement of the Prophet forecast (PREREG section 4, H5).

Stage Q  configuration self-search: quadratic surrogate + QAOA (6 qubits) over 6 Prophet
         knobs, 24 evaluations, 50 seeds, against random search and against the same
         surrogate minimised exactly (isolates what QAOA contributes).
Stage K  residual self-boosting with quantum-kernel ridge regression, rounds accepted
         only if the development MAE falls; classical twin = RBF kernel ridge.
-> results/RESULT_RSI.json
"""
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import spearmanr

from qps.data import DEV, TEST, UNIVERSE, classical_features, load_prices, month_ends, panel_labels, rank_u
from qps.prophet_feats import batch
from qps.quantum import bit_matrix, gram, qaoa_optimize, qaoa_state, ramp_start, zz_states

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
KNOBS = list(itertools.product((0, 1), repeat=6))
KEY = lambda k: "".join(map(str, k))


def sign_flip_p(d, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(d))
    null = np.abs((rng.choice([-1, 1], size=(n, len(d))) * np.asarray(d)).mean(axis=1))
    return float((1 + (null >= obs - 1e-15).sum()) / (n + 1))


# ------------------------------------------------------------------ Stage Q
def surrogate_features(bits):
    b = np.asarray(bits, float)
    pairs = [b[:, i] * b[:, j] for i, j in itertools.combinations(range(6), 2)]
    return np.column_stack([np.ones(len(b)), b] + pairs)


def fmqa_loop(landscape, seed, solver="qaoa", n_init=8, n_iter=16):
    rng = np.random.default_rng(seed)
    allb = np.array(KNOBS)
    Phi_all = surrogate_features(allb)
    order = rng.permutation(64)
    seen = list(order[:n_init])
    for it in range(n_iter):
        y = landscape[seen]
        yz = (y - y.mean()) / (y.std() + 1e-12)
        Phi = Phi_all[seen]
        w = solve(Phi.T @ Phi + 1e-3 * np.eye(Phi.shape[1]), Phi.T @ yz, assume_a="pos")
        pred = Phi_all @ w
        unseen = [i for i in range(64) if i not in seen]
        if solver == "exact":
            cand = min(unseen, key=lambda i: pred[i])
        else:
            B = bit_matrix(6)                                  # basis state b: qubit i = knob i
            idx_of = {tuple(r): i for i, r in enumerate(allb)}
            perm = np.array([idx_of[tuple(r)] for r in B])   # basis state -> knob index
            cost = pred[perm]
            C = (cost - cost.mean()) / (cost.std() + 1e-12)
            res = qaoa_optimize(C, 2, [ramp_start(2)], maxiter=200)
            prob = np.abs(qaoa_state(C, res.x[:2], res.x[2:])) ** 2
            samp = perm[rng.choice(64, size=1024, p=prob / prob.sum())]
            cands = [i for i in dict.fromkeys(samp.tolist()) if i not in seen]
            if cands:
                cand = min(cands, key=lambda i: pred[i])
            else:  # every sampled config already evaluated: most probable unseen one
                pu = {perm[b]: prob[b] for b in range(64)}
                cand = max(unseen, key=lambda i: pu[i])
        seen.append(int(cand))
    best = min(seen, key=lambda i: landscape[i])
    return seen, int(best)


# ------------------------------------------------------------------ Stage K
def krr_walk(df, Kfull, f_prev, alpha, dates_all):
    """Walk-forward kernel ridge on residuals r - f_prev; returns correction per row (NaN if none)."""
    corr = np.full(len(df), np.nan)
    resid = df["fwd_ret"].values - f_prev
    for t in dates_all:
        tr = np.where((df["label_end"] <= t).values & np.isfinite(resid))[0]
        if tr.size < 36:
            continue
        assert (df["date"].values[tr] < np.datetime64(t)).all()
        te = np.where((df["date"] == t).values)[0]
        Ktr = Kfull[np.ix_(tr, tr)]
        c = solve(Ktr + alpha * np.eye(tr.size), resid[tr], assume_a="pos")
        corr[te] = Kfull[np.ix_(te, tr)] @ c
    return corr


def month_mae(df, f, mask):
    d = df.loc[mask].assign(err=np.abs(df["fwd_ret"].values[mask] - f[mask]))
    return d.groupby("date")["err"].mean()


def metrics(df, f, mask):
    sub = df.loc[mask].assign(f=f[mask])
    mae = float(np.abs(sub["fwd_ret"] - sub["f"]).mean())
    hit = float((np.sign(sub["f"]) == np.sign(sub["fwd_ret"])).mean())
    ics = sub.groupby("date").apply(lambda d: spearmanr(d["f"], d["fwd_ret"]).correlation
                                    if d["f"].std() > 0 else np.nan)
    return dict(mae=mae, direction_hit=hit, mean_ic=float(np.nanmean(ics.values)))


if __name__ == "__main__":
    t0 = time.time()
    px = load_prices()
    me = month_ends(px)
    lab = panel_labels(px, me)

    # ---------------- Stage Q
    tab = pd.read_csv(os.path.join(OUT, "rsi_dev_table.csv"), parse_dates=["date"], dtype={"knobs": str})
    tab = tab.merge(lab[["date", "asset", "fwd_ret"]], on=["date", "asset"])
    tab["ae"] = (tab["p_ret"] - tab["fwd_ret"]).abs()
    land = tab.groupby("knobs")["ae"].mean()
    fails = tab.groupby("knobs")["p_ret"].apply(lambda s: int(s.isna().sum()))
    landscape = np.array([land.get(KEY(k), np.inf) if fails.get(KEY(k), 0) == 0 else np.inf for k in KNOBS])
    dev_best = int(np.argmin(landscape))
    base_idx = KNOBS.index((0, 0, 0, 0, 0, 0))
    loops = {}
    for solver in ("qaoa", "exact"):
        found, best_vals = [], []
        for seed in range(50):
            seen, best = fmqa_loop(landscape, seed, solver)
            found.append(int(dev_best in seen))
            best_vals.append(float(landscape[best]))
            if seed == 0:
                loops[f"{solver}_seed0_best"] = KEY(KNOBS[best])
        loops[solver] = dict(found_dev_best_rate=float(np.mean(found)), mean_best_dev_mae=float(np.mean(best_vals)))
    rng = np.random.default_rng(123)
    rnd = [int(dev_best in rng.permutation(64)[:24]) for _ in range(50)]
    rnd_best = [float(np.min(landscape[rng.permutation(64)[:24]])) for _ in range(50)]
    loops["random"] = dict(found_dev_best_rate=float(np.mean(rnd)), analytic=24 / 64,
                           mean_best_dev_mae=float(np.mean(rnd_best)))
    carried = KNOBS[[KEY(k) for k in KNOBS].index(loops["qaoa_seed0_best"])]
    stage_q = dict(landscape={KEY(k): float(v) for k, v in zip(KNOBS, landscape)},
                   failures={k: int(v) for k, v in fails.items() if v},
                   base=KEY(KNOBS[base_idx]), base_dev_mae=float(landscape[base_idx]),
                   dev_best=KEY(KNOBS[dev_best]), dev_best_mae=float(landscape[dev_best]),
                   loops=loops, carried=KEY(carried), carried_dev_mae=float(landscape[KNOBS.index(carried)]))
    print("stage Q:", json.dumps({k: v for k, v in stage_q.items() if k != "landscape"}, indent=1), flush=True)

    # ---------------- features for base and carried configs
    base = pd.read_csv(os.path.join(OUT, "features_base.csv"), parse_dates=["date"])
    if carried == (0, 0, 0, 0, 0, 0):
        featQ = base.copy()
    else:
        path = os.path.join(OUT, f"features_{KEY(carried)}.csv")
        if not os.path.exists(path):
            dates = sorted(base["date"].unique())
            featQ = batch(px, [(pd.Timestamp(t), a, carried) for t in dates for a in UNIVERSE])
            featQ.to_csv(path, index=False)
        featQ = pd.read_csv(path, parse_dates=["date"])
    assert featQ["p_ret"].notna().all()
    cf = classical_features(px, me)
    df = (featQ.merge(cf, on=["date", "asset"]).merge(lab, on=["date", "asset"], how="left")
          .merge(base[["date", "asset", "p_ret"]].rename(columns={"p_ret": "p_ret_base"}), on=["date", "asset"]))
    df["asset"] = pd.Categorical(df["asset"], categories=UNIVERSE, ordered=True)
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    U = rank_u(df, ["p_ret", "p_resid", "mom_12_1", "vol_21"])
    sel = json.load(open(os.path.join(OUT, "DEV_SELECTION.json")))["selected"]
    lam, gamma = sel["qsvm"][0], sel["rbf"][0]
    psi = zz_states(lam * np.pi * U)
    kernels = {"quantum": gram(psi, psi)}
    sq = ((U[:, None, :] - U[None, :, :]) ** 2).sum(-1)
    kernels["rbf"] = np.exp(-gamma * sq)
    del sq
    has_label = df["fwd_ret"].notna().values
    dates_all = sorted(df.loc[has_label, "date"].unique())
    dev_mask = has_label & (df["date"] >= DEV[0]).values & (df["date"] <= DEV[1]).values
    test_mask = has_label & (df["date"] >= TEST[0]).values & (df["date"] <= TEST[1]).values

    def dev_mae(f):
        return float(np.abs(df["fwd_ret"].values[dev_mask] - f[dev_mask]).mean())

    stage_k = {}
    finals = {}
    for kname, Kfull in kernels.items():
        f0 = df["p_ret"].values.astype(float)
        trials = {}
        for alpha in (0.1, 1.0, 10.0):
            c = krr_walk(df, Kfull, f0, alpha, dates_all)
            trials[alpha] = (dev_mae(np.where(np.isfinite(c), f0 + c, np.nan)), c)
        alpha = min(trials, key=lambda a: trials[a][0])
        rounds, f, cur = [], f0.copy(), dev_mae(f0)
        c = trials[alpha][1]
        for r in range(1, 4):
            if r > 1:
                c = krr_walk(df, Kfull, f, alpha, dates_all)
            f_new = np.where(np.isfinite(c), f + c, np.nan)
            new = dev_mae(f_new)
            accepted = bool(np.isfinite(new) and new < cur)
            rounds.append(dict(round=r, dev_mae_before=cur, dev_mae_after=new, accepted=accepted))
            if not accepted:
                break
            f, cur = f_new, new
        stage_k[kname] = dict(alpha=alpha, alpha_trials={str(a): v[0] for a, v in trials.items()},
                              rounds=rounds, accepted_rounds=sum(r["accepted"] for r in rounds))
        finals[kname] = f

    # ---------------- test evaluation
    fb = df["p_ret_base"].values
    out = dict(stage_Q=stage_q, stage_K=stage_k)
    ev = {"prophet_base": fb, "prophet_carried_config": df["p_ret"].values,
          "rsi_quantum": finals["quantum"], "rsi_rbf_twin": finals["rbf"],
          "random_walk_zero": np.zeros(len(df))}
    hist = np.full(len(df), np.nan)
    for t in dates_all + [df["date"].max()]:
        prev = df.loc[(df["label_end"] <= t).values & has_label]
        m = prev.groupby("asset", observed=True)["fwd_ret"].mean()
        te = np.where((df["date"] == t).values)[0]
        hist[te] = df["asset"].iloc[te].map(m).astype(float).values
    ev["historical_mean"] = hist
    out["test_metrics"] = {k: metrics(df, v, test_mask) for k, v in ev.items()}
    out["dev_metrics"] = {k: metrics(df, v, dev_mask) for k, v in ev.items()}
    pd.DataFrame({k: month_mae(df, v, test_mask) for k, v in ev.items()}).to_csv(
        os.path.join(OUT, "rsi_month_mae_test.csv"))
    base_m = month_mae(df, fb, test_mask)
    h5 = {}
    for k in ("rsi_quantum", "rsi_rbf_twin", "prophet_carried_config", "random_walk_zero"):
        d = (month_mae(df, ev[k], test_mask) - base_m).values
        h5[k] = dict(mae_diff_vs_base=float(d.mean()), p_signflip=sign_flip_p(d))
    q = h5["rsi_quantum"]
    out["H5"] = dict(per_arm=h5, verdict="POS" if (q["mae_diff_vs_base"] < 0 and q["p_signflip"] < 0.05) else "CLOSED-NEG",
                     overfit_gap=dict(dev_gain=out["dev_metrics"]["prophet_base"]["mae"] - out["dev_metrics"]["rsi_quantum"]["mae"],
                                      test_gain=out["test_metrics"]["prophet_base"]["mae"] - out["test_metrics"]["rsi_quantum"]["mae"]))
    out["seconds"] = time.time() - t0
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(out, open(os.path.join(OUT, "RESULT_RSI.json"), "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("stage_Q",)}, indent=2, default=float))
