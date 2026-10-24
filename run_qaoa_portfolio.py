"""H3 (QAOA solver) and H4 (portfolio) — PREREG section 4.

Each test month: 18-qubit top-K mean-variance QUBO from the QSVM scores, brute-force
optimum, QAOA p=3 (warm-started from the previous month), 1,024 shots.
-> results/qaoa_months.csv, results/RESULT_QAOA.json
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from qps.data import UNIVERSE, load_prices
from qps.quantum import bit_matrix, qaoa_optimize, qaoa_state, ramp_start

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
N, K, GAMMA_RISK, P, SHOTS, COST_BP = 18, 4, 1.0, 3, 1024, 10.0


def build_cost(mu, S, B):
    c = GAMMA_RISK / (2 * K)
    A = 1.5 * np.max(np.abs(mu) + (GAMMA_RISK / K) * np.abs(S).sum(axis=1))
    E = -(B @ mu) + c * np.einsum("bi,bi->b", B @ S, B) + A * (B.sum(axis=1) - K) ** 2
    return E, A


def sharpe(r):
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(12))


def block_boot_ci(a, b, reps=2000, block=6, seed=0, q=(5, 95)):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(reps):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        diffs.append(sharpe(a[idx]) - sharpe(b[idx]))
    return [float(np.percentile(diffs, q[0])), float(np.percentile(diffs, q[1]))]


def run_portfolio(weights_by_month, rets_by_month):
    """Monthly rebalance with drift-aware one-way turnover cost."""
    prev, out = None, []
    for w, r in zip(weights_by_month, rets_by_month):
        if prev is None:
            turnover = 1.0
        else:
            drift = prev * (1 + r_prev)
            drift = drift / drift.sum()
            turnover = 0.5 * np.abs(w - drift).sum()
        gross = float(w @ r)
        out.append(gross - COST_BP * 1e-4 * turnover)
        prev, r_prev = w, r
    return np.array(out)


if __name__ == "__main__":
    px = load_prices()
    rets = px.pct_change()
    sc = pd.read_csv(os.path.join(OUT, "test_scores.csv"), parse_dates=["date"])
    sc["asset"] = pd.Categorical(sc["asset"], categories=UNIVERSE, ordered=True)
    sc = sc.sort_values(["date", "asset"])
    B = bit_matrix(N).astype(float)
    feasible = B.sum(axis=1) == K
    rows, prev_x = [], None
    t0 = time.time()
    for m, (t, d) in enumerate(sc.groupby("date")):
        q = d["qsvm"].values
        mu = (q - q.mean()) / q.std()
        hist = rets.loc[:t].iloc[-252:][UNIVERSE]
        S = hist.cov().values
        S = S / np.mean(np.diag(S))
        E, A = build_cost(mu, S, B)
        b_opt = int(np.argmin(E))
        assert feasible[b_opt], "penalty too small: global optimum infeasible"
        Ef = E[feasible]
        C = (E - E.mean()) / E.std()
        starts = [ramp_start(P)] + ([prev_x] if prev_x is not None else [])
        res = qaoa_optimize(C, P, starts, maxiter=300)
        prev_x = res.x
        psi = qaoa_state(C, res.x[:P], res.x[P:])
        prob = np.abs(psi) ** 2
        prob = prob / prob.sum()
        rng = np.random.default_rng(m)
        samp = rng.choice(prob.size, size=SHOTS, p=prob)
        fs = samp[feasible[samp]]
        flagged = fs.size == 0
        if flagged:
            b_sol = int(np.flatnonzero(feasible)[np.argmax(prob[feasible])])
        else:
            b_sol = int(fs[np.argmin(E[fs])])
        sel = lambda b: np.array([(b >> i) & 1 for i in range(N)], float)
        x_sol, x_opt = sel(b_sol), sel(b_opt)
        greedy = np.zeros(N); greedy[np.argsort(-mu, kind="stable")[:K]] = 1
        prophet = np.zeros(N); prophet[np.argsort(-d["p_ret"].values, kind="stable")[:K]] = 1
        mom = np.zeros(N); mom[np.argsort(-d["mom_12_1"].values, kind="stable")[:K]] = 1
        rows.append(dict(
            date=t, success=int(b_sol == b_opt), p_opt=float(prob[b_opt]),
            p_feasible=float(prob[feasible].sum()), feasible_shots=int(fs.size), flagged=int(flagged),
            approx_ratio=float((Ef.max() - E[b_sol]) / (Ef.max() - Ef.min())),
            nfev=int(res.nfev), expect_norm=float(res.fun),
            w_qaoa=(x_sol / K).tolist(), w_opt=(x_opt / K).tolist(), w_greedy=(greedy / K).tolist(),
            w_prophet=(prophet / K).tolist(), w_mom=(mom / K).tolist(),
            fwd=d["fwd_ret"].values.tolist()))
        if m % 10 == 0:
            print(f"{t.date()} success={rows[-1]['success']} p_opt={rows[-1]['p_opt']:.3f} "
                  f"nfev={res.nfev} {time.time()-t0:.0f}s", flush=True)
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(OUT, "qaoa_months.csv"), index=False)

    n = len(mdf)
    succ = int(mdf["success"].sum())
    base = 1 - (1 - 1 / 3060) ** SHOTS
    binom_p = float(stats.binomtest(succ, n, base, alternative="greater").pvalue)
    h3 = dict(success_rate=succ / n, successes=succ, months=n, random_baseline=base, binom_p=binom_p,
              mean_p_opt=float(mdf["p_opt"].mean()), uniform_p_opt=1 / 2 ** N,
              mean_p_feasible=float(mdf["p_feasible"].mean()), uniform_p_feasible=3060 / 2 ** N,
              mean_approx_ratio=float(mdf["approx_ratio"].mean()), flagged_months=int(mdf["flagged"].sum()),
              verdict="PASS" if (succ / n >= 0.90 and binom_p < 0.05) else "FAIL")

    R = np.array(mdf["fwd"].tolist())
    ew = np.full(N, 1 / N)
    port = {}
    for name, col in [("qaoa", "w_qaoa"), ("bruteforce_opt", "w_opt"), ("greedy_mu", "w_greedy"),
                      ("prophet_top4", "w_prophet"), ("momentum_top4", "w_mom")]:
        port[name] = run_portfolio([np.array(w) for w in mdf[col]], R)
    port["equal_weight_18"] = run_portfolio([ew] * n, R)
    perf = {}
    for k, r in port.items():
        eq = np.cumprod(1 + r)
        perf[k] = dict(sharpe=sharpe(r), ann_return=float(eq[-1] ** (12 / n) - 1),
                       ann_vol=float(np.std(r, ddof=1) * np.sqrt(12)),
                       max_drawdown=float((eq / np.maximum.accumulate(eq) - 1).min()))
    ci = block_boot_ci(port["qaoa"], port["equal_weight_18"])
    h4 = dict(sharpe_qaoa=perf["qaoa"]["sharpe"], sharpe_ew=perf["equal_weight_18"]["sharpe"],
              diff=perf["qaoa"]["sharpe"] - perf["equal_weight_18"]["sharpe"], ci90=ci,
              verdict="POS" if ci[0] > 0 else "CLOSED-NEG")
    out = dict(H3=h3, H4=h4, performance=perf, seconds=time.time() - t0,
               written=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    pd.DataFrame(port, index=mdf["date"]).to_csv(os.path.join(OUT, "portfolio_returns.csv"))
    json.dump(out, open(os.path.join(OUT, "RESULT_QAOA.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
