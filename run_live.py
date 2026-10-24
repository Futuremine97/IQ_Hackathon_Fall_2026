"""Live forecast as of the latest close (fresh yfinance data). NOT investment advice.

Per ETF: Prophet 30-day forecast with 80 % interval and the BULLISH/BEARISH/NEUTRAL
(+-3 %) signal; QSVM / RBF cross-sectional ranks (trained on the frozen panel's resolved
labels with the dev-selected hyper-parameters); QAOA top-4 portfolio (+ brute-force check).
-> results/LIVE_<date>.json and results/LIVE_<date>.md
"""
import json
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVC

from qps.data import UNIVERSE, rank_u
from qps.panel import FEATS, build_panel
from qps.prophet_feats import fit_one
from qps.quantum import bit_matrix, gram, qaoa_optimize, qaoa_state, ramp_start, zz_states
from run_qaoa_portfolio import K, N, P, build_cost

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def fresh_prices():
    raw = yf.download(UNIVERSE, start="2023-01-01", auto_adjust=True, progress=False)
    px = raw["Close"][UNIVERSE].sort_index()
    last = px.index[-1]
    missing = [a for a in UNIVERSE if pd.isna(px.loc[last, a])]
    if missing:  # Yahoo can leave today's daily close empty after the bell: use 1-minute bars
        intr = yf.download(missing, period="1d", interval="1m", auto_adjust=True, progress=False)["Close"]
        for a in missing:
            px.loc[last, a] = float(intr[a].dropna().iloc[-1]) if isinstance(intr, pd.DataFrame) else float(intr.dropna().iloc[-1])
    return px.dropna()


if __name__ == "__main__":
    px = fresh_prices()
    t = px.index[-1]
    rows = []
    for a in UNIVERSE:
        f = fit_one(px[a], t, uncertainty=1000)
        i = px.index.get_loc(t)
        rows.append(dict(asset=a, close=float(px[a].iloc[-1]), **f,
                         mom_12_1=float(px[a].iloc[i - 21] / px[a].iloc[i - 252] - 1),
                         vol_21=float(np.log(px[a]).diff().iloc[i - 20:i + 1].std())))
    live = pd.DataFrame(rows)
    live["date"] = t
    live["signal"] = np.where(live["p_ret"] >= 0.03, "BULLISH", np.where(live["p_ret"] <= -0.03, "BEARISH", "NEUTRAL"))

    # train on the frozen panel (all resolved labels) with the dev-selected hyper-parameters
    _, df, U = build_panel()
    sel = json.load(open(os.path.join(OUT, "DEV_SELECTION.json")))["selected"]
    tr = df["fwd_ret"].notna().values
    U_live = rank_u(live, FEATS)
    lam, Cq = sel["qsvm"]
    psi_tr, psi_live = zz_states(lam * np.pi * U[tr]), zz_states(lam * np.pi * U_live)
    qsvm = SVC(kernel="precomputed", C=Cq).fit(gram(psi_tr, psi_tr), df["y"].values[tr])
    live["qsvm_score"] = qsvm.decision_function(gram(psi_live, psi_tr))
    g, Cr = sel["rbf"]
    rbf = SVC(kernel="rbf", gamma=g, C=Cr).fit(U[tr], df["y"].values[tr])
    live["rbf_score"] = rbf.decision_function(U_live)
    live["qsvm_rank"] = live["qsvm_score"].rank(ascending=False).astype(int)
    live["rbf_rank"] = live["rbf_score"].rank(ascending=False).astype(int)

    # QAOA top-4 portfolio on the QSVM scores
    q = live["qsvm_score"].values
    mu = (q - q.mean()) / q.std()
    S = px.pct_change().iloc[-252:][UNIVERSE].cov().values
    S = S / np.mean(np.diag(S))
    B = bit_matrix(N).astype(float)
    E, _ = build_cost(mu, S, B)
    b_opt = int(np.argmin(E))
    C = (E - E.mean()) / E.std()
    res = qaoa_optimize(C, P, [ramp_start(P)], maxiter=300)
    prob = np.abs(qaoa_state(C, res.x[:P], res.x[P:])) ** 2
    samp = np.random.default_rng(0).choice(prob.size, size=1024, p=prob / prob.sum())
    feas = samp[B[samp].sum(axis=1) == K]
    b_sol = int(feas[np.argmin(E[feas])]) if feas.size else b_opt
    pick = lambda b: [UNIVERSE[i] for i in range(N) if (b >> i) & 1]

    bt = json.load(open(os.path.join(OUT, "RESULT.json")))["arms"]
    caveat = (f"Backtest 2014-01..2026-08: Prophet IC {bt['M0_prophet']['mean_ic']:.3f} (p={bt['M0_prophet']['ic_p']:.2f}), "
              f"top-half accuracy QSVM {bt['M1_qsvm']['acc_pct']:.1f} % vs RBF {bt['M3_rbf']['acc_pct']:.1f} % "
              "(coin = 50 %): these ranks have no demonstrated predictive value. Not investment advice.")
    out = dict(as_of=str(t.date()), generated=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               qaoa_top4=pick(b_sol), bruteforce_top4=pick(b_opt), qaoa_matches_optimum=b_sol == b_opt,
               qaoa_p_optimum=float(prob[b_opt] / prob.sum()),
               table=live.drop(columns=["date"]).round(5).to_dict(orient="records"),
               caveat=caveat)
    json.dump(out, open(os.path.join(OUT, f"LIVE_{t.date()}.json"), "w"), indent=2, default=str)

    lines = [f"# Live forecast as of {t.date()} (close) — NOT investment advice", "",
             "| ETF | close | Prophet 30d | 80 % interval | signal | QSVM rank | RBF rank |",
             "|---|---:|---:|---|---|---:|---:|"]
    for r in live.sort_values("qsvm_rank").itertuples():
        lines.append(f"| {r.asset} | {r.close:,.2f} | {100*r.p_ret:+.2f} % | {100*r.p_lo:+.1f} .. {100*r.p_hi:+.1f} % "
                     f"| {r.signal} | {r.qsvm_rank} | {r.rbf_rank} |")
    lines += ["", f"QAOA top-4 (equal weight): {', '.join(out['qaoa_top4'])} "
              f"(brute-force optimum {'matched' if out['qaoa_matches_optimum'] else 'NOT matched: ' + ', '.join(out['bruteforce_top4'])}, "
              f"P(optimum) in the QAOA state {out['qaoa_p_optimum']:.3f})", "", out["caveat"]]
    open(os.path.join(OUT, f"LIVE_{t.date()}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
