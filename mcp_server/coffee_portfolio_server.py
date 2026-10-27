"""Q-BREW: an MCP server for this repository's Prophet + quantum-kernel + QAOA pipeline.

The README machine as tools an agent can call. Every quantum circuit is an exact classical
statevector simulation (qps/quantum.py, checked against Qiskit in results/CONTROLS_SIM.json).

  list_universe     the 18 ETFs, the frozen price file and the coffee presets
  prophet_forecast  Prophet 30-day forecast, 80 % interval, BULLISH / BEARISH / NEUTRAL (+-3 %)
  qsvm_rank         4-qubit quantum-kernel SVM ranking of the 18 ETFs, next to its RBF-SVM twin
  brew_portfolio    coffee preset -> risk aversion -> top-K mean-variance QUBO solved by QAOA,
                    reported next to the exact optimum and the equal-weight basket
  study_results     the pre-registered backtest verdicts, so an agent sees the caveats

Prices come from the frozen file in data/ (offline, reproducible) unless as_of="live", which
downloads fresh closes with yfinance. Not investment advice.

Run (stdio):  python mcp_server/coffee_portfolio_server.py
"""
from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qps.data import PRICES, PRICES_SHA256, UNIVERSE, load_prices, rank_u  # noqa: E402
from qps.panel import FEATS, build_panel  # noqa: E402
from qps.prophet_feats import HORIZON, fit_one  # noqa: E402
from qps.quantum import bit_matrix, gram, qaoa_optimize, qaoa_state, ramp_start, zz_states  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
MAX_QUBITS = 18
# risk aversion gamma in  E(x) = -mu.x + gamma/(2K) x'Sx + A (sum x - K)^2   (run_qaoa_portfolio.py)
COFFEE = {
    "espresso": (0.25, "strong: follows the return signal, small variance penalty"),
    "americano": (1.0, "the study's setting (run_qaoa_portfolio.py, GAMMA_RISK = 1)"),
    "latte": (3.0, "milder: three times the variance penalty"),
    "decaf": (10.0, "barely any kick: close to the minimum-variance basket"),
}
NOT_ADVICE = "Paper study, not investment advice."

mcp = MCPServer(
    "q-brew",
    instructions="Prophet + quantum-kernel + QAOA tools from the IQ Hackathon Fall 2026 study. "
                 "Call study_results before presenting any output as predictive: in the "
                 "pre-registered 2014-2026 backtest no quantum method beat its classical twin "
                 "and equal weight beat the QAOA portfolio.")


# ------------------------------------------------------------------------------ data
@functools.cache
def frozen_prices() -> pd.DataFrame:
    return load_prices()


@functools.cache
def panel():
    _, df, U = build_panel()
    return df, U


@functools.cache
def cached_prophet() -> pd.DataFrame:
    """Prophet features of the study (month-ends of the frozen file, base configuration)."""
    pf = pd.read_csv(os.path.join(RESULTS, "features_base.csv"), parse_dates=["date"])
    return pf.set_index(["date", "asset"])[["p_ret", "p_resid"]]


@functools.cache
def dev_selection() -> dict:
    return json.load(open(os.path.join(RESULTS, "DEV_SELECTION.json")))["selected"]


def live_prices(tickers) -> pd.DataFrame:
    import yfinance as yf
    start = (pd.Timestamp.today() - pd.Timedelta(days=1100)).strftime("%Y-%m-%d")
    with contextlib.redirect_stdout(sys.stderr):
        raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    px = raw["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    missing = [a for a in tickers if a not in px.columns or px[a].isna().all()]
    if missing:
        raise ToolError(f"no Yahoo prices for {missing}")
    return px[list(tickers)].sort_index()


def prices_at(as_of: str | None, tickers) -> pd.DataFrame:
    """Daily closes of `tickers` up to as_of (rows with any gap dropped)."""
    tickers = [t.upper() for t in tickers]
    if as_of == "live":
        px = live_prices(tickers)
    else:
        px = frozen_prices()
        bad = [a for a in tickers if a not in px.columns]
        if bad:
            raise ToolError(f"{bad} are not in the frozen file; use tickers from {UNIVERSE} or as_of='live'")
        px = px[tickers]
        if as_of is not None:
            px = px.loc[:pd.Timestamp(as_of)]
    px = px.dropna()
    if len(px) < 260:
        raise ToolError("need at least 260 trading days of prices up to as_of "
                         f"(the frozen file covers {frozen_prices().index[0].date()}..{frozen_prices().index[-1].date()})")
    return px


def features_at(px: pd.DataFrame, use_cache: bool) -> pd.DataFrame:
    """The study's four features at the last row of px: Prophet return / residual, 12-1 momentum,
    21-day volatility (same formulas as qps.data.classical_features)."""
    t = px.index[-1]
    i = len(px) - 1
    logr = np.log(px).diff()
    cache = cached_prophet() if use_cache else None
    rows = []
    for a in px.columns:
        if cache is not None and (t, a) in cache.index:
            p_ret, p_resid = cache.loc[(t, a)]
        else:
            with contextlib.redirect_stdout(sys.stderr):
                f = fit_one(px[a], t)
            p_ret, p_resid = f["p_ret"], f["p_resid"]
        rows.append(dict(date=t, asset=a, p_ret=float(p_ret), p_resid=float(p_resid),
                         mom_12_1=float(px[a].iloc[i - 21] / px[a].iloc[i - 252] - 1.0),
                         vol_21=float(logr[a].iloc[i - 20:i + 1].std())))
    return pd.DataFrame(rows)


def qsvm_table(as_of: str | None) -> tuple[pd.DataFrame, dict]:
    """QSVM and RBF-SVM scores of the 18 ETFs at as_of, trained on every resolved label
    (label end <= as_of) of the frozen panel with the dev-selected hyper-parameters."""
    live = as_of == "live"
    px = prices_at(as_of, UNIVERSE)
    t = px.index[-1]
    feat = features_at(px, use_cache=not live)
    df, U = panel()
    tr = (df["label_end"] <= t).values & df["fwd_ret"].notna().values
    if tr.sum() < 18 * 24:
        raise ToolError("fewer than 24 resolved training months before as_of; pick a date after 2009")
    y = df["y"].values
    sel = dev_selection()
    lam, c_q = sel["qsvm"]
    gamma, c_r = sel["rbf"]
    u_t = rank_u(feat, FEATS)
    psi_tr, psi_t = zz_states(lam * np.pi * U[tr]), zz_states(lam * np.pi * u_t)
    qsvm = SVC(kernel="precomputed", C=c_q).fit(gram(psi_tr, psi_tr), y[tr])
    feat["qsvm_score"] = qsvm.decision_function(gram(psi_t, psi_tr))
    rbf = SVC(kernel="rbf", gamma=gamma, C=c_r).fit(U[tr], y[tr])
    feat["rbf_score"] = rbf.decision_function(u_t)
    feat["qsvm_rank"] = feat["qsvm_score"].rank(ascending=False, method="first").astype(int)
    feat["rbf_rank"] = feat["rbf_score"].rank(ascending=False, method="first").astype(int)
    meta = dict(as_of=str(t.date()), training_samples=int(tr.sum()),
                training_months=int(df.loc[tr, "date"].nunique()),
                hyper_parameters=dict(qsvm_lambda=lam, qsvm_C=c_q, rbf_gamma=gamma, rbf_C=c_r))
    realised = df[(df["date"] == t) & df["fwd_ret"].notna()]
    if not live and len(realised) == 18:
        r = realised.set_index("asset")["y"]
        for m in ("qsvm", "rbf"):
            top = set(feat.nsmallest(9, f"{m}_rank")["asset"])
            meta[f"{m}_top_half_accuracy_realised"] = float(np.mean([(a in top) == bool(r[a]) for a in UNIVERSE]))
    return feat, meta


# ------------------------------------------------------------------------------ QUBO
def qubo_energies(mu: np.ndarray, S: np.ndarray, B: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """E(x) for every bitstring row of B; gamma = 1 is run_qaoa_portfolio.build_cost."""
    c = gamma / (2 * k)
    A = 1.5 * np.max(np.abs(mu) + (gamma / k) * np.abs(S).sum(axis=1))
    return -(B @ mu) + c * np.einsum("bi,bi->b", B @ S, B) + A * (B.sum(axis=1) - k) ** 2


def basket(b: int, assets) -> list[str]:
    return [a for i, a in enumerate(assets) if (b >> i) & 1]


# ------------------------------------------------------------------------------ tools
@mcp.tool()
def list_universe() -> dict:
    """The 18 ETFs of the study, the frozen price file and the coffee presets of brew_portfolio."""
    px = frozen_prices()
    return dict(universe=UNIVERSE, price_file=os.path.relpath(PRICES, ROOT), price_sha256=PRICES_SHA256,
                first_date=str(px.index[0].date()), last_date=str(px.index[-1].date()),
                coffee_presets={k: dict(risk_aversion=g, meaning=m) for k, (g, m) in COFFEE.items()},
                as_of="'YYYY-MM-DD' inside the frozen file, omitted = its last day, or 'live' (yfinance, needs internet)",
                note=NOT_ADVICE)


@mcp.tool()
def prophet_forecast(ticker: str = "SPY", as_of: str | None = None) -> dict:
    """Prophet 30-day forecast with the study's configuration (730-day window, weekly seasonality,
    yearly seasonality when the window spans >= 400 days, changepoint prior 0.05, 80 % interval).
    Signal: BULLISH if the forecast return >= +3 %, BEARISH if <= -3 %, else NEUTRAL.
    ticker: one of the 18 ETFs for frozen dates; any Yahoo ticker with as_of='live'."""
    px = prices_at(as_of, [ticker])
    t = px.index[-1]
    s = px.iloc[:, 0]
    with contextlib.redirect_stdout(sys.stderr):
        f = fit_one(s, t, uncertainty=1000)
    sig = "BULLISH" if f["p_ret"] >= 0.03 else "BEARISH" if f["p_ret"] <= -0.03 else "NEUTRAL"
    h1 = json.load(open(os.path.join(RESULTS, "RESULT.json")))["H1"]
    return dict(ticker=ticker.upper(), as_of=str(t.date()), close=float(s.iloc[-1]), horizon_days=HORIZON,
                forecast_return=f["p_ret"], interval_80=[f["p_lo"], f["p_hi"]], signal=sig,
                backtest_caveat=(f"Across 18 ETFs, 2014-2026, this forecast's rank correlation with next-month "
                                 f"returns was {h1['mean_ic']:.3f} (p = {h1['ic_p']:.2f}). {NOT_ADVICE}"))


@mcp.tool()
def qsvm_rank(as_of: str | None = None) -> dict:
    """Rank the 18 ETFs for 'top half of the 18 next month' with the 4-qubit quantum-kernel SVM
    (ZZ feature map, exact fidelity kernel) and its classical RBF-SVM twin, trained on every
    label resolved by as_of. Month-ends of the frozen file reuse the study's Prophet features;
    other dates refit Prophet for each ETF (about 10-20 s). For past month-ends the realised
    top-half accuracy is included."""
    feat, meta = qsvm_table(as_of)
    h2 = json.load(open(os.path.join(RESULTS, "RESULT.json")))["H2"]
    cols = ["asset", "qsvm_rank", "rbf_rank", "qsvm_score", "rbf_score"] + FEATS
    table = feat.sort_values("qsvm_rank")[cols].round(5).to_dict(orient="records")
    return dict(**meta, ranking=table, predicted_top_half_qsvm=[r["asset"] for r in table[:9]],
                backtest_caveat=(f"Walk-forward 2014-2026: QSVM minus RBF top-half accuracy "
                                 f"{h2['gain_pt']:+.1f} pt (p = {h2['p_signflip']:.3f}); a coin scores 50 %. "
                                 f"{NOT_ADVICE}"))


@mcp.tool()
def brew_portfolio(coffee: str = "americano", k: int = 4, signal: str = "qsvm",
                   assets: list[str] | None = None, as_of: str | None = None,
                   qaoa_layers: int = 3, maxiter: int = 150, shots: int = 1024, seed: int = 0) -> dict:
    """Coffee in, portfolio out. The coffee preset sets the risk aversion (espresso 0.25,
    americano 1 = the study, latte 3, decaf 10) of a top-k mean-variance QUBO over the assets
    (one qubit per asset, at most 18): reward = the standardised `signal` ('qsvm' as in the study,
    'prophet' = Prophet forecast return, or 'momentum' = 12-1 momentum), risk = the last 252 days'
    covariance. QAOA (X mixer, linear-ramp start, COBYLA) is simulated exactly and sampled
    `shots` times; the lowest-energy valid sample is the basket. The exact optimum by brute force
    and the equal-weight basket are returned next to it. 18 qubits take about 10-20 s."""
    coffee = coffee.lower().replace("-", " ").strip()
    if coffee not in COFFEE:
        raise ToolError(f"unknown coffee {coffee!r}; choose one of {list(COFFEE)}")
    gamma = COFFEE[coffee][0]
    assets = [a.upper() for a in (assets or UNIVERSE)]
    n = len(assets)
    if len(set(assets)) != n or not 2 <= n <= MAX_QUBITS:
        raise ToolError(f"need 2-{MAX_QUBITS} distinct assets")
    if not 1 <= k < n:
        raise ToolError("k must satisfy 1 <= k < number of assets")
    if not (1 <= qaoa_layers <= 5 and 10 <= maxiter <= 1000 and 1 <= shots <= 100_000):
        raise ToolError("qaoa_layers 1-5, maxiter 10-1000, shots 1-100000")

    if signal == "qsvm":
        bad = [a for a in assets if a not in UNIVERSE]
        if bad:
            raise ToolError(f"the QSVM is trained on the 18 study ETFs; {bad} are not among them "
                             "(use signal='prophet' or 'momentum')")
        feat, _ = qsvm_table(as_of)
        score = feat.set_index("asset").loc[assets, "qsvm_score"].values
        px = prices_at(as_of, assets)
    elif signal in ("prophet", "momentum"):
        px = prices_at(as_of, assets)
        feat = features_at(px, use_cache=as_of != "live") if signal == "prophet" else None
        score = (feat.set_index("asset").loc[assets, "p_ret"].values if signal == "prophet"
                 else (px.iloc[-22] / px.iloc[-253] - 1.0).values)
    else:
        raise ToolError("signal must be 'qsvm', 'prophet' or 'momentum'")
    t = px.index[-1]
    mu = (score - score.mean()) / score.std() if score.std() > 0 else np.zeros(n)
    cov = px.pct_change().iloc[-252:].cov().to_numpy(copy=True)
    S = cov / np.mean(np.diag(cov))

    B = bit_matrix(n).astype(float)
    feasible = B.sum(axis=1) == k
    E = qubo_energies(mu, S, B, k, gamma)
    b_opt = int(np.argmin(E))
    assert feasible[b_opt], "penalty too small: global optimum infeasible"
    C = (E - E.mean()) / E.std()
    res = qaoa_optimize(C, qaoa_layers, [ramp_start(qaoa_layers)], maxiter=maxiter)
    prob = np.abs(qaoa_state(C, res.x[:qaoa_layers], res.x[qaoa_layers:])) ** 2
    prob = prob / prob.sum()
    samp = np.random.default_rng(seed).choice(prob.size, size=shots, p=prob)
    valid = samp[feasible[samp]]
    fallback = valid.size == 0
    b_sol = int(np.flatnonzero(feasible)[np.argmax(prob[feasible])]) if fallback else int(valid[np.argmin(E[valid])])
    Ef = E[feasible]
    n_valid = math.comb(n, k)

    def vol(b):
        x = B[b] / k
        return float(np.sqrt(x @ cov @ x * 252))

    ew = np.full(n, 1 / n)
    h4 = json.load(open(os.path.join(RESULTS, "RESULT_QAOA.json")))["H4"]
    return dict(
        coffee=coffee, risk_aversion=gamma, meaning=COFFEE[coffee][1], as_of=str(t.date()), signal=signal,
        assets=assets, k=k, qubits=n,
        portfolio={a: 1 / k for a in basket(b_sol, assets)},
        exact_optimum={a: 1 / k for a in basket(b_opt, assets)},
        qaoa_found_exact_optimum=b_sol == b_opt,
        approximation_ratio=float((Ef.max() - E[b_sol]) / (Ef.max() - Ef.min())) if Ef.max() > Ef.min() else 1.0,
        p_optimum_in_qaoa_state=float(prob[b_opt]), p_optimum_uniform_valid=1 / n_valid,
        p_valid_in_qaoa_state=float(prob[feasible].sum()), valid_baskets=n_valid,
        chance_random_valid_draws_hit_optimum=1 - (1 - 1 / n_valid) ** shots,
        valid_shots=int(valid.size), no_valid_shot_fallback=fallback,
        qaoa=dict(layers=qaoa_layers, cobyla_evaluations=int(res.nfev), normalised_energy=float(res.fun), shots=shots),
        ex_ante_volatility=dict(portfolio=vol(b_sol), exact_optimum=vol(b_opt),
                                equal_weight_all=float(np.sqrt(ew @ cov @ ew * 252))),
        signal_by_asset=dict(zip(assets, np.round(score, 5).tolist())),
        backtest_caveat=(f"Pre-registered backtest 2014-2026 (18 ETFs, americano, QSVM signal): Sharpe "
                         f"QAOA {h4['sharpe_qaoa']:.2f} vs equal weight {h4['sharpe_ew']:.2f}. {NOT_ADVICE}"))


@mcp.tool()
def study_results() -> dict:
    """Headline verdicts of the pre-registered study (read from results/*.json)."""
    r = json.load(open(os.path.join(RESULTS, "RESULT.json")))
    q = json.load(open(os.path.join(RESULTS, "RESULT_QAOA.json")))
    ta = json.load(open(os.path.join(ROOT, "turnaround", "results", "RESULT.json")))
    return dict(
        etf_ranking=dict(prophet_rank_correlation=r["H1"]["mean_ic"], prophet_p=r["H1"]["ic_p"],
                         qsvm_minus_rbf_accuracy_pt=r["H2"]["gain_pt"], qsvm_minus_rbf_p=r["H2"]["p_signflip"],
                         vqc_minus_logistic_pt=r["secondary_vqc_vs_logit"]["vqc_minus_logit_pt"],
                         test_months=r["n_test_months"], verdicts=dict(H1=r["H1"]["verdict"], H2=r["H2"]["verdict"])),
        qaoa_solver=dict(months_optimum_found=q["H3"]["success_rate"], random_valid_baseline=q["H3"]["random_baseline"],
                         mean_p_optimum=q["H3"]["mean_p_opt"], verdict=q["H3"]["verdict"]),
        portfolio=dict(sharpe_qaoa=q["H4"]["sharpe_qaoa"], sharpe_equal_weight=q["H4"]["sharpe_ew"],
                       ci90_difference=q["H4"]["ci90"], verdict=q["H4"]["verdict"]),
        turnaround=dict(qsvm_minus_rbf_auc=ta["TA_H2"]["auc_diff"], ci95=ta["TA_H2"]["auc_ci95"],
                        qaoa_best_of_16_found_optimum=ta["TA_H3"]["S3_best16_rate"], verdict_quantum=ta["TA_H2"]["verdict"]),
        summary=("No quantum method beat its classical twin; equal weight beat the QAOA portfolio; "
                 "all circuits are exact classical simulations at 4-18 qubits. " + NOT_ADVICE))


if __name__ == "__main__":
    mcp.run()
