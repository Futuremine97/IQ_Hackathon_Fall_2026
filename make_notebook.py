"""Builds quantum_prophet_stocks.ipynb (walkthrough + live demos; numbers read from results/)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Prophet + Quantum ML + QAOA on stock (ETF) data

**IQ Hackathon Fall 2026 (Beginners) — project notebook.** Combines the three Training
Session II notebooks on one real problem:

| notebook | used here for |
|---|---|
| `QML_1_Lecture.ipynb` (quantum kernels, ZZ feature map, QSVM, VQC) | ranking 18 ETFs for next month from Prophet + momentum features |
| `QAOA_ibm_notebook.ipynb` (max-cut -> Ising -> QAOA) | picking a 4-ETF portfolio from an 18-qubit QUBO |
| `vqe_lecture.ipynb` (variational loop) | the same classical-optimiser-in-the-loop idea drives the VQC and QAOA |

Plus **Prophet** (Meta's time-series forecaster) as the feature generator, and a
**recursive self-improvement** loop that tries to make Prophet itself better.

Everything was **pre-registered** (`PREREG.md`, sha256 in `PREREG.sha256`) before any
result existed, and evaluated walk-forward on 2014-01..2026-08 with no look-ahead.
Heavy runs live in the `run_*.py` scripts; this notebook loads their results and
re-runs small pieces live. *Paper backtest only — not investment advice.*
""")
code("""
# Install every package this notebook needs (fresh environment or Google Colab).
# In Colab, first get the code and data:
#   !git clone https://github.com/Futuremine97/IQ_Hackathon_Fall_2026.git
#   %cd IQ_Hackathon_Fall_2026
%pip install -q -r requirements.txt
""")
code("""
import json, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import logging; [logging.getLogger(n).setLevel(logging.ERROR) for n in ("cmdstanpy", "prophet")]
from qps.data import load_prices, UNIVERSE
from qps.panel import build_panel
from qps.quantum import zz_states, gram, bit_matrix
R = lambda f: json.load(open(os.path.join("results", f)))
px = load_prices()          # frozen file, sha256-checked
print(px.shape, px.index[0].date(), "->", px.index[-1].date())
(px / px.iloc[0]).plot(figsize=(10, 4), legend=False, logy=True, title="18 ETFs, growth of $1 (log scale)"); plt.show()
""")
md("""
## 1. Prophet forecast (same configuration as our earlier forecaster)
`Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
changepoint_prior_scale=0.05, interval_width=0.8)` on the last 730 days of closes,
30-day horizon. The signal rule is BULLISH if the forecast is >= +3 %, BEARISH if <= -3 %.
""")
code("""
from prophet import Prophet
t = pd.Timestamp("2026-09-30"); s = px["SPY"].loc[t - pd.Timedelta(days=730):t]
m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
            changepoint_prior_scale=0.05, interval_width=0.8).fit(pd.DataFrame({"ds": s.index, "y": s.values}))
fc = m.predict(m.make_future_dataframe(periods=30))
ax = plt.figure(figsize=(10, 4)).gca()
ax.plot(s.index, s.values, color="k", lw=1, label="SPY close")
ax.plot(fc.ds, fc.yhat, label="Prophet yhat"); ax.fill_between(fc.ds, fc.yhat_lower, fc.yhat_upper, alpha=.2, label="80 % interval")
real = px["SPY"].loc[t:t + pd.Timedelta(days=30)]; ax.plot(real.index, real.values, color="crimson", lw=2, label="what happened")
ax.axvline(t, ls="--", c="gray"); ax.legend(); ax.set_title("SPY: Prophet 30-day forecast from 2026-09-30"); plt.show()
""")
md("""
## 2. The prediction task
Every month-end *t* and every ETF we compute 4 features, all from data up to *t*:
Prophet's 30-day forecast return `p_ret`, Prophet's in-sample residual `p_resid`,
12-1 month momentum and 21-day volatility. Target: will this ETF finish **above the
median of the 18 next month** (exactly 9 winners per month, so a coin scores 50 %).
Features are converted to cross-sectional ranks in (0,1) -> 4 numbers -> **4 qubits**.
""")
code("""
_, df, U = build_panel()
df[["date", "asset", "p_ret", "p_resid", "mom_12_1", "vol_21", "fwd_ret", "y"]].tail(18).round(4)
""")
md("""
## 3. Quantum kernel (QML notebook, Parts 2-5)
ZZ feature map with the notebook's product data map, scaled by a bandwidth lambda
(Part 5: large lambda -> exponential kernel concentration). Our batched statevector
kernel is checked against `FidelityStatevectorKernel` to 1e-15 (`results/CONTROLS_SIM.json`).
""")
code("""
from qiskit.circuit.library import zz_feature_map
product_map = lambda x: x[0] if len(x) == 1 else x[0] * x[1]
display(zz_feature_map(4, reps=2, data_map_func=product_map).decompose().draw("mpl", fold=-1))
sel = R("DEV_SELECTION.json")["selected"]; lam = sel["qsvm"][0]
idx = np.where((df.date >= "2013-01-01") & (df.date <= "2013-12-31"))[0]
order = idx[np.argsort(df.y.values[idx], kind="stable")]
Kq = gram(zz_states(lam*np.pi*U[order]), zz_states(lam*np.pi*U[order]))
Kr = np.exp(-sel["rbf"][0] * ((U[order][:, None] - U[order][None]) ** 2).sum(-1))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, K, tl in [(ax[0], Kq, f"quantum ZZ kernel (lambda={lam})"), (ax[1], Kr, f"RBF kernel (gamma={sel['rbf'][0]})")]:
    a.imshow(K, cmap="viridis"); a.set_title(tl + "\\nsorted: losers | winners")
for l in [0.1, 0.3, 0.5, 1.0]:
    K = gram(zz_states(l*np.pi*U[idx]), zz_states(l*np.pi*U[idx])); ax[2].hist(K[np.triu_indices_from(K, 1)], bins=40, alpha=.5, label=f"lambda={l}")
ax[2].set_title("off-diagonal kernel values"); ax[2].legend(); plt.show()
""")
md("""
## 4. Walk-forward results (152 test months x 18 ETFs)
Hyper-parameters were chosen on 2009-2013 only. Arms: Prophet rule, momentum rule,
QSVM, VQC, RBF-SVM, logistic regression.
""")
code("""
res = R("RESULT.json"); arms = pd.DataFrame(res["arms"]).T
display(arms.round(3))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].bar(arms.index, arms.acc_pct, yerr=1.96 * arms.acc_se_pct, color=["#888", "#888", "#7b3fe4", "#3f8fe4", "#3f8fe4", "#7b3fe4"])
ax[0].axhline(50, c="k", ls="--"); ax[0].set_ylim(44, 58); ax[0].set_ylabel("top-half accuracy %"); ax[0].tick_params(axis="x", rotation=30)
ts = pd.read_csv("results/test_scores.csv", parse_dates=["date"])
from qps.panel import top9_accuracy
for col, c in [("qsvm", "#7b3fe4"), ("rbf", "#3f8fe4"), ("p_ret", "#888")]:
    a = ts.groupby("date").apply(lambda d: top9_accuracy(d[col].values, d.y.values))
    ax[1].plot(a.index, a.expanding().mean() * 100, label=col, color=c)
ax[1].axhline(50, c="k", ls="--"); ax[1].legend(); ax[1].set_title("running accuracy"); plt.show()
print("H1 Prophet channel:", res["H1"]); print("H2 QSVM vs RBF:", res["H2"])
print("controls VOID:", res["controls_void"], "| planted-signal check:", R("CONTROLS.json")["PC3_planted_acc"])
""")
md("""
## 5. QAOA portfolio (QAOA notebook, max-cut -> portfolio QUBO)
Each month: choose exactly K=4 of 18 ETFs to minimise
`-mu.x + (1/8) x^T S x + A (sum x - 4)^2` (mu = QSVM scores, S = 1-year covariance).
First a small **8-ETF instance with Qiskit's own `QAOAAnsatz` + `StatevectorEstimator`**,
exactly as in the training notebook; then the full 18-qubit backtest.
""")
code("""
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from scipy.optimize import minimize
from run_qaoa_portfolio import build_cost
sub = UNIVERSE[:8]; t = ts.date.max(); d = ts[ts.date == t].set_index("asset").loc[sub]
mu = ((d.qsvm - d.qsvm.mean()) / d.qsvm.std()).values
S = px[sub].pct_change().loc[:t].iloc[-252:].cov().to_numpy(copy=True); S = S / np.mean(np.diag(S))
B = bit_matrix(8).astype(float); E, _ = build_cost(mu, S, B); E = (E - E.mean()) / E.std()
# exact Pauli-Z expansion of the diagonal cost (Walsh transform), as SparsePauliOp
terms = []
for Sm in range(256):
    sel_ = np.array([(Sm >> i) & 1 for i in range(8)]); c = float(E @ ((-1.0) ** (B @ sel_))) / 256
    if abs(c) > 1e-12: terms.append(("".join("Z" if sel_[7 - q] else "I" for q in range(8)), c))
H = SparsePauliOp.from_list(terms); ans = QAOAAnsatz(H, reps=2)
est = StatevectorEstimator()
f = lambda x: float(est.run([(ans, H, x)]).result()[0].data.evs)
opt = minimize(f, [0.4, 0.2, 0.2, 0.4], method="COBYLA", options={"maxiter": 200})
circ = ans.assign_parameters(opt.x); circ.measure_all()
counts = StatevectorSampler(seed=1).run([circ], shots=2000).result()[0].data.meas.get_counts()
best = int(np.argmin(E)); top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
print("brute-force optimum:", [sub[i] for i in range(8) if (best >> i) & 1])
for bits, n in top:
    b = int(bits, 2); print(f"{bits}  {n:4d} shots  {'<- optimum' if b == best else ''}  {[sub[i] for i in range(8) if (b >> i) & 1]}")
""")
code("""
if os.path.exists("results/RESULT_QAOA.json"):
    q = R("RESULT_QAOA.json"); print("H3 QAOA solver:", q["H3"]); print("H4 portfolio:", q["H4"])
    display(pd.DataFrame(q["performance"]).T.round(3))
else:
    print("run_qaoa_portfolio.py has not finished yet (152 sequential 18-qubit QAOA runs)")
from IPython.display import Image
display(Image("figures/fig_portfolios.png"))
""")
md("""
## 5b. Regimes, 9 sector ETFs (9 qubits), and POMDP / GRPO agents
* **Regimes**: volatility regime (18 ETFs), bull/bear month (SPY), risk-on/off (SPY vs TLT),
  against persistence, VIX, Prophet and Markov-switching baselines.
* **9 sector SPDRs** shrink the QAOA problem to 9 qubits (84 valid portfolios).
* **POMDP** belief agent (2-regime HMM filter) and **GRPO** (group-relative policy optimisation,
  Plackett-Luce policy, 5 seeds, shuffled-reward control) on the same top-K decision.
""")
code("""
from IPython.display import Image
rg, sc, rl = R("RESULT_REGIME.json"), R("RESULT_SECTOR.json"), R("RESULT_RL.json")
print("Regime quantum-vs-RBF test:", rg["RG_H1"])
print("9-qubit QAOA:", {k: sc["qaoa_9q"][k] for k in ("S1_argmax_rate", "S2_mean_p_opt", "S3_best16_rate", "S3_random_baseline", "verdict")})
print("GRPO / POMDP Sharpe (18 ETFs):", {k: round(rl["U18"]["portfolio"][k]["sharpe"], 2) for k in ("grpo", "pomdp", "equal_weight")})
display(Image("figures/fig_regime.png")); display(Image("figures/fig_sector_rl.png"))
""")
md("""
## 6. Recursive self-improvement of Prophet (RSI)
*Stage Q:* Prophet tunes its own configuration (6 on/off knobs = 6 qubits): fit a
quadratic surrogate to the configurations tried so far, let QAOA propose the next one,
repeat (24 evaluations of 64). *Stage K:* a quantum-kernel ridge model learns Prophet's
own residuals and adds a correction; a new round is kept only if the 2009-2013 error falls.
Then the improved forecaster is scored once on 2014-2026 against plain Prophet.
""")
code("""
r = R("RESULT_RSI.json"); land = pd.Series(r["stage_Q"]["landscape"]).sort_values()
ax = land.plot.bar(figsize=(14, 3), color=["crimson" if k == r["stage_Q"]["base"] else "#7b3fe4" for k in land.index])
ax.set_ylabel("dev MAE"); ax.set_ylim(land.min() * 0.95, land.quantile(0.9)); ax.set_title("all 64 Prophet configurations (red = original)"); plt.show()
print("search loops:", json.dumps(r["stage_Q"]["loops"], indent=1))
print("stage K:", json.dumps(r["stage_K"], indent=1, default=str))
display(pd.DataFrame(r["test_metrics"]).T.round(4)); print("H5:", r["H5"])
""")
md("## 7. Live forecast (latest close, fresh data) — not investment advice")
code("""
import glob
live = sorted(glob.glob("results/LIVE_*.md"))
from IPython.display import Markdown; display(Markdown(open(live[-1]).read()) if live else "run `python run_live.py` first")
""")
md("## 8. Verdicts (read from the result files)")
code("""
qa = R("RESULT_QAOA.json") if os.path.exists("results/RESULT_QAOA.json") else None
v = dict(H1_prophet_informative=R("RESULT.json")["H1"]["verdict"], H2_qsvm_vs_rbf=R("RESULT.json")["H2"]["verdict"],
         H3_qaoa_finds_optimum=qa["H3"]["verdict"] if qa else "running", H4_portfolio_vs_equal_weight=qa["H4"]["verdict"] if qa else "running",
         H5_self_improvement=R("RESULT_RSI.json")["H5"]["verdict"])
display(pd.Series(v, name="verdict").to_frame())
display(Image("figures/fig_stock_classifiers.png")); display(Image("figures/fig_self_improvement.png"))
""")
md("""
**How to read this.** A CLOSED-NEG verdict is a real result, not a failure of the
notebook: it says the pre-registered effect was not there. All quantum parts here are
exact classical simulations (4-18 qubits), so even a positive result would be a
statement about a kernel or a heuristic, never a quantum advantage.
""")
nb["cells"] = C
nbf.write(nb, "quantum_prophet_stocks.ipynb")
print("written", len(C), "cells")
