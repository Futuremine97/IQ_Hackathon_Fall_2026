# PREREG — Prophet + QML + QAOA on an 18-ETF monthly panel (2026-10-23)

Written and hashed BEFORE any feature, model or backtest result was computed.
Paper backtest only. No live orders. Not investment advice.

## 0. Question

1. Does a Prophet forecast (daily closes, the same configuration as our earlier
   Prophet forecaster) carry cross-sectional information about next-month ETF returns?
2. Does a 4-qubit quantum-kernel SVM (ZZ feature map, the hackathon QML notebook) or
   a variational quantum classifier (VQC) built on Prophet + classical features rank
   ETFs better than classical kernels on the SAME features?
3. Can QAOA (the hackathon QAOA notebook, here on a portfolio QUBO) pick the exact
   optimum of a monthly top-K mean-variance QUBO, and does the resulting portfolio
   beat equal weight?
4. Can a recursive self-improvement loop (QAOA-driven configuration search + quantum-
   kernel residual boosting, each round accepted only if it improves the development
   period) make the Prophet forecast more accurate out of sample?

## 1. Data (frozen)

* `data/prices_2005-01-01_2026-10-17.csv`, adjusted daily closes (yfinance), sha256
  `8d92ddeca2d56fd319929a4b62543c1674999a145f39d2d2d2d7dd41c14cf028`. 18 ETFs:
  XLK XLF XLE XLV XLI XLP XLY XLU XLB SPY QQQ IWM EFA EEM TLT IEF GLD VNQ
  (all listed before 2005 — no survivorship selection).
* Decision dates = last trading day of each month. Features at t use closes <= t.
* Label month t: simple return close(t) -> close(next month-end). Cross-sectional
  target y = 1 for the 9 ETFs above the median of that month (balanced 9/9).
* Training set at decision t = all (date, ETF) samples whose label END date <= t
  (asserted in code for every decision).
* Development (hyper-parameter selection only): decisions 2009-01 .. 2013-12 (60).
  Test (evaluated once, after all choices are frozen): 2014-01 .. 2026-08 (152 months,
  2,736 decisions). 2026-09 has no complete label -> used only for the live forecast.

## 2. Features

Prophet (fit per ETF per month-end on the trailing 730 calendar days of closes, y =
price): `Prophet(daily_seasonality=False, weekly_seasonality=True,
yearly_seasonality=span>=400 days, changepoint_prior_scale=0.05, interval_width=0.8)`,
horizon 30 calendar days.
  * `p_ret`   = yhat(t+30d)/close(t) - 1
  * `p_resid` = (close(t) - yhat(t))/close(t)   (in-sample fitted value at t)
  * `p_slope` = (trend(t+30d) - trend(t))/close(t)          (descriptive)
Classical: `mom_12_1` = close(t-21)/close(t-252) - 1; `vol_21` = std of the last 21
daily log returns; `mom_1` = close(t)/close(t-21) - 1 (descriptive).

Model input (4 features -> 4 qubits): cross-sectional ranks of
[p_ret, p_resid, mom_12_1, vol_21] per date, u = (rank - 0.5)/18 in (0,1).

## 3. Arms (same inputs, same training windows)

| id | arm | trained? | dev grid |
|---|---|---|---|
| M0 | Prophet rule: score = p_ret | no | - |
| M5 | Momentum rule: score = mom_12_1 | no | - |
| M1 | QSVM: ZZ map (4 qubits, reps 2, full entanglement, data map phi_i = x_i, phi_ij = x_i x_j), x = lambda*pi*u, exact statevector kernel, SVC(precomputed) | yes | lambda {0.1,0.2,0.3,0.5,1.0} x C {0.1,1,10} |
| M3 | RBF-SVC on u | yes | gamma {0.1,0.3,1,3,10} x C {0.1,1,10} |
| M4 | Logistic regression on u | yes | C {0.01,0.1,1,10} |
| M2 | VQC: same ZZ map (lambda_v) + RealAmplitudes(4, reps 3), parity read-out, cross-entropy, COBYLA maxiter 150, warm start month to month, init seed 7, exact statevector | yes | lambda_v {0.3,0.5,1.0} |

Expanding window. Dev selection = highest dev mean accuracy; ties -> first in the
order listed. Scores -> predicted label: top 9 per date.

## 4. Hypotheses and kill conditions (test period only)

* **H1 Prophet channel.** Monthly Spearman IC of M0 vs next-month return; mean over
  152 months, two-sided t-test. INFORMATIVE iff p < 0.05; else CLOSED-NEG ("Prophet
  output uninformative for cross-sectional 1-month ETF ranking"). Also M0 accuracy vs a
  200-shuffle within-month label null band (descriptive).
* **H2 Quantum kernel (headline).** Per-month accuracy difference M1 - M3, mean over
  152 months, two-sided sign-flip permutation test (10,000 flips, seed 0).
  POS iff gain >= 1.0 pt AND p < 0.05; else CLOSED-NEG. A POS is a kernel-choice
  result on a 4-qubit classically simulated kernel — it is NOT a quantum-advantage
  claim under any outcome. Secondary label: M2 (VQC) - M4 (logistic), same test.
* **H3 QAOA solver.** Each test month: QUBO over 18 binary x_i minimise
  E = -sum mu_i x_i + (g/(2K)) x^T S x + A (sum x_i - K)^2 with mu = cross-sectional
  z-score of the M1 decision function, S = trailing-252-day daily-return covariance
  divided by its mean diagonal, K = 4, g = 1, A = 1.5 * max_i(|mu_i| + (g/K) sum_j |S_ij|).
  QAOA p = 3, X mixer, |+>^18 start, energy normalised to zero mean / unit std over all
  2^18 states, COBYLA maxiter 300 from two starts (linear ramp; previous month's
  optimum), 1,024 shots sampled from the exact final distribution (seed = month index);
  solution = lowest-energy feasible sample (if none feasible: most probable feasible
  state, flagged). Brute force over 2^18 gives the optimum (asserted feasible).
  PASS iff solution == optimum in >= 90 % of test months AND the rate beats the
  generous random baseline (1,024 uniform feasible draws: 1-(1-1/3060)^1024 = 0.284,
  one-sided binomial p < 0.05).
* **H4 Portfolio.** QAOA top-4 equal weight, monthly, 10 bp per unit one-way turnover,
  vs equal-weight 18 (same cost rule). Annualised Sharpe of monthly returns (rf = 0).
  Circular block bootstrap (block 6, 2,000 reps, seed 0) 90 % CI of the Sharpe
  difference. POS iff the CI lower bound > 0; else CLOSED-NEG.
* **H5 Recursive self-improvement (RSI) of the Prophet forecast.**
  Target: next-month return per ETF; loss = absolute error; baseline = M0's p_ret.
  - Stage Q (configuration self-search): 6 binary knobs, baseline = 000000:
    k1 changepoint_prior_scale 0.05 / 0.005; k2 seasonality additive / multiplicative;
    k3 yearly on / off; k4 weekly on / off; k5 lookback 730 / 365 days; k6 growth linear / flat.
    Dev objective = MAE on 20 quarterly dev month-ends (Mar/Jun/Sep/Dec 2009-2013) x 18.
    All 64 configurations are evaluated on dev (ground-truth landscape). Loop: 8 random
    initial configs, then 16 rounds of {fit quadratic surrogate (ridge 1e-3 on bits and
    pairwise products) -> QAOA p = 2 on 6 qubits minimises it -> evaluate best unseen
    sampled config}. 24 evaluations total; repeated for 50 seeds. Label: fraction of
    seeds reaching the dev-best config vs random search (24/64 = 0.375).
    Config carried forward = the seed-0 loop's best-found config.
  - Stage K (residual self-boosting): round r fits quantum-kernel ridge regression
    (same ZZ map, lambda = M1's dev choice; alpha {0.1,1,10} chosen on dev round 1) on
    the walk-forward residuals of round r-1, f_r = f_{r-1} + correction. Rounds 1..3;
    a round is ACCEPTED only if it lowers dev MAE (walk-forward); stop at first reject.
    Classical twin: RBF kernel ridge (gamma = M3's dev choice), same protocol.
  - Kill: final RSI forecaster's test MAE vs baseline Prophet, per-month mean-abs-error
    difference, sign-flip test. POS iff MAE lower AND p < 0.05; else CLOSED-NEG.
    Reported alongside: random walk (forecast 0) and expanding historical mean MAE,
    directional accuracy, IC, and dev-gain vs test-gain (overfitting gap).

H2 and H5 are the row-closing tests (Holm over the two at 0.05). H1, H3, H4 and the
secondary comparisons are labels.

## 5. Controls (CONTROLS.json is written before RESULT.json)

* PC1 kernel: our batched statevector Gram == qiskit-machine-learning
  FidelityStatevectorKernel on 60 random points, max |diff| < 1e-10.
* PC2 QSVM learns when signal exists: qiskit `ad_hoc_data` (notebook setup) test acc >= 0.9.
* PC3 planted signal: labels = top-9 of (u_p_ret + u_mom + N(0, 0.35)) on the real
  feature panel -> M1 and M3 test accuracy >= 60 %.
* PC4 QAOA simulator == qiskit `QAOAAnsatz` + `Statevector` expectation (notebook's
  5-node max-cut and an 8-asset portfolio QUBO, random angles), |diff| < 1e-9.
* PC5 VQC forward pass == qiskit statevector parity probabilities, |diff| < 1e-10.
* NC1 within-month label shuffle (20 shuffles, frozen hyper-parameters): M1 and M3
  test accuracy must lie within 50 +- 3 pt (else VOID: leakage).
* Look-ahead: assertion that every training label end date <= decision date.
VOID (no verdict) if PC1, PC4, PC5 or NC1 fails.

## 6. Claim ceiling

Historical simulation only; no "beats the market", no live-trading or return claim, no
quantum-advantage claim (4-18 qubits, classically simulated, exact statevectors).
