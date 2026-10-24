# PREREG — finding turnaround stocks with quantum algorithms (2026-10-24)

Question: can quantum algorithms help find turnaround stocks? Written and hashed before any price
was downloaded or any result computed. Paper study only; not investment advice. All quantum
circuits are exact classical simulations (<= 12 qubits).

## Universe and data
Tickers = the offline S&P 500 constituent snapshot (~2025, GICS sectors) shipped with our earlier
trading app, plus SPY as the benchmark. Daily adjusted close and volume from Yahoo Finance via
yfinance, 2005-01-01 .. 2026-10-23, downloaded once and frozen with sha256.
**Survivorship bias (stated now):** the list contains only today's survivors; firms that went
bankrupt or were removed are missing, so turnaround success rates are inflated. Only
comparisons between methods on the same universe are interpretable; absolute returns are not.

## Events and labels
Decision dates = month-ends. A stock-month is a CANDIDATE (distressed) iff its close <= 0.60 x its
max close over the previous 252 trading days, it has >= 504 trading days of history, and its
adjusted close >= 1. Label y = 1 (turnaround) iff its return over the next 126 trading days exceeds
SPY's over the same days. Label end = that 126th day; a training sample is usable at t only if its
label end <= t (asserted). Decisions: dev 2009-01..2013-12 (hyper-parameters only), test
2014-01 .. last month-end whose 126-day label completes inside the data.

## Features (8 -> 8 qubits), mapped to (0,1) by the training window's empirical CDF
f1 drawdown = close / 252-day max - 1; f2 21-day return; f3 63-day return; f4 rebound = close /
252-day min - 1; f5 63-day volatility of daily log returns; f6 volume ratio = mean volume 21 d /
mean volume 252 d; f7 Prophet 30-day forecast return (same Prophet configuration as the ETF study,
730-day window); f8 SPY drawdown (close / 252-day max - 1), shared by all candidates of a month.

## Arms
Rules (no training): all candidates (base rate), deepest drawdown (most negative f1), reversal
(highest f2), off-the-low (highest f4), Prophet rule (highest f7).
Classical: logistic regression (C {0.01,0.1,1,10}), RBF-SVM (gamma {0.1,0.3,1,3} x C {0.1,1,10}),
gradient boosting (HistGradientBoosting, max_depth {2,3} x learning_rate {0.05,0.1}, 200 iterations).
Quantum: QSVM (ZZ map, 8 qubits, reps 2, full entanglement, product data map, x = lambda*pi*u,
lambda {0.1,0.25,0.5,1.0} x C {0.1,1,10}); VQC (same map, RealAmplitudes(8, reps 2), parity
read-out, COBYLA 150, warm start, lambda {0.25,0.5,1.0}).
Lattice learning (label only): ILL rule classifier, candidate partitions = joint terciles of every
feature subset of size <= 2 (36 rules), <= 4 rules, min gap 1e-3.
Expanding training window; dev selection by pooled dev AUC (ties -> first in grid order).
Months with < 6 candidates are skipped for the selection metrics (kept for AUC).

## Metrics
* Pooled test AUC; 95 % CI by month-block bootstrap (2,000 reps, seed 0).
* Selection alpha per month: mean 126-day excess return (vs SPY) of the top-4 candidates by score
  minus the mean over all candidates that month. Paired arms compared by sign-flip (10,000) over months.

## Hypotheses
* TA-H2 (row-closing, quantum vs classical): QSVM - RBF-SVM. POS iff pooled AUC difference >= 0.01
  with bootstrap CI excluding 0 AND selection-alpha difference > 0 with sign-flip p < 0.05; else CLOSED-NEG.
* TA-H1 (label): best dev-selected classical model's test AUC CI excludes 0.5 ("signal exists").
* Labels: VQC - logistic; GBM - RBF; ILL - QSVM; every model vs the rules; Prophet rule vs base rate.
* TA-H3 QAOA (label): each test month with >= 12 candidates, take the top 12 by QSVM score; QUBO over
  12 bits: minimise -sum mu_i x_i + (1/(2K)) x^T S x + 0.5 * (# same-sector pairs selected)
  + A (sum x - K)^2, K = 4, mu = z-scored QSVM scores, S = 252-day daily-return correlation matrix,
  A = 1.5 max_i(|mu_i| + (1/K) sum_j |S_ij| + 0.5 * 11). QAOA p = 3, X mixer, ramp start only (months
  run independently), COBYLA 300. Metrics as in the 9-qubit study: S1 argmax == optimum (rate),
  S2 P(optimum) vs 1/495 (C(12,4)), S3 best of 16 shots == optimum vs random 16 valid draws
  (1-(1-1/495)^16 = 0.032), one-sided binomial. PASS iff S1 >= 90 % and S3 p < 0.05.
  Selection alpha of the QAOA (S1 readout) picks and of the brute-force optimum, vs greedy top-4.
* Controls: shuffled labels within month (10 shuffles) -> QSVM and RBF AUC within 0.5 +- 0.03;
  planted label (y = top half of f2 + f4 + noise) -> QSVM and RBF AUC >= 0.65; look-ahead asserts.
  VOID if the shuffle control fails.
