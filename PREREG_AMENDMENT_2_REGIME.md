# PREREG AMENDMENT 2 — regime prediction (2026-10-23)

Scope: also compare regime-prediction performance. Written and hashed before any regime label or model was
computed. Base PREREG (50943aae…) and AMENDMENT_1 (71cb6aba…) unchanged.
Same frozen prices (sha256 8d92ddec…) plus the frozen daily VIX column of
`data/macro_2005-01-01_2026-10-17.csv` (sha256 fa019dde…). Decision dates, dev (2009-2013) and test
(2014-01..2026-08) periods, expanding windows and the "label end <= decision date" rule are as in
the base PREREG.

## Regime targets (label of decision month-end t covers (t, next month-end])
* RG1 volatility regime, panel of 18 ETFs (2,736 test decisions): y = 1 iff the ETF's realised
  volatility next month (std of daily log returns) exceeds the median of its own monthly realised
  volatilities over all months ending <= t.
* RG2 bull/bear month, market (SPY, 152 test months): y = 1 iff SPY's next-month return > 0.
* RG3 risk-on/off, market (152 test months): y = 1 iff SPY's next-month return > TLT's.

## Features (4 -> 4 qubits); each mapped to (0,1) by the training window's empirical CDF
* RG1: log(current-month realised vol / the same expanding median), VIX percentile (VIX at t vs all
  VIX closes <= t), own 1-month return, |Prophet residual p_resid|.
* RG2, RG3: SPY Prophet p_ret, SPY p_ret - TLT p_ret, VIX percentile, SPY log(current vol / median).

## Arms
Learned (dev-selected on 2009-2013, same grids as the base PREREG): QSVM (ZZ map, product data map),
VQC (RealAmplitudes reps 3, COBYLA 150, warm start), RBF-SVM, logistic regression, ILL rule
classifier (joint-tercile partitions of feature subsets, <= 4 rules, min gap 1e-3).
Fixed rules: majority class of the training window; persistence (RG1: current vol above the
median; RG2/RG3: last month's regime repeats); RG1 VIX rule (VIX percentile > 0.5); RG2/RG3 Prophet
rule (RG2: SPY p_ret > 0; RG3: SPY p_ret > TLT p_ret); RG2/RG3 Markov-switching model (statsmodels
MarkovRegression, 2 regimes, switching mean and variance, on the trailing 756 daily returns of SPY
for RG2 and of SPY - TLT for RG3; predict the sign of the expected mean over the next 21 days from
the filtered state probabilities and the transition matrix; a failed fit falls back to persistence
and is counted).
QAOA is not used: regime prediction is not an optimisation problem here.

## Tests (accuracy; month-paired sign-flip, 10,000 flips; RG1 uses per-month mean accuracy)
* RG-H1 (closing test for this amendment): QSVM - RBF-SVM. POS iff, after Holm over RG1-RG3, some
  task has gain >= 1.0 pt and adjusted p < 0.05; else CLOSED-NEG.
* Labels: every learned arm vs persistence and vs majority; VQC - logistic; ILL - QSVM;
  Prophet rule vs majority (RG2, RG3); Markov-switching vs persistence (RG2, RG3); AUC where a
  score exists; balanced accuracy.
* Controls: within-month shuffled RG1 labels (10 shuffles) -> QSVM and RBF within 50 +- 3 pt of the
  shuffled base rate; persistence must beat majority on RG1 (volatility clustering is a known
  effect — if it does not, the RG1 label is VOID).
