# PREREG AMENDMENT 3 — sector-ETF universe (9 assets -> 9 qubits) (2026-10-23)

Motivation: a sector-ETF universe has fewer assets. Written and hashed before any sector-universe result.
Base PREREG and AMENDMENTS 1-2 unchanged. Same frozen prices, Prophet features (features_base.csv),
dev 2009-2013 / test 2014-01..2026-08, expanding windows, label-end rule.

Universe: the 9 original Select Sector SPDRs XLK XLF XLE XLV XLI XLP XLY XLU XLB (all trading since
1998). XLRE (2015) and XLC (2018) are excluded: they did not exist for the whole test period.

## Classifiers (labels)
Cross-sectional target within the 9: y = 1 for the 4 ETFs strictly above the median next-month
return (the median ETF and below = 0). Features = within-9 ranks of [p_ret, p_resid, mom_12_1,
vol_21], u = (rank - 0.5)/9. Arms and dev grids as in the base PREREG (QSVM, VQC, RBF-SVM,
logistic, Prophet rule, momentum rule, ILL rules). Prediction = top 4 by score. Accuracy per month
(coin expectation 41/81 = 50.6 %). Test: QSVM - RBF, sign-flip (label only).

## QAOA on 9 qubits (the motivating question)
QUBO as in H3 with N = 9, K = 3 (C(9,3) = 84 feasible portfolios, 2^9 = 512 states), mu = z-scored
sector-QSVM scores, g = 1, same penalty rule, p = 3, X mixer, COBYLA maxiter 300, two starts
(ramp; previous month's optimum). Because 1,024 uniform feasible draws would hit the optimum with
probability 1-(1-1/84)^1024 ~ 1, the 1,024-shot criterion is uninformative here; registered
metrics instead:
* S1 argmax readout: the single most probable basis state == brute-force optimum (rate over 152).
* S2 mean P(optimum) in the final state vs 1/84 (uniform over feasible) and 1/512 (uniform).
* S3 best of 16 shots == optimum, vs the generous random baseline (16 uniform feasible draws:
  1-(1-1/84)^16 = 0.175), one-sided binomial.
PASS-9q iff S1 >= 90 % AND S3 binomial p < 0.05. Also reported next to the running 18-qubit H3 run.
## Portfolio (label)
QAOA top-3 (S1 readout) equal weight vs equal-weight 9 sectors, 10 bp costs, Sharpe, block bootstrap
CI90 (block 6, 2,000 reps).
