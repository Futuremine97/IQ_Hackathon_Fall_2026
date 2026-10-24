# Turnaround stocks — quantum vs classical

Pre-registered in `PREREG.md` (sha256 in `PREREG.sha256`, hashed before any data was downloaded).
Data in `data/` (gzipped; sha256 of the uncompressed files in `data/SHA256SUMS`).

**Question.** Among S&P 500 stocks that have fallen at least 40 % below their 52-week high, can a
quantum classifier tell which ones will recover (beat SPY over the next 6 months) better than
classical methods, and can QAOA pick a better diversified basket?

**Universe caveat (stated before the run).** The tickers are today's S&P 500 members (481 with
data), so companies that failed or were removed are missing (survivorship bias). Absolute
turnaround rates and returns are therefore inflated; only method-vs-method comparisons are meaningful.

**Setup.** 5,124 candidate stock-months (2,284 of them in the 2008-09 crisis, used for training).
Test 2014-01..2026-03: 2,232 candidates in 138 months, 56.4 % of which beat SPY. Eight features
(drawdown, 1- and 3-month return, rebound off the 52-week low, volatility, volume surge, Prophet
30-day forecast, SPY drawdown) → 8 qubits. Hyper-parameters chosen on 2009-2013 only.

**Controls passed.** Shuffled labels → AUC 0.501 (QSVM) / 0.510 (RBF); a planted signal is found
(0.717 / 0.723); every training label ends before its decision date (asserted).

![turnaround](figures/fig_turnaround.png)

| method | test AUC (95 % month-block CI) | top-4 picks minus all candidates (6-month excess return, pt) |
|---|---|---|
| QSVM (8 qubits) | 0.549 [0.516, 0.583] | +1.7 |
| VQC (8 qubits) | 0.502 [0.477, 0.530] | −0.5 |
| RBF-SVM | 0.538 [0.503, 0.573] | +2.4 |
| logistic regression | 0.547 [0.518, 0.581] | +1.4 |
| gradient boosting | 0.537 [0.501, 0.570] | +4.4 |
| rule: furthest off the 52-week low | **0.575** [0.524, 0.622] | **+5.6** |
| rule: deepest drawdown | 0.544 | +0.3 |
| rule: 1-month reversal | 0.520 | +2.2 |
| rule: Prophet forecast | 0.522 | +1.5 |

* A weak signal exists (best classical model AUC 0.538, CI just above 0.5).
* **Quantum vs classical:** QSVM − RBF AUC +0.011, CI [−0.003, +0.026] includes 0; the top-4 selection
  is 0.7 pt *worse* (p = 0.61) → no quantum advantage. The VQC is at chance level.
* The simplest rule — pick the stocks that have already rebounded most from their 52-week low — has the
  best AUC and selection value; no learned model beats it.
* **12-qubit QAOA** (choose a sector- and correlation-diversified 4 of the top 12, 59 months): the most
  likely state is the optimum in 96.6 % of months, but P(optimum) = 0.13 % is *below* uniform sampling of
  the 495 valid baskets (0.20 %), and 16 shots never found it (0/59 vs 3.2 % for random). The QAOA basket,
  the exact optimum and the plain top-4 earn about the same (+3.2 / +3.0 / +3.3 pt): diversification added nothing.

Reproduce: `python run_turnaround.py` (several hours: 5,124 Prophet fits plus walk-forward kernel SVMs),
then `python make_figures.py`. Paper study, not investment advice.
