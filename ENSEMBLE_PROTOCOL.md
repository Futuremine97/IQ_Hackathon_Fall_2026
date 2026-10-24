# Quantum ensemble selector: exploratory protocol

Written before running `run_ensemble_selector.py`. This is an exploratory reuse of
an already examined dataset, not a new confirmatory preregistration.

Question: can a four-qubit quantum fidelity kernel learn which equal-weight pair
of six saved ETF forecasters will have the lowest next-month Brier loss?
Experts: Prophet, momentum, QSVM, RBF SVM, logistic regression and VQC. Convert
each month's scores to (rank - 0.5)/18; these are rank scores, not calibrated
probabilities. Enumerate all 15 pairs. Primary loss is mean squared error of the
pair's mean rank score against the top-half label. Secondary metric: top-9 accuracy.
The scope of optimality is these 15 equal-weight pairs and this loss, not all
possible ensemble sizes, continuous weights, or future performance.

Use saved walk-forward base predictions without refitting them. Verify their
labels and returns against the frozen price panel. At decision t, use only past
labels with label_end <= t. Four context features: preceding three months' average
expert loss, preceding three months' quantum-minus-classical expert loss,
last resolved month's average asset return, and its cross-sectional dispersion.
Scale with training means and standard deviations, then arctan to (0,1), identically
for quantum and classical kernels. Predict centered losses for all 15 pairs by
kernel ridge regression and select the lowest predicted loss (stable ties).

Warm-up: 2014-2015 (minimum 18 usable context rows). Select hyperparameters using
walk-forward mean selected-pair Brier loss in 2016-2018 only. Quantum: ZZ product
map, two repetitions, scale in {0.1,0.3,1}; classical: RBF gamma in {0.1,1,10}.
Both ridge penalties in {0.01,0.1,1}: nine configurations each. Freeze for evaluation
2019-01 through 2026-08; expand training monthly with resolved labels.

Baselines: all-six equal weight, fixed best pair from pre-2019 data, expanding
historical best pair, trailing-12-month best pair, and uniform random pair's exact
expected metrics. An explicitly non-deployable hindsight oracle chooses the
lowest realized Brier loss each month. Report optimum hit rate (ties allowed),
regret to that oracle, and paired Brier difference vs RBF. A 6-month circular block
bootstrap (10,000 replicates, fixed seed 2026) gives a 95% interval. Evidence for
QML benefit requires that interval to be strictly positive for RBF minus quantum,
and QML mean loss below all-six, fixed, expanding, trailing, and random baselines.

Controls: normalized positive-semidefinite quantum Gram matrix, a synthetic
kernel-generated target learned better than its constant predictor, chronological
assertions, unchanged past predictions after perturbing future target rows, and
100 independent within-month asset-label shuffles evaluated with the original
selections (a scoring sanity check, not a retrained learning null). Failures void
the result. Simulation only: no hardware or computational advantage claim.

Reference: Havlicek et al. (2019),
https://www.nature.com/articles/s41586-019-0980-2 .
