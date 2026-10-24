# QML ensemble selector — exploratory result

Evaluation: 2019-01-31 to 2026-08-31, 92 months.

Six experts; choose one of 15 equal-weight pairs. Four-qubit ZZ fidelity kernel
ridge predicts next-month pair loss using resolved historical outcomes only.
Nine configurations per kernel, selected on 2016–2018; evaluation begins in 2019.

| Arm | Rank-score Brier ↓ | Top-9 accuracy | Pair optimum hit | Regret ↓ |
|---|---:|---:|---:|---:|
| hindsight_oracle | 0.235348 | 60.63% | 100.0% | 0.000000 |
| all_six | 0.276494 | 52.42% | — | 0.041146 |
| fixed | 0.292611 | 49.88% | 13.0% | 0.057263 |
| expanding | 0.294113 | 50.24% | 3.3% | 0.058765 |
| trailing12 | 0.297005 | 50.24% | 5.4% | 0.061657 |
| random_pair_expectation | 0.297597 | 50.74% | — | 0.062249 |
| rbf | 0.297816 | 49.76% | 8.7% | 0.062468 |
| quantum | 0.301850 | 49.52% | 4.3% | 0.066502 |

Verdict: **NO_QML_BENEFIT_DEMONSTRATED**.

RBF minus QML Brier = -0.004034; paired 6-month circular-block
bootstrap 95% interval [-0.014311, 0.005659]. Positive favors QML.

All implemented controls passed: input labels/returns, chronological training,
future-target invariance, PSD/normalized Gram, planted kernel signal,
and fixed-selection shuffled-label accuracy (49.75%).

The hindsight oracle uses future outcomes and cannot be deployed. Its optimum
is restricted to the 15 pairs under Brier loss; accuracy can favor another pair.
All-six can have negative regret because it lies outside the pair candidate set.

Limitations: previously examined data, no fresh holdout, only 92 monthly decisions,
uncalibrated rank scores, fixed pair size/weights/context, noiseless classical
simulation. The bootstrap conditions on the frozen hyperparameter selection.
Base models were not refitted here. No return, trading, hardware, global-optimality
or quantum-speedup claim. The shuffle is a scoring sanity check, not a retrained null.

Reproduce from repository root: `python run_ensemble_selector.py`.
See `ENSEMBLE_PROTOCOL.md`, `monthly.csv`, and `RESULT.json` for details.

Method reference: https://www.nature.com/articles/s41586-019-0980-2
