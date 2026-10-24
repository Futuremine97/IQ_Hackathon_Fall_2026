"""Build the reproducible ensemble-selection experiment notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''
# Can QML choose the optimal ensemble?

**IQ Hackathon Fall 2026 — exploratory extension**

기존 6개 예측 모델 중 2개를 선택하는 QML 선택기를 고전 선택기 및 전체 평균과 비교합니다.
‘최적’은 다음 달 실제 Brier 손실이 가장 작은 **15개 동일 가중치 모델 쌍 중 하나**로 정의합니다.
미래 결과를 아는 oracle은 선택기의 평가 기준으로만 사용합니다.

The four-qubit quantum kernel learns a context-dependent loss estimate for each
pair. It does not guarantee a globally optimal ensemble or a quantum speedup.
This notebook reruns the experiment, displays its executed results, and checks provenance.

**Data status:** the underlying dataset and earlier results were already examined.
This is an exploratory extension, not an independent holdout or a new confirmatory study.
''')
md('''
## 1. Environment and source

Open this notebook from the repository root. In a fresh Python environment,
uncomment the minimal dependency installation below. Base forecasts and frozen
prices are included in the repository; this experiment requires no network data.
For Colab, first clone the repository and change into its root using the commented commands.
''')
code('''
# Fresh environment / Colab setup (uncomment if needed):
# !git clone https://github.com/Futuremine97/IQ_Hackathon_Fall_2026.git
# %cd IQ_Hackathon_Fall_2026
# %pip install numpy pandas scipy matplotlib

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

ROOT = Path.cwd()
assert (ROOT / 'run_ensemble_selector.py').exists(), 'Run from the repository root.'
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__, 'Pandas:', pd.__version__)
display(Markdown((ROOT / 'ENSEMBLE_PROTOCOL.md').read_text()))
''')
md('''
## 2. Candidate pairs and chronological evaluation

Experts: Prophet, momentum, QSVM, RBF SVM, logistic regression, and VQC. Map
each expert's 18 scores to `(rank - 0.5)/18`, then average the two selected experts.
These scores are **not calibrated probabilities**. Pair loss is the mean squared
error against next-month top-half labels; accuracy classifies the nine highest scores.

At date $t$, historical labels must satisfy `label_end <= t` and `date < t`.
The four inputs are lagged mean expert loss, lagged quantum-minus-classical loss,
last resolved mean return, and last resolved cross-sectional return dispersion.
Scaling is fitted only on each training window.

* Warm-up: 2014–2015; minimum 18 usable training months.
* Tuning: walk-forward 2016–2018, nine configurations per kernel.
* Evaluation: 2019-01 through 2026-08; frozen hyperparameters, expanding training.

The quantum model uses a four-qubit ZZ fidelity kernel with two feature-map repetitions.
Both kernels use centered multi-output kernel ridge regression to predict all 15 losses.
The pair with the smallest predicted loss is selected before the next-month outcome.
''')
code('''
from run_ensemble_selector import EXPERTS, PAIR_NAMES, load_inputs, main

x, losses, ends, dates, pair_scores, expert_scores, labels, source_hash = load_inputs()
display(pd.DataFrame({'pair_id': range(len(PAIR_NAMES)), 'experts': PAIR_NAMES}))
print('Usable monthly contexts:', len(dates), '| Context dimensions:', x.shape[1])
print('Saved base-forecast SHA256:', source_hash)
assert x.shape[1] == 4 and losses.shape[1] == 15
''')
md('''
## 3. Rerun the complete experiment

This cell regenerates `results/ensemble_selector/RESULT.json`, `monthly.csv`, and
`REPORT.md`. Seeds, grids, split dates, and the decision rule are fixed in source.
The original base forecasters are loaded from their saved walk-forward predictions.
''')
code('''
main()
OUT = ROOT / 'results' / 'ensemble_selector'
result = json.loads((OUT / 'RESULT.json').read_text())
monthly = pd.read_csv(OUT / 'monthly.csv', parse_dates=['date'])
assert result['input_scores_sha256'] == source_hash
assert result['protocol_sha256'] == hashlib.sha256((ROOT / 'ENSEMBLE_PROTOCOL.md').read_bytes()).hexdigest()
assert not monthly.duplicated(['date', 'arm']).any()
print('Evaluation:', result['eval_start'], 'to', result['eval_end'])
print('Months:', result['n_eval_months'])
print('Selected hyperparameters (kernel scale, ridge):', result['selected_hyperparameters'])
''')
md('''
## 4. Results and baselines

Lower Brier loss is better. The hindsight oracle knows next month's realized
losses and cannot be deployed. The random-pair baseline reports exact expected
metrics over all 15 pairs. The all-six average lies outside the pair candidate set.
The fixed pair is chosen using only data resolved before the evaluation begins.
''')
code('''
summary = pd.DataFrame(result['arms']).T.sort_values('brier')
display(summary.rename(columns={'brier': 'Brier loss ↓', 'accuracy': 'Top-9 accuracy',
                                'oracle_hit': 'Optimal-pair hit', 'regret': 'Brier regret ↓'})
        .style.format({'Brier loss ↓': '{:.6f}', 'Top-9 accuracy': '{:.2%}',
                       'Optimal-pair hit': '{:.2%}', 'Brier regret ↓': '{:.6f}'}, na_rep='—'))
print('Verdict:', result['verdict'])
''')
code('''
deployable = summary.drop(index='hindsight_oracle')
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
colors = ['#7c3aed' if arm == 'quantum' else '#64748b' for arm in deployable.index]
axes[0].barh(deployable.index, deployable.brier, color=colors)
axes[0].invert_yaxis()
axes[0].set(xlabel='Mean rank-score Brier loss (lower is better)', title='2019–2026: 92 months')
wide = monthly.pivot(index='date', columns='arm', values='brier')
for arm, color in [('quantum', '#7c3aed'), ('rbf', '#0284c7'), ('all_six', '#059669')]:
    axes[1].plot(wide.index, wide[arm].expanding().mean(), label=arm, color=color)
axes[1].set(xlabel='Decision month', ylabel='Expanding mean Brier loss', title='Evaluation trajectory')
axes[1].legend()
plt.show()
''')
md('''
## 5. Uncertainty, optimum selection, and controls

The primary comparison is paired **RBF minus QML Brier loss**. Positive values
favor QML. The 95% interval uses 10,000 circular block-bootstrap replicates with
six-month blocks. The interval conditions on the chosen hyperparameters.
Evidence of QML benefit additionally requires lower mean loss than every deployable baseline.

An optimal-pair hit means the chosen pair ties the lowest realized Brier loss
among the 15 pairs. This does not imply best accuracy or investment return.
''')
code('''
delta = result['rbf_minus_quantum_brier']
low, high = result['block_bootstrap_95_ci']
print(f'RBF minus QML Brier: {delta:.6f}; 95% block-bootstrap CI [{low:.6f}, {high:.6f}]')
print(f"QML optimal-pair hit rate: {summary.loc['quantum', 'oracle_hit']:.2%}")
display(pd.Series(result['controls'], name='Control result').to_frame())
display(monthly.loc[monthly.arm == 'quantum', 'pair'].value_counts().rename('Months selected').to_frame())
assert result['controls']['future_target_invariance']
assert result['controls']['fixed_selection_shuffle_pass']
assert result['controls']['planted_mse'] < result['controls']['planted_constant_mse']
''')
md('''
## 6. Interpretation and reproducibility

**이번 실험에서는 QML 선택기의 개선을 확인하지 못했습니다.**
The quantum selector has higher Brier loss than RBF and the all-six average;
the RBF comparison interval includes zero. This is evidence about this specific
setup, not a universal impossibility result for quantum ensemble selection.

Controls verify frozen labels and returns, chronological training, future-target
invariance, a normalized PSD quantum Gram matrix, and learnability of a synthetic
kernel signal. Shuffled labels with fixed selections check scoring sanity;
they are not a retrained selector null experiment.

Limits: previously examined data; no fresh holdout; 92 monthly evaluations;
uncalibrated rank scores; only equal-weight pairs; one fixed context design;
noiseless classical quantum simulation. No hardware speedup or trading claim.

Files: `ENSEMBLE_PROTOCOL.md`, `run_ensemble_selector.py`, and
`results/ensemble_selector/{RESULT.json,monthly.csv,REPORT.md}`.
Rebuild this notebook with `python make_ensemble_notebook.py`, then run all cells.

Quantum feature-space reference:
[Havlíček et al., Nature (2019)](https://www.nature.com/articles/s41586-019-0980-2).
''')
nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python', 'version': '3.12'}}
nbf.validate(nb)
nbf.write(nb, ROOT / 'quantum_ensemble_selector.ipynb')
print('Wrote quantum_ensemble_selector.ipynb')
