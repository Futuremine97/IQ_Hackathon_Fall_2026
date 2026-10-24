"""Exploratory QML meta-selection; design frozen in ENSEMBLE_PROTOCOL.md."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve

from qps.data import load_prices, month_ends, panel_labels
from qps.quantum import gram, zz_states

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'ensemble_selector'
EXPERTS = ['p_ret', 'mom_12_1', 'qsvm', 'rbf', 'logit', 'vqc']
PAIRS = list(itertools.combinations(range(6), 2))
PAIR_NAMES = ['+'.join(EXPERTS[j] for j in p) for p in PAIRS]
GRID = {'quantum': list(itertools.product([0.1, 0.3, 1.0], [0.01, 0.1, 1.])),
        'rbf': list(itertools.product([0.1, 1., 10.], [0.01, 0.1, 1.]))}


def accuracy(scores, labels):
    order = np.argsort(-scores, axis=-1, kind='stable')
    pred = np.zeros_like(scores)
    np.put_along_axis(pred, order[..., :9], 1, axis=-1)
    return np.mean(pred == labels, axis=-1)


def kernel(a, b, kind, scale):
    if kind == 'quantum':
        return gram(zz_states(scale * np.pi * a), zz_states(scale * np.pi * b))
    return np.exp(-scale * ((a[:, None] - b[None]) ** 2).sum(axis=-1))


def predict(x, losses, ends, dates, i, kind, hp):
    tr = np.flatnonzero((dates < dates[i]) & (ends <= dates[i]))
    assert len(tr) >= 18
    assert np.all(ends[tr] <= dates[i]) and np.all(dates[tr] < dates[i])
    mean, sd = x[tr].mean(axis=0), x[tr].std(axis=0)
    sd = np.maximum(sd, 1e-8)
    a = .5 + np.arctan((x[tr] - mean) / sd) / np.pi
    b = .5 + np.arctan((x[i:i+1] - mean) / sd) / np.pi
    scale, ridge = hp
    k = kernel(a, a, kind, scale)
    target_mean = losses[tr].mean(axis=0)
    weights = solve(k + ridge * np.eye(len(tr)), losses[tr] - target_mean,
                    assume_a='pos')
    return (kernel(b, a, kind, scale) @ weights)[0] + target_mean


def load_inputs():
    source = ROOT / 'results' / 'test_scores.csv'
    df = pd.read_csv(source, parse_dates=['date'])
    assert not df.duplicated(['date', 'asset']).any()
    px = load_prices()
    labels = panel_labels(px, month_ends(px))
    checked = df.merge(labels, on=['date', 'asset'], suffixes=('', '_frozen'),
                       validate='one_to_one')
    assert len(checked) == len(df)
    np.testing.assert_allclose(checked.fwd_ret, checked.fwd_ret_frozen, atol=1e-12)
    np.testing.assert_array_equal(checked.y, checked.y_frozen)
    dates, ends, probs, ys, returns = [], [], [], [], []
    for t, d in checked.groupby('date', sort=True):
        d = d.sort_values('asset')  # deterministic tie order
        assert len(d) == 18 and d.y.sum() == 9
        dates.append(t)
        ends.append(d.label_end.iloc[0])
        probs.append(((d[EXPERTS].rank(method='first') - .5) / 18).to_numpy().T)
        ys.append(d.y.to_numpy())
        returns.append(d.fwd_ret.to_numpy())
    dates, ends = np.array(dates, dtype='datetime64[ns]'), np.array(ends, dtype='datetime64[ns]')
    probs, ys, returns = np.array(probs), np.array(ys), np.array(returns)
    expert_loss = ((probs - ys[:, None]) ** 2).mean(axis=-1)
    pair_scores = np.stack([probs[:, list(p)].mean(axis=1) for p in PAIRS], axis=1)
    losses = ((pair_scores - ys[:, None]) ** 2).mean(axis=-1)
    x, usable = [], []
    for i, t in enumerate(dates):
        past = np.flatnonzero((dates < t) & (ends <= t))
        if len(past) < 3:
            continue
        recent, last = past[-3:], past[-1]
        quantum = expert_loss[recent][:, [2, 5]].mean()
        classical = expert_loss[recent][:, [0, 1, 3, 4]].mean()
        x.append([expert_loss[recent].mean(), quantum - classical,
                  returns[last].mean(), returns[last].std()])
        usable.append(i)
    take = np.array(usable)
    return (np.array(x), losses[take], ends[take], dates[take], pair_scores[take],
            probs[take], ys[take], hashlib.sha256(source.read_bytes()).hexdigest())


def circuit_controls():
    rng = np.random.default_rng(2026)
    x = rng.uniform(0, 1, (100, 4))
    k = kernel(x, x, 'quantum', .3)
    np.testing.assert_allclose(np.diag(k), 1, atol=1e-12)
    assert np.linalg.eigvalsh(k).min() > -1e-10
    target = k[:, :8] @ rng.normal(size=8)
    w = solve(k[:70, :70] + .01 * np.eye(70), target[:70], assume_a='pos')
    mse = np.mean((k[70:, :70] @ w - target[70:]) ** 2)
    constant = np.mean((target[:70].mean() - target[70:]) ** 2)
    assert mse < constant
    return {'gram_normalized_psd': True, 'planted_mse': float(mse),
            'planted_constant_mse': float(constant)}


def main():
    OUT.mkdir(exist_ok=True)
    protocol_hash = hashlib.sha256((ROOT / 'ENSEMBLE_PROTOCOL.md').read_bytes()).hexdigest()
    controls = circuit_controls()
    x, losses, ends, dates, scores, probs, ys, source_hash = load_inputs()
    dev = np.flatnonzero((dates >= np.datetime64('2016-01-01')) &
                         (dates < np.datetime64('2019-01-01')))
    test = np.flatnonzero(dates >= np.datetime64('2019-01-01'))
    chosen, tuning = {}, []
    for kind, grid in GRID.items():
        trials = []
        for hp in grid:
            selections = [int(np.argmin(predict(x, losses, ends, dates, i, kind, hp))) for i in dev]
            loss = float(np.mean(losses[dev, selections]))
            trials.append((loss, hp))
            tuning.append({'kind': kind, 'hp': hp, 'dev_brier': loss})
        chosen[kind] = min(trials, key=lambda t: t[0])[1]
        print(f'{kind} selected {chosen[kind]}', flush=True)

    # Fixed pair is learned strictly before the first evaluation decision.
    before_test = ends <= dates[test[0]]
    fixed = int(np.argmin(losses[before_test].mean(axis=0)))
    rows, selected_quantum = [], []
    for i in test:
        past = np.flatnonzero((dates < dates[i]) & (ends <= dates[i]))
        selections = {k: int(np.argmin(predict(x, losses, ends, dates, i, k, chosen[k])))
                      for k in chosen}
        selections.update(fixed=fixed,
                          expanding=int(np.argmin(losses[past].mean(axis=0))),
                          trailing12=int(np.argmin(losses[past[-12:]].mean(axis=0))),
                          hindsight_oracle=int(np.argmin(losses[i])))
        pair_acc = accuracy(scores[i], ys[i][None, :])
        selected_quantum.append(selections['quantum'])
        for name, j in selections.items():
            rows.append({'date': str(dates[i])[:10], 'arm': name, 'pair': PAIR_NAMES[j],
                         'brier': losses[i, j], 'accuracy': pair_acc[j],
                         'oracle_hit': float(np.isclose(losses[i,j], losses[i].min(), atol=1e-12, rtol=0)),
                         'regret': losses[i, j] - losses[i].min()})
        all_score = probs[i].mean(axis=0)
        for name, loss, acc in [
                ('all_six', np.mean((all_score - ys[i]) ** 2), accuracy(all_score, ys[i])),
                ('random_pair_expectation', losses[i].mean(), pair_acc.mean())]:
            rows.append({'date': str(dates[i])[:10], 'arm': name, 'pair': '',
                         'brier': loss, 'accuracy': float(acc), 'oracle_hit': np.nan,
                         'regret': loss - losses[i].min()})

    # A future-target perturbation must not change an earlier decision's output.
    probe = int(test[len(test)//2])
    changed = losses.copy()
    changed[ends > dates[probe]] += 123.
    for kind in chosen:
        np.testing.assert_array_equal(
            predict(x, losses, ends, dates, probe, kind, chosen[kind]),
            predict(x, changed, ends, dates, probe, kind, chosen[kind]))
    controls['future_target_invariance'] = True
    controls['frozen_labels_returns_match'] = True
    controls['chronology_assertions'] = True

    rng = np.random.default_rng(2026)
    selected_scores = scores[test, selected_quantum]
    shuffled = []
    for _ in range(100):
        shuffled_y = np.array([rng.permutation(y) for y in ys[test]])
        shuffled.append(float(accuracy(selected_scores, shuffled_y).mean()))
    controls['fixed_selection_shuffle_mean_accuracy'] = float(np.mean(shuffled))
    assert abs(np.mean(shuffled) - .5) < .02
    controls['fixed_selection_shuffle_pass'] = True

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'monthly.csv', index=False)
    summary = frame.groupby('arm')[['brier', 'accuracy', 'oracle_hit', 'regret']].mean()
    pivot = frame.pivot(index='date', columns='arm', values='brier')
    difference = (pivot.rbf - pivot.quantum).to_numpy()
    starts = rng.integers(0, len(test), size=(10000, int(np.ceil(len(test)/6))))
    idx = ((starts[:, :, None] + np.arange(6)) % len(test)).reshape(10000, -1)[:, :len(test)]
    ci = np.percentile(difference[idx].mean(axis=1), [2.5, 97.5])
    baselines = ['all_six', 'fixed', 'expanding', 'trailing12', 'random_pair_expectation']
    positive = bool(ci[0] > 0 and all(summary.loc['quantum', 'brier'] < summary.loc[b, 'brier']
                                    for b in baselines))
    result = {'protocol_sha256': protocol_hash, 'input_scores_sha256': source_hash,
              'exploratory': True, 'simulation_only': True, 'qubits': 4,
              'n_pairs': 15, 'n_dev_months': len(dev), 'n_eval_months': len(test),
              'eval_start': str(dates[test[0]])[:10], 'eval_end': str(dates[test[-1]])[:10],
              'selected_hyperparameters': chosen, 'tuning': tuning, 'controls': controls,
              'arms': json.loads(summary.to_json(orient='index')),
              'rbf_minus_quantum_brier': float(difference.mean()),
              'block_bootstrap_95_ci': ci.tolist(),
              'verdict': 'QML_BENEFIT_IN_EXPLORATORY_SAMPLE' if positive else 'NO_QML_BENEFIT_DEMONSTRATED'}
    (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    report = ['# QML ensemble selector — exploratory result', '',
              f"Evaluation: {result['eval_start']} to {result['eval_end']}, {len(test)} months.", '',
              'Six experts; choose one of 15 equal-weight pairs. Four-qubit ZZ fidelity kernel',
              'ridge predicts next-month pair loss using resolved historical outcomes only.',
              'Nine configurations per kernel, selected on 2016–2018; evaluation begins in 2019.', '',
              '| Arm | Rank-score Brier ↓ | Top-9 accuracy | Pair optimum hit | Regret ↓ |',
              '|---|---:|---:|---:|---:|']
    for arm, r in summary.sort_values('brier').iterrows():
        hit = '—' if pd.isna(r.oracle_hit) else f'{r.oracle_hit:.1%}'
        report.append(f'| {arm} | {r.brier:.6f} | {r.accuracy:.2%} | {hit} | {r.regret:.6f} |')
    report += ['', f"Verdict: **{result['verdict']}**.", '',
               f'RBF minus QML Brier = {difference.mean():.6f}; paired 6-month circular-block',
               f'bootstrap 95% interval [{ci[0]:.6f}, {ci[1]:.6f}]. Positive favors QML.', '',
               'All implemented controls passed: input labels/returns, chronological training,',
               'future-target invariance, PSD/normalized Gram, planted kernel signal,',
               f'and fixed-selection shuffled-label accuracy ({np.mean(shuffled):.2%}).', '',
               'The hindsight oracle uses future outcomes and cannot be deployed. Its optimum',
               'is restricted to the 15 pairs under Brier loss; accuracy can favor another pair.',
               'All-six can have negative regret because it lies outside the pair candidate set.', '',
               'Limitations: previously examined data, no fresh holdout, only 92 monthly decisions,',
               'uncalibrated rank scores, fixed pair size/weights/context, noiseless classical',
               'simulation. The bootstrap conditions on the frozen hyperparameter selection.',
               'Base models were not refitted here. No return, trading, hardware, global-optimality',
               'or quantum-speedup claim. The shuffle is a scoring sanity check, not a retrained null.', '',
               'Reproduce from repository root: `python run_ensemble_selector.py`.',
               'See `ENSEMBLE_PROTOCOL.md`, `monthly.csv`, and `RESULT.json` for details.', '',
               'Method reference: https://www.nature.com/articles/s41586-019-0980-2', '']
    (OUT / 'REPORT.md').write_text('\n'.join(report))
    print(summary.to_string(), flush=True)
    print(result['verdict'], ci.tolist(), flush=True)


if __name__ == '__main__':
    main()
