"""Turnaround figure -> figures/fig_turnaround.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e1"
QC, CC, LC, WEAK, MID = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ad", "#8a897f"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
r = json.load(open(os.path.join(RES, "RESULT.json")))
ts = pd.read_csv(os.path.join(RES, "test_scores.csv"), parse_dates=["date"])
ts = ts[ts["y"] >= 0]
arms = [("QSVM (8 qubits)", "qsvm", QC), ("VQC (8 qubits)", "vqc", QC), ("RBF-SVM", "rbf", CC), ("logistic", "logit", CC),
        ("gradient boosting", "gbm", CC), ("rule: deepest drawdown", "rule_deepest", MID),
        ("rule: 1-month reversal", "rule_reversal", MID), ("rule: off the 52-w low", "rule_offlow", MID),
        ("rule: Prophet forecast", "rule_prophet", MID)]


def alpha_series(col):
    out = []
    for _, g in ts.groupby("date"):
        if len(g) >= 6:
            out.append(g.nlargest(4, col)["excess"].mean() - g["excess"].mean())
    return np.array(out)


fig, axs = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]})
ax = axs[0]
for i, (lab, k, col) in enumerate(arms):
    a = r["arms"][k]
    ax.barh(-i, a["auc"] - 0.4, left=0.4, height=0.62, color=col)
    lo, hi = a["auc_ci95"]
    ax.errorbar(a["auc"], -i, xerr=[[a["auc"] - lo], [hi - a["auc"]]], fmt="none", ecolor=INK, capsize=3, elinewidth=1.1)
    ax.text(hi + 0.004, -i, f"{a['auc']:.3f}", va="center", fontsize=8.5, color=INK2)
ax.axvline(0.5, color=INK, lw=1, ls="--")
ax.set_yticks([-i for i in range(len(arms))], [a[0] for a in arms], fontsize=9)
ax.set_xlim(0.4, 0.68)
ax.grid(axis="y", visible=False)
ax.set_xlabel("AUC for '6-month return beats SPY' (95 % month-block CI)")
ax.set_title(f"Which beaten-down stocks turn around?\n{r['n_test_events']} candidates, {r['n_test_months']} months "
             f"(base rate {100*r['base_rate_test']:.0f} %)", loc="left", fontsize=10)
ax = axs[1]
for i, (lab, k, col) in enumerate(arms):
    s = alpha_series(k) * 100
    m, h = s.mean(), 1.96 * s.std(ddof=1) / np.sqrt(len(s))
    ax.barh(-i, m, height=0.62, color=col)
    ax.errorbar(m, -i, xerr=h, fmt="none", ecolor=INK, capsize=3, elinewidth=1.1)
    ax.text(m + h + 0.3, -i, f"{m:+.1f}", va="center", ha="left", fontsize=8.5, color=INK2)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks([-i for i in range(len(arms))], [""] * len(arms))
ax.grid(axis="y", visible=False)
ax.set_xlabel("top-4 picks minus all candidates: 6-month excess return (pt, 95 % CI)")
c = r["comparisons"]["qsvm-rbf"]
ax.set_title(f"Selection value per month\nQSVM − RBF: AUC {c['auc_diff']:+.3f}, alpha {c['alpha_diff_pct']:+.1f} pt (p={c['alpha_p']:.2g})",
             loc="left", fontsize=10)
ax = axs[2]
q = r["TA_H3"]
if q["months"]:
    items = [("argmax\n= optimum", 100 * q["S1_argmax_rate"], None), ("best of 16\n= optimum", 100 * q["S3_best16_rate"], 100 * q["S3_random"]),
             ("P(optimum)", 100 * q["S2_mean_p_opt"], 100 * q["uniform_valid"])]
    for i, (lab, v, base) in enumerate(items):
        ax.bar(i, v, color=QC, width=0.6)
        ax.text(i, v + 1.5, f"{v:.1f} %", ha="center", fontsize=8.5, color=INK2)
        if base is not None:
            ax.plot([i - 0.32, i + 0.32], [base, base], color=INK, lw=2)
            ax.text(i + 0.34, base, f"random {base:.1f}", fontsize=7.5, color=INK2, va="center")
    ax.set_xticks(range(3), [it[0] for it in items], fontsize=8.5)
    ax.set_ylim(0, 108)
    al = q["alpha_pct"]
    ax.set_title(f"12-qubit QAOA: diversified top-4 of 12 ({q['months']} months)\n"
                 f"alpha: QAOA {al['alpha_qaoa']:+.1f} · exact {al['alpha_opt']:+.1f} · greedy {al['alpha_greedy']:+.1f} pt",
                 loc="left", fontsize=10)
fig.text(0.01, -0.02, "Universe = current S&P 500 members (survivorship bias: failed firms are missing), so absolute returns are inflated; "
         "only method-vs-method comparisons are meaningful. Paper study, not investment advice.", fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_turnaround.png"), dpi=150, bbox_inches="tight")
print("written")
