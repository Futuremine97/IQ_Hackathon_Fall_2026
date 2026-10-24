"""Figures for the stock / Prophet project -> figures/*.png (reads results/)."""
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
QC, CC, LC, EXACT, WEAK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#b5b4ad"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
R = lambda f: json.load(open(os.path.join(RES, f))) if os.path.exists(os.path.join(RES, f)) else None


def top9(score, y):
    pred = np.zeros(len(y), int)
    pred[np.argsort(-np.asarray(score), kind="stable")[:9]] = 1
    return float((pred == y).mean())


def mean_ci(a):
    a = np.asarray(a, float)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(a.size)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


# ---------------------------------------------------------------- figure A: classifiers
ts = pd.read_csv(os.path.join(RES, "test_scores.csv"), parse_dates=["date"])
arms = [("Prophet rule", "p_ret", EXACT), ("momentum 12-1", "mom_12_1", WEAK), ("QSVM (4 qubits)", "qsvm", QC),
        ("VQC (4 qubits)", "vqc", QC), ("RBF-SVM", "rbf", CC), ("logistic", "logit", CC)]
acc = {c: ts.groupby("date").apply(lambda d: top9(d[c].values, d["y"].values)) for _, c, _ in arms}
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axs[0]
for i, (lab, c, col) in enumerate(arms):
    m, h = mean_ci(acc[c].values)
    ax.barh(-i, 100 * m - 40, left=40, height=0.62, color=col, alpha=0.9 if c != "vqc" and c != "logit" else 0.55)
    ax.errorbar(100 * m, -i, xerr=100 * h, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    ax.text(100 * (m + h) + 0.3, -i, f"{100*m:.1f} %", va="center", fontsize=9, color=INK2)
ax.axvline(50, color=INK, lw=1, ls="--")
ax.text(50.1, 0.55, "coin = 50 %", fontsize=8, color=INK2)
ax.set_yticks([-i for i in range(len(arms))], [a[0] for a in arms])
ax.set_xlim(44, 57)
ax.grid(axis="y", visible=False)
ax.set_xlabel("top-half accuracy, 2014-01..2026-08 (152 months × 18 ETFs, 95 % CI)")
ax.set_title("Which ETFs beat the median next month?\nblue = quantum, orange = classical ML",
             loc="left", fontsize=10)
ax = axs[1]
for lab, c, col, ls in [("QSVM", "qsvm", QC, "-"), ("RBF-SVM", "rbf", CC, "-"), ("Prophet rule", "p_ret", EXACT, "--")]:
    s = acc[c].expanding().mean() * 100
    ax.plot(s.index, s.values, color=col, lw=2, ls=ls, label=f"{lab} ({s.values[-1]:.1f} %)")
ax.axhline(50, color=INK, lw=1, ls=":")
ax.set_ylabel("running mean accuracy (%)")
ax.set_ylim(44, 60)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
ax.set_title("Running accuracy over the test period (no arm separates from 50 %)", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_stock_classifiers.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- figure B: recursive self-improvement
rsi = R("RESULT_RSI.json")
mm = os.path.join(RES, "rsi_month_mae_test.csv")
if rsi:
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axs[0]
    bars = [("Prophet (original config)", "prophet_base", EXACT),
            ("+ config self-search (QAOA)", "prophet_carried_config", "#86b6ef"),
            ("+ quantum-kernel residual round", "rsi_quantum", QC),
            ("+ RBF-kernel residual round (twin)", "rsi_rbf_twin", CC),
            ("random walk (forecast 0)", "random_walk_zero", WEAK),
            ("historical mean", "historical_mean", WEAK)]
    month = pd.read_csv(mm, index_col=0) if os.path.exists(mm) else None
    for i, (lab, k, col) in enumerate(bars):
        v = 100 * rsi["test_metrics"][k]["mae"]
        ax.barh(-i, v, height=0.62, color=col)
        if month is not None:
            m, h = mean_ci(month[k].values)
            ax.errorbar(100 * m, -i, xerr=100 * h, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
        ax.text(v + 0.25, -i + 0.22, f"{v:.2f} %", fontsize=9, color=INK2)
    ax.set_yticks([-i for i in range(len(bars))], [b[0] for b in bars])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean absolute error of the next-month return forecast (%, test 2014-2026; lower is better)")
    h5 = rsi["H5"]["per_arm"]
    ax.set_title(f"Self-improvement lowers Prophet's error ({100*h5['rsi_quantum']['mae_diff_vs_base']:+.2f} pt, "
                 f"p={h5['rsi_quantum']['p_signflip']:.3f}) …\n… but the classical twin is better and a forecast of 0 beats every Prophet variant",
                 loc="left", fontsize=10)
    ax = axs[1]
    loops = rsi["stage_Q"]["loops"]
    items = [("surrogate\n+ QAOA", loops["qaoa"]["found_dev_best_rate"], QC),
             ("surrogate\n+ exact argmin", loops["exact"]["found_dev_best_rate"], CC),
             ("random search", loops["random"]["found_dev_best_rate"], WEAK)]
    for i, (lab, p, col) in enumerate(items):
        p_, lo, hi = wilson(round(p * 50), 50)
        ax.bar(i, 100 * p, color=col, width=0.62)
        ax.errorbar(i, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none", ecolor=INK, capsize=4)
        ax.text(i, 100 * hi + 2, f"{100*p:.0f} %", ha="center", fontsize=9, color=INK2)
    ax.axhline(100 * 24 / 64, color=INK, lw=1, ls=":")
    ax.text(2.45, 100 * 24 / 64 + 1.5, "chance 37.5 %", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(3), [i[0] for i in items])
    ax.set_ylim(0, 100)
    ax.set_ylabel("seeds reaching the best of 64 configs (%, n=50)")
    ax.set_title("Stage Q: finding Prophet's best configuration\nin 24 of 64 evaluations", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_self_improvement.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- figure C: portfolios
qm = os.path.join(RES, "portfolio_returns.csv")
pr = pd.read_csv(qm, index_col=0, parse_dates=True) if os.path.exists(qm) else None
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
curves = [("equal weight 18", "equal_weight_18", EXACT, "-"), ("QUBO optimum, QSVM scores", "bruteforce_opt", QC, "-"),
          ("QAOA pick, QSVM scores", "qaoa", QC, "--"), ("top-4 QSVM scores (no risk term)", "greedy_mu", "#86b6ef", "-"),
          ("top-4 Prophet forecast", "prophet_top4", WEAK, "-"), ("top-4 momentum", "momentum_top4", "#8a897f", "-")]
for lab, c, col, ls in (curves if pr is not None else []):
    eq = (1 + pr[c]).cumprod()
    ax.plot(eq.index, eq.values, color=col, lw=2, ls=ls, label=lab)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([1, 1.5, 2, 3])
ax.set_ylabel("growth of $1 (log), after 10 bp costs")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Paper portfolios 2014-2026, 18 ETFs, 10 bp costs", loc="left", fontsize=10)
ax = axs[1]
sh = {lab: float(pr[c].mean() / pr[c].std() * np.sqrt(12)) for lab, c, _, _ in curves} if pr is not None else {}
cols = [col for _, _, col, _ in curves]
ax.barh(range(len(sh)), list(sh.values()), color=cols, height=0.6)
for i, v in enumerate(sh.values()):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(sh)), list(sh.keys()), fontsize=8)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("annualised Sharpe ratio")
ax.set_title("Nothing beats equal weight", loc="left", fontsize=10)
fig.tight_layout()
if pr is not None:
    fig.savefig(os.path.join(FIG, "fig_portfolios.png"), dpi=150)
plt.close(fig)
# ---------------------------------------------------------------- figure D: regime prediction
reg = R("RESULT_REGIME.json")
if reg:
    MID = "#8a897f"
    names = {"qsvm": ("QSVM (4 qubits)", QC), "vqc": ("VQC (4 qubits)", QC), "rbf": ("RBF-SVM", CC),
             "logit": ("logistic", CC), "persistence": ("persistence", EXACT),
             "vix_rule": ("VIX rule", MID), "prophet_rule": ("Prophet rule", MID),
             "markov_switching": ("Markov-switching", MID), "majority": ("majority class", WEAK)}
    titles = {"RG1": "RG1 volatility regime\n(18 ETFs × 152 months)", "RG2": "RG2 bull / bear month (SPY)\n(152 months)",
              "RG3": "RG3 risk-on / off (SPY vs TLT)\n(152 months)"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), sharex=False)
    for ax, task in zip(axs, ("RG1", "RG2", "RG3")):
        tb = reg["tasks"][task]
        keys = [k for k in names if k in tb["arms"]]
        for i, k in enumerate(keys):
            m, h = mean_ci(tb["per_month"][k])
            lab, col = names[k]
            ax.barh(-i, 100 * m, height=0.62, color=col, alpha=0.55 if k in ("vqc", "logit") else 0.95)
            ax.errorbar(100 * m, -i, xerr=100 * h, fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
            ax.text(100 * (m + h) + 0.8, -i, f"{100*m:.1f}", va="center", fontsize=8.5, color=INK2)
        maj = 100 * np.mean(tb["per_month"]["majority"])
        ax.axvline(maj, color=INK, lw=1, ls=":")
        ax.set_yticks([-i for i in range(len(keys))], [names[k][0] for k in keys], fontsize=9)
        ax.set_xlim(30, 100)
        ax.grid(axis="y", visible=False)
        c = tb["comparisons"]["qsvm-rbf"]
        ax.set_title(f"{titles[task]}\nQSVM − RBF {c['diff_pt']:+.1f} pt (p={c['p']:.2g})", loc="left", fontsize=10)
        ax.set_xlabel("accuracy (%, 95 % CI; dotted = majority class)")
    fig.suptitle(f"Regime prediction, test 2014-01..2026-08 — RG-H1 (quantum vs RBF): {reg['RG_H1']['verdict']}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_regime.png"), dpi=150)
    plt.close(fig)
# ---------------------------------------------------------------- figure E: sectors, QAOA 9q, POMDP, GRPO
rl, sec = R("RESULT_RL.json"), R("RESULT_SECTOR.json")
if rl and sec:
    MID, RLC = "#8a897f", "#26251f"          # 4th+ identity -> neutral ink (no 4th hue passes all-pairs); bars are text-labelled
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    arms = [("QSVM", "qsvm", QC), ("RBF-SVM", "rbf", CC), ("logistic", "logit", CC),
            ("Prophet rule", "p_ret", MID), ("POMDP belief agent", "pomdp", RLC), ("GRPO (5 seeds)", "grpo", RLC),
            ("GRPO, shuffled reward", "grpo_shuffled", WEAK)]
    ax = axs[0]
    for j, (uni, off, hatch) in enumerate((("U18", -0.2, None), ("U9", 0.2, "//"))):
        for i, (lab, k, col) in enumerate(arms):
            v = rl[uni]["accuracy_pct"][k]
            ax.barh(-i + off, v - 40, left=40, height=0.38, color=col, hatch=hatch, edgecolor=SURF, lw=0)
            ax.text(v + 0.2, -i + off, f"{v:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(50, color=INK, lw=1, ls="--")
    ax.set_yticks([-i for i in range(len(arms))], [a[0] for a in arms], fontsize=9)
    ax.set_xlim(44, 56)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("top-half accuracy (%), solid = 18 ETFs, hatched = 9 sectors")
    ax.set_title("Ranking accuracy, test 2014-2026 (coin ≈ 50 %)", loc="left", fontsize=10)
    ax = axs[1]
    parms = [("equal weight", "equal_weight", MID), ("QSVM top-K", "qsvm_topK", QC), ("RBF top-K", "rbf_topK", CC),
             ("Prophet top-K", "p_ret_topK", MID), ("POMDP top-K", "pomdp", RLC),
             ("GRPO top-K", "grpo", RLC), ("GRPO shuffled", "grpo_shuffled", WEAK)]
    for j, (uni, off, hatch) in enumerate((("U18", -0.2, None), ("U9", 0.2, "//"))):
        for i, (lab, k, col) in enumerate(parms):
            v = rl[uni]["portfolio"][k]["sharpe"]
            ax.barh(-i + off, v, height=0.38, color=col, hatch=hatch, edgecolor=SURF, lw=0)
            ax.text(v + 0.01, -i + off, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
    v9 = sec["portfolio"]["qaoa_top3"]["sharpe"]
    ax.barh(-len(parms) + 0.2, v9, height=0.38, color=QC, hatch="//", edgecolor=SURF, lw=0)
    ax.text(v9 + 0.01, -len(parms) + 0.2, f"{v9:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks([-i for i in range(len(parms) + 1)], [p[0] for p in parms] + ["9-qubit QAOA top-3"], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("annualised Sharpe after 10 bp costs (solid = 18 ETFs K=4, hatched = 9 sectors K=3)")
    ax.set_title("Portfolios: nothing beats equal weight", loc="left", fontsize=10)
    ax = axs[2]
    q = sec["qaoa_9q"]
    items = [("argmax\n= optimum", 100 * q["S1_argmax_rate"], None),
             ("best of 16\n= optimum", 100 * q["S3_best16_rate"], 100 * q["S3_random_baseline"]),
             ("P(optimum)", 100 * q["S2_mean_p_opt"], 100 * q["uniform_feasible"]),
             ("P(valid)", 100 * q["mean_p_feasible"], 100 * 84 / 512)]
    for i, (lab, v, base) in enumerate(items):
        ax.bar(i, v, color=QC, width=0.6)
        ax.text(i, v + 1.5, f"{v:.1f} %", ha="center", fontsize=8.5, color=INK2)
        if base is not None:
            ax.plot([i - 0.32, i + 0.32], [base, base], color=INK, lw=2)
            ax.text(i + 0.34, base, f"random {base:.1f}", fontsize=7.5, color=INK2, va="center")
    ax.set_xticks(range(len(items)), [it[0] for it in items], fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of 152 months / probability (%)")
    ax.set_title("9-qubit QAOA, 9 sector ETFs\n(84 valid portfolios; black = random)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_sector_rl.png"), dpi=150)
    plt.close(fig)
print("written:", sorted(os.listdir(FIG)))
