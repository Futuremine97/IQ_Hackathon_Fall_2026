"""AMENDMENT_4: POMDP belief agent and GRPO vs the existing arms, on U18 (K=4) and U9 sectors (K=3).
Needs results/test_scores.csv, results/sector_scores.csv, results/RESULT_REGIME.json.
-> results/RESULT_RL.json, results/rl_portfolio_returns.csv"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

from qps.data import ROOT, TEST, UNIVERSE, load_prices, month_ends
from qps.panel import build_panel
from qps.rl_agents import Grpo, poly_features, pomdp_scores
from run_backtest import sign_flip_p
from run_qaoa_portfolio import block_boot_ci, run_portfolio, sharpe
import run_sector as RS

OUT = os.path.join(ROOT, "results")
G = {}


def topk_acc(score, y, k):
    pred = np.zeros(len(y), int)
    pred[np.argsort(-np.asarray(score), kind="stable")[:k]] = 1
    return float((pred == y).mean())


def month_returns(px, assets):
    me = month_ends(px)
    last = px.index[-1]
    complete = me[me < last] if (last + pd.offsets.BDay(1)).month == last.month else me
    return px.loc[complete, assets].pct_change().dropna()


def _init(g):
    G.update(g)


def pomdp_job(a):
    uni, t = a
    mr = G[f"mret_{uni}"]
    X = mr.loc[:t].values
    b, sc, p = pomdp_scores(X)
    out = dict(uni=uni, date=t, scores=sc.tolist(), belief=b.tolist())
    if uni == "U18":
        i_spy, i_tlt = UNIVERSE.index("SPY"), UNIVERSE.index("TLT")
        mu, sd = p["mu"], np.sqrt(p["var"])
        out["p_spy_up"] = float(sum(b[s] * norm.cdf(mu[s, i_spy] / sd[s, i_spy]) for s in range(2)))
        out["p_spy_gt_tlt"] = float(sum(b[s] * norm.cdf((mu[s, i_spy] - mu[s, i_tlt]) /
                                                        np.sqrt(sd[s, i_spy] ** 2 + sd[s, i_tlt] ** 2)) for s in range(2)))
    return out


def grpo_job(a):
    uni, seed, shuffle = a
    df, U, K = G[f"df_{uni}"], G[f"U_{uni}"], G[f"K_{uni}"]
    dates_all = sorted(df["date"].unique())
    Phi_all = {pd.Timestamp(t): poly_features(U[idx]) for t, idx in df.groupby("date").indices.items()}
    R_all = {pd.Timestamp(t): df["fwd_ret"].values[idx] for t, idx in df.groupby("date").indices.items()}
    agent = Grpo(next(iter(Phi_all.values())).shape[1], K, seed=seed, shuffle_reward=shuffle)
    out, first = [], True
    for t in G[f"test_{uni}"]:
        tr_dates = [pd.Timestamp(d) for d in dates_all
                    if (df.loc[df["date"] == d, "label_end"] <= t).all() and pd.Timestamp(d) < t
                    and df.loc[df["date"] == d, "fwd_ret"].notna().all()]
        Phi = np.stack([Phi_all[d] for d in tr_dates])
        R = np.stack([R_all[d] for d in tr_dates])
        for _ in range(30 if first else 10):
            agent.step(Phi, R)
        first = False
        out.append(agent.scores(Phi_all[t]).tolist())
    return dict(uni=uni, seed=seed, shuffle=shuffle, scores=out)


if __name__ == "__main__":
    t0 = time.time()
    px, df18, U18 = build_panel()
    _, df9, U9 = RS.build()
    test18 = [pd.Timestamp(t) for t in sorted(df18.loc[df18["fwd_ret"].notna(), "date"].unique())
              if pd.Timestamp(TEST[0]) <= t <= pd.Timestamp(TEST[1])]
    test9 = RS.dates(df9, TEST)
    g = dict(df_U18=df18, U_U18=U18, K_U18=4, df_U9=df9, U_U9=U9, K_U9=3, test_U18=test18, test_U9=test9,
             mret_U18=month_returns(px, UNIVERSE), mret_U9=month_returns(px, RS.SECTORS))
    with Pool(max(1, (os.cpu_count() or 2) - 4), initializer=_init, initargs=(g,)) as pool:
        pj = pool.map(pomdp_job, [("U18", t) for t in test18] + [("U9", t) for t in test9], chunksize=2)
        gj = pool.map(grpo_job, [(u, s, sh) for u in ("U18", "U9") for s in range(5) for sh in (False, True)], chunksize=1)
    out = {"amendment_sha256": open(os.path.join(ROOT, "PREREG_AMENDMENT_4_POMDP_GRPO.sha256")).read().split()[0]}
    port_all = {}
    for uni, df, tests, K, kacc, ext in (("U18", df18, test18, 4, 9, "test_scores.csv"),
                                        ("U9", df9, test9, 3, 4, "sector_scores.csv")):
        ts = pd.read_csv(os.path.join(OUT, ext), parse_dates=["date"])
        assets = UNIVERSE if uni == "U18" else RS.SECTORS
        ts["asset"] = pd.Categorical(ts["asset"], categories=assets, ordered=True)
        ts = ts.sort_values(["date", "asset"]).reset_index(drop=True)
        assert len(ts) == len(tests) * len(assets)
        ts["pomdp"] = np.concatenate([r["scores"] for r in pj if r["uni"] == uni])
        for sh in (False, True):
            runs = [r for r in gj if r["uni"] == uni and r["shuffle"] == sh]
            for r in runs:
                ts[f"grpo{'_shuf' if sh else ''}_s{r['seed']}"] = np.concatenate(r["scores"])
        y = ts["y"].values
        arms = ["qsvm", "rbf", "logit", "p_ret", "pomdp"] + [f"grpo_s{s}" for s in range(5)] + \
               [f"grpo_shuf_s{s}" for s in range(5)] + (["vqc"] if "vqc" in ts else [])
        acc = {c: ts.groupby("date").apply(lambda d: topk_acc(d[c].values, d["y"].values, kacc)).values for c in arms}
        acc["grpo"] = np.mean([acc[f"grpo_s{s}"] for s in range(5)], axis=0)
        acc["grpo_shuffled"] = np.mean([acc[f"grpo_shuf_s{s}"] for s in range(5)], axis=0)
        cmp = {f"{a}-{b}": dict(diff_pt=100 * float((acc[a] - acc[b]).mean()), p=sign_flip_p(acc[a] - acc[b]))
               for a, b in (("qsvm", "grpo"), ("qsvm", "pomdp"), ("rbf", "grpo"), ("rbf", "pomdp"),
                            ("grpo", "grpo_shuffled"), ("pomdp", "p_ret"))}
        # portfolios (greedy top-K)
        R = np.stack([d["fwd_ret"].values for _, d in ts.groupby("date")])
        n = len(R)
        def picks(col):
            return [np.isin(np.arange(len(assets)), np.argsort(-d[col].values, kind="stable")[:K]).astype(float) / K
                    for _, d in ts.groupby("date")]
        port = {"pomdp": run_portfolio(picks("pomdp"), R), "equal_weight": run_portfolio([np.full(len(assets), 1 / len(assets))] * n, R)}
        for s in range(5):
            port[f"grpo_s{s}"] = run_portfolio(picks(f"grpo_s{s}"), R)
            port[f"grpo_shuf_s{s}"] = run_portfolio(picks(f"grpo_shuf_s{s}"), R)
        port["grpo"] = np.mean([port[f"grpo_s{s}"] for s in range(5)], axis=0)
        port["grpo_shuffled"] = np.mean([port[f"grpo_shuf_s{s}"] for s in range(5)], axis=0)
        for c in ("qsvm", "rbf", "p_ret"):
            port[f"{c}_topK"] = run_portfolio(picks(c), R)
        perf = {k: dict(sharpe=sharpe(v), ci90_vs_ew=block_boot_ci(v, port["equal_weight"]))
                for k, v in port.items() if not k[-1].isdigit()}
        perf["grpo_per_seed_sharpe"] = [sharpe(port[f"grpo_s{s}"]) for s in range(5)]
        h1 = "POS" if perf["grpo"]["ci90_vs_ew"][0] > 0 else "CLOSED-NEG"
        h2 = "POS" if perf["pomdp"]["ci90_vs_ew"][0] > 0 else "CLOSED-NEG"
        ctrl_void = perf["grpo_shuffled"]["ci90_vs_ew"][0] > 0
        out[uni] = dict(K=K, accuracy_pct={c: 100 * float(acc[c].mean()) for c in
                                           ("qsvm", "rbf", "logit", "p_ret", "pomdp", "grpo", "grpo_shuffled")},
                        grpo_per_seed_acc_pct=[100 * float(acc[f"grpo_s{s}"].mean()) for s in range(5)],
                        comparisons=cmp, portfolio=perf, AM4_H1_grpo=h1, AM4_H2_pomdp=h2,
                        grpo_shuffle_control_void=bool(ctrl_void),
                        per_month_acc={c: acc[c].tolist() for c in ("qsvm", "rbf", "pomdp", "grpo", "grpo_shuffled")})
        port_all.update({f"{uni}_{k}": v for k, v in port.items() if not k[-1].isdigit()})
    # POMDP on regimes RG2 / RG3
    reg = json.load(open(os.path.join(OUT, "RESULT_REGIME.json")))
    _, _, rg2 = __import__("run_regime").build()
    rg2t = rg2[rg2["date"].isin(test18)].sort_values("date")
    pr = {pd.Timestamp(r["date"]): r for r in pj if r["uni"] == "U18"}
    regime = {}
    for task, key, ycol in (("RG2", "p_spy_up", "y2"), ("RG3", "p_spy_gt_tlt", "y3")):
        corr = np.array([int((pr[pd.Timestamp(t)][key] > 0.5) == bool(y)) for t, y in zip(rg2t["date"], rg2t[ycol])], float)
        pm = reg["tasks"][task]["per_month"]
        regime[task] = dict(pomdp_acc_pct=100 * corr.mean(),
                            vs={k: dict(diff_pt=100 * float((corr - np.array(pm[k])).mean()),
                                        p=sign_flip_p(corr - np.array(pm[k])))
                                for k in ("majority", "persistence", "markov_switching", "qsvm", "rbf")},
                            per_month=corr.tolist())
    out["regimes_pomdp"] = regime
    pd.DataFrame(port_all, index=pd.Index(test18, name="date")).to_csv(os.path.join(OUT, "rl_portfolio_returns.csv"))
    out["seconds"] = time.time() - t0
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(out, open(os.path.join(OUT, "RESULT_RL.json"), "w"), indent=2, default=float)
    for uni in ("U18", "U9"):
        o = out[uni]
        print(uni, "acc", {k: round(v, 1) for k, v in o["accuracy_pct"].items()})
        print(uni, "sharpe", {k: round(v["sharpe"], 2) for k, v in o["portfolio"].items() if isinstance(v, dict)},
              "H1", o["AM4_H1_grpo"], "H2", o["AM4_H2_pomdp"], "ctrl void", o["grpo_shuffle_control_void"])
        print(uni, {k: (round(v["diff_pt"], 2), round(v["p"], 4)) for k, v in o["comparisons"].items()})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_month"} for k, v in regime.items()}, indent=1))
