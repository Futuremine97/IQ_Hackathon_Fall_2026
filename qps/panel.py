"""Merged (date, asset) panel, walk-forward splits and the classifier arms."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .data import DEV, ROOT, TEST, UNIVERSE, classical_features, load_prices, month_ends, panel_labels, rank_u
from .quantum import Vqc, gram, zz_states

FEATS = ["p_ret", "p_resid", "mom_12_1", "vol_21"]


def build_panel(features_csv: str = os.path.join(ROOT, "results", "features_base.csv")):
    px = load_prices()
    me = month_ends(px)
    pf = pd.read_csv(features_csv, parse_dates=["date"])
    assert pf["p_ret"].notna().all(), "failed Prophet fits present"
    cf = classical_features(px, me)
    lab = panel_labels(px, me)
    df = pf.merge(cf, on=["date", "asset"], how="inner").merge(lab, on=["date", "asset"], how="left")
    df["asset"] = pd.Categorical(df["asset"], categories=UNIVERSE, ordered=True)
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    assert (df.groupby("date").size() == 18).all()
    U = rank_u(df, FEATS)
    return px, df, U


def decision_dates(df: pd.DataFrame, period) -> list:
    d = df.loc[df["fwd_ret"].notna(), "date"].unique()
    return [pd.Timestamp(t) for t in sorted(d) if pd.Timestamp(period[0]) <= t <= pd.Timestamp(period[1])]


def train_mask(df: pd.DataFrame, t: pd.Timestamp) -> np.ndarray:
    m = (df["label_end"] <= t).values & df["fwd_ret"].notna().values
    assert (df.loc[m, "label_end"] <= t).all() and (df.loc[m, "date"] < t).all()
    return m


def top9_accuracy(score: np.ndarray, y: np.ndarray) -> float:
    pred = np.zeros(18, int)
    pred[np.argsort(-score, kind="stable")[:9]] = 1
    return float((pred == y).mean())


def ic(score: np.ndarray, r: np.ndarray) -> float:
    return float(spearmanr(score, r).correlation)


# --------------------------------------------------------------------------- arms
class QsvmArm:
    def __init__(self, U, lam, C):
        self.psi = zz_states(lam * np.pi * U)
        self.C = C

    def score(self, tr, te, y):
        K = gram(self.psi[tr], self.psi[tr])
        svm = SVC(kernel="precomputed", C=self.C).fit(K, y[tr])
        return svm.decision_function(gram(self.psi[te], self.psi[tr]))


class RbfArm:
    def __init__(self, U, gamma, C):
        self.U, self.gamma, self.C = U, gamma, C

    def score(self, tr, te, y):
        svm = SVC(kernel="rbf", gamma=self.gamma, C=self.C).fit(self.U[tr], y[tr])
        return svm.decision_function(self.U[te])


class LogitArm:
    def __init__(self, U, C):
        self.U, self.C = U, C

    def score(self, tr, te, y):
        m = LogisticRegression(C=self.C, max_iter=1000).fit(self.U[tr], y[tr])
        return m.decision_function(self.U[te])


class VqcArm:
    """Warm-started month to month (PREREG M2): call score() in date order."""

    def __init__(self, U, lam, seed=7):
        self.psi = zz_states(lam * np.pi * U)
        self.model = Vqc(4, reps=3, maxiter=150, seed=seed)

    def score(self, tr, te, y):
        self.model.fit(self.psi[tr], y[tr])
        return self.model.proba1(self.psi[te])
