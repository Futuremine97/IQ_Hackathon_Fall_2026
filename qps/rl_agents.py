"""POMDP belief agent (2-regime diagonal-Gaussian HMM, Baum-Welch) and GRPO (Plackett-Luce policy,
group-relative advantages). Pure numpy; see PREREG_AMENDMENT_4_POMDP_GRPO.md."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


# ------------------------------------------------------------------ POMDP / HMM
def _log_emis(X, mu, var):
    return -0.5 * (np.log(2 * np.pi * var)[None] + (X[:, None, :] - mu[None]) ** 2 / var[None]).sum(-1)


def hmm_fit(X, n_iter=200, restarts=5, seed=0, tol=1e-6):
    """EM for a 2-state HMM with diagonal Gaussian emissions. Returns the best (loglik, params)."""
    T, N = X.shape
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        if r == 0:   # split months by cross-asset mean return
            m = X.mean(1)
            g = np.stack([m <= np.median(m), m > np.median(m)], 1).astype(float)
        else:
            g = rng.dirichlet([1, 1], size=T)
        mu = (g.T @ X) / g.sum(0)[:, None]
        var = np.maximum((g.T @ X ** 2) / g.sum(0)[:, None] - mu ** 2, 1e-6)
        A = np.array([[0.9, 0.1], [0.1, 0.9]])
        pi = np.array([0.5, 0.5])
        prev = -np.inf
        for _ in range(n_iter):
            le = _log_emis(X, mu, var)
            la = np.empty((T, 2))
            la[0] = np.log(pi) + le[0]
            for t in range(1, T):
                la[t] = logsumexp(la[t - 1][:, None] + np.log(A), axis=0) + le[t]
            lb = np.zeros((T, 2))
            for t in range(T - 2, -1, -1):
                lb[t] = logsumexp(np.log(A) + (le[t + 1] + lb[t + 1])[None, :], axis=1)
            ll = logsumexp(la[-1])
            lg = la + lb - ll
            gam = np.exp(lg)
            xi = np.zeros((2, 2))
            for t in range(T - 1):
                xi += np.exp(la[t][:, None] + np.log(A) + (le[t + 1] + lb[t + 1])[None, :] - ll)
            pi = np.clip(gam[0], 1e-6, None); pi /= pi.sum()
            A = xi / xi.sum(1, keepdims=True)
            A = np.clip(A, 1e-6, None); A /= A.sum(1, keepdims=True)
            w = gam.sum(0)
            mu = (gam.T @ X) / w[:, None]
            var = np.maximum((gam.T @ X ** 2) / w[:, None] - mu ** 2, 1e-6)
            if ll - prev < tol:
                break
            prev = ll
        if best is None or ll > best[0]:
            filt = np.exp(la[-1] - logsumexp(la[-1]))
            best = (ll, dict(A=A, mu=mu, var=var, belief=filt))
    return best


def pomdp_scores(X_hist):
    """Fit on the monthly-return history and return (predictive belief, expected next-month returns, params)."""
    _, p = hmm_fit(X_hist)
    b_next = p["belief"] @ p["A"]
    return b_next, b_next @ p["mu"], p


# ------------------------------------------------------------------ GRPO
def poly_features(U):
    """[u, u_a u_b, u^2] for a (N, 4) block of rank features, centred within the month."""
    n = U.shape[1]
    cols = [U] + [U[:, [a]] * U[:, [b]] for a in range(n) for b in range(a + 1, n)] + [U ** 2]
    F = np.hstack(cols)
    return F - F.mean(0, keepdims=True)


class Grpo:
    def __init__(self, dim, K, G=16, beta=0.04, lr=0.05, seed=0, shuffle_reward=False):
        self.theta = np.zeros(dim)
        self.K, self.G, self.beta, self.lr = K, G, beta, lr
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle_reward
        self.m = np.zeros(dim); self.v = np.zeros(dim); self.t = 0

    def step(self, Phi, R):
        """One full-batch GRPO step. Phi: (M, N, D) features, R: (M, N) realised next-month returns."""
        M, N, D = Phi.shape
        z = Phi @ self.theta                                             # (M, N)
        gumb = -np.log(-np.log(self.rng.random((M, self.G, N))))
        acts = np.argsort(-(z[:, None, :] + gumb), axis=2)[:, :, :self.K]  # PL samples (Gumbel top-k)
        rew = np.take_along_axis(np.broadcast_to(R[:, None, :], (M, self.G, N)), acts, axis=2).mean(2)
        if self.shuffle:
            rew = np.take_along_axis(rew, self.rng.permuted(np.tile(np.arange(self.G), (M, 1)), axis=1), axis=1)
        adv = (rew - rew.mean(1, keepdims=True)) / (rew.std(1, keepdims=True) + 1e-8)
        mask = np.ones((M, self.G, N))
        glog = np.zeros((M, self.G, D))
        zz = np.broadcast_to(z[:, None, :], (M, self.G, N))
        for j in range(self.K):
            logits = np.where(mask > 0, zz, -np.inf)
            p = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
            a = acts[:, :, j]
            phi_a = Phi[np.arange(M)[:, None], a]                           # (M, G, D)
            glog += phi_a - np.einsum("mgn,mnd->mgd", p, Phi)
            np.put_along_axis(mask, a[:, :, None], 0.0, axis=2)
        g_pg = (adv[:, :, None] * glog).mean((0, 1))
        p1 = np.exp(z - logsumexp(z, axis=1, keepdims=True))
        dkl = p1 * (np.log(p1 + 1e-300) - (p1 * np.log(p1 + 1e-300)).sum(1, keepdims=True))
        g_kl = np.einsum("mn,mnd->d", dkl, Phi) / M
        g = g_pg - self.beta * g_kl                                     # ascent direction
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g ** 2
        mh, vh = self.m / (1 - 0.9 ** self.t), self.v / (1 - 0.999 ** self.t)
        self.theta += self.lr * mh / (np.sqrt(vh) + 1e-8)
        return float(rew.mean())

    def scores(self, Phi_month):
        return Phi_month @ self.theta
