"""Exact statevector building blocks used by every arm.

* zz_states:   batched statevectors of qiskit's zz_feature_map with the notebook's
               product data map (phi_i = x_i, phi_ij = x_i * x_j), full entanglement.
* Vqc:         ZZ feature map + RealAmplitudes ansatz, parity read-out (as in
               qiskit-machine-learning's VQC for two classes), trained by COBYLA.
* qaoa_*:      numpy QAOA on a diagonal cost vector (X mixer, |+> start), the same
               unitary as qiskit's QAOAAnsatz: prod_l exp(-i b_l sum X) exp(-i g_l C).

All three are cross-checked against qiskit in run_controls.py (PC1, PC4, PC5).
Bit convention everywhere: basis index b, qubit i <-> bit (b >> i) & 1 (qiskit order).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- kernel
def _bits(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    out = np.array([[1.0]])
    for _ in range(n):
        out = np.kron(h, out)
    return out


def zz_states(X: np.ndarray, reps: int = 2, alpha: float = 2.0) -> np.ndarray:
    """Statevectors (N, 2^n) of zz_feature_map(n, reps, entanglement='full',
    data_map_func=product_map) evaluated at the rows of X.

    One repetition = H on every qubit, then the diagonal phase
    exp(i*alpha*[sum_i phi_i b_i + sum_{i<j} phi_ij (b_i xor b_j)]) up to a global phase
    (P(alpha*phi) on each qubit; CX . P(alpha*phi_ij) . CX on each pair)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[1]
    B = _bits(n).astype(float)                       # (2^n, n)
    pairs = list(itertools.combinations(range(n), 2))
    phase = X @ B.T                                  # sum_i x_i b_i
    for i, j in pairs:
        xor = (B[:, i] != B[:, j]).astype(float)     # (2^n,)
        phase = phase + np.outer(X[:, i] * X[:, j], xor)
    D = np.exp(1j * alpha * phase)                   # (N, 2^n)
    H = _hadamard(n)
    psi = np.zeros((X.shape[0], 2 ** n), dtype=complex)
    psi[:, 0] = 1.0
    for _ in range(reps):
        psi = (psi @ H.T) * D
    return psi


def gram(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    """Fidelity kernel |<psi_a|psi_b>|^2."""
    return np.abs(np.conj(psi_a) @ psi_b.T) ** 2


# --------------------------------------------------------------------------- VQC
class Vqc:
    """Variational quantum classifier on precomputed feature states.

    p(class 1 | x) = sum over odd-parity basis states of |U(theta) psi(x)|^2,
    U = RealAmplitudes(n, reps) (qiskit, default entanglement)."""

    def __init__(self, n_qubits: int = 4, reps: int = 3, maxiter: int = 150, seed: int = 7):
        from qiskit.circuit.library import real_amplitudes
        self.ansatz = real_amplitudes(n_qubits, reps=reps)
        self.params = list(self.ansatz.parameters)
        self.maxiter = maxiter
        rng = np.random.default_rng(seed)
        self.theta = rng.uniform(-np.pi, np.pi, len(self.params))
        self.odd = (_bits(n_qubits).sum(axis=1) % 2 == 1)

    def unitary(self, theta: np.ndarray) -> np.ndarray:
        from qiskit.quantum_info import Operator
        return Operator(self.ansatz.assign_parameters(dict(zip(self.params, theta)))).data

    def proba1(self, psi: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        U = self.unitary(self.theta if theta is None else theta)
        amp = psi @ U.T
        return (np.abs(amp[:, self.odd]) ** 2).sum(axis=1)

    def fit(self, psi: np.ndarray, y: np.ndarray) -> "Vqc":
        y = np.asarray(y, dtype=float)
        eps = 1e-9

        def loss(theta):
            p = np.clip(self.proba1(psi, theta), eps, 1 - eps)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(loss, self.theta, method="COBYLA",
                       options={"maxiter": self.maxiter, "rhobeg": 0.5})
        self.theta = res.x
        return self


# --------------------------------------------------------------------------- QAOA
def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i beta X) on every qubit."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.copy()
    for i in range(n):                               # qubit i = middle axis of the view
        t = psi.reshape(2 ** (n - 1 - i), 2, 2 ** i)
        a0 = t[:, 0, :].copy()
        t[:, 0, :] *= c
        t[:, 0, :] += s * t[:, 1, :]
        t[:, 1, :] *= c
        t[:, 1, :] += s * a0
    return psi


def qaoa_state(cost: np.ndarray, betas, gammas) -> np.ndarray:
    n = int(np.log2(cost.size))
    psi = np.full(cost.size, 1 / np.sqrt(cost.size), dtype=complex)
    for b, g in zip(betas, gammas):
        psi = psi * np.exp(-1j * g * cost)
        psi = apply_mixer(psi, b, n)
    return psi


def qaoa_expectation(cost: np.ndarray, x: np.ndarray) -> float:
    p = len(x) // 2
    psi = qaoa_state(cost, x[:p], x[p:])
    return float(np.real(np.vdot(psi, cost * psi)))


def qaoa_optimize(cost: np.ndarray, p: int, starts, maxiter: int = 300):
    """Minimise <C> over (betas, gammas) with COBYLA from each start; keep the best."""
    best = None
    for x0 in starts:
        res = minimize(lambda x: qaoa_expectation(cost, x), np.asarray(x0, float),
                       method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        if best is None or res.fun < best.fun:
            best = res
    return best


def ramp_start(p: int) -> np.ndarray:
    """Linear-ramp (annealing-like) initial angles: beta decreasing, gamma increasing."""
    k = (np.arange(p) + 0.5) / p
    return np.concatenate([0.6 * (1 - k), 0.6 * k])


def bit_matrix(n: int) -> np.ndarray:
    return _bits(n)
