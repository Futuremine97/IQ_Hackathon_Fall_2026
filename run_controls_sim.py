"""Simulator-level controls PC1, PC2, PC4, PC5 (PREREG section 5) -> results/CONTROLS_SIM.json"""
import json
import os
import time

import numpy as np

from qps.quantum import (Vqc, bit_matrix, gram, qaoa_expectation, qaoa_state, zz_states)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
res = {}


def product_map(x):  # the notebook's data map
    return x[0] if len(x) == 1 else x[0] * x[1]


# PC1 kernel ------------------------------------------------------------------------
from qiskit.circuit.library import zz_feature_map, QAOAAnsatz, real_amplitudes
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

X = rng.uniform(0, np.pi, size=(60, 4)) * 0.5
fm = zz_feature_map(4, reps=2, data_map_func=product_map)
K_ref = FidelityStatevectorKernel(feature_map=fm).evaluate(X)
K_ours = gram(zz_states(X), zz_states(X))
res["PC1_kernel_max_abs_diff"] = float(np.abs(K_ref - K_ours).max())
res["PC1_pass"] = res["PC1_kernel_max_abs_diff"] < 1e-10

# PC2 ad hoc dataset (notebook setup) -------------------------------------------------
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.svm import SVC

algorithm_globals.random_seed = 12345
Xtr, ytr, Xte, yte = ad_hoc_data(training_size=20, test_size=10, n=2, gap=0.3, one_hot=False)
fm2 = zz_feature_map(2, reps=2)  # default data map, as in the notebook
k2 = FidelityStatevectorKernel(feature_map=fm2)
svm = SVC(kernel="precomputed").fit(k2.evaluate(Xtr), ytr)
res["PC2_adhoc_test_acc"] = float(svm.score(k2.evaluate(Xte, Xtr), yte))
res["PC2_pass"] = res["PC2_adhoc_test_acc"] >= 0.9

# PC4 QAOA simulator vs qiskit QAOAAnsatz ----------------------------------------------
def walsh_to_pauli(cost):
    """Exact Z-string decomposition of a diagonal cost: C = sum_S c_S prod_{i in S} Z_i."""
    n = int(np.log2(cost.size))
    B = bit_matrix(n)
    terms = []
    for S in range(cost.size):
        sel = [(S >> i) & 1 for i in range(n)]
        sign = (-1.0) ** (B @ np.array(sel))
        c = float(cost @ sign) / cost.size
        if abs(c) > 1e-13:
            label = "".join("Z" if sel[n - 1 - q] else "I" for q in range(n))
            terms.append((label, c))
    return SparsePauliOp.from_list(terms)


def qiskit_expect(cost_op, betas, gammas):
    p = len(betas)
    ans = QAOAAnsatz(cost_operator=cost_op, reps=p)
    vals = {}
    for prm in ans.parameters:
        idx = int(prm.name.split("[")[1].rstrip("]"))
        vals[prm] = betas[idx] if prm.name.startswith("β") else gammas[idx]
    sv = Statevector(ans.assign_parameters(vals))
    return float(np.real(sv.expectation_value(cost_op))), sv.data


pc4 = []
# (a) notebook 5-node max-cut, H_C = sum_(i,j) Z_i Z_j
edges = [(0, 1), (0, 2), (0, 4), (1, 2), (2, 3), (3, 4)]
B5 = bit_matrix(5)
z = 1 - 2 * B5
cost_mc = sum(z[:, i] * z[:, j] for i, j in edges).astype(float)
op_mc = SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0) for i, j in edges], num_qubits=5)
# (b) random 8-asset portfolio QUBO
n8 = 8
mu = rng.normal(size=n8)
A = rng.normal(size=(n8, n8)); S = A @ A.T / n8
B8 = bit_matrix(n8).astype(float)
cost_pf = -(B8 @ mu) + 0.125 * np.einsum("bi,ij,bj->b", B8, S, B8) + 2.0 * (B8.sum(1) - 3) ** 2
for name, cost, op in [("maxcut5", cost_mc, op_mc), ("portfolio8", cost_pf, walsh_to_pauli(cost_pf))]:
    for trial in range(3):
        p = trial + 1
        bet, gam = rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi, np.pi, p)
        e_ref, sv_ref = qiskit_expect(op, bet, gam)
        psi = qaoa_state(cost, bet, gam)
        e_ours = qaoa_expectation(cost, np.concatenate([bet, gam]))
        fid = abs(np.vdot(sv_ref, psi)) ** 2
        pc4.append(dict(instance=name, p=p, expect_qiskit=e_ref, expect_ours=e_ours,
                        abs_diff=abs(e_ref - e_ours), state_fidelity=float(fid)))
res["PC4_cases"] = pc4
res["PC4_max_abs_diff"] = max(c["abs_diff"] for c in pc4)
res["PC4_min_fidelity"] = min(c["state_fidelity"] for c in pc4)
res["PC4_pass"] = res["PC4_max_abs_diff"] < 1e-9 and res["PC4_min_fidelity"] > 1 - 1e-9

# PC5 VQC forward pass ------------------------------------------------------------------
v = Vqc(4, reps=3, seed=7)
Xv = rng.uniform(0, np.pi, size=(20, 4)) * 0.5
p_ours = v.proba1(zz_states(Xv))
p_ref = []
for x in Xv:
    circ = fm.assign_parameters(x).compose(v.ansatz.assign_parameters(dict(zip(v.params, v.theta))))
    probs = Statevector(circ).probabilities()
    p_ref.append(sum(pr for b, pr in enumerate(probs) if bin(b).count("1") % 2 == 1))
res["PC5_max_abs_diff"] = float(np.abs(np.array(p_ref) - p_ours).max())
res["PC5_pass"] = res["PC5_max_abs_diff"] < 1e-10

res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
json.dump(res, open(os.path.join(OUT, "CONTROLS_SIM.json"), "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "PC4_cases"}, indent=2))
