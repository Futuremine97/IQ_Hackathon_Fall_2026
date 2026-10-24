# PREREG AMENDMENT 4 — POMDP and GRPO agents (2026-10-23)

Scope: add POMDP and GRPO (Group Relative Policy Optimization) agents to the comparison.
Written and hashed before any POMDP/GRPO result. Earlier PREREG/AMENDMENTS unchanged.
[REDACTED: one internal cross-reference sentence, see REDACTIONS.md] A lesson from earlier work is
kept as a control: GRPO must beat a shuffled-reward twin.

Decision task (same as the existing arms): every month-end t, score the assets and hold the top K
(equal weight) next month. U18 = the 18 ETFs, K = 4 (accuracy = top 9 of 18); U9 = the 9 sector
SPDRs, K = 3 (accuracy = top 4 of 9, as AMENDMENT 3). Test 2014-01..2026-08, expanding windows,
label-end <= t for every training sample (asserted).

## POMDP agent (Bayes-filtering belief agent)
Hidden market regime s in {0,1}, Markov transition matrix P; observation = the vector of the
universe's monthly returns; emission N(mu_s, diag sigma_s^2). Fit by EM (Baum-Welch, 200 iterations
max, 5 restarts, best log-likelihood) on all monthly returns with month end <= t (from 2005-02).
Belief b_t = forward-filtered P(s_t | o_1..o_t); predictive b_{t+1} = P^T b_t;
score_i = sum_s b_{t+1}(s) mu_{s,i}. Actions do not change the regime, so ignoring costs the
Bayes-optimal policy is myopic: hold the top K scores.
Regime use (labels): RG2 P(SPY > 0) = sum_s b(s) Phi(mu_s,SPY / sigma_s,SPY) > 0.5; RG3 same for
SPY - TLT (diagonal Gaussians -> difference mean and summed variance); fitted on U18 returns.

## GRPO agent (critic-free policy gradient with group-relative advantages, DeepSeek 2024)
Policy: Plackett-Luce over assets, logits z_i = theta . phi_i, phi_i = the 4 rank features, their 6
pairwise products and 4 squares (14 dims, centred). Action = K distinct assets sampled without
replacement. Reward = realised next-month mean return of the K assets (training months only).
Per training month: G = 16 sampled actions, advantage A_g = (r_g - mean) / (std + 1e-8);
one on-policy step (importance ratio 1, so PPO clipping is inactive), loss = -mean(A_g log pi(a_g))
+ beta KL(first-pick softmax || uniform reference), beta = 0.04, Adam lr 0.05.
Training: 30 epochs over the training months at the first test month, then 10 warm-started epochs
per month. Test action = greedy top K of z. Five seeds (0-4); reported mean and per-seed spread.
Control: shuffled-reward twin (rewards permuted within each group) — must not beat equal weight;
if it does the GRPO arm is VOID.

## Tests (labels; no row-closing power)
* AM4-H1 GRPO and AM4-H2 POMDP: top-K portfolio Sharpe - equal-weight Sharpe, 10 bp costs, circular
  block bootstrap CI90 (block 6, 2,000 reps). POS iff CI90 lower bound > 0.
* Accuracy vs the existing arms (QSVM, RBF, logistic, ILL, Prophet rule): month-paired sign-flip.
* RG2 / RG3: POMDP belief accuracy vs majority, persistence, Markov-switching, QSVM.
