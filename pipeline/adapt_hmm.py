#!/usr/bin/env python3
"""Unsupervised test-time adaptation of the counted joint estimator.

The emission table P(observation | state) is the part of the estimator that
encodes in-domain response bias: how often each model says "yes" in each state.
Out of domain that bias moves (the world model answers "yes" on 40% of
work-free California seconds) while the transition structure -- how long
approaches and zones last -- is a property of roads, not of models.  So we keep
the transition rates and re-estimate only the emission, by EM on UNLABELED
target-domain evidence, with a Dirichlet prior centred on the in-domain table:

  e'[s, o]  ∝  lam * N * e_src[s, o]  +  sum_t gamma_t(s) * 1[o_t = o]

N is the amount of target evidence, so lam is the weight of the source table
relative to the target data (lam = 0.5: equal weight, fixed a priori).  gamma
comes from forward-backward with the same tempered likelihood the filter uses
(exponent dt / tau), and the adapted table is then used by the unchanged causal
filter.  No target label is used anywhere.

Two protocols:
  transductive   adapt on all unlabeled target videos, then filter each
  leave-one-out  adapt on the OTHER target videos only (stricter: the video
                 being scored contributes nothing to its own adaptation)
"""
import numpy as np

import joint_filter as J


def trans_matrix(rate, dt):
    T = np.eye(4) + np.asarray(rate) * dt
    T = np.clip(T, 1e-9, None)
    return T / T.sum(axis=1, keepdims=True)


def posteriors(obs, dts, log_emit, rate, prior):
    """Forward-backward with tempered emissions; returns gamma [n, 4]."""
    n = len(obs)
    kappa = np.asarray(dts) / J.EVIDENCE_TAU
    lik = np.exp(log_emit[:, obs].T * kappa[:, None])            # [n, 4]
    alpha = np.zeros((n, 4)); c = np.zeros(n)
    Ts = [trans_matrix(rate, d) for d in dts]
    a = prior * lik[0]; c[0] = a.sum(); alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ Ts[t]) * lik[t]; c[t] = a.sum() or 1e-300; alpha[t] = a / c[t]
    beta = np.ones((n, 4))
    for t in range(n - 2, -1, -1):
        b = Ts[t + 1] @ (lik[t + 1] * beta[t + 1]); beta[t] = b / (c[t + 1] or 1e-300)
    g = alpha * beta
    return g / g.sum(axis=1, keepdims=True)


def adapt(params, seqs, lam=0.5, iters=5):
    """seqs: list of (obs_index array, dt array).  Returns adapted params."""
    e_src = np.asarray(params["emission"])
    rate, prior = np.asarray(params["rate"]), np.asarray(params["prior"])
    N = sum(len(o) for o, _ in seqs)
    e = e_src.copy()
    for _ in range(iters):
        counts = np.zeros_like(e)
        le = np.log(e + 1e-9)
        for o, d in seqs:
            g = posteriors(o, d, le, rate, prior)
            np.add.at(counts.T, o, g)
        # prior mass spread over states in proportion to the source marginal
        e = lam * N * prior[:, None] * e_src + counts
        e = e / e.sum(axis=1, keepdims=True)
    out = dict(params); out["emission"] = e.tolist()
    return out
