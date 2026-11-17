"""Parameterised variants of the secure-scheme attack, for benchmarking (not submitted)."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy

from a4 import _aux_statistics


def _cost(k, n, p, like):
    """Misfit of an observed count k (out of n documents) against an expected fraction p."""
    if like == "gauss":
        return (k - n * p) ** 2 / (2 * n * p * (1 - p))
    if like == "dev":  # binomial deviance
        return xlogy(k, k) - k * np.log(n * p) + xlogy(n - k, n - k) - (n - k) * np.log(n * (1 - p))
    raise ValueError(like)


def secure(queries, keyword_space, aux_documents, num_documents,
           lam=1.0, rounds=10, like="gauss", mask_self=False):
    n, Q, K = num_documents, len(queries), len(keyword_space)
    C = _aux_statistics(keyword_space, aux_documents)
    f = np.diag(C)

    R = np.zeros((n, Q))
    for i, (_token, documents) in enumerate(queries):
        R[documents, i] = 1.0
    overlap = R.T @ R
    count = np.diag(overlap)

    freq_cost = _cost(count[:, None], n, f[None, :], like)
    _, assignment = linear_sum_assignment(freq_cost)

    keep = (1.0 - np.eye(Q))[:, None, :]
    for _ in range(rounds):
        Cp = C[:, assignment]
        cost = _cost(overlap[:, None, :], n, Cp[None, :, :], like) * keep
        if mask_self:
            cost = cost * (assignment[None, None, :] != np.arange(K)[None, :, None])
        _, new_assignment = linear_sum_assignment(freq_cost + lam * cost.sum(axis=2))
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment

    return [keyword_space[k] for k in assignment]


def injection(plan="none", **opts):
    """Returns (choose_injected_documents, attack_injection) for the injection benchmark."""
    import a4

    if plan == "none":  # no injection: the plain secure attack, as a baseline
        return (lambda keyword_space, max_documents, aux: [],
                lambda queries, keyword_space, aux, n, injected: a4.attack_secure(queries, keyword_space, aux, n))

    if plan == "probe":  # harness self-test: one document with every keyword, found by its column count
        def attack(queries, keyword_space, aux, n, injected):
            hits = np.bincount(np.concatenate([d for _t, d in queries]).astype(int), minlength=n)
            return a4.attack_secure(queries, keyword_space, aux, n), {"columns": [int(hits.argmax())]}
        return (lambda keyword_space, max_documents, aux: [set(keyword_space)]), attack

    raise ValueError(plan)
