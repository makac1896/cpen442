"""Assignment 4: attacks against searchable symmetric encryption.

Complete the functions below. You play the role of the honest-but-curious server.
You can use the functions in sse.py (e.g., h, h_key, aes_ctr, access_pattern_to_documents).

Common arguments:
  index:         the Encrypted Search Index (ESI), {query token: (nonce, encrypted access pattern)}
  queries:       the queries the server saw, in order: a list of (token, document IDs)
  keyword_space: every keyword that Alice's index can contain (a list of strings)
  aux_documents: the auxiliary dataset, documents similar to Alice's (a list of sets of keywords)
  num_documents: the number of documents in Alice's index

The query-recovery attacks return one keyword guess per query, in the same order as `queries`.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy

from sse import h, h_key, aes_ctr, access_pattern_to_documents


def attack_unkeyed(index, queries, keyword_space) -> list[str]:
    """Section 3: the "unkeyed" scheme."""
    token_to_keyword = {h(b"token" + w.encode()): w for w in keyword_space}
    return [token_to_keyword[token] for token, _documents in queries]


def _aux_statistics(keyword_space, aux_documents):
    """Fraction of aux documents containing each keyword (diagonal) or each pair of keywords."""
    index = {w: k for k, w in enumerate(keyword_space)}
    A = np.zeros((len(aux_documents), len(keyword_space)))
    for d, doc in enumerate(aux_documents):
        for w in doc:
            if w in index:
                A[d, index[w]] = 1.0
    C = (A.T @ A) / len(aux_documents)
    eps = 1e-4
    return np.clip(C, eps, 1 - eps)


def _misfit(k, n, p):
    """Binomial deviance: how badly k matches in n documents fit a keyword with expected fraction p."""
    return xlogy(k, k) - k * np.log(n * p) + xlogy(n - k, n - k) - (n - k) * np.log(n * (1 - p))


def attack_secure(queries, keyword_space, aux_documents, num_documents, rounds=30, lam=1.0) -> list[str]:
    """Section 4: the "secure" scheme."""
    n, Q, K = num_documents, len(queries), len(keyword_space)
    C = _aux_statistics(keyword_space, aux_documents)
    f = np.diag(C)

    R = np.zeros((n, Q))
    for i, (_token, documents) in enumerate(queries):
        R[documents, i] = 1.0
    overlap = R.T @ R  # documents shared by each pair of queries
    count = np.diag(overlap)

    freq_cost = _misfit(count[:, None], n, f[None, :])
    _, assignment = linear_sum_assignment(freq_cost)

    other_query = (1.0 - np.eye(Q))[:, None, :]
    for _ in range(rounds):
        Cp = C[:, assignment]  # co-occurrence of each keyword with the current guesses
        cost = _misfit(overlap[:, None, :], n, Cp[None, :, :]) * other_query
        cost = cost * (assignment[None, None, :] != np.arange(K)[None, :, None])  # skip the keyword's own guess
        _, new_assignment = linear_sum_assignment(freq_cost + lam * cost.sum(axis=2))
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment

    return [keyword_space[k] for k in assignment]


def choose_injected_documents(keyword_space, max_documents, aux_documents) -> list[set[str]]:
    """Section 5, part 1: the documents you inject into Alice's index (at most max_documents)."""
    return []


def attack_injection(queries, keyword_space, aux_documents, num_documents, injected_documents) -> list[str]:
    """Section 5, part 2: the "secure" scheme, after injecting documents.
    injected_documents are the documents you chose in part 1. Alice shuffles all her documents
    (yours included) before building the index, so you do not know their IDs."""
    return [keyword_space[0] for _ in queries]
