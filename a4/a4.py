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

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import xlogy

from sse import h, h_key, aes_ctr, access_pattern_to_documents

MAX_PLANTED = 3  # injected documents we use
SHORTLIST = 1  # likely columns tried per injected document, besides the best-agreeing one
POOL_SIZE = 30  # columns considered as injected documents
TOP_SETS = 3  # column sets compared by the full fit
INDEPENDENCE_SCALE = 0.1  # correlation with query counts that counts as one unit of suspicion
PRE_ROUNDS = 5  # refinement rounds used to compare column choices


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


def _match(overlap, n, C, allowed=None, rounds=30, lam=1.0):
    """Assign each query a distinct keyword. Returns the assignment and its total cost.
    allowed[i][w] says whether query i may be keyword w; forbidden pairs get a huge cost."""
    Q, K = len(overlap), len(C)
    f = np.diag(C)
    count = np.diag(overlap)
    forbidden = np.zeros((Q, K)) if allowed is None else 1e8 * ~allowed

    freq_cost = _misfit(count[:, None], n, f[None, :]) + forbidden
    _, assignment = linear_sum_assignment(freq_cost)

    other_query = (1.0 - np.eye(Q))[:, None, :]
    total = freq_cost[np.arange(Q), assignment].sum()
    for _ in range(rounds):
        Cp = C[:, assignment]  # co-occurrence of each keyword with the current guesses
        cost = _misfit(overlap[:, None, :], n, Cp[None, :, :]) * other_query
        cost = cost * (assignment[None, None, :] != np.arange(K)[None, :, None])  # skip the keyword's own guess
        cost = freq_cost + lam * cost.sum(axis=2)
        _, new_assignment = linear_sum_assignment(cost)
        total = cost[np.arange(Q), new_assignment].sum()
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
    return assignment, total


def _result_matrix(queries, num_documents):
    """Document-by-query 0/1 matrix: R[d][i] = 1 if document d matched query i."""
    R = np.zeros((num_documents, len(queries)))
    for i, (_token, documents) in enumerate(queries):
        R[documents, i] = 1.0
    return R


def attack_secure(queries, keyword_space, aux_documents, num_documents, rounds=30, lam=1.0) -> list[str]:
    """Section 4: the "secure" scheme."""
    C = _aux_statistics(keyword_space, aux_documents)
    R = _result_matrix(queries, num_documents)
    assignment, _ = _match(R.T @ R, num_documents, C, rounds=rounds, lam=lam)
    return [keyword_space[k] for k in assignment]


def choose_injected_documents(keyword_space, max_documents, aux_documents) -> list[set[str]]:
    """Section 5, part 1: the documents you inject into Alice's index (at most max_documents)."""
    d = min(max_documents, MAX_PLANTED)
    if d <= 0:
        return []
    C = _aux_statistics(keyword_space, aux_documents)
    by_frequency = sorted(range(len(keyword_space)), key=lambda k: (-C[k, k], k))
    documents = [set() for _ in range(d)]
    for rank, k in enumerate(by_frequency):
        code = rank % 2 ** d  # neighbours in frequency get different codes
        for j in range(d):
            if code >> j & 1:
                documents[j].add(keyword_space[k])
    return documents


def _injection_core(queries, keyword_space, aux_documents, num_documents, injected_documents, rounds=30,
                    shortlist=SHORTLIST, pool_size=POOL_SIZE, top_sets=TOP_SETS, tau=INDEPENDENCE_SCALE):
    """Returns the assignment and, for each injected document, the column it was found in."""
    d = len(injected_documents)
    Q, K = len(queries), len(keyword_space)
    C = _aux_statistics(keyword_space, aux_documents)
    R = _result_matrix(queries, num_documents)
    overlap = R.T @ R
    if d == 0:  # nothing injected: the plain attack
        assignment, _ = _match(overlap, num_documents, C, rounds=rounds)
        return assignment, []
    keyword_code = np.array([[w in doc for doc in injected_documents] for w in keyword_space]).reshape(K, d)
    bits = 1 << np.arange(d)

    def solve(columns, rounds):
        """Restrict every query to the keywords whose code matches the columns' pattern."""
        columns = list(columns)
        natural_overlap = overlap - R[columns].T @ R[columns]
        allowed = (R[columns].T.astype(bool)[:, None, :] == keyword_code[None, :, :]).all(axis=2)
        return _match(natural_overlap, num_documents - d, C, allowed, rounds)

    # Queries are a random subset of the keywords, so injected document j matches about |S_j| * Q / K of them
    hits = R.sum(axis=1)
    share = keyword_code.mean(axis=0)
    spread = np.sqrt(Q * share * (1 - share) * (K - Q) / max(K - 1, 1))
    plausible_for = np.abs(hits[:, None] - Q * share[None, :]) <= 4 * spread[None, :] + 1  # (columns, documents)
    plausible = plausible_for.any(axis=1)
    if plausible.sum() < d:
        plausible[:] = True

    # Option 1: a first guess from the plain attack says which columns behave like each injected document
    guess, _ = _match(overlap, num_documents, C, rounds=rounds)
    predicted = keyword_code[guess].astype(float)
    agreement = R @ predicted + (1 - R) @ (1 - predicted)
    per_document = []
    for j in range(d):
        by_agreement = np.argsort(-agreement[:, j])
        likely = by_agreement[plausible_for[by_agreement, j]]
        per_document.append(list(dict.fromkeys([*likely[:shortlist], by_agreement[0]])))
    options = {order for order in itertools.product(*per_document) if len(set(order)) == d}

    # Option 2: no guess needed. We dealt keywords to the documents by frequency, so an injected document
    # matches queries independently of how common their keyword is (unlike a natural one), and together
    # the documents split the queries into code groups of known expected sizes.
    count = np.log(R.sum(axis=0) + 1)
    count = (count - count.mean()) / (count.std() + 1e-12)
    centred = R - R.mean(axis=1, keepdims=True)
    correlation = np.abs(centred @ count) / (np.linalg.norm(centred, axis=1) * np.linalg.norm(count) + 1e-12)
    correlation[~plausible] = np.inf
    pool = np.argsort(correlation)[:pool_size]
    group = np.bincount(keyword_code @ bits, minlength=2 ** d)
    expected = Q * group / K
    variance = expected * (1 - group / K) * (K - Q) / max(K - 1, 1) + 0.25

    def suspicion(columns):
        observed = np.bincount(R[list(columns)].T.astype(int) @ bits, minlength=2 ** d)
        return (((observed - expected) ** 2) / variance).sum() + ((correlation[list(columns)] / tau) ** 2).sum()

    for columns in sorted(itertools.combinations(pool, d), key=suspicion)[:top_sets]:
        options.update(itertools.permutations(columns))

    # the full fit decides between all the candidates (it also fixes which column is which document)
    best = min(options, key=lambda order: solve(order, PRE_ROUNDS)[1])
    assignment, _ = solve(best, rounds)
    return assignment, list(best)


def attack_injection(queries, keyword_space, aux_documents, num_documents, injected_documents) -> list[str]:
    """Section 5, part 2: the "secure" scheme, after injecting documents.
    injected_documents are the documents you chose in part 1. Alice shuffles all her documents
    (yours included) before building the index, so you do not know their IDs."""
    assignment, _ = _injection_core(queries, keyword_space, aux_documents, num_documents, injected_documents)
    return [keyword_space[k] for k in assignment]
