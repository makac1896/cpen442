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

from sse import h, h_key, aes_ctr, access_pattern_to_documents


def attack_unkeyed(index, queries, keyword_space) -> list[str]:
    """Section 3: the "unkeyed" scheme."""
    return [keyword_space[0] for _ in queries]


def attack_secure(queries, keyword_space, aux_documents, num_documents) -> list[str]:
    """Section 4: the "secure" scheme."""
    return [keyword_space[0] for _ in queries]


def choose_injected_documents(keyword_space, max_documents, aux_documents) -> list[set[str]]:
    """Section 5, part 1: the documents you inject into Alice's index (at most max_documents)."""
    return []


def attack_injection(queries, keyword_space, aux_documents, num_documents, injected_documents) -> list[str]:
    """Section 5, part 2: the "secure" scheme, after injecting documents.
    injected_documents are the documents you chose in part 1. Alice shuffles all her documents
    (yours included) before building the index, so you do not know their IDs."""
    return [keyword_space[0] for _ in queries]
