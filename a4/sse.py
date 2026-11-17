"""Searchable symmetric encryption (SSE) for Assignment 4.

Alice (the client) has documents, and each document is a set of keywords.
She builds an Encrypted Search Index (ESI) and uploads it to a server. To search
for a keyword, she sends the server a query token, and the server returns the
documents that contain that keyword.

The ESI is a dictionary {query token: (nonce, encrypted access pattern)}, with one
entry per keyword. The access pattern of a keyword is a binary vector with one bit
per document: bit i is 1 if document i contains the keyword.

There are two versions of the scheme (see SCHEMES below). Do not modify this
file: we will use an unmodified copy to grade your attacks.
"""

import hashlib
import hmac
import os

from Crypto.Cipher import AES

SCHEMES = ("unkeyed", "secure")


def h(data: bytes) -> bytes:
    """Hash function: SHA-256."""
    return hashlib.sha256(data).digest()


def h_key(key: bytes, data: bytes) -> bytes:
    """Keyed hash function: HMAC-SHA256."""
    return hmac.new(key, data, hashlib.sha256).digest()


def aes_ctr(key: bytes, nonce: bytes, data: bytes) -> bytes:
    """AES-CTR encryption (and decryption, which is the same operation)."""
    return AES.new(key, AES.MODE_CTR, nonce=nonce).encrypt(data)


def documents_to_access_pattern(document_ids: list[int], num_documents: int) -> bytes:
    bits = bytearray((num_documents + 7) // 8)
    for i in document_ids:
        bits[i // 8] |= 1 << (i % 8)
    return bytes(bits)


def access_pattern_to_documents(access_pattern: bytes) -> list[int]:
    return [i for i in range(len(access_pattern) * 8) if access_pattern[i // 8] >> (i % 8) & 1]


class Client:
    """Alice. She knows her documents, the keyword space, and her secret keys."""

    def __init__(self, documents: list[set[str]], keyword_space: list[str], scheme: str):
        assert scheme in SCHEMES
        self.documents = documents
        self.keyword_space = keyword_space
        self.scheme = scheme
        self.key_token = os.urandom(16)  # K1, used for tokens
        self.key_list = os.urandom(16)   # K2, used for the access patterns

    def token_and_key(self, keyword: str) -> tuple[bytes, bytes]:
        """The token t_w that identifies the keyword's entry, and the key k_w of its access pattern."""
        w = keyword.encode()
        if self.scheme == "unkeyed":
            return h(b"token" + w), h(b"key" + w)[:16]
        # "secure": keyed hashes.
        return h_key(self.key_token, w), h_key(self.key_list, w)[:16]

    def build_index(self) -> dict[bytes, tuple[bytes, bytes]]:
        index = {}
        for keyword in self.keyword_space:
            matches = [i for i, doc in enumerate(self.documents) if keyword in doc]
            access_pattern = documents_to_access_pattern(matches, len(self.documents))
            token, key = self.token_and_key(keyword)
            nonce = os.urandom(8)
            index[token] = (nonce, aes_ctr(key, nonce, access_pattern))
        return index

    def search(self, keyword: str, server: "Server") -> list[int]:
        token, key = self.token_and_key(keyword)
        return server.search(token, key)


class Server:
    """The honest-but-curious server. It keeps everything it sees."""

    def __init__(self, index: dict[bytes, tuple[bytes, bytes]]):
        self.index = index
        self.queries = []  # one (token, matching document IDs) pair per search

    def search(self, token: bytes, key: bytes) -> list[int]:
        nonce, ciphertext = self.index[token]
        documents = access_pattern_to_documents(aes_ctr(key, nonce, ciphertext))
        self.queries.append((token, documents))
        return documents


def load_keywords(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_documents(path: str) -> list[set[str]]:
    """One document per line, keywords separated by spaces."""
    with open(path) as f:
        return [set(line.split()) for line in f]
