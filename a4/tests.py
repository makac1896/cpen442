"""Runs every scenario of Assignment 4 and prints your score in each one.

    python tests.py                       # tests your a4.py
    python tests.py a4_solution           # tests another module with the same functions
    python tests.py a4 --trials 3         # fewer trials (default: 10), e.g. while debugging

Each scenario: Alice builds her index with one of the schemes, uploads it to the
server, and searches for some keywords. Then your attack gets what the server saw.
Each trial splits the emails at random into Alice's documents and the auxiliary
dataset, and Alice searches for a fresh set of keywords; we average over the trials.
We grade with this same code, over 100 trials, but on other datasets (not Enron),
each with its own keyword space: do not hard-code anything specific to Enron.
"""

import argparse
import importlib
import math
import random
import time

from sse import Client, Server, load_documents, load_keywords

# We will evaluate your solution over 100 trials (100 random splits) and average the results to be as fair
# as possible. This takes a while, so you can play around with the number of trials when evaluating your own
# solutions. Single splits vary a lot, so use at least 10 trials before trusting your score.
NUM_TRIALS = 10

NUM_QUERIES = 100
MAX_INJECTED = 3

POINTS = {"unkeyed": 10, "secure": 50, "injection": 40}
TIME_LIMITS = {"secure": 3, "injection": 10}  # seconds per trial; unkeyed has none
TIME_PENALTY = 10


def simulate(scheme, documents, keyword_space, query_keywords):
    client = Client(documents, keyword_space, scheme)
    server = Server(client.build_index())
    for keyword in query_keywords:
        client.search(keyword, server)
    return server


def accuracy(guesses, query_keywords):
    """Fraction of queries whose guess is exactly the right keyword (a plain string)."""
    return sum(type(g) is str and g == k for g, k in zip(guesses, query_keywords)) / len(query_keywords)


def points(scenario, acc, seconds):
    """Points from the query recovery rate (QRR, in percent), rounded to the nearest half point:
    unkeyed QRR / 10, secure max(0, QRR - 45) (95% earns 50, 100% earns 55), injection
    max(0, QRR - 55) (95% earns 40, 100% earns 45). Then the time penalty, if any."""
    qrr = 100 * acc
    score = {"unkeyed": qrr / 10, "secure": qrr - 45, "injection": qrr - 55}[scenario]
    score = max(0.0, math.floor(score * 2 + 0.5 + 1e-9) / 2)  # halves round up
    if seconds > TIME_LIMITS.get(scenario, float("inf")):
        score = max(0.0, score - TIME_PENALTY)
    return score


def run_all(attacker, document_paths=("data/aux-documents.txt", "data/client-documents.txt"),
            keywords_path="data/keywords.txt", seed=0, num_trials=NUM_TRIALS):
    """Returns {scenario: (average accuracy, average seconds per trial)}. The seconds are the time
    spent in your functions. The document files are pooled; trial t uses random.Random(seed + t) to
    split the pool in half (auxiliary dataset, Alice's documents), draw each scenario's query
    keywords (independently, so no scenario reveals another's queries), and shuffle Alice's documents
    after the injection."""
    keyword_space = load_keywords(keywords_path)
    pool = [doc for path in document_paths for doc in load_documents(path)]

    totals = {scenario: [0.0, 0.0] for scenario in ("unkeyed", "secure", "injection")}
    for trial in range(num_trials):
        rng = random.Random(seed + trial)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        aux_documents, documents = shuffled[:len(pool) // 2], shuffled[len(pool) // 2:]
        query_keywords = {scenario: rng.sample(keyword_space, NUM_QUERIES) for scenario in totals}

        start = time.perf_counter()
        injected = attacker.choose_injected_documents(keyword_space, MAX_INJECTED, aux_documents)[:MAX_INJECTED]
        choose_seconds = time.perf_counter() - start
        injected = [set(doc) & set(keyword_space) for doc in injected]
        injected_documents = documents + injected
        rng.shuffle(injected_documents)  # Alice stores her documents in a random order

        server = simulate("unkeyed", documents, keyword_space, query_keywords["unkeyed"])
        start = time.perf_counter()
        guesses = attacker.attack_unkeyed(server.index, server.queries, keyword_space)
        add(totals["unkeyed"], accuracy(guesses, query_keywords["unkeyed"]), time.perf_counter() - start)

        server = simulate("secure", documents, keyword_space, query_keywords["secure"])
        start = time.perf_counter()
        guesses = attacker.attack_secure(server.queries, keyword_space, aux_documents, len(documents))
        add(totals["secure"], accuracy(guesses, query_keywords["secure"]), time.perf_counter() - start)

        # The injection time counts choosing the documents too.
        server = simulate("secure", injected_documents, keyword_space, query_keywords["injection"])
        start = time.perf_counter()
        guesses = attacker.attack_injection(server.queries, keyword_space, aux_documents,
                                            len(injected_documents), injected)
        add(totals["injection"], accuracy(guesses, query_keywords["injection"]), choose_seconds + time.perf_counter() - start)

    return {scenario: (acc / num_trials, seconds / num_trials) for scenario, (acc, seconds) in totals.items()}


def add(total, acc, seconds):
    total[0] += acc
    total[1] += seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("module", nargs="?", default="a4", help="module with the attacks (default: a4)")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS,
                        help=f"number of trials, each a new random split with fresh queries (default: {NUM_TRIALS})")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first trial (default: 0)")
    args = parser.parse_args()
    total = 0.0
    results = run_all(importlib.import_module(args.module), seed=args.seed, num_trials=args.trials)
    for scenario, (acc, seconds) in results.items():
        score = points(scenario, acc, seconds)
        total += score
        limit = TIME_LIMITS.get(scenario)
        late = f"  over the {limit}s limit: -{TIME_PENALTY}" if limit and seconds > limit else ""
        print(f"{scenario:13s} {acc:6.1%}  {seconds:6.2f}s/trial  ({score:.1f}/{POINTS[scenario]} points){late}")
    print(f"{'total':13s} {'':6s}  {'':6s}{'':7s}  ({total:.1f}/{sum(POINTS.values())} points)")
