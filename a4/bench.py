"""Benchmark for the secure-scheme attack: compare candidates on identical trials.

    python bench.py "lam=1" "lam=2" "lam=1,like=dev"              # dev seeds, default profile
    python bench.py --seeds holdout --profile small_aux "lam=1" "lam=2"

The first candidate is the baseline. Trials use the same split and queries as tests.py
(seed s, same rng order), so numbers are comparable with `python tests.py`.
Reports per candidate: mean QRR +- SE, paired delta vs baseline +- SE (wins/losses), worst trial,
seconds per trial, and QRR by frequency tercile of the true keyword.
"""

import argparse
import ast
import os
import pickle
import random
import time

import numpy as np

import lab
from a4 import _aux_statistics
from sse import load_documents, load_keywords
from tests import NUM_QUERIES, accuracy, simulate

SEEDS = {"dev": range(0, 30), "holdout": range(1000, 1030)}
PROFILES = {"default": (1.0, 1.0), "small_aux": (0.25, 1.0), "small_client": (1.0, 0.25)}  # (aux, client) fraction kept
CACHE = ".bench_cache"


def make_trial(seed, profile, pool, keyword_space):
    path = f"{CACHE}/{profile}-{seed}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    aux, documents = shuffled[:len(pool) // 2], shuffled[len(pool) // 2:]
    for _ in range(2):  # same draws as tests.py: unkeyed, then secure
        truth = rng.sample(keyword_space, NUM_QUERIES)
    aux_frac, client_frac = PROFILES[profile]
    aux, documents = aux[:int(len(aux) * aux_frac)], documents[:int(len(documents) * client_frac)]
    server = simulate("secure", documents, keyword_space, truth)
    trial = (server.queries, aux, len(documents), truth)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(trial, fh)
    return trial


def parse(spec):
    opts = {}
    for part in filter(None, spec.split(",")):
        key, value = part.split("=")
        opts[key] = ast.literal_eval(value) if value[0].isdigit() or value in ("True", "False") else value
    return opts


def evaluate(opts, trials, keyword_space):
    accs, secs, buckets = [], [], np.zeros((3, 2))
    for queries, aux, n, truth in trials:
        start = time.perf_counter()
        guesses = lab.secure(queries, keyword_space, aux, n, **opts)
        secs.append(time.perf_counter() - start)
        accs.append(accuracy(guesses, truth))
        freq = np.diag(_aux_statistics(keyword_space, aux))
        tercile = np.digitize(freq, np.quantile(freq, [1 / 3, 2 / 3]))
        position = {w: i for i, w in enumerate(keyword_space)}
        for guess, true in zip(guesses, truth):
            buckets[tercile[position[true]]] += (guess == true, 1)
    return np.array(accs), np.mean(secs), buckets[:, 0] / buckets[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--seeds", default="dev", choices=SEEDS)
    parser.add_argument("--profile", default="default", choices=PROFILES)
    args = parser.parse_args()

    keyword_space = load_keywords("data/keywords.txt")
    pool = load_documents("data/aux-documents.txt") + load_documents("data/client-documents.txt")
    trials = [make_trial(s, args.profile, pool, keyword_space) for s in SEEDS[args.seeds]]

    print(f"seeds={args.seeds} ({len(trials)} trials)  profile={args.profile}")
    print(f"{'candidate':34s} {'QRR':>13s} {'delta vs base':>22s} {'worst':>6s} {'<50%':>5s} {'s/trial':>8s}   low/mid/high freq")
    base = None
    for spec in args.specs:
        accs, secs, terciles = evaluate(parse(spec), trials, keyword_space)
        if base is None:
            base = accs
        diff = accs - base
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        delta = "(baseline)" if accs is base else \
            f"{100 * diff.mean():+5.2f}+-{100 * se(diff):.2f} ({(diff > 0).sum()}W/{(diff < 0).sum()}L)"
        print(f"{spec:34s} {100 * accs.mean():5.1f}+-{100 * se(accs):4.2f}% {delta:>22s} "
              f"{100 * accs.min():5.1f}% {(accs < 0.5).sum():5d} {secs:8.2f}   " + "/".join(f"{100 * t:.0f}" for t in terciles))


if __name__ == "__main__":
    main()
