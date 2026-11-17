"""Benchmark for the attacks on the secure scheme: compare candidates on identical trials.

    python bench.py "lam=1" "lam=2,like=dev"                                  # secure mode (Q2)
    python bench.py --mode injection "plan=none" "plan=probe"                  # injection mode (Q3)
    python bench.py --seeds holdout --profile small_aux --mode injection "plan=none"

The first candidate is the baseline. Trials use the same split, query draws and shuffle as
tests.py (seed s, same rng order), so numbers are comparable with `python tests.py`.

Reports per candidate: mean QRR +- SE, paired delta vs baseline +- SE (wins/losses), worst trial,
number of collapsed trials (<50%), seconds per trial, and QRR by frequency tercile of the true
keyword. Injection mode counts choose + attack time, like tests.py, and also reports how often the
candidate located the injected documents among Alice's shuffled ones (only if its attack returns
`(guesses, {"columns": [...]})`, one column per injected document, in injection order).

Profiles stress the settings the graders may use: small_aux / small_client keep a quarter of the
aux / Alice's documents, long_docs merges three documents into one (longer natural documents).
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
from tests import MAX_INJECTED, NUM_QUERIES, accuracy, simulate

SEEDS = {"dev": range(0, 30), "holdout": range(1000, 1030)}
PROFILES = {  # (aux fraction kept, client fraction kept, documents merged into one)
    "default": (1.0, 1.0, 1),
    "small_aux": (0.25, 1.0, 1),
    "small_client": (1.0, 0.25, 1),
    "long_docs": (1.0, 1.0, 3),
}
CACHE = ".bench_cache"


def split(seed, profile, pool):
    """Same split and query draws as tests.py. Returns (aux, documents, secure truth, injection truth, rng state)."""
    aux_frac, client_frac, merge = PROFILES[profile]
    if merge > 1:
        pool = [set().union(*pool[i:i + merge]) for i in range(0, len(pool) - merge + 1, merge)]
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    aux, documents = shuffled[:len(pool) // 2], shuffled[len(pool) // 2:]
    return aux, documents, rng


def secure_trial(seed, profile, pool, keyword_space):
    path = f"{CACHE}/{profile}-{seed}.pkl"
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    aux, documents, rng = split(seed, profile, pool)
    for _ in range(2):  # tests.py draws unkeyed, then secure
        truth = rng.sample(keyword_space, NUM_QUERIES)
    aux_frac, client_frac, _ = PROFILES[profile]
    aux, documents = aux[:int(len(aux) * aux_frac)], documents[:int(len(documents) * client_frac)]
    server = simulate("secure", documents, keyword_space, truth)
    trial = (server.queries, aux, len(documents), truth)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(trial, fh)
    return trial


def injection_trial(seed, profile, pool, keyword_space):
    aux, documents, rng = split(seed, profile, pool)
    for _ in range(3):  # unkeyed, secure, then injection
        truth = rng.sample(keyword_space, NUM_QUERIES)
    aux_frac, client_frac, _ = PROFILES[profile]
    aux, documents = aux[:int(len(aux) * aux_frac)], documents[:int(len(documents) * client_frac)]
    return aux, documents, truth, rng.getstate()


def parse(spec):
    opts = {}
    for part in filter(None, spec.split(",")):
        key, value = part.split("=")
        opts[key] = ast.literal_eval(value) if value[0].isdigit() or value in ("True", "False") else value
    return opts


def tally(buckets, guesses, truth, aux, keyword_space):
    freq = np.diag(_aux_statistics(keyword_space, aux))
    tercile = np.digitize(freq, np.quantile(freq, [1 / 3, 2 / 3]))
    position = {w: i for i, w in enumerate(keyword_space)}
    for guess, true in zip(guesses, truth):
        buckets[tercile[position[true]]] += (guess == true, 1)


def run_secure(opts, trials, keyword_space):
    accs, secs, buckets = [], [], np.zeros((3, 2))
    for queries, aux, n, truth in trials:
        start = time.perf_counter()
        guesses = lab.secure(queries, keyword_space, aux, n, **opts)
        secs.append(time.perf_counter() - start)
        accs.append(accuracy(guesses, truth))
        tally(buckets, guesses, truth, aux, keyword_space)
    return np.array(accs), np.mean(secs), buckets[:, 0] / buckets[:, 1], None


def run_injection(opts, trials, keyword_space):
    choose, attack = lab.injection(**opts)
    accs, secs, buckets, found = [], [], np.zeros((3, 2)), []
    for aux, documents, truth, state in trials:
        start = time.perf_counter()
        injected = choose(keyword_space, MAX_INJECTED, aux)[:MAX_INJECTED]
        seconds = time.perf_counter() - start
        injected = [set(doc) & set(keyword_space) for doc in injected]
        shuffled = documents + injected
        rng = random.Random()
        rng.setstate(state)
        rng.shuffle(shuffled)
        positions = [next(i for i, d in enumerate(shuffled) if d is doc) for doc in injected]

        server = simulate("secure", shuffled, keyword_space, truth)
        start = time.perf_counter()
        result = attack(server.queries, keyword_space, aux, len(shuffled), injected)
        secs.append(seconds + time.perf_counter() - start)
        guesses, diagnostics = result if isinstance(result, tuple) else (result, {})
        accs.append(accuracy(guesses, truth))
        tally(buckets, guesses, truth, aux, keyword_space)
        columns = diagnostics.get("columns")
        if columns is not None:
            found.append((set(columns) == set(positions), list(columns) == positions))
    columns_rate = np.mean(found, axis=0) if found else None
    return np.array(accs), np.mean(secs), buckets[:, 0] / buckets[:, 1], columns_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--mode", default="secure", choices=["secure", "injection"])
    parser.add_argument("--seeds", default="dev", choices=SEEDS)
    parser.add_argument("--profile", default="default", choices=PROFILES)
    args = parser.parse_args()

    keyword_space = load_keywords("data/keywords.txt")
    pool = load_documents("data/aux-documents.txt") + load_documents("data/client-documents.txt")
    make = secure_trial if args.mode == "secure" else injection_trial
    run = run_secure if args.mode == "secure" else run_injection
    trials = [make(s, args.profile, pool, keyword_space) for s in SEEDS[args.seeds]]

    print(f"mode={args.mode}  seeds={args.seeds} ({len(trials)} trials)  profile={args.profile}")
    print(f"{'candidate':34s} {'QRR':>13s} {'delta vs base':>22s} {'worst':>6s} {'<50%':>5s} {'s/trial':>8s}"
          f"   low/mid/high freq   cols set/order")
    base = None
    for spec in args.specs:
        accs, secs, terciles, columns = run(parse(spec), trials, keyword_space)
        if base is None:
            base = accs
        diff = accs - base
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        delta = "(baseline)" if accs is base else \
            f"{100 * diff.mean():+5.2f}+-{100 * se(diff):.2f} ({(diff > 0).sum()}W/{(diff < 0).sum()}L)"
        cols = "-" if columns is None else f"{100 * columns[0]:.0f}%/{100 * columns[1]:.0f}%"
        print(f"{spec:34s} {100 * accs.mean():5.1f}+-{100 * se(accs):4.2f}% {delta:>22s} "
              f"{100 * accs.min():5.1f}% {(accs < 0.5).sum():5d} {secs:8.2f}   "
              + "/".join(f"{100 * t:.0f}" for t in terciles).ljust(18) + f"  {cols}")


if __name__ == "__main__":
    main()
