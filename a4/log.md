# Assignment 4 – engineering log

Working notes: decisions, trade-offs, what was dropped and why, results, assumptions, open questions.
Newest state of each question is at the top of its section; the history is below it.
Q3 (file injection) has not been started.

## Status

| Question | Code | Report | Latest result |
|---|---|---|---|
| Q1 `attack_unkeyed` | done | done | 100% QRR (10/10) |
| Q2 `attack_secure` | done (v1, tuned) | done | 95.6% on `tests.py` (10 trials), 0.34 s/trial |
| Q3 injection | not started | not started | |

## Repo and tooling decisions

- One directory per assignment (`a4/`), data under `a4/data/`, report under `a4/report/`.
- `tests.py` is the provided harness. The only change is that its data paths now point into `data/`. `sse.py` is untouched, because graders use their own copy.
- **Only numpy, scipy, pycryptodome and the standard library** in `a4.py`, per the handout. `a4.py` imports nothing from `lab.py` or `bench.py`.
- `lab.py` holds parameterised variants of the Q2 attack for experiments; `bench.py` is the benchmark. Neither is submitted.
- Comments in code are kept light on purpose: a one-line docstring per function and a few inline comments on non-obvious lines.
- `.bench_cache/` (pickled simulated trials) is gitignored.

## Q1 – `attack_unkeyed`

**Approach.** The unkeyed scheme computes `t_w = h("token" || w)` with no secret. Build `{h(b"token"+w): w}` for all 200 keywords in the keyword space and look every query token up in it.

**Why this and nothing else.**
- The token is a deterministic function of the keyword alone and the keyword space is small and public, so enumerating it is trivial. The hash being one-way is irrelevant.
- The `index` argument and the returned document IDs are not needed. (The key `h("key"+w)[:16]` is also computable, so the whole index could be decrypted, but the token lookup already answers the question.)
- Exact answer, not a statistical guess: SHA-256 collisions among 200 keywords are not a practical concern.

**Result.** 100.0% QRR in every run, ~0 s.

**Assumptions / open items.**
- Every query token is assumed to be in the table (true for the provided `sse.py`). A token not found would raise `KeyError`. I left it that way because the scheme guarantees it; a fallback would only hide a bug.

## Q2 – `attack_secure`

### Final approach (what is in `a4.py`)

1. **Aux statistics.** From the auxiliary documents, build `C[w][v]` = fraction of aux documents containing both `w` and `v` (diagonal = frequency of `w`). Clipped to `[1e-4, 1-1e-4]` so no probability is 0 or 1.
2. **Observed statistics.** From the queries, build the `n x Q` document/query matrix `R`; `overlap = R^T R` gives, for each pair of queries, the number of documents they share (diagonal = result count).
3. **Misfit function.** For an observed count `k` out of `n` documents and an expected fraction `p`: the binomial deviance `k·log(k/(n·p)) + (n−k)·log((n−k)/(n·(1−p)))`. It is 0 when `k = n·p` and grows as they diverge.
4. **Initial assignment.** Cost of "query i is keyword w" = misfit of query i's count against `f[w]`. Solve the one-to-one assignment (Hungarian algorithm, `linear_sum_assignment`).
5. **Refinement (up to 30 rounds).** For each (query i, keyword w), add up over every other query j the misfit between `overlap[i][j]` and `C[w][guess(j)]`, using the current guess for j. New cost = count cost + `lam` × overlap cost (`lam = 1`). Re-solve the assignment. Stop early if it no longer changes.
6. **Self-overlap masking.** When computing the cost for (i, w), skip any j whose current guess is already `w`, because `C[w][w]` is the keyword's frequency, not a co-occurrence.

### Why these choices

- **One-to-one assignment, not per-query argmax.** The 100 queries are 100 *different* keywords, so a keyword used for one query is unavailable to the others. Solving it jointly (Hungarian) uses that constraint; greedy matching would not.
- **Counts alone are weak, overlaps carry the signal.** Frequency-only matching gives 11.6% QRR (dev). The overlap refinement lifts it to ~90%+.
- **Iterate on own guesses.** The overlap cost for query i depends on what the other queries are believed to be, so it is a fixed-point iteration. The first round is only as good as the frequency-only guess, which is why it takes ~20 rounds to settle.
- **Binomial deviance over Gaussian.** The Gaussian approximation `(k−np)²/(2np(1−p))` is poor for small counts/probabilities, which is exactly where many keyword pairs live. Deviance is the actual binomial log-likelihood (up to a constant), and it measured better.
- **Masking self-overlap.** Fixed a real inconsistency in the cost, but on its own it was unstable (a 23% trial); it helped only together with deviance.
- **Pure numpy.** The cost tensor is 100 × 200 × 100 ≈ 2M terms per round, so 30 rounds fit comfortably under the 3 s limit.

### Benchmark protocol (so future approaches are comparable)

- Same trials for every candidate; split and query draws replicate `tests.py` (seed `s`, same rng order).
- **dev** seeds 0–29 for choosing, **holdout** seeds 1000–1029 for confirming. Variants were chosen on dev; holdout was only run on a few finalists.
- **Stress profiles** (the graders use other, unseen datasets): `small_aux` keeps ¼ of the aux documents, `small_client` keeps ¼ of Alice's documents.
- Metrics: mean QRR ± standard error; paired Δ vs baseline ± SE with wins/losses per trial (a change counts only if |Δ| > ~2 SE); worst trial; number of collapsed trials (QRR < 50%); seconds per trial against the 3 s limit; QRR by frequency tercile of the true keyword.

### Results (QRR %, mean ± SE over 30 trials unless noted)

Dev seeds, default profile, first version = `rounds=10, lam=1`, Gaussian, no masking:

| change | QRR | paired Δ vs baseline | worst |
|---|---|---|---|
| baseline (rounds=10) | 88.0 ± 1.79 | – | 55 |
| `lam=0` (frequency only) | 11.6 ± 0.56 | −76.4 | 6 |
| `lam` 0.25 / 0.5 / 2 / 4 / 8 | 88.2 / 89.3 / 87.2 / 87.4 / 87.8 | all within ~1 SE | 41–58 |
| `rounds` 0 / 1 / 2 / 5 | 11.6 / 14.9 / 27.7 / 67.7 | much worse | 6–30 |
| `rounds` 20 or 50 | 90.6 ± 1.01 | +2.67 ± 1.48 (7W/0L) | 75 |

Dev seeds, new baseline `rounds=30`:

| variant | QRR | paired Δ | worst |
|---|---|---|---|
| Gaussian, no masking (baseline) | 90.6 ± 1.01 | – | 75 |
| deviance | 92.5 ± 0.82 | +1.87 ± 0.76 (19W/7L) | 80 |
| masking only | 91.4 ± 2.49 | +0.77 ± 2.52 (23W/3L) | 23 |
| **deviance + masking (shipped)** | **95.3 ± 0.44** | **+4.63 ± 0.80 (28W/1L)** | **89** |
| Gaussian `lam` 0.5 / 2 | 91.2 / 90.4 | within noise | 70 / 62 |
| deviance `lam` 0.5 / 2 (no masking) | 88.1 / 91.9 | −2.6 ± 3.1 / +1.3 ± 0.7 | 27 / 76 |

Holdout seeds 1000–1029:

| profile | rounds=10 Gaussian | rounds=30 Gaussian | **shipped** |
|---|---|---|---|
| default | 77.1 ± 5.12 | 79.8 ± 5.10 (4 collapses) | **91.0 ± 2.90** (1 collapse, worst 20) |
| small_aux | 38.2 | 48.7 | **64.0** |
| small_client | 44.4 | 54.1 | **71.5** |

Official harness after porting: `python tests.py --trials 10` → secure 95.6%, 0.34 s/trial (50.5/50 points).
Time of the shipped version in the benchmark: 0.32–0.38 s/trial (default profile), up to 0.67 s in `small_aux`. The limit is 3 s.

### What was dropped and why

| dropped | why |
|---|---|
| Gaussian misfit | worse than binomial deviance on every comparison |
| `rounds=10` | many trials had not converged; 20+ is identical to 50 |
| `lam` as a tuned knob | flat from 0.25 to 8 at the baseline; kept at 1. (Not re-swept on the final variant, see open questions.) |
| masking without deviance | unstable alone; only helps in combination |
| count-only matching | 11.6% QRR |

### Assumptions

- Alice's documents and the aux documents are random halves of the same pool, so keyword frequencies and co-occurrences are comparable between them (true in `tests.py`; the grading datasets are described as similar real collections).
- The queries are 100 distinct keywords chosen uniformly, so there is no prior favouring some keywords; the assignment is a pure one-to-one matching.
- Nothing Enron-specific is used: no hard-coded keywords or frequencies, and no files are read.

### Open questions / next experiments

- **Collapse failures.** On holdout, ~1 in 30 trials still scores well under 50%. The iteration probably settles in a bad fixed point. Candidate fixes: a better initial assignment (the frequency-only start is only ~12% accurate), damping the updates, or a few restarts and keeping the best total cost.
- **`lam` was not re-tuned with deviance + masking.**
- **Dev vs holdout gap** (95.3 vs 91.0): variants were selected on dev, so dev is optimistic. Use holdout as the headline estimate; a third seed range would be a cleaner test set.
- **Aux sampling noise** is ignored in the misfit (the aux fractions are treated as exact). Inflating the variance for it only matters relative to the count term, which `lam` partly covers.
- **Epsilon clipping (1e-4)** and zero-count keywords are untested edge cases; the grading datasets may have rarer keywords than Enron.
- **Unused information:** only counts and pairwise overlaps are used; higher-order co-occurrence (triples) and per-document identity are not.
- **Different dataset shapes** (more/fewer documents, different aux/client ratio) are only approximated by the stress profiles.

## Q3 – injection

Not started. Ideas recorded from planning (untested):
- 3 injected documents give 2³ = 8 membership codes, which splits the keyword space into ~25-word groups.
- Choose the groups from aux statistics so that easily confused keywords get different codes.
- The injected documents must be recognisable among Alice's shuffled documents (they match many queries; natural documents match few), and the order of the three must be resolved.
- Counts in the secure-attack machinery must account for the injected documents, and `num_documents` includes them.
