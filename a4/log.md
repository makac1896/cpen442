# Assignment 4 – engineering log

Working notes: decisions, trade-offs, what was dropped and why, results, assumptions, open questions.
Newest state of each question is at the top of its section; the history is below it.
All three questions are implemented and written up.

## Status

| Question | Code | Report | Latest result |
|---|---|---|---|
| Q1 `attack_unkeyed` | done | done | 100% QRR (10/10) |
| Q2 `attack_secure` | done (v1, tuned) | done | 95.6% on `tests.py` (10 trials), 0.32 s/trial |
| Q3 injection | done (v1, tuned) | done | 99.0% on `tests.py` (10 trials), 2.08 s/trial |

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

### Final approach (what is in `a4.py`)

**`choose_injected_documents`** (3 documents, i.e. a 3-bit code per keyword):
1. Sort the keywords by their frequency in the aux data.
2. Deal them out round-robin over the 8 codes (`rank mod 8`). Document `j` contains every keyword whose code has bit `j` set, so each document has half the keyword space (100 keywords) and each code group has ~25 keywords.

Why round-robin by frequency: keywords with similar frequency are the ones the secure attack confuses most, and this puts neighbours in frequency into different groups. It also makes an injected document match queries *independently of how common the keyword is*, which the column identification below exploits.

**`attack_injection`:**
1. **Find the injected columns** among the `n+3` shuffled columns. Two independent routes propose candidate sets, and the full fit cost decides (below).
2. **Decode:** each query's code is its pattern over the three columns, which restricts it to the ~25 keywords with that code.
3. **Match:** remove the three columns (so counts and overlaps are those of Alice's real documents, `n = num_documents − 3`) and run the same assignment as Q2 (`_match`) with the code restriction (forbidden pairs cost 1e8).

**Column identification in detail** (the hard part: the IDs are shuffled and we do not know which column is which document):
- *Hit-count prior.* Queries are a random subset of the keywords, so a document with `|S_j|` keywords matches about `|S_j|·Q/K` queries (hypergeometric spread). Columns outside ±4 sd are implausible.
- *Route A (agreement).* Run the plain Q2 attack ignoring the injection, then score each column by how often "matches the query" agrees with "the guessed keyword is in injected document j". Candidates: the best plausible column per document plus the best raw one.
- *Route B (no guess needed).* Among plausible columns take the 30 least correlated with the query result counts (injected documents are frequency-independent, natural ones are not). Score each triple by how unbalanced the 8 code groups are (expected sizes are known) plus the correlation. Candidates: the 3 best triples, in all 6 orders.
- *Decision.* Run the code-restricted matching for every candidate (5 rounds) and keep the one with the lowest total cost. This also fixes the order of the three documents. It is a very strong test: in the one case I inspected the true columns cost 10,941 vs 48,352 for a wrong set.

### Why this and not something simpler (what was tried and dropped)

All on 30 trials; "cols" = % of trials where the injected columns were found (set / right order).

| version | dev | holdout | long_docs | small_aux | small_client | verdict |
|---|---|---|---|---|---|---|
| v1: take the 3 columns with most hits, order by fit cost | 98.8 (100%) | 96.5 (97%) | **7.2 (0%)** | 83.2 (87%) | 93.8 (100%) | dropped: natural documents reach 40–48 hits (more than my estimate of ≤35 from document sizes, because the tail across 12k documents is long), overlapping the injected ones' ~50; hopeless when natural documents are long |
| v2: agreement with the plain attack's guesses only | 97.0 (97%) | | | | | dropped: one trial where the plain attack was only 37% accurate. More passes (2, 3) did not help (fixed point); trying the best 2–4 by agreement did not help either (true column ranked 18th) |
| v2b: agreement + hit-count prior (shortlist) | 98.8 (100%) | 99.0 (100%) | 92.2 (100%) | 92.1 (100%) | **81.6 (87%)** | fails when the plain attack collapses (~10% accurate with only ~3k natural documents) |
| v3: independence + code-group balance only | 95.9 (97%) → 98.8 (100%) with the correlation term | 97.2 (97%) | **62.4 (53%)** | 92.1 (100%) | 93.8 (100%) | fails when natural documents are long (true columns not in the 30-column pool; pool 50 and 5 sets only reached 70.0) |
| **shipped: A and B together, decided by fit cost** | **98.8 (100%)** | **99.0 (100%)** | **92.2 (100%)** | **92.1 (100%)** | **93.8 (100%)** | the two routes fail in different places and the fit cost is a reliable judge |

Reference (no injection, plain Q2 attack): 94.9 / 95.5 / 66.2 / 71.0 / 74.1.

Other measurements:
- `tau` (independence scale) 0.05 / 0.1 / 0.2: identical on dev. Shortlist 1 / 2 / 3: identical. `top_sets=1` is worse (96.8, one miss) so the fit check across 3 sets earns its cost.
- Zero collapsed trials (<50%) in every setting for the shipped version. Worst single trial 82–97% depending on the setting.
- Time: 2.1–2.6 s/trial (limit 10 s). Most of it is ~25 restricted solves of 5 rounds each.
- Official harness: `python tests.py --trials 10` → injection 99.0%, 2.08 s (44.0/40 points).

### Assumptions
- The 100 queries are a uniformly random subset of the keywords (stated in the handout). Used for the hit-count prior; if no column is plausible it falls back to considering all columns, and Route A always also tries the best raw column.
- Injected documents are exactly the ones returned by `choose_injected_documents` (intersected with the keyword space, as `tests.py` does). `attack_injection` re-derives each keyword's code from the `injected_documents` argument and does not rely on anything stored from part 1.
- At most 3 documents are used (`MAX_PLANTED`), so at most 3! = 6 orders per set.
- Nothing is read from files and nothing is Enron-specific.

### Caveats and open questions
- **Holdout is not clean.** I looked at holdout numbers while choosing between v1/v2b/v3, so the 99.0% there is less independent than it looks. A third seed range would be a fair final test.
- **Only the Enron data**, plus stress profiles. `long_docs` (three documents merged) is my stand-in for news-like data, not a real dataset.
- **Remaining errors** are mostly keywords that look identical to the aux data inside the same code group; a smarter keyword-to-code assignment (using co-occurrence, not only frequency) might help.
- `lam` and `rounds` were tuned for Q2 and not re-tuned for the code-restricted problem.
- **Time:** 2.1–2.6 s of 10 s. If the grading machine is much slower this is the thing to watch; `top_sets`, `PRE_ROUNDS` and the candidate count can be reduced.
- The probe candidate `plan=probe` and the `plan=none` baseline in `lab.py` are for the harness only.

### Benchmark

`python bench.py --mode injection "plan=none" "plan=probe"` (add `--seeds holdout`, `--profile ...`).

- Same trials as `tests.py`: same split, same three query draws, same shuffle after injection. Verified: running `tests.run_all` with a no-injection shim over seeds 0-29 gives 94.9%, identical to the benchmark's `plan=none`.
- Time counts `choose_injected_documents` + `attack_injection`, as in `tests.py` (limit 10 s).
- Extra diagnostics: if an attack returns `(guesses, {"columns": [...]})` (one column per injected document, in injection order), the benchmark reports how often it found the injected documents (as a set) and in the right order. The benchmark knows the true positions because it tracks the injected set objects through the shuffle.
- New profile `long_docs`: three documents merged into one, to test what happens when natural documents are long (news-like data) and the injected documents are harder to tell apart from them.
- Candidates live in `lab.injection(plan=...)`. `plan=none` (no injection, plain secure attack) is the baseline. `plan=probe` is a harness self-test (one document with all keywords, found by its column count): it located the column in 100% of trials on every profile.

Reference QRR for `plan=none` (dev seeds, 30 trials), the numbers an injection attack has to beat:

| profile | QRR | collapsed (<50%) | s/trial |
|---|---|---|---|
| default | 94.9 ± 0.48 | 0 | 0.43 |
| long_docs | 66.2 ± 3.98 | 5 | 0.60 |
| small_aux | 71.0 ± 1.50 | 0 | 0.60 |
| small_client | 74.1 ± 2.74 | 2 | 0.55 |

Natural document sizes on Enron (keywords per document, out of 200): mean 13.6, median 10, p95 41, p99 58, max 70. With about half the keywords queried, a natural document appears in about 7 queries on average (at most ~35), a planted document with ~100 keywords in ~50.
