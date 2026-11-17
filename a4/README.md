# Assignment 4 – Searchable Encryption attacks

Honest-but-curious server attacks against a searchable symmetric encryption scheme
(query recovery from the "unkeyed" scheme, the "secure" scheme, and file injection).

## Layout

```
a4/
├── a4.py              # YOUR code: the four attack functions (submitted)
├── sse.py             # the scheme (provided, do not modify)
├── tests.py           # scoring harness (provided; paths adjusted to data/)
├── data/              # keywords + Enron-derived document sets
├── report/a4.tex      # answer template -> compile to a4.pdf
├── assignment.pdf     # assignment handout
└── requirements.txt
```

## Setup and run

```bash
cd a4
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests.py --trials 3     # quick check
python tests.py                # 10 trials
```

Run from inside `a4/` (the harness reads `data/...` relative to the cwd).

## Report

```bash
cd a4/report && pdflatex a4.tex
```

Submit `a4.py` and `a4.pdf`.

## Status

- [ ] `attack_unkeyed`
- [ ] `attack_secure`
- [ ] `choose_injected_documents` / `attack_injection`
- [ ] report
