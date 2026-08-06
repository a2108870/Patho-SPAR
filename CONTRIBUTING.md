# Contributing to Patho-SPAR

Contributions that improve correctness, reproducibility, documentation, or
platform compatibility are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

Run the complete local validation suite:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
python -m build
```

Pull requests should:

- describe the motivation and behavioral impact;
- include tests for code changes;
- avoid committing datasets, checkpoints, generated outputs, or credentials;
- preserve the public attack API unless a breaking change is justified;
- keep paper-specific claims consistent with the released configuration.

## Reporting results

When reporting reproduction results, include the package version, commit SHA,
dataset split, checkpoint digest, preprocessing, random seed, attack
configuration, and software environment.

## Double-blind review

The repository is anonymized while the manuscript is under review. Please do
not add identifying author information to issues, pull requests, commits, or
documentation during this period.
