# Workflow Test Suite

This directory contains unit-style checks that keep `workflow.yaml` consistent as it evolves.

## What these tests cover
- YAML loads correctly without YAML 1.1 boolean coercion issues (e.g., `on`).
- Required top-level sections and input groups exist.
- Job graph validity (missing `needs`, dependency cycles).
- Template references to `inputs.*` and `needs.*` resolve to known definitions.

## Running the tests
1. Install test dependencies:
   - `python -m pip install -r requirements-dev.txt`
   - or `uv pip install -r requirements-dev.txt`
2. Run the unit checks:
   - `python -m pytest -q tests`
   - or `uv run -m pytest -q tests`
   - or `./scripts/run_tests.sh` to run unit checks plus the YAML smoke test.

## Optional: existing local smoke tests
There is a broader local workflow tester in `scripts/test_local.sh` that validates YAML
syntax and other components:
- `./scripts/test_local.sh yaml`
