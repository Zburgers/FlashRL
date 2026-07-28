# Contributing to FlashRL

FlashRL welcomes correctness fixes, reproducible experiments, simulator
extensions, analysis improvements, and focused documentation.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/smoke_test.py
pytest -q
```

Before opening a pull request:

```bash
ruff check flashrl scripts tests
ruff format --check flashrl scripts tests
pytest -q
python scripts/smoke_test.py
python -m build
python scripts/verify_wheel.py dist/*.whl
git diff --check
```

## Code changes

- Add a failing test that expresses the intended behavior before production
  code.
- Preserve `terminated` and `truncated` separately. Only true termination masks
  TD bootstrap.
- Version any observation, action, reward, result, manifest, or checkpoint
  contract change.
- Keep deterministic components on private seeded RNGs.
- Fail fast on invalid simulator state; never convert implementation errors
  into plausible training transitions.
- Keep the core install free of research-only dependencies.

## Experiment changes

An experiment contribution must include:

- a committed configuration under `experiments/`;
- an exact environment-frame budget;
- at least three training seeds for pilot evidence and five for release claims;
- checkpoint-selection seeds separate from evaluation seeds;
- per-run manifests and checkpoint hashes;
- episode-first, then independent-run statistics;
- negative results and discarded hypotheses;
- wall-clock and sample-cost reporting;
- an honest random and rule-policy comparison.

Do not tune on final evaluation seeds or select a method from one maximum score.
Generated `runs/` stay out of Git. Publish validated checkpoints and complete run
bundles as release assets with SHA-256 checksums.

## Pull requests and Git

Create a focused branch from current `master`, use small imperative commits, and
avoid mixing unrelated cleanup with behavioral changes. Fill out the pull
request evidence checklist and link the issue being resolved. CI must pass on
all supported Python versions and on the built wheel.

Large binary checkpoints do not belong in the normal source tree. A compact
visual replay may be committed under `reports/demo/` when it directly supports
a published result.

## Scope

V2 is a deterministic simulator benchmark. Real-browser control, PPO, or a new
game can be proposed, but it must satisfy the same environment, artifact,
evaluation, and test contracts before it is advertised as supported.

