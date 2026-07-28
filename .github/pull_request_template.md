## Outcome

Describe the user-visible or research outcome.

Closes #

## Evidence

- [ ] A failing test was observed before the implementation.
- [ ] `ruff check flashrl scripts tests`
- [ ] `ruff format --check flashrl scripts tests`
- [ ] `pytest -q`
- [ ] `python scripts/smoke_test.py`
- [ ] `git diff --check`
- [ ] Built-wheel verification completed when packaging changed.

## Experiment integrity

- [ ] Not applicable.
- [ ] Exact frame budget and independent train seeds are documented.
- [ ] Selection and final evaluation seeds are disjoint.
- [ ] Raw results trace to manifest and checkpoint hashes.
- [ ] Negative results and limitations are reported.

## Compatibility

List any schema version change, migration, or deliberate V2 de-scope decision.

