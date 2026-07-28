# FlashRL V2 release audit

Date: 2026-07-28
Candidate version: 2.0.0

This matrix maps GitHub issues #2 through #14 to direct repository or runtime
evidence. A deferred feature is complete only when no V2 command or result
claims to support it.

| Issue | Resolution | Direct evidence |
| ---: | --- | --- |
| #2 | Browser environment deferred; V2 is simulator-only | `DinoEnv` rejects every backend except `sim`; train/evaluate parsers expose only `sim`; `tests/test_env_contract.py`; README and REPORT scope statements |
| #3 | Termination and truncation preserved through TD targets and N-step replay | `tests/test_dqn_targets.py`, `tests/test_dqn_replay.py`, `tests/test_training_roundtrip.py` |
| #4 | Atomic, distinct best and last checkpoints with fixed-seed model selection | `tests/test_checkpoints.py`, `tests/test_training_roundtrip.py`, `tests/test_evaluation.py` |
| #5 | Versioned identity, checkpoint-derived evaluation, and run-first aggregation | `tests/test_identity.py`, `tests/test_artifacts.py`, `tests/test_evaluation.py`, `tests/test_aggregation.py`; final run and manifest hashes in `reports/v2_benchmark_runs.csv` |
| #6 | Bird altitude is observable; feature-specific bounds hold over long trajectories | `tests/test_env_contract.py`, `tests/test_baselines.py`; feature units and transforms in `docs/environment_api.md` |
| #7 | One versioned simulator action contract | action transitions in `docs/environment_api.md`; schema validation in `tests/test_checkpoints.py`; policy behavior in `tests/test_baselines.py` |
| #8 | Simulator defects and invalid observations abort instead of becoming transitions | `tests/test_training_safety.py`; explicit observation validation before action selection and replay insertion |
| #9 | Release-gating unit, determinism, optimizer, round-trip, smoke, hygiene, and CI checks | 89-test suite; `tests/test_dqn_components.py`; `tests/test_training_roundtrip.py`; `scripts/smoke_test.py`; `.github/workflows/ci.yml` |
| #10 | Installable package, split extras, package resources, license, and wheel qualification | `pyproject.toml`, `LICENSE`, `scripts/verify_wheel.py`, `tests/test_packaging.py`; CI wheel job |
| #11 | V1 scripts, binary drivers, generated models/logs, duplicate entry points, and contradictory reports removed | `tests/test_repo_hygiene.py`; historical rationale in `docs/history/v1.md`; one `flashrl` CLI |
| #12 | PPO explicitly deferred from V2 | no PPO module or CLI/result option; core dependencies exclude Stable-Baselines3; README, REPORT, and changelog agree |
| #13 | Required DQN matrix and final learned-policy evidence published | five-seed component matrix in `reports/v2_component_report.md`; five-seed final result in `reports/v2_benchmark_report.md`; 100 held-out episodes per policy; median and best GIFs under `reports/demo/` |
| #14 | Simulator-only, DQN-focused release definition satisfied | all rows above; release qualification commands below |

## Optional environment checker qualification

Gymnasium’s checker is part of the automated suite for every observation mode.
In addition, an isolated system-site-packages virtual environment installed
only Stable-Baselines3 2.9.0 and ran:

```python
from stable_baselines3.common.env_checker import check_env
from flashrl.envs import DinoEnv

for mode in ("state", "vision", "hybrid"):
    env = DinoEnv(obs_mode=mode, backend="sim", max_episode_steps=20, seed=17)
    check_env(env, warn=True)
    env.close()
```

Observed result: state PASS, vision PASS, hybrid PASS.

## Performance evidence

- final learned mean: 301.90;
- learned 95% interval across five training seeds: [270.61, 335.98];
- random on common final seeds: 126.18;
- rule on common final seeds: 284.17;
- learned improvement over random: 139.3%;
- exact final budget: 120,000 frames per trained policy;
- complete DQN component matrix: four algorithms, five train seeds each, 100
  new held-out episodes per trained policy;
- every published run row contains full checkpoint and manifest SHA-256 values.

The learned point estimate exceeds rule, but rule remains inside the learned
training-seed interval. The release therefore claims a reliable improvement
over random, not a reliable improvement over rule.

## Release qualification

Run from a clean detached worktree at the candidate commit:

```bash
python -m build
python -m twine check dist/*
python scripts/verify_wheel.py dist/*.whl
pytest -q
python scripts/smoke_test.py
flashrl doctor
git diff --check
git status --short
```

Live GitHub Actions, release asset URLs, and the merged/tagged commit are
verified after publication. The V2 release bundle contains raw CSV/JSONL,
configs, metrics, finalized manifests, and artifact hashes; the representative
median checkpoint is a separate checksummed release asset.
