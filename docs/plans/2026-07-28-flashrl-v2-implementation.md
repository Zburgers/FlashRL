# FlashRL V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and release a correct, reproducible, research-oriented simulator-only RL benchmark with a trained multi-seed DQN result and a live web demo.

**Architecture:** Keep the existing flat `flashrl` package, split environment and artifact identities into small versioned modules, and make the checkpoint manifest the source of truth for evaluation. Expose one argparse-based `flashrl` CLI, retain the deterministic simulator as the sole V2 backend, and provide a local read-only web demo driven by simulator telemetry.

**Tech Stack:** Python 3.10+, Gymnasium, NumPy, PyTorch, Pillow, setuptools, pytest, vanilla HTML/CSS/JavaScript, GitHub Actions.

---

## Working Rules

- Work only on `feat/flashrl-v2-release` or a child branch, never `master`.
- Preserve `reports/benchmark_summary2.md` and `results/dqn_state.*` until the
  cleanup task explicitly classifies them as invalid schema-v1 artifacts.
- Follow red-green-refactor for every behavior change.
- Keep generated training output under ignored `runs/` or a temporary directory.
- Commit after each task with only the files named by that task.
- Do not run the final compute benchmark until correctness, schema, and CI tests
  pass.

### Task 1: Version and Correct the Environment Contract

**Files:**
- Create: `flashrl/schemas.py`
- Modify: `flashrl/envs/dino_env.py`
- Modify: `flashrl/agents/baselines.py`
- Modify: `docs/environment_api.md`
- Test: `tests/test_env_contract.py`
- Test: `tests/test_baselines.py`

**Step 1: Write failing schema and altitude tests**

Add tests proving:

```python
def test_state_distinguishes_bird_altitude():
    low = DinoEnv(obs_mode="state", seed=1)
    high = DinoEnv(obs_mode="state", seed=1)
    low.reset(seed=1)
    high.reset(seed=1)
    low.obstacles[0] = Obstacle(100, 34, 24, 2, y=36)
    high.obstacles[0] = Obstacle(100, 34, 24, 2, y=68)
    assert not np.array_equal(low._state_vector(), high._state_vector())


def test_rule_policy_uses_bird_altitude():
    # Construct observations differing only in normalized altitude.
    assert policy.act(low_bird_state) == DUCK_PRESS
    assert policy.act(high_bird_state) == NOOP
```

Also add complete-trajectory determinism and a long-horizon test that replaces
obstacles with a collision-free sequence and asserts
`env.observation_space.contains(obs)` at every step.

**Step 2: Run tests and verify expected failures**

Run:

```bash
pytest tests/test_env_contract.py tests/test_baselines.py -q
```

Expected: FAIL because altitude is absent, bounds are generic, and schema
constants do not exist.

**Step 3: Add explicit schemas and feature bounds**

Define immutable version constants and action IDs in `flashrl/schemas.py`:

```python
ENVIRONMENT_ID = "FlashRL-DinoSim-v2"
ENVIRONMENT_VERSION = 2
OBSERVATION_SCHEMA_VERSION = 2
ACTION_SCHEMA_VERSION = 2
REWARD_SCHEMA_VERSION = 2
SIMULATOR_VERSION = 2

NOOP = 0
JUMP = 1
DUCK = 2
RELEASE = 3
```

Add nearest-obstacle bottom and top features, remove unbounded raw score from
the policy observation, and define feature-specific `low`/`high` arrays. Update
the rule policy to use obstacle altitude rather than obstacle height.

**Step 4: Make simulator failures fail fast**

Remove the broad `try/except` around simulator stepping. Reserve typed
truncation reasons for future external backends; simulator invariant errors
must propagate and cannot generate a replay transition.

**Step 5: Run tests**

Run:

```bash
pytest tests/test_env_contract.py tests/test_baselines.py -q
python scripts/smoke_test.py
```

Expected: PASS with deterministic in-space observations.

**Step 6: Commit**

```bash
git add flashrl/schemas.py flashrl/envs/dino_env.py \
  flashrl/agents/baselines.py docs/environment_api.md \
  tests/test_env_contract.py tests/test_baselines.py
git commit -m "fix: version and correct simulator contracts"
```

### Task 2: Correct Replay, N-Step, and DQN Targets

**Files:**
- Modify: `flashrl/agents/dqn/replay.py`
- Modify: `flashrl/agents/dqn/train.py`
- Test: `tests/test_dqn_replay.py`
- Test: `tests/test_dqn_targets.py`

**Step 1: Write failing transition tests**

Specify the new transition shape:

```python
Transition(
    obs=obs,
    action=0,
    reward=1.0,
    next_obs=next_obs,
    terminated=False,
    truncated=True,
    discount=0.99,
)
```

Add tests proving terminals do not bootstrap, truncations do bootstrap, a
short flushed N-step transition uses `gamma ** actual_steps`, and an N-step
terminal preserves its ending signals.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_dqn_replay.py tests/test_dqn_targets.py -q
```

Expected: FAIL because `Transition` contains only `done` and optimize always
uses `gamma ** cfg.n_step`.

**Step 3: Implement explicit endings and discounts**

Replace `done` with `terminated`, `truncated`, and `discount`. Keep episode loop
control as `terminated or truncated`; compute targets as:

```python
target = rewards + discounts * next_q * (~terminated).float()
```

Make `NStepBuffer` calculate the effective discount for every emitted
transition and flush safely at both ending types.

**Step 4: Make replay deterministic**

Give prioritized replay its own seeded `np.random.Generator` instead of global
`np.random.choice`. Test identical samples from buffers initialized with the
same seed.

**Step 5: Verify green**

Run:

```bash
pytest tests/test_dqn_replay.py tests/test_dqn_targets.py -q
pytest -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add flashrl/agents/dqn/replay.py flashrl/agents/dqn/train.py \
  tests/test_dqn_replay.py tests/test_dqn_targets.py
git commit -m "fix: preserve DQN bootstrap semantics"
```

### Task 3: Add Versioned Run Manifests

**Files:**
- Create: `flashrl/artifacts.py`
- Create: `flashrl/identity.py`
- Modify: `flashrl/agents/dqn/train.py`
- Test: `tests/test_artifacts.py`
- Test: `tests/test_identity.py`

**Step 1: Write failing identity tests**

Test that canonical JSON ordering produces the same SHA-256 identity, a changed
learning rate changes identity, files receive stable hashes, and a manifest
round-trips without losing schema versions.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_artifacts.py tests/test_identity.py -q
```

Expected: ERROR because the modules do not exist.

**Step 3: Implement canonical identity and manifests**

Add:

```python
def canonical_json(value: Any) -> str: ...
def sha256_bytes(value: bytes) -> str: ...
def sha256_file(path: Path) -> str: ...
def hyperparameter_hash(config: Mapping[str, Any]) -> str: ...
def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None: ...
```

Create a versioned manifest dataclass containing algorithm ID, hyperparameter
hash, training seed, Git state, environment/schema versions, frame count,
timing, and artifact hashes.

**Step 4: Integrate training identity**

Derive algorithm IDs such as `dqn`, `double_dqn`,
`dueling_double_dqn`, and
`dueling_double_dqn_per_n3` from enabled components. Write
`manifest.json` before training and finalize it after artifacts exist.

**Step 5: Verify green and commit**

Run:

```bash
pytest tests/test_artifacts.py tests/test_identity.py -q
```

Then:

```bash
git add flashrl/artifacts.py flashrl/identity.py \
  flashrl/agents/dqn/train.py tests/test_artifacts.py tests/test_identity.py
git commit -m "feat: add versioned experiment manifests"
```

### Task 4: Implement Safe Best, Last, and Resume Checkpoints

**Files:**
- Modify: `flashrl/artifacts.py`
- Modify: `flashrl/agents/dqn/train.py`
- Modify: `flashrl/benchmark/train.py`
- Test: `tests/test_checkpoints.py`
- Test: `tests/test_training_roundtrip.py`

**Step 1: Write failing checkpoint tests**

Test separate `best.pt` and `last.pt`, atomic replacement, format rejection,
environment-schema rejection, and a worse final policy not replacing the best
checkpoint. Add a tiny train/resume test that proves frame counts continue and
optimizer/RNG state are restored.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_checkpoints.py tests/test_training_roundtrip.py -q
```

Expected: FAIL because training writes one non-atomic `checkpoint.pt`.

**Step 3: Implement atomic checkpoint IO**

Write checkpoints to a same-directory temporary path, `fsync`, then
`os.replace`. Store format version, role, selected frame, schema versions,
manifest ID, model/optimizer state, Python/NumPy/Torch RNG state, and selection
protocol.

**Step 4: Add deterministic held-out selection**

Add configuration fields for selection episodes, interval, and seed base.
Evaluate without exploration on fixed held-out seeds. Save `best.pt` only when
the aggregate selection mean improves; always save `last.pt` at completion.

**Step 5: Add `--resume`**

Validate configuration compatibility, restore state, append metrics, and
continue from the stored episode/frame. Never silently change experiment
identity on resume.

**Step 6: Verify and commit**

Run:

```bash
pytest tests/test_checkpoints.py tests/test_training_roundtrip.py -q
```

Then:

```bash
git add flashrl/artifacts.py flashrl/agents/dqn/train.py \
  flashrl/benchmark/train.py tests/test_checkpoints.py \
  tests/test_training_roundtrip.py
git commit -m "feat: add reproducible checkpoint selection and resume"
```

### Task 5: Replace the Result and Aggregation Contract

**Files:**
- Create: `flashrl/results.py`
- Modify: `flashrl/benchmark/evaluate.py`
- Modify: `flashrl/benchmark/aggregate.py`
- Test: `tests/test_evaluation.py`
- Test: `tests/test_aggregation.py`

**Step 1: Write failing evaluation tests**

Test that evaluating `best.pt` automatically carries training run ID,
algorithm ID, train frames, schema versions, hyperparameter hash, checkpoint
role, and checkpoint SHA-256. Test that per-episode timing is not cumulative.

**Step 2: Write failing aggregation tests**

Test rejection when environment, reward, observation, action, hyperparameter,
or result schemas differ. Test episode-to-run statistics followed by
run-to-experiment statistics, with separate run and episode counts and a
bootstrap 95% confidence interval across training seeds.

**Step 3: Verify red**

Run:

```bash
pytest tests/test_evaluation.py tests/test_aggregation.py -q
```

Expected: FAIL under the schema-v1 result implementation.

**Step 4: Implement schema-v2 results**

Use one authoritative field list in `flashrl/results.py`. Give every episode
its actual reset seed, explicit training/evaluation commits, per-episode wall
time, termination reason, and manifest/checkpoint identity.

Make checkpoint evaluation infer observation/action compatibility and reject
conflicting overrides.

**Step 5: Implement two-level aggregation**

Validate identity fields before grouping. Write raw-run and experiment-summary
CSV plus a Markdown report. Bootstrap across independent training runs, not
pooled episodes.

**Step 6: Verify and commit**

Run:

```bash
pytest tests/test_evaluation.py tests/test_aggregation.py -q
```

Then:

```bash
git add flashrl/results.py flashrl/benchmark/evaluate.py \
  flashrl/benchmark/aggregate.py tests/test_evaluation.py \
  tests/test_aggregation.py
git commit -m "feat: make benchmark results traceable and safe"
```

### Task 6: Build the Installable Canonical CLI

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `flashrl/cli.py`
- Create: `flashrl/doctor.py`
- Modify: `flashrl/__init__.py`
- Modify: `flashrl/benchmark/train.py`
- Modify: `flashrl/benchmark/evaluate.py`
- Modify: `flashrl/benchmark/aggregate.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_doctor.py`

**Step 1: Write failing CLI tests**

Invoke `flashrl.cli.main()` with `--version`, `train --help`,
`evaluate --help`, `aggregate --help`, `demo --help`, and `doctor`.
Assert stable exit codes and clear diagnostics.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_cli.py tests/test_doctor.py -q
```

Expected: FAIL because the modules and unified parser do not exist.

**Step 3: Implement one parser**

Register:

```toml
[project.scripts]
flashrl = "flashrl.cli:main"
```

Create subparsers that call command functions without rebuilding parsers or
calling subprocesses. Keep only NumPy, Gymnasium, Pillow, and PyTorch in core
dependencies. Put test/build tools in `dev`, Playwright in `browser`, and
Stable-Baselines3 in `ppo` extras.

**Step 4: Add doctor checks**

Report Python/package versions, supported device, writable artifact directory,
Git identity when available, and optional-extra availability.

**Step 5: Verify and commit**

Run:

```bash
pytest tests/test_cli.py tests/test_doctor.py -q
python -m pip install -e '.[dev]'
flashrl --help
```

Then:

```bash
git add pyproject.toml LICENSE flashrl/cli.py flashrl/doctor.py \
  flashrl/__init__.py flashrl/benchmark tests/test_cli.py \
  tests/test_doctor.py
git commit -m "feat: ship installable FlashRL CLI"
```

### Task 7: Clean the Supported Repository Surface

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `REPORT.md`
- Modify: `requirements.txt`
- Delete: `config.py`
- Delete: `utils.py`
- Delete: `menu.py`
- Delete: `test_script.py`
- Delete: `dino_env.py`
- Delete: `dqn_train.py`
- Delete: `dqn_eval.py`
- Delete: `chromedriver-win64/`
- Delete: committed `__pycache__/`
- Delete: `data/models/`
- Delete: generated logs and debug artifacts
- Delete or archive: schema-v1 `results/*_smoke.*`
- Test: `tests/test_repo_hygiene.py`

**Step 1: Write a failing hygiene test**

Reject committed Python caches, model binaries, ChromeDriver, logs, generated
runs, legacy root scripts, and schema-v1 supported results. Verify one canonical
CLI and no supported import from root `utils.py` or `config.py`.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_repo_hygiene.py -q
```

Expected: FAIL listing current V1 artifacts.

**Step 3: Preserve only useful history**

Move concise historical context to `docs/history/v1.md`; do not move generated
binaries elsewhere in Git. Mark the pre-existing untracked schema-v1 benchmark
files invalid in a local note, then remove or replace them only after their
origin and lack of reproducibility are recorded.

**Step 4: Remove legacy artifacts**

Use exact `git rm` targets. Extend `.gitignore` for build output, results,
recordings, and checkpoints while retaining fixture allow-lists.

**Step 5: Verify and commit**

Run:

```bash
pytest tests/test_repo_hygiene.py -q
git status --short
```

Then:

```bash
git add -A
git commit -m "refactor: remove unsupported V1 repository surface"
```

### Task 8: Add Release-Gating CI and Wheel Tests

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `scripts/verify_wheel.py`
- Modify: `pyproject.toml`
- Test: `tests/test_packaging.py`

**Step 1: Write failing package metadata tests**

Validate Python version classifiers, extras, entrypoint, license, package data,
and CI triggers for pull requests and `master`.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_packaging.py -q
```

Expected: FAIL until all release metadata and jobs exist.

**Step 3: Implement CI**

Add least-privilege jobs for:

- core tests on Python 3.10, 3.11, 3.12, and 3.13;
- lint/format checks;
- build plus `twine check`;
- installation of the wheel into a clean virtual environment;
- CLI/smoke execution outside the source checkout;
- repository hygiene.

Pin official actions by stable major version and enable pip caching.

**Step 4: Verify locally**

Run:

```bash
python -m build
python -m twine check dist/*
python scripts/verify_wheel.py dist/*.whl
pytest -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add .github/workflows/ci.yml scripts/verify_wheel.py \
  pyproject.toml tests/test_packaging.py
git commit -m "ci: gate FlashRL releases on wheel verification"
```

### Task 9: Build the Real-Time Learned-Policy Demo

**Required skill:** Read and apply `design-taste-frontend` before implementing
the interface.

**Files:**
- Create: `flashrl/demo/__init__.py`
- Create: `flashrl/demo/server.py`
- Create: `flashrl/demo/static/index.html`
- Create: `flashrl/demo/static/app.js`
- Create: `flashrl/demo/static/styles.css`
- Modify: `flashrl/cli.py`
- Modify: `pyproject.toml`
- Modify: `flashrl/agents/dqn/train.py`
- Test: `tests/test_demo.py`

**Step 1: Write failing server tests**

Start the server on an ephemeral loopback port and test:

- `/api/status` returns version and policy identity;
- `/api/frame` returns bounded simulator telemetry and Q-values;
- reset accepts only integer seeds in range;
- pause/speed controls validate input;
- arbitrary checkpoint paths cannot be submitted over HTTP;
- static assets load from an installed package.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_demo.py -q
```

Expected: ERROR because the demo package does not exist.

**Step 3: Implement the simulator session**

Run one environment on a controlled worker loop. Return a JSON-safe snapshot
containing Dino and obstacle geometry, score, action, reward terms, Q-values,
policy identity, and ending reason. Keep checkpoint selection local to the CLI.

**Step 4: Implement the web interface**

Build a responsive custom Canvas visualization with a restrained research-console
visual language. Include real-time game rendering, score/speed/action cards,
Q-value bars, rolling charts, seed and speed controls, pause/reset, keyboard
policy override, and deterministic showcase-seed selection.

**Step 5: Add recording**

Expose a local CLI option to save deterministic demo frames and assemble a GIF
when Pillow is installed. Record manifest identity and seed beside the media.

**Step 6: Verify and commit**

Run:

```bash
pytest tests/test_demo.py -q
flashrl demo --policy rule --no-open --port 0
```

Inspect the live page in a browser and capture a representative screenshot.

Then:

```bash
git add flashrl/demo flashrl/cli.py flashrl/agents/dqn/train.py \
  pyproject.toml tests/test_demo.py
git commit -m "feat: add live learned-policy simulator demo"
```

### Task 10: Add Research Experiment Configuration and Analysis

**Files:**
- Create: `experiments/README.md`
- Create: `experiments/pilot.yaml`
- Create: `experiments/v2_benchmark.yaml`
- Create: `flashrl/experiments.py`
- Create: `flashrl/analysis.py`
- Modify: `flashrl/cli.py`
- Test: `tests/test_experiments.py`
- Test: `tests/test_analysis.py`

**Step 1: Write failing experiment tests**

Test deterministic matrix expansion for vanilla, Double, Dueling Double, PER,
N-step, and combined candidates across seeds. Test duplicate identity
rejection, resumable job state, and dry-run command output.

**Step 2: Write failing analysis tests**

Use synthetic data to test learning-curve interpolation by environment frames,
area under the learning curve, bootstrap confidence intervals, effect size,
failure taxonomy, and representative median/best seed selection.

**Step 3: Verify red**

Run:

```bash
pytest tests/test_experiments.py tests/test_analysis.py -q
```

Expected: ERROR because experiment modules do not exist.

**Step 4: Implement matrix runner**

Use a small documented configuration format parsed without a mandatory YAML
dependency; JSON-compatible YAML files may be loaded with the standard `json`
module. Support `--dry-run`, `--resume`, bounded local workers, and one manifest
per job.

**Step 5: Implement research analysis**

Generate CSV, Markdown, and SVG plots without requiring a hosted service. Keep
all reported values traceable to run manifests.

**Step 6: Verify and commit**

Run:

```bash
pytest tests/test_experiments.py tests/test_analysis.py -q
flashrl experiment experiments/pilot.yaml --dry-run
```

Then:

```bash
git add experiments flashrl/experiments.py flashrl/analysis.py \
  flashrl/cli.py tests/test_experiments.py tests/test_analysis.py
git commit -m "feat: add reproducible RL experiment runner"
```

### Task 11: Run Pilot RL Experiments and Tune Performance

**Files:**
- Modify: `experiments/pilot.yaml`
- Create: `reports/v2_pilot_report.md`
- Generate ignored: `runs/v2-pilot/`

**Step 1: Establish pilot protocol**

Use at least three training seeds per candidate, a common held-out seed set,
and a bounded frame budget. Include:

- vanilla DQN;
- Double DQN;
- Dueling Double DQN;
- Dueling Double with PER;
- Dueling Double with three-step returns;
- Dueling Double with PER and three-step returns.

**Step 2: Run the pilot**

Run:

```bash
flashrl experiment experiments/pilot.yaml --resume
```

Expected: every job has a finalized manifest, `last.pt`, and selected `best.pt`.

**Step 3: Analyze**

Run:

```bash
flashrl analyze runs/v2-pilot --out reports/v2_pilot_report.md
```

Compare held-out mean, confidence interval, AUC/sample efficiency, failure
types, and wall-clock cost. Do not select a configuration from one maximum
score.

**Step 4: Iterate scientifically**

Change one factor at a time for promising candidates: learning rate, epsilon
schedule, target cadence, warmup, N-step horizon, or reward scaling. Record
discarded hypotheses and evidence in the pilot report.

**Step 5: Lock the final protocol and commit**

Update `experiments/v2_benchmark.yaml` with the evidence-selected recommended
configuration and fixed budget.

```bash
git add experiments/pilot.yaml experiments/v2_benchmark.yaml \
  reports/v2_pilot_report.md
git commit -m "research: select V2 DQN protocol from pilot ablations"
```

### Task 12: Run and Publish the Final Multi-Seed Benchmark

**Files:**
- Create: `reports/v2_benchmark_report.md`
- Create: `reports/v2_benchmark_runs.csv`
- Create: `reports/v2_benchmark_summary.csv`
- Create: `reports/figures/learning_curves.svg`
- Create: `reports/figures/failure_taxonomy.svg`
- Create: `reports/figures/score_distributions.svg`
- Create: `reports/demo/README.md`
- Create: representative recording under `reports/demo/`
- Generate ignored: `runs/v2-benchmark/`

**Step 1: Verify prerequisite gates**

Run:

```bash
pytest -q
python -m build
python scripts/verify_wheel.py dist/*.whl
git diff --check
```

Expected: PASS before compute starts.

**Step 2: Run final protocol**

Run:

```bash
flashrl experiment experiments/v2_benchmark.yaml --resume
```

Use at least five independent training seeds and at least 100 held-out
evaluation episodes per trained policy. Evaluate random and rule policies on
the exact same episode seed set.

**Step 3: Generate report and figures**

Run:

```bash
flashrl analyze runs/v2-benchmark \
  --out reports/v2_benchmark_report.md \
  --publish-data reports/
```

Require every report row to contain or link manifest and checkpoint hashes.

**Step 4: Gate the learned result**

Assert the selected learned policy reliably beats random across seeds. Report
rule-policy comparison honestly. If it does not beat random, return to Task 11
instead of publishing a release.

**Step 5: Generate demo evidence**

Select representative median and best held-out seeds from analysis:

```bash
flashrl demo --checkpoint <best.pt> --seed <median-seed> \
  --record reports/demo/median.gif
```

Record the exact command and identity in `reports/demo/README.md`.

**Step 6: Commit**

```bash
git add reports/v2_benchmark_report.md reports/v2_benchmark_runs.csv \
  reports/v2_benchmark_summary.csv reports/figures reports/demo
git commit -m "research: publish reproducible FlashRL V2 benchmark"
```

### Task 13: Rewrite User and Contributor Documentation

**Files:**
- Modify: `README.md`
- Modify: `REPORT.md`
- Create: `CONTRIBUTING.md`
- Create: `CHANGELOG.md`
- Create: `.github/ISSUE_TEMPLATE/bug.yml`
- Create: `.github/ISSUE_TEMPLATE/experiment.yml`
- Create: `.github/pull_request_template.md`
- Modify: `docs/benchmark_protocol.md`
- Modify: `docs/environment_api.md`

**Step 1: Add documentation assertions**

Extend CLI and hygiene tests so every README command either appears in tested
examples or is generated from `--help`. Reject browser/PPO claims in V2 docs.

**Step 2: Verify red**

Run:

```bash
pytest tests/test_cli.py tests/test_repo_hygiene.py -q
```

Expected: FAIL against stale documentation.

**Step 3: Rewrite from evidence**

Lead README with the tested quick start, live demo, result table, learning-curve
figure, and reproduction command. Make REPORT a technical research report
covering hypothesis, methods, algorithms, protocol, results, limitations, and
future browser transfer work.

Add contributor workflow, experiment standards, release history, and structured
templates.

**Step 4: Verify and commit**

Run:

```bash
pytest -q
python scripts/smoke_test.py
```

Then:

```bash
git add README.md REPORT.md CONTRIBUTING.md CHANGELOG.md .github \
  docs tests/test_cli.py tests/test_repo_hygiene.py
git commit -m "docs: publish FlashRL V2 research and demo guide"
```

### Task 14: Completion Audit and GitHub Release

**Files:**
- Modify as evidence requires.

**Step 1: Audit every issue acceptance criterion**

Create a temporary matrix for issues #2 through #14 linking each criterion to a
test, artifact, command output, or deliberate V2 de-scope decision. Fix any
criterion without direct evidence.

**Step 2: Run clean release verification**

From a clean detached worktree at the release commit:

```bash
python -m build
python scripts/verify_wheel.py dist/*.whl
pytest -q
python scripts/smoke_test.py
flashrl doctor
git status --short
```

Expected: all checks pass and the worktree remains clean.

**Step 3: Push and open a PR**

```bash
git push -u origin feat/flashrl-v2-release
gh pr create --base master --head feat/flashrl-v2-release \
  --title "Release FlashRL V2 reproducible RL benchmark" \
  --body-file <generated-pr-evidence.md>
```

**Step 4: Verify live CI and review**

Use:

```bash
gh pr checks --watch
gh pr view --comments
```

Address real failures or review findings with focused commits. Do not merge
until every required check is green.

**Step 5: Merge and reconcile branches**

Merge through GitHub, fetch the resulting `master`, fast-forward the local
`master`, and verify remote/local commit identity. Delete only safely merged
feature branches. Preserve any branch containing unique work until its commits
are incorporated.

**Step 6: Close issues with evidence**

Comment on and close issues #2 through #13 only where the acceptance matrix
proves completion or an issue explicitly accepts V2 de-scope. Close #14 last.

**Step 7: Tag and publish release**

Create the V2 tag from verified merged `master`. Publish release notes and
attach benchmark summaries, representative demo recording, manifest bundle,
and checksums.

**Step 8: Verify remote truth**

Run:

```bash
gh run list --limit 10
gh release view v2.0.0
git ls-remote origin refs/heads/master refs/tags/v2.0.0
gh issue list --state open
git branch -a -vv
git status --short --branch
```

The task is complete only when the merged commit, green live CI, release,
assets, tag, issues, and branch state all match the audited evidence.
