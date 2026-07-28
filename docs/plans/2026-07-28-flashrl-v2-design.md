# FlashRL V2 Release Design

Date: 2026-07-28
Status: Approved

## Product Direction

FlashRL V2 will be a small, honest, research-oriented reinforcement-learning
benchmark built around a deterministic Dino simulator. It will prioritize
correct learning, reproducible experiments, useful algorithm comparisons, and
an impressive live demonstration over a broad but unfinished feature surface.

The supported V2 environment backend is the Python simulator. The existing
Playwright paths are not browser-backed environments because their observations,
reward, physics, and termination still come from the simulator. They will be
removed from the supported CLI and documentation. A browser may visualize the
simulator, but it will not be described as the environment source of truth.

PPO and real browser control are deferred to V2.x unless they can satisfy the
same manifest, evaluation, and test contracts as the custom DQN agents.

## Alternatives Considered

### Real-browser V2

Replacing the bundled game and implementing a deterministic JavaScript bridge
would create a compelling browser-control project, but it would consume the
release effort before the existing DQN and benchmark correctness problems were
fixed. It is a good V2.x transfer-learning project, not the V2 release scope.

### Multi-game framework

Generalizing the environment and agent APIs to several games would demonstrate
breadth but would leave every experiment shallow. The current repository needs
one deeply validated benchmark first.

### Simulator-first research benchmark

This is the selected approach. The simulator is deterministic, fast enough for
ablation studies, and available in clean and headless environments. It enables
credible learning curves, multi-seed comparisons, automated tests, and a live
visualization without misrepresenting browser support.

## Environment Contract

The environment will expose versioned observation, action, reward, simulator,
and environment schemas.

State observations will include every control-relevant property of the nearest
obstacles, including bird altitude. Every feature will have a documented unit
and feature-specific bound or bounded transform. Long successful trajectories
must remain inside the declared Gymnasium space.

Episode termination and time-limit truncation remain separate throughout the
environment, replay buffer, N-step construction, metrics, checkpoints, and
results. True terminal states do not bootstrap. Time limits do.

The action schema will define exact behavior for each action. V2 will use one
simulator action contract, validated when checkpoints are loaded. Internal
simulator errors fail fast and never become replay transitions. Time limits and
typed external interruptions remain reportable truncations.

## Agent and Experiment Surface

FlashRL will retain independently selectable DQN components so the repository
can run meaningful ablations:

- vanilla DQN;
- Double DQN;
- Double plus Dueling DQN;
- prioritized replay;
- N-step returns;
- the strongest validated combination found during experimentation.

Correctness work precedes performance work. After correctness is locked, the
project will run pilot experiments to select training budgets and promising
configurations. Final experiments use at least five independent training seeds
and a common held-out evaluation seed set. Comparisons will include random and
rule-based baselines under identical environment conditions.

Performance work may tune learning rate, exploration duration, replay warmup,
target update cadence, N-step horizon, replay prioritization, network capacity,
and reward scaling. Every retained change must be justified by held-out
multi-seed evidence rather than a single best episode.

The final report will include learning curves, sample efficiency, per-seed and
across-seed statistics, confidence intervals, failure taxonomy, compute time,
and an honest comparison with both baselines. A learned policy must reliably
beat random before V2 is considered successful. Beating the rule policy is a
claim only if held-out evidence supports it.

## Artifact and Identity Model

Every training run writes an immutable, versioned manifest containing:

- run and experiment IDs;
- canonical algorithm ID and hyperparameter hash;
- training seed and evaluation protocol;
- Git commit and dirty-state marker;
- environment and schema versions;
- complete configuration;
- frame count and timing;
- artifact paths and SHA-256 hashes.

Training writes separate atomic checkpoints:

- `last.pt` for resuming the latest state;
- `best.pt` selected by deterministic held-out evaluation;
- optional periodic checkpoints.

Checkpoint metadata records its role, selected step, selection metric, model
schema, optimizer state, RNG state, and manifest identity. Loading validates
the supported format and environment compatibility before model construction.

Evaluation derives compatible settings and training identity from the
checkpoint instead of relying on manually repeated flags. Results distinguish
training and evaluation commits and contain explicit per-episode timing.

Aggregation validates schemas and experiment identity. It summarizes episodes
within a trained run first, then summarizes independent runs across training
seeds. Incompatible backends, environment versions, reward versions,
hyperparameters, or checkpoints cannot be silently combined.

## Canonical CLI

An installable `flashrl` command will provide:

- `flashrl train`;
- `flashrl evaluate`;
- `flashrl aggregate`;
- `flashrl demo`;
- `flashrl doctor`;
- `flashrl --version`.

The package uses modern `pyproject.toml` metadata. Core simulator installation
contains only simulator/runtime requirements. Development, browser research,
and optional algorithm integrations use dependency extras.

## Live Demo

`flashrl demo` will launch a local real-time demonstration backed by the
deterministic simulator. The packaged web interface will show:

- an animated Dino canvas sourced from current simulator state;
- current score, speed, seed, action, reward, and episode status;
- the selected algorithm and checkpoint identity;
- Q-values and the chosen action for trained DQN policies;
- rolling reward and score charts;
- pause, reset, speed, seed, and policy controls;
- deterministic replay of representative evaluation seeds.

The demo server will expose read-only simulator telemetry and bounded control
commands. It must not execute arbitrary paths or deserialize user-controlled
objects. Checkpoints are explicitly selected by the local operator. A
rule-policy mode will keep the demo useful before a trained artifact is chosen.

The demo will also support recording deterministic representative episodes so
the release can include portable visual evidence.

## Error Handling

Configuration and compatibility failures produce actionable typed errors before
training starts. Checkpoint and manifest writes use temporary files followed by
atomic replacement. Interrupted runs retain a valid last checkpoint and an
append-safe metric stream.

Simulator invariant failures abort the command. Benchmark aggregation rejects
invalid or incompatible inputs and explains the conflicting identity fields.
The demo isolates one episode failure, reports it visibly, and stops advancing
instead of fabricating telemetry.

## Testing and Release Gates

Tests will cover:

- Gymnasium contracts for every supported observation mode;
- complete-trajectory determinism;
- long-horizon observation bounds;
- bird-altitude observability and baseline behavior;
- action semantics;
- terminal versus truncated DQN targets;
- one-step and N-step effective discounts;
- uniform and prioritized replay determinism;
- a real optimizer update;
- atomic best/last checkpoint behavior;
- train/save/load/resume/evaluate round trips;
- schema and compatibility rejection;
- safe aggregation and seed-level statistics;
- wheel installation outside the source tree;
- canonical CLI behavior;
- demo telemetry and HTTP controls.

GitHub Actions will test supported Python versions, the minimal core install,
the built wheel, static quality checks, and release fixtures. The release cannot
be tagged while required checks are missing or failing.

## Repository and Git Strategy

V1 scripts, committed browser binaries, stale checkpoints, logs, caches, and
contradictory reports will be removed from the supported tree. Valuable
historical context may be retained under an explicitly historical document,
but generated artifacts will not remain mixed with source.

Implementation will use a dedicated feature branch and focused commits grouped
by correctness, artifacts, packaging, demo, experiments, and release evidence.
Existing untracked benchmark files are preserved until they are either archived
as invalid historical output or replaced by current-schema evidence.

After verification, the feature branch will be pushed and reviewed through a
pull request. Remote issues will be closed only when their acceptance criteria
have direct evidence. Stale local branches will be reconciled after the release
branch is safely published. V2 is tagged only after the merged commit, live CI,
release notes, and release assets are verified.

## Definition of Done

FlashRL V2 is complete when:

1. supported learning and environment semantics are correct and versioned;
2. install, CLI, CI, tests, manifests, and aggregation pass from a clean wheel;
3. the repository contains one clear supported architecture;
4. multi-seed held-out experiments produce traceable performance evidence;
5. a trained learned policy reliably beats random;
6. the live demo runs from the installed package with a validated checkpoint;
7. required GitHub issues are resolved with evidence;
8. the PR, branches, tag, release, assets, and live Actions run are verified.
