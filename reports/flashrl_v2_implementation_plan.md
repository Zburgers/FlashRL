# FlashRL V2 Implementation Plan

Date: 2026-06-10

## Goal

Turn FlashRL into a reproducible reinforcement learning benchmark for lightweight browser games, starting with Chrome Dino and expanding to additional environments after the benchmark harness is real.

## Non-Negotiable Principles

- Documentation must match code.
- Observation mode must match model architecture.
- Every result must be reproducible from a clean clone.
- Baselines come before advanced algorithms.
- Browser/game control must be deterministic enough for benchmark use.
- Saved checkpoints without metrics are not results.

## Phase 0: Repo Cleanup And Reproducibility

Target duration: 2-4 days.

Tasks:

- Remove or quarantine stale README claims.
- Replace README commands with actual commands: `python dqn_train.py`, `python dqn_eval.py`, `python test_script.py`, or new CLI entrypoints.
- Add `pyproject.toml` or pinned `requirements.lock`.
- Fix dependency mismatch: either import Gymnasium or add legacy `gym`; preferred answer is Gymnasium.
- Document `playwright install chromium`.
- Add `.gitignore` entries for `__pycache__/`, TensorBoard logs, debug frames, local `.env`, local checkpoints.
- Move heavyweight checkpoints to releases or external storage; keep only tiny smoke-test fixtures if needed.
- Add `scripts/smoke_test.py` that imports modules, checks env spaces without launching long training, and validates docs commands.
- Add CI for lint/import/smoke tests.

Exit criteria:

- Fresh virtualenv can run `pip install -r requirements.txt`, `playwright install chromium`, and import all modules.
- README quick start succeeds.
- No stale non-existent commands remain.

## Phase 1: Gymnasium Migration And Environment Correctness

Target duration: 4-6 days.

Tasks:

- Replace `import gym` with `import gymnasium as gym`.
- Implement `reset(self, *, seed=None, options=None) -> (obs, info)`.
- Implement `step(self, action) -> (obs, reward, terminated, truncated, info)`.
- Add `metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}`.
- Add `render_mode`, `max_episode_steps`, `headless`, `game_url`, `obs_mode`, `action_mode`, `seed`, `fixed_timestep_ms`.
- Load a local controlled game by default, not `chrome://dino`.
- Support `chrome://dino` only as an experimental backend.
- Replace silent default observations with explicit error states or controlled reset retries.
- Add robust browser lifecycle management and idempotent `close()`.
- Add deterministic game seeding through JS injection for local game mode.

Exit criteria:

- `gymnasium.utils.env_checker.check_env(DinoEnv(...))` passes for each observation mode.
- Reset/step behavior is tested without relying on manual browser interaction.

## Phase 2: Baseline Agents And Benchmark Harness

Target duration: 5-7 days.

Tasks:

- Implement `agents/random_agent.py`.
- Implement `agents/rule_based_dino.py`.
- Implement `benchmark/evaluate.py` with fixed seeds and CSV/JSONL output.
- Implement `benchmark/train.py` wrapper that stores config, commit hash, dependency versions, environment metadata, and checkpoint path.
- Add result schema:
  - `run_id`
  - `seed`
  - `game`
  - `obs_mode`
  - `action_mode`
  - `agent`
  - `episode`
  - `score`
  - `survival_time_s`
  - `steps`
  - `obstacles_cleared`
  - `death_type`
  - `wall_clock_s`
  - `train_frames`
  - `checkpoint`
- Add reproducible plots from result files, not hand-edited images.

Exit criteria:

- Random and rule-based baselines run for 100 eval episodes across at least 5 seeds.
- Baseline results are committed as small CSV examples or generated in CI smoke mode.

## Phase 3: Rainbow DQN And PPO

Target duration: 8-12 days.

Tasks:

- Refactor DQN into reusable modules:
  - networks
  - replay buffers
  - schedules
  - trainers
  - evaluators
- Implement state-mode vanilla DQN with MLP.
- Implement vision-mode vanilla DQN with CNN.
- Add Double DQN.
- Add Dueling DQN.
- Add PER.
- Add N-step returns.
- Add NoisyNet.
- Add C51 or QR-DQN.
- Add combined Rainbow config.
- Add PPO via Stable-Baselines3 or a small CleanRL-style script after Gymnasium compatibility is proven.

Exit criteria:

- Each DQN variant can run a short smoke train and benchmark eval.
- Full benchmark table compares random, rule-based, vanilla DQN, Double DQN, Rainbow, PPO.

## Phase 4: Vision/State/Hybrid Comparison

Target duration: 5-8 days.

Tasks:

- Finalize three observation modes:
  - `state`: normalized structured game internals.
  - `vision`: grayscale or RGB pixel frames.
  - `hybrid`: dict observation with image plus state features.
- Make action semantics identical across modes.
- Run all major algorithms across modes where appropriate.
- Add ablations:
  - frame stack size
  - crop region
  - grayscale vs binary threshold
  - state features with/without speed and obstacle type

Exit criteria:

- Report sample efficiency and final score by observation mode.
- Claims about vision are backed by actual pixel observations.

## Phase 5: Additional Games

Target duration: 7-14 days.

Tasks:

- Add Flappy Bird, Snake, 2048, Breakout clone, and custom Dino curriculum environment.
- Standardize each environment under one package namespace.
- Reuse benchmark harness across games.
- Add per-game rule-based baselines where feasible.

Exit criteria:

- At least three games support random baseline, rule-based baseline, and one learning baseline.

## Phase 6: Dashboard, Videos, README Polish

Target duration: 4-7 days.

Tasks:

- Build a static benchmark dashboard from result files.
- Add videos generated from evaluation episodes.
- Add README result table with exact run IDs.
- Publish model cards/checkpoint metadata.
- Add "how to reproduce" commands for each table.

Exit criteria:

- A user can reproduce the headline table from a clean clone.
- Every visual/video maps to a result file and checkpoint.

## 30-Day Implementation Plan

### Week 1

- Finish Phase 0 and most of Phase 1.
- Deliver clean install, Gymnasium API migration, local-game default, and environment checker.

### Week 2

- Finish Phase 1 and Phase 2.
- Deliver random/rule-based baselines, benchmark harness, result schema, and first reproducible baseline table.

### Week 3

- Implement clean vanilla DQN, Double DQN, and Dueling DQN for state mode.
- Add vision-mode CNN only after pixel observations are verified.

### Week 4

- Add PER and N-step.
- Add PPO baseline.
- Run the first real comparison: random, rule-based, vanilla DQN, Double DQN, Dueling DQN, PER/N-step DQN, PPO.
- Update README with honest, reproducible results.

## Definition Of Done For V2

- Clean clone works.
- Gymnasium API passes checks.
- State, vision, and hybrid modes are explicit.
- Baselines are reproducible.
- Results are stored as machine-readable files.
- Documentation no longer claims unverified scores or non-existent files.
- At least one DQN variant beats random and is compared against a rule-based baseline.

