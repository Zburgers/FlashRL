# FlashRL Backlog

## P0: Make The Repo Honest And Runnable

- Replace stale README quick start with actual commands.
- Remove references to `train_agent.py`, `run_agent.py`, `server.py`, `agent/`, `environment/`, and missing utility scripts unless those files are created.
- Replace TensorFlow/Keras README examples with PyTorch examples or remove them.
- Add clean-clone setup instructions including `playwright install chromium`.
- Fix `requirements.txt`/imports mismatch by migrating to Gymnasium.
- Add CI smoke test for imports and CLI help.

## P0: Fix Observation/Model Mismatch

- Split environment observation modes into `state`, `vision`, and `hybrid`.
- Add MLP network for state observations.
- Use CNN only for real image observations.
- Add tests asserting observation shape matches selected network.
- Remove fake image preprocessing of structured vectors.

## P0: Migrate DinoEnv To Gymnasium

- Change imports to Gymnasium.
- Implement `reset(seed=None, options=None) -> (obs, info)`.
- Implement `step(action) -> (obs, reward, terminated, truncated, info)`.
- Add `metadata`, `render_mode`, and idempotent `close()`.
- Add environment checker tests.

## P1: Stabilize Browser/Game Backend

- Default to local deterministic Dino HTML/JS backend.
- Add seeded RNG to local game.
- Make `chrome://dino` optional.
- Replace silent fallback states with explicit reset failure handling.
- Add browser crash handling that returns `truncated=True`.
- Add max episode step/time limit.

## P1: Improve Action Space

- Add full action mode with `NOOP`, `JUMP_PRESS`, `DUCK_PRESS`, `RELEASE`.
- Add key-up/key-down semantics.
- Add tests for action effects.
- Track action names in `info`.

## P1: Build Benchmark Harness

- Add `benchmark/evaluate.py`.
- Add `benchmark/train.py`.
- Add fixed train/eval seeds.
- Export CSV and JSONL results.
- Add metrics: mean, median, best score, survival time, obstacles cleared, death type, sample efficiency, wall-clock time.
- Add held-out generalization configs.

## P1: Add Baselines

- Implement random policy.
- Implement rule-based Dino policy.
- Require DQN results to compare against both.

## P2: Refactor DQN

- Move networks to `flashrl/agents/dqn/networks.py`.
- Move replay to `flashrl/agents/dqn/replay.py`.
- Move training loop to `flashrl/agents/dqn/train.py`.
- Add config dataclasses or YAML.
- Log loss, Q-values, epsilon, gradient norm, episode metrics.
- Store checkpoint metadata: config, commit hash, obs mode, action mode, seed, train frames.

## P2: Algorithm Upgrades

- Add Double DQN.
- Add Dueling DQN.
- Add PER.
- Add N-step returns.
- Add NoisyNet.
- Add C51 or QR-DQN.
- Add Rainbow config after components are individually validated.
- Add PPO baseline after Gymnasium compatibility is stable.

## P2: Repo Hygiene

- Remove committed `__pycache__/`.
- Move large checkpoints out of git or to release assets.
- Remove `chromedriver-win64` unless Selenium path is intentionally restored.
- Add `.gitignore` for logs/debug frames/checkpoints.
- Add license file or remove license badge/claim.

## P3: Multi-Game Expansion

- Add custom Dino curriculum runner.
- Add Flappy Bird environment.
- Add Snake environment.
- Add Breakout clone environment.
- Add 2048 environment.
- Reuse common benchmark protocol across games.

## P3: Dashboard And Reporting

- Generate benchmark markdown tables from result files.
- Add static dashboard.
- Add evaluation videos tied to run IDs.
- Add reproducibility appendix for every headline score.

