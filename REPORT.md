# FlashRL Implementation Report

Date: 2026-06-10

## Summary

FlashRL has been rebuilt from a fragile Chrome Dino DQN demo into a reproducible RL benchmark pipeline. The old code mixed structured observations with an image CNN; that mismatch is removed. The new code makes observation mode explicit and chooses a matching model architecture.

## Implemented Pipeline

- `flashrl.envs.DinoEnv`: Gymnasium API with `reset(seed, options)` and `step()` returning `(obs, reward, terminated, truncated, info)`.
- Observation modes:
  - `state`: 11 normalized structured features.
  - `vision`: stacked `84x84` grayscale frames.
  - `hybrid`: dict with `image` and `state`.
- Action modes:
  - `minimal`: noop, jump.
  - `full`: noop, jump press, duck press, release.
- Backends:
  - `sim`: deterministic simulator used by default for training and tests.
  - `browser`/`chrome`: optional Playwright hooks.
- Baselines:
  - random policy.
  - rule-based Dino policy.
- DQN:
  - state MLP.
  - vision CNN.
  - hybrid CNN plus MLP encoder.
  - Double DQN target selection.
  - dueling value/advantage heads.
  - optional prioritized replay.
  - optional N-step returns.
- Benchmark harness:
  - one row per evaluation episode.
  - CSV and JSONL output.
  - run ID, commit hash, seed, eval seed, backend, observation mode, action mode, score, steps, survival time, death type, checkpoint path, config path.
- PPO state-mode training entrypoint through Stable-Baselines3.
- Markdown aggregation from result CSV files.

## Verification Completed

Commands run:

```bash
python -m py_compile dino_env.py dqn_train.py dqn_eval.py scripts/smoke_test.py $(rg --files flashrl tests -g '*.py')
python scripts/smoke_test.py
python scripts/smoke_test.py --train-smoke
pytest -q
```

Observed results:

- Smoke evaluation completed on simulator backend.
- One-episode DQN train smoke wrote a checkpoint and config under `runs/smoke/...`.
- Test suite passed: `5 passed`.
- Additional five-episode smoke benchmarks were written to `results/*_smoke.csv` and summarized in `reports/benchmark_smoke_summary.md`.

Smoke benchmark summary:

| agent | obs_mode | action_mode | episodes | mean_score | median_score | best_score |
| --- | --- | --- | --- | --- | --- | --- |
| dqn | state | full | 5 | 70.704 | 57.286 | 124.377 |
| random | state | full | 5 | 110.285 | 112.117 | 121.309 |
| rule | state | full | 5 | 404.165 | 333.822 | 661.289 |

The DQN row is from a deliberately short 10-episode training run and is only evidence that training, checkpoint loading, and evaluation work. It is not a trained-performance claim.

## What This Does Not Claim

This report does not claim that DQN has beaten the rule-based baseline yet. A full benchmark needs multi-seed training and held-out evaluation result files. Historical `.pth` files are not considered validated results because their metadata is incomplete.

## Reproduction Commands

Install:

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

Run smoke checks:

```bash
python scripts/smoke_test.py
pytest -q
```

Evaluate baselines:

```bash
python -m flashrl.benchmark.evaluate --agent random --episodes 100 --out results/random_state.csv
python -m flashrl.benchmark.evaluate --agent rule --episodes 100 --out results/rule_state.csv
```

Train DQN:

```bash
python -m flashrl.benchmark.train --episodes 500 --max-episode-steps 1000 --output-dir runs
```

Evaluate trained DQN:

```bash
python -m flashrl.benchmark.evaluate \
  --agent dqn \
  --checkpoint runs/<run_id>/checkpoint.pt \
  --episodes 100 \
  --obs-mode state \
  --action-mode full \
  --backend sim \
  --out results/dqn_state.csv
```

Train PPO:

```bash
python -m flashrl.benchmark.train_ppo --timesteps 100000 --seed 0 --output-dir runs/ppo_seed_0
```

Aggregate benchmark rows:

```bash
python -m flashrl.benchmark.aggregate 'results/*.csv' --out reports/benchmark_summary.md
```

## Next Work

- Run the full five-seed benchmark.
- Add confidence intervals to the aggregation script.
- Harden the optional browser backend with deterministic JS stepping.
- Add PPO through Stable-Baselines3 after the full DQN baseline table exists.
