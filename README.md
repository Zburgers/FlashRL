# FlashRL

FlashRL is a reproducible reinforcement-learning pipeline for a Dino-style browser game benchmark. The project now separates environment observations from model architecture:

- `state`: normalized structured game state, trained with an MLP DQN.
- `vision`: real rendered pixel frames, trained with a CNN DQN.
- `hybrid`: image frames plus structured state, trained with a two-encoder DQN.

The default backend is a deterministic simulator (`--backend sim`) so training, evaluation, and tests run without a browser. Browser-backed play is available as an experimental path (`--backend browser` or `--backend chrome`) after installing Playwright browsers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

For simulator-only training and tests, the Playwright browser install is optional.

## Verify The Repo

```bash
python scripts/smoke_test.py
pytest -q
```

Run a tiny training smoke:

```bash
python scripts/smoke_test.py --train-smoke
```

## Evaluate Baselines

Random baseline:

```bash
python -m flashrl.benchmark.evaluate \
  --agent random \
  --episodes 20 \
  --seed 0 \
  --eval-seed 1000 \
  --out results/random_state.csv
```

Rule-based baseline:

```bash
python -m flashrl.benchmark.evaluate \
  --agent rule \
  --episodes 20 \
  --seed 0 \
  --eval-seed 1000 \
  --out results/rule_state.csv
```

Each evaluation writes CSV and JSONL rows with score, survival time, steps, death type, seed, observation mode, action mode, backend, checkpoint path, and commit hash.

## Train DQN

State-mode Double Dueling DQN:

```bash
python -m flashrl.benchmark.train \
  --algorithm dqn \
  --obs-mode state \
  --action-mode full \
  --backend sim \
  --episodes 200 \
  --max-episode-steps 1000 \
  --seed 0 \
  --output-dir runs
```

Add prioritized replay and 3-step returns:

```bash
python -m flashrl.benchmark.train \
  --obs-mode state \
  --prioritized-replay \
  --n-step 3 \
  --episodes 200 \
  --max-episode-steps 1000 \
  --output-dir runs
```

Vision mode:

```bash
python -m flashrl.benchmark.train \
  --obs-mode vision \
  --episodes 200 \
  --max-episode-steps 1000 \
  --output-dir runs
```

Hybrid mode:

```bash
python -m flashrl.benchmark.train \
  --obs-mode hybrid \
  --episodes 200 \
  --max-episode-steps 1000 \
  --output-dir runs
```

The trainer stores `config.json`, `train_metrics.csv`, and `checkpoint.pt` in `runs/<run_id>/`.

## Evaluate A DQN Checkpoint

```bash
python -m flashrl.benchmark.evaluate \
  --agent dqn \
  --checkpoint runs/<run_id>/checkpoint.pt \
  --episodes 100 \
  --seed 0 \
  --eval-seed 10000 \
  --obs-mode state \
  --action-mode full \
  --backend sim \
  --out results/dqn_state.csv
```

Use the same `obs-mode`, `action-mode`, and backend used during training.

## PPO Comparison

PPO is available for state observations through Stable-Baselines3:

```bash
python -m flashrl.benchmark.train_ppo \
  --timesteps 100000 \
  --seed 0 \
  --output-dir runs/ppo_seed_0
```

Evaluate PPO with a custom wrapper is still a follow-up; the saved SB3 model is intended as the PPO training artifact for comparison work.

## Aggregate Results

```bash
python -m flashrl.benchmark.aggregate 'results/*.csv' --out reports/benchmark_summary.md
```

## Compatibility Entrypoints

These top-level scripts call the package CLIs:

```bash
python dqn_train.py --episodes 50
python dqn_eval.py --agent rule --episodes 20
```

## Current Implementation

Implemented:

- Gymnasium-compatible `DinoEnv`.
- Deterministic simulator backend for reproducible local training.
- Optional browser/chrome backend hooks through Playwright.
- State, vision, and hybrid observation modes.
- Minimal and full action modes.
- Random and rule-based baselines.
- Benchmark CSV/JSONL output.
- DQN, Double DQN, dueling heads, prioritized replay, and N-step returns.
- PPO state-mode training entrypoint.
- Benchmark aggregation to markdown.
- Smoke tests and environment contract tests.

Not claimed:

- No headline score is claimed until result files are produced from a full benchmark run.
- Historical checkpoints under `data/models/` are not treated as validated benchmark results because they do not contain enough metadata.

## Recommended Full Benchmark

Run at least five seeds before writing a result table:

```bash
for seed in 0 1 2 3 4; do
  python -m flashrl.benchmark.evaluate --agent random --episodes 100 --seed "$seed" --eval-seed "$((10000 + seed * 1000))" --out "results/random_seed_${seed}.csv"
  python -m flashrl.benchmark.evaluate --agent rule --episodes 100 --seed "$seed" --eval-seed "$((10000 + seed * 1000))" --out "results/rule_seed_${seed}.csv"
  python -m flashrl.benchmark.train --episodes 500 --max-episode-steps 1000 --seed "$seed" --output-dir runs
done
```

Then evaluate each DQN checkpoint with held-out eval seeds and compare against random and rule-based baselines.
