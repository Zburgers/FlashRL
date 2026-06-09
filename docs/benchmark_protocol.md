# FlashRL Benchmark Protocol

## Goals

The benchmark must answer:

- Does the agent beat random?
- Does it beat a simple rule-based policy?
- How sample-efficient is it?
- Does it generalize beyond the exact training game distribution?
- Are results reproducible across seeds?

## Algorithms To Compare

Minimum table:

- Random policy
- Rule-based Dino policy
- Vanilla DQN
- Double DQN
- Dueling Double DQN
- Rainbow DQN
- PPO
- Recurrent PPO, optional for partially observed settings

## Seeds

Use at least 5 training seeds for development:

```text
0, 1, 2, 3, 4
```

Use 10 seeds for reported benchmark tables:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Evaluation seeds must be separate from training seeds.

## Evaluation Episodes

- Development: 20 episodes per seed.
- Reported results: 100 episodes per seed.
- No exploration during DQN evaluation unless evaluating stochastic policies explicitly.
- PPO should report deterministic and stochastic evaluation separately if both are used.

## Metrics

Primary metrics:

- Mean score
- Median score
- Best score
- Interquartile range
- Survival time
- Obstacles cleared
- Death type distribution

Efficiency metrics:

- Training frames to threshold score
- Wall-clock training time
- Environment steps per second
- GPU/CPU used

Reliability metrics:

- Reset failure count
- Browser crash count
- Truncated episodes
- Invalid observations

## Result Schema

Write one row per evaluation episode:

```text
run_id,git_commit,seed,eval_seed,game,backend,obs_mode,action_mode,agent,algorithm,phase,episode,score,median_score_so_far,survival_time_s,steps,obstacles_cleared,death_type,terminated,truncated,train_frames,wall_clock_train_s,wall_clock_eval_s,checkpoint_path,config_path
```

Use CSV for simple analysis and JSONL for richer metadata.

## Train/Test Split

Training environment:

- Default obstacle distribution.
- Default speed schedule.
- Default visual theme.
- Fixed but seedable RNG.

Test environment:

- Held-out obstacle RNG seeds.
- Held-out speed multipliers.
- Held-out visual variations.

Never tune hyperparameters on final test seeds.

## Generalization Tests

Run each trained policy on:

- `speed_easy`: 0.8x speed
- `speed_hard`: 1.2x speed
- `spacing_dense`: smaller obstacle gaps
- `spacing_sparse`: larger obstacle gaps
- `bird_low`: increased low-bird frequency
- `bird_high`: increased high-bird frequency
- `visual_day_night`: day/night and contrast changes
- `visual_noisy`: mild background/noise variation for vision agents

## Failure Taxonomy

Classify death by:

- `early_jump`: jumped too early and landed on obstacle.
- `late_jump`: failed to jump in time.
- `overjump`: unnecessary jump caused collision with next obstacle.
- `bird_no_duck`: did not duck under bird.
- `duck_collision`: ducked at wrong time.
- `release_error`: held duck/jump too long.
- `env_reset_failure`: environment failed before meaningful episode.
- `browser_failure`: browser/backend crashed.
- `unknown`: cannot classify.

## Reporting Format

Each benchmark report must include:

- Environment version and backend.
- Observation mode.
- Action mode.
- Algorithm config.
- Number of train seeds.
- Number of eval seeds and episodes.
- Training frames.
- Wall-clock time.
- Mean/median/best score with confidence intervals.
- Comparison to random and rule-based baselines.

## Minimum Acceptance Thresholds

Before claiming a learning result:

- Agent beats random by a statistically meaningful margin.
- Agent is compared against the rule-based baseline.
- Evaluation uses held-out seeds.
- Result file and checkpoint metadata are present.
- Commands are documented and runnable from a clean clone.

