# FlashRL V2 benchmark protocol

## Purpose

A performance result must answer whether learning beats random, how it compares
with a state-aware rule controller, how sample-efficient it is, and how much it
varies across independent training seeds.

## Scope

The V2 certification surface is `FlashRL-DinoSim-v2` with `backend="sim"`.
State, simulator-pixel vision, and hybrid observations are separate
experiments. The published V2 result is state mode with the full action schema.
Real-browser control and PPO are deferred.

## Identity controls

Every comparison locks:

- environment, simulator, observation, action, and reward versions;
- observation and action modes;
- fixed timestep and maximum episode length;
- algorithm and canonical hyperparameter hash;
- exact environment-frame budget;
- checkpoint role and SHA-256;
- training and evaluation commits.

Aggregation rejects incompatible identity fields.

## Seeds

Use separate domains for:

1. environment sequences encountered during training;
2. deterministic greedy checkpoint selection;
3. pilot held-out evaluation;
4. final held-out evaluation.

Pilot evidence requires at least three independent training seeds. A release
claim requires at least five. Final evaluation uses at least 100 common unseen
episode seeds per trained policy. Exploration is disabled.

Never tune hyperparameters or choose checkpoints on final evaluation seeds.

## Sample budget

Use `total_train_frames` as the primary learning budget. Episode counts are only
a safety ceiling because better policies survive longer and otherwise receive
more frames. Report exact frames and wall-clock time.

## Checkpoint selection

`best.pt` is selected by mean greedy score on a fixed selection set.
`last.pt` is the latest continuation state. Selection interval, seed base,
episode count, selected frame, score, and role are checkpoint metadata.

Final evaluation always uses selected weights and a seed set disjoint from
selection.

## Required outputs

- committed experiment configuration;
- finalized manifest for every training run;
- complete config and training metrics;
- raw episode-level CSV and JSONL;
- best and last checkpoint hashes;
- run-level mean, median, standard deviation, and best;
- across-run mean, median, standard deviation, and 95% bootstrap interval;
- learning curves and AUC against environment frames;
- ending-reason taxonomy;
- random and rule results on identical held-out conditions;
- representative portable replay;
- exact reproduction command and compute cost.

## Statistical unit

Episodes from one checkpoint are not independent training replications. Average
episodes within a trained run first. Then summarize run means across training
seeds. The FlashRL interval uses a deterministic 10,000-sample percentile
bootstrap over independent run means.

Do not pool episodes from several checkpoints into one nominal sample count.
Keep the number of train runs and evaluation episodes separate.

## Success gates

- every value traces to a current manifest and artifact hash;
- schemas and identities validate before aggregation;
- every learned run uses the same exact frame budget;
- at least one learned configuration beats random across training seeds;
- rule comparison is reported even if learning loses;
- a rule advantage is claimed only when held-out multi-seed uncertainty
  supports it;
- smoke runs remain separate from performance evidence;
- commands run from the installed package.

## V2 reference protocol

The release configuration is
[experiments/v2_benchmark.yaml](../experiments/v2_benchmark.yaml):

- train seeds 11, 29, 47, 71, 89;
- 120,000 frames per policy;
- selection seeds beginning at 50,000;
- final seeds 200,000 through 200,099;
- Dueling Double DQN, uniform replay, three-step returns;
- learning rate `5e-4`, target cadence 500, update cadence four;
- 100 final episodes per policy and baseline.

Generate evidence with:

```bash
flashrl experiment experiments/v2_benchmark.yaml --resume
flashrl analyze runs/v2-benchmark \
  --out reports/v2_benchmark_report.md \
  --publish-data reports
```

