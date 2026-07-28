# FlashRL experiment report

Source directory: `runs/v2-benchmark`

## Method

Only finalized manifests with held-out per-episode results are included. Scores are averaged within each trained policy first, then independent training-run means are summarized across seeds. The 95% interval is a seeded 10,000-sample percentile bootstrap over those run means.

Training sample cost is indexed by environment frames. Learning-curve area uses the same frame axis, and every published run row carries its manifest path and selected checkpoint SHA-256.

## Held-out results

| Algorithm | Runs | Episodes | Mean | 95% CI | Mean AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| dueling_double_dqn_n3 | 5 | 500 | 301.90 | [270.61, 335.98] | 36225757 |
| random | 1 | 100 | 126.18 | [126.18, 126.18] | 0 |
| rule | 1 | 100 | 284.17 | [284.17, 284.17] | 0 |

## Baseline comparisons

- `dueling_double_dqn_n3` beats random: 301.90 vs 126.18 (+175.72, +139.3%).
- `dueling_double_dqn_n3` beats rule: 301.90 vs 284.17 (+17.73, +6.2%).

Confidence intervals above resample independent training-run means. A baseline point inside the learned interval is not evidence of a reliable learned-policy advantage over that baseline.

## Failure taxonomy

Ending reasons across all held-out episodes for the selected learned configuration:

| Ending reason | Count | Rate |
| --- | ---: | ---: |
| bird_no_duck | 263 | 52.6% |
| early_jump | 158 | 31.6% |
| late_jump | 79 | 15.8% |

## Representative learned runs

- **Median:** `v2-benchmark-recommended-seed89` (mean 300.01, seed 89, checkpoint `758f47a4ce58`).
- **Best:** `v2-benchmark-recommended-seed47` (mean 363.82, seed 47, checkpoint `ff0d16e843b4`).

Episode scores are averaged within each trained policy before independent training-run means are summarized. Learning AUC uses environment frames on the x-axis. Checkpoint and manifest identities below make every value traceable.

## Run provenance

| Run | Seed | Mean | Frames | Checkpoint SHA-256 | Manifest SHA-256 / path |
| --- | ---: | ---: | ---: | --- | --- |
| v2-benchmark-random-baseline | 0 | 126.18 | 0 | `` | `487db19efb63` / `runs/v2-benchmark/v2-benchmark-random-baseline/manifest.json` |
| v2-benchmark-recommended-seed11 | 11 | 319.14 | 120000 | `259e36ba64fc` | `7b60b0bb5157` / `runs/v2-benchmark/v2-benchmark-recommended-seed11/manifest.json` |
| v2-benchmark-recommended-seed29 | 29 | 269.29 | 120000 | `5271c21a593b` | `5b3f9fa1e1e7` / `runs/v2-benchmark/v2-benchmark-recommended-seed29/manifest.json` |
| v2-benchmark-recommended-seed47 | 47 | 363.82 | 120000 | `ff0d16e843b4` | `9a566942db35` / `runs/v2-benchmark/v2-benchmark-recommended-seed47/manifest.json` |
| v2-benchmark-recommended-seed71 | 71 | 257.24 | 120000 | `9502618982f5` | `cf072c0eeb93` / `runs/v2-benchmark/v2-benchmark-recommended-seed71/manifest.json` |
| v2-benchmark-recommended-seed89 | 89 | 300.01 | 120000 | `758f47a4ce58` | `cde5c98588e8` / `runs/v2-benchmark/v2-benchmark-recommended-seed89/manifest.json` |
| v2-benchmark-rule-baseline | 0 | 284.17 | 0 | `` | `1da6b3182aae` / `runs/v2-benchmark/v2-benchmark-rule-baseline/manifest.json` |
