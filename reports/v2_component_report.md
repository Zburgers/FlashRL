# FlashRL experiment report

Source directory: `runs/v2-components`

## Method

Only finalized manifests with held-out per-episode results are included. Scores are averaged within each trained policy first, then independent training-run means are summarized across seeds. The 95% interval is a seeded 10,000-sample percentile bootstrap over those run means.

Training sample cost is indexed by environment frames. Learning-curve area uses the same frame axis, and every published run row carries its manifest path and selected checkpoint SHA-256.

## Held-out results

| Algorithm | Runs | Episodes | Mean | 95% CI | Mean AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| double_dqn | 5 | 500 | 199.37 | [175.20, 223.53] | 5922914 |
| dqn | 5 | 500 | 238.68 | [196.92, 283.87] | 6097610 |
| dueling_double_dqn | 5 | 500 | 255.79 | [225.62, 285.96] | 6987341 |
| dueling_double_dqn_n3 | 5 | 500 | 289.67 | [260.28, 320.35] | 7002571 |

## Failure taxonomy

Ending reasons across all held-out episodes for the selected learned configuration:

| Ending reason | Count | Rate |
| --- | ---: | ---: |
| bird_no_duck | 205 | 41.0% |
| early_jump | 239 | 47.8% |
| late_jump | 56 | 11.2% |

## Representative learned runs

- **Median:** `v2-components-recommended-n3-seed53` (mean 293.91, seed 53, checkpoint `ebfd4203a18a`).
- **Best:** `v2-components-recommended-n3-seed97` (mean 345.21, seed 97, checkpoint `57dbdfbba5fb`).

Episode scores are averaged within each trained policy before independent training-run means are summarized. Learning AUC uses environment frames on the x-axis. Checkpoint and manifest identities below make every value traceable.

## Run provenance

| Run | Seed | Mean | Frames | Checkpoint SHA-256 | Manifest SHA-256 / path |
| --- | ---: | ---: | ---: | --- | --- |
| v2-components-double-seed13 | 13 | 188.04 | 30000 | `0809fa0afc23` | `a2883cf1b39d` / `runs/v2-components/v2-components-double-seed13/manifest.json` |
| v2-components-double-seed31 | 31 | 242.54 | 30000 | `faf23690906b` | `2d0175ae6287` / `runs/v2-components/v2-components-double-seed31/manifest.json` |
| v2-components-double-seed53 | 53 | 165.73 | 30000 | `f88b5420035b` | `6a7b723079f5` / `runs/v2-components/v2-components-double-seed53/manifest.json` |
| v2-components-double-seed73 | 73 | 178.25 | 30000 | `3085e67c3fde` | `b7ee636ddb5c` / `runs/v2-components/v2-components-double-seed73/manifest.json` |
| v2-components-double-seed97 | 97 | 222.27 | 30000 | `fdfcf8ac231e` | `0c09569f9dd7` / `runs/v2-components/v2-components-double-seed97/manifest.json` |
| v2-components-dueling-double-seed13 | 13 | 289.27 | 30000 | `e5b4dd428487` | `10c52edb3edc` / `runs/v2-components/v2-components-dueling-double-seed13/manifest.json` |
| v2-components-dueling-double-seed31 | 31 | 214.44 | 30000 | `f04fec0a8744` | `7c7d0c8960dd` / `runs/v2-components/v2-components-dueling-double-seed31/manifest.json` |
| v2-components-dueling-double-seed53 | 53 | 280.32 | 30000 | `f6fca5619647` | `9cad2ab09675` / `runs/v2-components/v2-components-dueling-double-seed53/manifest.json` |
| v2-components-dueling-double-seed73 | 73 | 285.46 | 30000 | `00d2d0d6e6ec` | `06ac02790210` / `runs/v2-components/v2-components-dueling-double-seed73/manifest.json` |
| v2-components-dueling-double-seed97 | 97 | 209.45 | 30000 | `023ede763161` | `096f57d3eb92` / `runs/v2-components/v2-components-dueling-double-seed97/manifest.json` |
| v2-components-recommended-n3-seed13 | 13 | 272.21 | 30000 | `39afa0299b45` | `04ad2bbf810d` / `runs/v2-components/v2-components-recommended-n3-seed13/manifest.json` |
| v2-components-recommended-n3-seed31 | 31 | 242.40 | 30000 | `009bc2595cb0` | `9f22064481f5` / `runs/v2-components/v2-components-recommended-n3-seed31/manifest.json` |
| v2-components-recommended-n3-seed53 | 53 | 293.91 | 30000 | `ebfd4203a18a` | `3aa225aed391` / `runs/v2-components/v2-components-recommended-n3-seed53/manifest.json` |
| v2-components-recommended-n3-seed73 | 73 | 294.60 | 30000 | `0bfddd63baa2` | `722e151237e4` / `runs/v2-components/v2-components-recommended-n3-seed73/manifest.json` |
| v2-components-recommended-n3-seed97 | 97 | 345.21 | 30000 | `57dbdfbba5fb` | `10926fcdde9a` / `runs/v2-components/v2-components-recommended-n3-seed97/manifest.json` |
| v2-components-vanilla-seed13 | 13 | 187.58 | 30000 | `039879ff94e0` | `4e25274a0cb5` / `runs/v2-components/v2-components-vanilla-seed13/manifest.json` |
| v2-components-vanilla-seed31 | 31 | 324.27 | 30000 | `7e606e4ff5b9` | `422b518065ac` / `runs/v2-components/v2-components-vanilla-seed31/manifest.json` |
| v2-components-vanilla-seed53 | 53 | 186.84 | 30000 | `41582b6a336a` | `9c5a91ab4661` / `runs/v2-components/v2-components-vanilla-seed53/manifest.json` |
| v2-components-vanilla-seed73 | 73 | 258.95 | 30000 | `e0d294203d2f` | `4960167f98d6` / `runs/v2-components/v2-components-vanilla-seed73/manifest.json` |
| v2-components-vanilla-seed97 | 97 | 235.78 | 30000 | `a7d4f6ef9d8d` | `5e4f2df27ffa` / `runs/v2-components/v2-components-vanilla-seed97/manifest.json` |
