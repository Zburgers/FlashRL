# FlashRL experiment report

## Question and protocol

This pilot asked which independently selectable DQN components improve
generalization in the deterministic V2 simulator. It trained vanilla DQN,
Double DQN, Dueling Double DQN, prioritized replay (PER), three-step returns,
and PER plus three-step returns with three independent training seeds each.
Every checkpoint was selected on seeds beginning at 50,000 and evaluated on 30
unseen episodes beginning at seed 100,000.

The pilot used 250 training episodes per run. Because stronger policies survive
longer, actual sample cost ranged from 11,214 to 12,658 environment frames.
This is a documented limitation, not an equal-budget comparison. FlashRL now
supports `total_train_frames`; all follow-up tuning and final comparisons use
that exact sample budget.

On the identical 30 held-out episode seeds, the random baseline scored 148.42
and the deterministic rule baseline scored 305.68. The rule policy is a strong
hand-engineered controller and is not presented as a learning baseline.

Source directory: `runs/v2-pilot`

## Held-out results

| Algorithm | Runs | Episodes | Mean | 95% CI | Mean AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| double_dqn | 3 | 90 | 186.62 | [117.26, 232.96] | 2041412 |
| dqn | 3 | 90 | 183.56 | [152.93, 231.36] | 1985553 |
| dueling_double_dqn | 3 | 90 | 189.37 | [162.37, 214.75] | 1899622 |
| dueling_double_dqn_n3 | 3 | 90 | 220.67 | [192.46, 256.71] | 1985947 |
| dueling_double_dqn_per | 3 | 90 | 186.10 | [162.29, 215.14] | 1898458 |
| dueling_double_dqn_per_n3 | 3 | 90 | 237.21 | [207.04, 265.89] | 2175707 |

## Findings and next experiment

PER plus three-step returns led the pilot at 237.21 mean score, 60% above
random. Three-step returns without PER ranked second at 220.67. PER without
multi-step returns did not improve over Dueling Double DQN, so the evidence
suggests that return horizon is the main effect and PER may be an interaction.

The follow-up study fixes every run at 30,000 environment frames. It retests
PER's interaction, then changes one factor at a time: learning rate, exploration
duration, and return horizon. A final protocol will be selected from mean
held-out performance and seed variance, never from one maximum episode.

## Run provenance

| Run | Seed | Mean | Frames | Checkpoint SHA-256 | Manifest |
| --- | ---: | ---: | ---: | --- | --- |
| v2-pilot-double-seed11 | 11 | 117.26 | 11443 | `694a4d7882c9` | `runs/v2-pilot/v2-pilot-double-seed11/manifest.json` |
| v2-pilot-double-seed29 | 29 | 232.96 | 11752 | `85c8c2b61c89` | `runs/v2-pilot/v2-pilot-double-seed29/manifest.json` |
| v2-pilot-double-seed47 | 47 | 209.64 | 12658 | `9911dc799ff5` | `runs/v2-pilot/v2-pilot-double-seed47/manifest.json` |
| v2-pilot-dueling-double-seed11 | 11 | 162.37 | 11492 | `6ab4e792fe21` | `runs/v2-pilot/v2-pilot-dueling-double-seed11/manifest.json` |
| v2-pilot-dueling-double-seed29 | 29 | 214.75 | 11214 | `74b3442b5018` | `runs/v2-pilot/v2-pilot-dueling-double-seed29/manifest.json` |
| v2-pilot-dueling-double-seed47 | 47 | 190.98 | 12317 | `90ed03d5798e` | `runs/v2-pilot/v2-pilot-dueling-double-seed47/manifest.json` |
| v2-pilot-n3-seed11 | 11 | 256.71 | 11510 | `59586f54ac1c` | `runs/v2-pilot/v2-pilot-n3-seed11/manifest.json` |
| v2-pilot-n3-seed29 | 29 | 192.46 | 12094 | `6b037f8abf53` | `runs/v2-pilot/v2-pilot-n3-seed29/manifest.json` |
| v2-pilot-n3-seed47 | 47 | 212.86 | 12316 | `98329aad7055` | `runs/v2-pilot/v2-pilot-n3-seed47/manifest.json` |
| v2-pilot-per-n3-seed11 | 11 | 265.89 | 12171 | `b1d8f9685402` | `runs/v2-pilot/v2-pilot-per-n3-seed11/manifest.json` |
| v2-pilot-per-n3-seed29 | 29 | 238.72 | 12616 | `a75f4a114377` | `runs/v2-pilot/v2-pilot-per-n3-seed29/manifest.json` |
| v2-pilot-per-n3-seed47 | 47 | 207.04 | 12047 | `1198ba7bd2eb` | `runs/v2-pilot/v2-pilot-per-n3-seed47/manifest.json` |
| v2-pilot-per-seed11 | 11 | 215.14 | 11733 | `9fd38143ddfe` | `runs/v2-pilot/v2-pilot-per-seed11/manifest.json` |
| v2-pilot-per-seed29 | 29 | 180.88 | 11592 | `4fb4efaa26e2` | `runs/v2-pilot/v2-pilot-per-seed29/manifest.json` |
| v2-pilot-per-seed47 | 47 | 162.29 | 11569 | `dca7d584ce58` | `runs/v2-pilot/v2-pilot-per-seed47/manifest.json` |
| v2-pilot-vanilla-seed11 | 11 | 152.93 | 11326 | `3ff586659bb7` | `runs/v2-pilot/v2-pilot-vanilla-seed11/manifest.json` |
| v2-pilot-vanilla-seed29 | 29 | 166.40 | 11832 | `6c65bb1931c7` | `runs/v2-pilot/v2-pilot-vanilla-seed29/manifest.json` |
| v2-pilot-vanilla-seed47 | 47 | 231.36 | 12568 | `98cc8b5d9760` | `runs/v2-pilot/v2-pilot-vanilla-seed47/manifest.json` |
