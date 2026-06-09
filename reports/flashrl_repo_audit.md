# FlashRL Repository Audit

Date: 2026-06-10
Scope: README.md, REPORT.md, dino_env.py, dqn_train.py, dqn_eval.py, config.py, utils.py, requirements.txt, assets, saved logs, saved checkpoints.

## Executive Summary

FlashRL is not currently a serious RL benchmark. It is a fragile Chrome Dino demo with stale documentation, a broken dependency story, and a core observation/model mismatch that makes the claimed pixel-based DQN pipeline misleading.

The most important fact: `DinoEnv` returns a 3-dimensional structured observation `[trex_y, obstacle_x, obstacle_width]`, but `dqn_train.py` and `dqn_eval.py` treat that observation as if it were an image frame and feed it into an Atari-style CNN over stacked `84x84` frames. The code only compiles because preprocessing resizes the 3-number state into an artificial image-like tensor. That is not screen capture, not computer vision, and not a meaningful pixel DQN benchmark.

The README and REPORT should be treated as stale marketing material, not as reproducible documentation. They mention non-existent scripts, non-existent package structure, TensorFlow/Keras snippets in a PyTorch repo, missing web UI files, and unsupported performance claims.

## Current Repo Diagnosis

### Repository Contents Observed

- Top-level code: `dino_env.py`, `dqn_train.py`, `dqn_eval.py`, `config.py`, `utils.py`, `menu.py`, `test_script.py`.
- Assets: `assets/dino_game.html`, `assets/game.js`, `assets/jquery.min.js`.
- Saved artifacts: `data/debug_frames/*.jpg`, `data/dino_debug_*.log`, `data/models/*.pth`, `data/rewards_plot.png`, TensorBoard event files under `logs/`.
- Windows-specific binary: `chromedriver-win64/chromedriver.exe`.
- No `train_agent.py`, `run_agent.py`, `server.py`, `agent/`, `environment/`, `utils/test_chrome_connection.py`, `utils/test_screen_capture.py`, `package.json`, `CONTRIBUTING.md`, or `LICENSE` found in the inspected checkout.

### What The Environment Actually Observes

`DinoEnv` exposes structured state, not pixels:

- `dino_env.py:97-99` sets `action_space = Discrete(2)` and `observation_space = Box(... shape=(3,))`.
- `dino_env.py:164-197` reads `Runner.instance_` internals through JavaScript.
- `dino_env.py:225-237` returns `np.array([state["y"], state["x"], state["width"]])`.
- `dino_env.py:239-257` reset returns the same 3-value vector.

There is no screenshot capture path in `DinoEnv`. The assets include a local Dino-like game, but the environment does not load it; it attempts `chrome://dino`.

### DQN Architecture Mismatch

`dqn_train.py` defines a convolutional DQN:

- `dqn_train.py:53-69` uses three conv layers followed by dense layers.
- `dqn_train.py:212-214` constructs it with `(FRAME_STACK, 84, 84)`.
- `dqn_train.py:233-239` calls `env.reset()`, then `preprocess_frame(frame)` even though `frame` is a 3-value vector.
- `dqn_train.py:260-266` repeats the same mistake for `env.step()`.

`utils.preprocess_frame()` resizes any non-image array to `84x84`, so the 3-value state is being spatially distorted into a fake image. The architecture and observation space do not match.

### Algorithm Classification

The current trainer is vanilla DQN with uniform replay and a target network. It is not Double DQN, Dueling DQN, PER, N-step, C51/QR-DQN, NoisyNet, Rainbow, PPO, or recurrent PPO.

Evidence:

- Uniform replay: `utils.py` samples random indices with `np.random.choice`.
- Target value uses `target_net(next_state_batch).max(1)[0]` in `dqn_train.py:152-158`, which is vanilla DQN max-over-target, not Double DQN online-action/target-evaluation split.
- No dueling heads, no priority weights, no distributional atoms/quantiles, no noisy layers, no N-step buffer.

## README And REPORT Mismatches

### Critical Documentation Mismatches

- README claims pixel data from the game screen (`README.md:37`), but the environment returns structured JS state.
- README quick start uses `python train_agent.py` and `python run_agent.py` (`README.md:81-85`); those files do not exist.
- README training command includes unsupported flags `--batch-size`, `--memory-size`, `--epsilon`, `--epsilon-min`, `--epsilon-decay`, `--gamma`, `--learning-rate` (`README.md:135-147`); `dqn_train.py` supports only `--no-cuda`, `--episodes`, and `--save-frames`.
- README testing command uses `models/dino_dqn_latest.h5` (`README.md:153-160`); actual checkpoints are PyTorch `.pth` files under `data/models/`.
- README claims `server.py` web UI (`README.md:162-170`); no `server.py` exists.
- README project structure (`README.md:172-196`) describes directories and files that are absent.
- README shows TensorFlow/Keras customization (`README.md:244-257`) while the actual implementation is PyTorch.
- README says DQN predicts jump, duck, do nothing (`README.md:214`), but the environment only supports wait and jump.
- REPORT claims best performance of `2,800+` (`REPORT.md:7`, `REPORT.md:13-18`), but saved artifacts do not provide reproducible evaluation evidence for this claim.
- REPORT claims stacked grayscale `84x84` frames (`REPORT.md:38-43`) while its own environment section correctly shows a structured `shape=(3,)` observation (`REPORT.md:81-85`, `REPORT.md:148-157`). The report contradicts itself.

### Dependency And Clean Clone Problems

- `requirements.txt` installs `gymnasium==0.29.1`, but `dino_env.py` imports `gym` and `from gym import spaces`. A clean clone following requirements is expected to fail unless `gym` is separately installed.
- Importing current code in this environment failed at `dqn_train` because `tensorboard` was not installed in the active interpreter. Static compile passed, but runtime import did not.
- Playwright browsers are not installed by `pip install -r requirements.txt`; users must run `playwright install chromium`, which is not documented.
- The README says `npm install`, but there is no `package.json`.
- Python support claim `3.8+` is dubious with pinned packages and no CI matrix.
- `chromedriver-win64/chromedriver.exe` is dead weight because the code uses Playwright, not Selenium/ChromeDriver. It is Windows-specific and should not be committed.

### Saved Artifacts

- `data/models/*.pth` totals about 155 MB. These are not accompanied by evaluation metadata, seeds, git commit hash, config snapshot, or score tables.
- Checkpoint metadata observed: `dqn_dino_best.pth` has `episode=364`, `frame_idx=17318`. That is not enough to validate the `2,800+` claim.
- Debug logs include historical reset failures: `net::ERR_INTERNET_DISCONNECTED`, `Runner is not defined`, and `Failed to detect game area`.
- TensorBoard logs exist, but there is no canonical export or results CSV.

## Severity-Ranked Issues

### P0: Code Cannot Support Claimed Pixel DQN

The environment returns 3 floats; the model expects images. This invalidates the central project claim. Fix by explicitly separating state, vision, and hybrid observation modes and matching model classes to each mode.

### P0: Clean Clone Is Not Reproducible

README commands are wrong, dependencies are mismatched, Playwright browser install is missing, and imports can fail. Fix before any algorithm work.

### P0: Gym API Is Obsolete

`DinoEnv` uses old Gym API: `reset() -> obs`, `step() -> obs, reward, done, info`. Gymnasium requires `reset(seed=None, options=None) -> (obs, info)` and `step() -> (obs, reward, terminated, truncated, info)`.

### P1: Environment Is Fragile And Non-Deterministic

The environment drives a real browser at wall-clock speed with `time.sleep()` delays, no deterministic seeding, no episode time limit, no stable local HTML URL, and weak reset verification. It also silently returns default non-terminal states when JavaScript state extraction fails, which can hide environment corruption.

### P1: Action Space Is Incomplete

The current `Discrete(2)` action space only supports wait and jump. Real Dino requires duck/release semantics for pterodactyls and jump duration control. At minimum support `NOOP`, `JUMP_PRESS`, `DUCK_PRESS`, `RELEASE`.

### P1: Evaluation Is Not A Benchmark

`dqn_eval.py` runs greedy evaluation but no seeds, no fixed episode count beyond config, no baseline comparison, no result file, no confidence intervals, no train/test split, and no failure taxonomy.

### P1: Reward Is Too Crude

Reward is `+0.1` per step and `-10` on crash. It does not expose score delta, obstacle cleared, death type, or survival time consistently. It may reward slow stepping artifacts rather than game skill.

### P2: Algorithm Implementation Is Minimal

Replay buffer is uniform only, target update is hardcoded every 1000 frames despite unused `TARGET_UPDATE`, epsilon is episode-based rather than frame-based, gradient clipping assumes gradients exist, optimizer LR ignores `config.LEARNING_RATE`, and checkpoints omit full experiment metadata.

### P2: Repo Hygiene Is Poor

Committed pycache, large checkpoints, old debug frames/logs, Windows ChromeDriver, and stale docs make the repo noisy and hard to maintain.

## Verification Performed

- Read key source/docs/artifacts listed in task.
- Ran `python -m py_compile dino_env.py dqn_train.py dqn_eval.py config.py utils.py menu.py test_script.py`: passed.
- Ran `python -c "import dino_env, dqn_train, dqn_eval; print('imports ok')"`: failed due missing `tensorboard` in active interpreter after Gym deprecation warning.
- Inspected checkpoint metadata with `torch.load`: checkpoints contain episode and frame index, not reproducible scores.

## Immediate Recommendation

Do not add Rainbow or PPO yet. First make the project honest and runnable:

1. Rewrite README/REPORT to match actual files.
2. Migrate to Gymnasium.
3. Split state/vision/hybrid environment modes.
4. Add random and rule-based baselines.
5. Add deterministic benchmark harness and result exports.
6. Only then clean up DQN and implement Double/Dueling/PER.

