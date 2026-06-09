# Browser Game Expansion Plan

## Selection Criteria

Add games that are lightweight, deterministic, seedable, fast to run headless, and useful for different RL failure modes. Do not expand before the Dino benchmark harness is stable.

## 1. Flappy Bird

Observation space:

- State: bird y, bird velocity, next pipe x, top/bottom pipe gap, pipe speed.
- Vision: cropped grayscale frame stack.
- Hybrid: image plus state vector.

Action space:

- `0 NOOP`
- `1 FLAP`

Reward:

- `+1` per pipe cleared.
- Small survival reward if timestep is fixed.
- `-1` or `-10` on crash.

Why useful:

- Simple delayed-control benchmark with tight timing and sparse events.
- Good for DQN, Double DQN, PPO, and recurrent PPO.

Baseline:

- Rule-based proportional controller that flaps below pipe center.

Expected difficulty:

- Low-medium.

## 2. Snake

Observation space:

- State: head position, direction, food vector, danger left/straight/right, body occupancy grid.
- Vision: grid image or canvas frame.
- Hybrid: grid plus summary state.

Action space:

- `0 STRAIGHT`
- `1 TURN_LEFT`
- `2 TURN_RIGHT`

Reward:

- `+1` food eaten.
- `-1` death.
- Small penalty per step to avoid loops.

Why useful:

- Tests planning, self-collision avoidance, and sparse rewards.
- Useful for DQN, Rainbow, PPO, and recurrent policies.

Baseline:

- BFS/greedy path-to-food with safety fallback.

Expected difficulty:

- Medium.

## 3. 2048

Observation space:

- State: 4x4 grid of log2 tile values.
- Vision: optional rendered board image.

Action space:

- `0 UP`
- `1 DOWN`
- `2 LEFT`
- `3 RIGHT`

Reward:

- Score delta from merges.
- Invalid move penalty.
- Game-over penalty optional.

Why useful:

- Different from reflex games: stochastic transitions, long-horizon planning, and value estimation.
- Good for DQN variants and expectimax/rule-based comparisons.

Baseline:

- Heuristic monotonicity/empty-cells policy or expectimax.

Expected difficulty:

- Medium-high.

## 4. Breakout Clone

Observation space:

- State: paddle x, ball x/y, ball velocity, brick map.
- Vision: Atari-like pixel frame stack.
- Hybrid: image plus ball/paddle state.

Action space:

- `0 NOOP`
- `1 LEFT`
- `2 RIGHT`

Reward:

- `+1` brick hit.
- `+bonus` level clear.
- `-1` life lost.

Why useful:

- Classic vision-control benchmark with delayed credit assignment.
- Directly comparable to Atari-style DQN design.

Baseline:

- Track ball x with paddle center.

Expected difficulty:

- Medium.

## 5. Custom Dino Curriculum Runner

Observation space:

- Same as Dino state/vision/hybrid modes.
- Explicit obstacle type, speed, and curriculum level.

Action space:

- `0 NOOP`
- `1 JUMP_PRESS`
- `2 DUCK_PRESS`
- `3 RELEASE`

Reward:

- Score delta.
- Obstacle cleared bonus.
- Crash penalty.
- Optional curriculum completion bonus.

Why useful:

- Provides controllable difficulty: cactus-only, variable gaps, birds, speed changes, visual noise.
- This should become the canonical benchmark environment instead of relying on `chrome://dino`.

Baseline:

- Rule-based thresholds per obstacle type and speed.

Expected difficulty:

- Low at early curriculum, high at full curriculum.

## Recommended Expansion Order

1. Custom Dino curriculum runner.
2. Flappy Bird.
3. Snake.
4. Breakout clone.
5. 2048.

This order keeps early work close to the current codebase while gradually adding new RL challenges.

