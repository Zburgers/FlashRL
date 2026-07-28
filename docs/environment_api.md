# FlashRL Environment API

## Objective

Define the stable Gymnasium API for the deterministic FlashRL Dino simulator.
FlashRL V2 intentionally supports only `backend="sim"`.

## Required Gymnasium Contract

The environment must follow Gymnasium, not legacy Gym:

```python
obs, info = env.reset(seed=seed, options=options)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

`terminated` means the game reached an MDP terminal state, such as a crash.
`truncated` means the configured time limit ended the episode. Internal
simulator failures raise immediately and never become training transitions.

## Constructor

```python
class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_mode: str = "state",
        action_mode: str = "full",
        render_mode: str | None = None,
        max_episode_steps: int = 5000,
        fixed_timestep_ms: int = 50,
        seed: int | None = None,
        backend: str = "sim",
    ):
        ...
```

## Observation Modes

### State Mode

Purpose: fast, low-dimensional algorithm development.

Observation space:

```python
spaces.Box(low=STATE_LOW, high=STATE_HIGH, shape=(12,), dtype=np.float32)
```

Recommended normalized fields:

- `trex_y`
- `trex_velocity_y`
- `is_jumping`
- `is_ducking`
- `game_speed`
- `distance_to_next_obstacle`
- `next_obstacle_width`
- `next_obstacle_height`
- `next_obstacle_type_id`
- `next_obstacle_bottom`
- `next_obstacle_top`
- `second_obstacle_distance`

Obstacle altitude prevents perceptual aliasing between low birds that require
ducking and high birds that are safe to ignore. Raw score and `crashed` are not
policy inputs: score belongs in `info`, while crashes use `terminated`.

### Vision Mode

Purpose: pixel-based RL over deterministic simulator renderings.

Observation space:

```python
spaces.Box(low=0, high=255, shape=(frame_stack, height, width), dtype=np.uint8)
```

Recommended defaults:

- grayscale
- cropped game area
- `84x84`
- frame stack 4
- deterministic preprocessing in a wrapper

Frames are rendered from simulator geometry. They are real image observations,
not structured state reshaped into an image tensor.

### Hybrid Mode

Purpose: compare perception-heavy and state-assisted agents.

Observation space:

```python
spaces.Dict({
    "image": spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),
    "state": spaces.Box(low=STATE_LOW, high=STATE_HIGH, shape=(12,), dtype=np.float32),
})
```

Use separate model encoders for image and state features, concatenate embeddings, then feed policy/Q heads.

## Action Modes

### Minimal

```text
0 NOOP
1 JUMP
```

Only acceptable for early cactus-only tests. It is not enough for full Dino because birds and jump-duration effects matter.

### Full

```text
0 NOOP
1 JUMP_PRESS
2 DUCK_PRESS
3 RELEASE
```

Simulator implementation:

- `JUMP_PRESS`: starts a jump when grounded.
- `DUCK_PRESS`: enters ducking while grounded.
- `RELEASE`: exits ducking.

### Factored Future Option

Use `MultiDiscrete([2, 2])` for jump-held and duck-held. This may better represent continuous key holds, but most baseline algorithms are simpler with `Discrete(4)`.

## Reward API

Return scalar reward and expose terms in `info`:

```python
info["reward_terms"] = {
    "survival": ...,
    "score_delta": ...,
    "obstacle_cleared": ...,
    "crash": ...,
}
```

Recommended starting reward:

```text
reward = 0.01 * score_delta + 1.0 * obstacle_cleared - 10.0 * crashed
```

Avoid pure per-step survival rewards unless timestep is fixed and logged.

## Info Schema

Each step should include:

- `score`
- `raw_score`
- `survival_time_s`
- `steps`
- `game_speed`
- `obstacles_cleared`
- `next_obstacle_type`
- `death_type`
- `backend`
- `obs_mode`
- `action_mode`
- `seed`

## Version Identity

V2 artifacts record the environment, simulator, observation, action, and reward
schema versions from `flashrl.schemas`. Checkpoints from incompatible schemas
must fail before inference.

## Testing Requirements

- Environment checker for every observation mode.
- Reset returns observation inside observation space.
- Step returns observation inside observation space.
- `terminated` is true on crash.
- `truncated` is true on max steps.
- A seed and complete action sequence produce the same full trajectory.
- Long successful trajectories remain inside every declared observation space.
- Low and high birds produce distinct state observations.
- Simulator exceptions propagate and abort training.
