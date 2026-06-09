# FlashRL Environment API

## Objective

Define a stable Gymnasium-compatible browser-game API for Chrome Dino and future lightweight browser games.

## Required Gymnasium Contract

The environment must follow Gymnasium, not legacy Gym:

```python
obs, info = env.reset(seed=seed, options=options)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

`terminated` means the game reached an MDP terminal state, such as a crash. `truncated` means an external limit ended the episode, such as `max_episode_steps` or browser failure.

## Constructor

```python
class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_mode: str = "state",
        action_mode: str = "full",
        render_mode: str | None = None,
        headless: bool = True,
        game_url: str | None = None,
        max_episode_steps: int = 5000,
        fixed_timestep_ms: int = 50,
        seed: int | None = None,
    ):
        ...
```

## Observation Modes

### State Mode

Purpose: fast, low-dimensional algorithm development.

Observation space:

```python
spaces.Box(low=-np.inf, high=np.inf, shape=(N,), dtype=np.float32)
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
- `second_obstacle_distance`
- `score`

Do not include `crashed` in the observation. Put crash state in `terminated` and `info`.

### Vision Mode

Purpose: true pixel-based RL.

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

The environment must capture real frames using `page.screenshot()` or canvas extraction, not structured JS state reshaped into images.

### Hybrid Mode

Purpose: compare perception-heavy and state-assisted agents.

Observation space:

```python
spaces.Dict({
    "image": spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8),
    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(N,), dtype=np.float32),
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

Implementation:

- `JUMP_PRESS`: key down/up or press Space/ArrowUp depending on desired jump duration model.
- `DUCK_PRESS`: key down ArrowDown.
- `RELEASE`: key up Space/ArrowUp/ArrowDown.

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

## Browser Backend Requirements

- Default to local controlled HTML/JS game for deterministic benchmarks.
- Use `chrome://dino` only as an optional experimental backend.
- Inject or expose seedable RNG in local game JS.
- Avoid unbounded `time.sleep()` as the timing mechanism; use fixed step advancement when possible.
- Explicitly fail after reset retries rather than silently returning default observations.
- `close()` must be idempotent and close Playwright context/browser/process.

## Testing Requirements

- Environment checker for every observation mode.
- Reset returns observation inside observation space.
- Step returns observation inside observation space.
- `terminated` is true on crash.
- `truncated` is true on max steps.
- `seed` produces repeatable obstacle sequences in local backend.
- Browser crash maps to `truncated=True` with `info["error"]`.

