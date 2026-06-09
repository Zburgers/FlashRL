"""Simple baseline policies for Dino."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


class RandomAgent:
    def __init__(self, action_space, seed: int | None = None) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        return int(self.rng.integers(self.action_space.n))


@dataclass
class RuleBasedDinoAgent:
    """Heuristic for normalized state observations.

    State order is defined in ``flashrl.envs.dino_env.STATE_KEYS``.
    """

    action_mode: str = "full"

    def act(self, obs) -> int:
        state = obs["state"] if isinstance(obs, dict) else obs
        is_jumping = state[2] > 0.5
        is_ducking = state[3] > 0.5
        speed = max(0.1, state[4])
        distance = state[5]
        height = state[7]
        type_id = state[8]
        jump_threshold = 0.24 + 0.24 * speed

        if self.action_mode == "minimal":
            return int((not is_jumping) and distance < jump_threshold)

        is_bird = type_id > 0.75
        low_bird = is_bird and height < 0.5
        if low_bird and distance < jump_threshold:
            return 2
        if is_ducking and (not low_bird or distance > jump_threshold):
            return 3
        if (not is_jumping) and (not is_bird) and distance < jump_threshold:
            return 1
        return 0
