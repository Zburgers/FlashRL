"""Simple baseline policies for Dino."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from flashrl.envs.dino_env import STATE_INDEX
from flashrl.schemas import DUCK, JUMP, NOOP, RELEASE


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
        is_jumping = state[STATE_INDEX["is_jumping"]] > 0.5
        is_ducking = state[STATE_INDEX["is_ducking"]] > 0.5
        speed = max(0.1, state[STATE_INDEX["game_speed"]])
        distance = state[STATE_INDEX["distance_to_next_obstacle"]]
        type_id = state[STATE_INDEX["next_obstacle_type_id"]]
        obstacle_bottom = state[STATE_INDEX["next_obstacle_bottom"]]
        jump_threshold = 0.24 + 0.24 * speed

        if self.action_mode == "minimal":
            return JUMP if (not is_jumping) and distance < jump_threshold else NOOP

        is_bird = type_id > 0.75
        low_bird = is_bird and obstacle_bottom < 0.6
        if low_bird and distance < jump_threshold:
            return DUCK
        if is_ducking and (not low_bird or distance > jump_threshold):
            return RELEASE
        if (not is_jumping) and (not is_bird) and distance < jump_threshold:
            return JUMP
        return NOOP
