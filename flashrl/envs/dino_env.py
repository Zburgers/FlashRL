"""Gymnasium-compatible Chrome Dino benchmark environment.

The default backend is a deterministic Python simulator. It mirrors the core
endless-runner mechanics closely enough for fast training, CI, and reproducible
baselines. Browser-backed play is optional because it requires Playwright
browsers and a display/headless Chromium runtime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import time
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw

ObsMode = Literal["state", "vision", "hybrid"]
ActionMode = Literal["minimal", "full"]
Backend = Literal["sim", "browser", "chrome"]


STATE_KEYS = (
    "trex_y",
    "trex_velocity_y",
    "is_jumping",
    "is_ducking",
    "game_speed",
    "distance_to_next_obstacle",
    "next_obstacle_width",
    "next_obstacle_height",
    "next_obstacle_type_id",
    "second_obstacle_distance",
    "score",
)


@dataclass
class Obstacle:
    x: float
    width: float
    height: float
    type_id: int
    y: float = 0.0
    cleared: bool = False


class DinoEnv(gym.Env):
    """Dino runner environment with state, vision, and hybrid observations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_mode: ObsMode = "state",
        action_mode: ActionMode = "full",
        render_mode: str | None = None,
        headless: bool = True,
        game_url: str | None = None,
        max_episode_steps: int = 5000,
        fixed_timestep_ms: int = 50,
        seed: int | None = None,
        backend: Backend = "sim",
        frame_stack: int = 4,
        frame_size: tuple[int, int] = (84, 84),
    ) -> None:
        super().__init__()
        if obs_mode not in {"state", "vision", "hybrid"}:
            raise ValueError(f"Unsupported obs_mode: {obs_mode}")
        if action_mode not in {"minimal", "full"}:
            raise ValueError(f"Unsupported action_mode: {action_mode}")
        if backend not in {"sim", "browser", "chrome"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.render_mode = render_mode
        self.headless = headless
        self.game_url = game_url
        self.max_episode_steps = int(max_episode_steps)
        self.fixed_timestep_ms = int(fixed_timestep_ms)
        self.backend = backend
        self.frame_stack = int(frame_stack)
        self.frame_size = frame_size
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._closed = False
        self._browser_objects: dict[str, Any] = {}

        self.action_space = spaces.Discrete(2 if action_mode == "minimal" else 4)
        self.observation_space = self._make_observation_space()
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self._reset_sim_state()

    @property
    def action_names(self) -> tuple[str, ...]:
        if self.action_mode == "minimal":
            return ("NOOP", "JUMP")
        return ("NOOP", "JUMP_PRESS", "DUCK_PRESS", "RELEASE")

    def _make_observation_space(self) -> spaces.Space:
        state_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(STATE_KEYS),),
            dtype=np.float32,
        )
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_stack, self.frame_size[0], self.frame_size[1]),
            dtype=np.uint8,
        )
        if self.obs_mode == "state":
            return state_space
        if self.obs_mode == "vision":
            return image_space
        return spaces.Dict({"image": image_space, "state": state_space})

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
        if options:
            self._apply_reset_options(options)
        self._reset_sim_state()
        if self.backend in {"browser", "chrome"}:
            self._ensure_browser()
            self._reset_browser()
        obs = self._observe()
        return obs, self._info(reward_terms={})

    def step(
        self, action: int
    ) -> tuple[np.ndarray | dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("DinoEnv is closed")
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            raise ValueError(f"Invalid action {action} for {self.action_space}")

        try:
            if self.backend in {"browser", "chrome"}:
                self._send_browser_action(action)
            reward, terminated, reward_terms = self._step_sim(action)
            obs = self._observe()
            truncated = self.steps >= self.max_episode_steps
            info = self._info(reward_terms=reward_terms)
            info["action_name"] = self.action_names[action]
            return obs, float(reward), bool(terminated), bool(truncated), info
        except Exception as exc:
            obs = self._observe()
            info = self._info(reward_terms={})
            info["error"] = str(exc)
            info["death_type"] = "browser_failure" if self.backend != "sim" else "sim_failure"
            return obs, -10.0, False, True, info

    def render(self) -> np.ndarray | None:
        frame = self._render_frame()
        if self.render_mode == "rgb_array":
            return np.repeat(frame[..., None], 3, axis=2)
        return None

    def close(self) -> None:
        if self._closed:
            return
        page = self._browser_objects.get("page")
        context = self._browser_objects.get("context")
        browser = self._browser_objects.get("browser")
        playwright = self._browser_objects.get("playwright")
        for obj in (page, context, browser):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        try:
            if playwright is not None:
                playwright.stop()
        except Exception:
            pass
        self._browser_objects.clear()
        self._closed = True

    def _apply_reset_options(self, options: dict[str, Any]) -> None:
        if "max_episode_steps" in options:
            self.max_episode_steps = int(options["max_episode_steps"])

    def _reset_sim_state(self) -> None:
        self.steps = 0
        self.elapsed_s = 0.0
        self.score = 0.0
        self.prev_score = 0.0
        self.trex_y = 0.0
        self.trex_vy = 0.0
        self.is_jumping = False
        self.is_ducking = False
        self.game_speed = 6.0
        self.obstacles_cleared = 0
        self.death_type = "none"
        self.obstacles: list[Obstacle] = []
        self._next_spawn_x = 420.0
        self._spawn_obstacle(initial=True)
        self._frames.clear()
        frame = self._render_frame()
        for _ in range(self.frame_stack):
            self._frames.append(frame)

    def _spawn_obstacle(self, initial: bool = False) -> None:
        spacing = float(self._rng.integers(220, 420))
        if initial:
            x = self._next_spawn_x
        else:
            last_x = max((obs.x for obs in self.obstacles), default=300.0)
            x = max(last_x + spacing, 600.0 + spacing * 0.25)
        roll = float(self._rng.random())
        if roll < 0.72:
            width = float(self._rng.integers(16, 36))
            height = float(self._rng.integers(30, 54))
            type_id = 1
            y = 0.0
        else:
            width = 34.0
            height = 24.0
            type_id = 2
            y = float(self._rng.choice([36.0, 68.0]))
        self.obstacles.append(Obstacle(x=x, width=width, height=height, type_id=type_id, y=y))

    def _step_sim(self, action: int) -> tuple[float, bool, dict[str, float]]:
        dt = self.fixed_timestep_ms / 1000.0
        self.steps += 1
        self.elapsed_s += dt
        self.prev_score = self.score
        self.game_speed = min(13.0, 6.0 + self.score / 900.0)

        if self.action_mode == "minimal":
            if action == 1 and not self.is_jumping:
                self.trex_vy = 12.0
                self.is_jumping = True
            self.is_ducking = False
        else:
            if action == 1 and not self.is_jumping:
                self.trex_vy = 12.0
                self.is_jumping = True
                self.is_ducking = False
            elif action == 2 and not self.is_jumping:
                self.is_ducking = True
            elif action == 3:
                self.is_ducking = False

        if self.is_jumping:
            self.trex_y += self.trex_vy
            self.trex_vy -= 0.72
            if self.trex_y <= 0:
                self.trex_y = 0.0
                self.trex_vy = 0.0
                self.is_jumping = False

        dx = self.game_speed * self.fixed_timestep_ms / 16.67
        for obstacle in self.obstacles:
            obstacle.x -= dx
        self.obstacles = [obs for obs in self.obstacles if obs.x + obs.width > -20]
        while len(self.obstacles) < 3:
            self._spawn_obstacle()

        obstacle_cleared = 0.0
        for obstacle in self.obstacles:
            if not obstacle.cleared and obstacle.x + obstacle.width < 50:
                obstacle.cleared = True
                self.obstacles_cleared += 1
                obstacle_cleared += 1.0

        self.score += self.game_speed * dt * 10.0
        score_delta = self.score - self.prev_score
        crashed = self._detect_collision()
        crash_penalty = -10.0 if crashed else 0.0
        reward = 0.01 * score_delta + obstacle_cleared + crash_penalty
        reward_terms = {
            "survival": 0.0,
            "score_delta": 0.01 * score_delta,
            "obstacle_cleared": obstacle_cleared,
            "crash": crash_penalty,
        }
        self._frames.append(self._render_frame())
        return reward, crashed, reward_terms

    def _detect_collision(self) -> bool:
        trex_x = 50.0
        trex_w = 28.0
        trex_h = 28.0 if self.is_ducking else 44.0
        trex_bottom = self.trex_y
        trex_top = self.trex_y + trex_h
        for obstacle in self.obstacles:
            overlap_x = trex_x < obstacle.x + obstacle.width and trex_x + trex_w > obstacle.x
            if not overlap_x:
                continue
            if obstacle.type_id == 1:
                if trex_bottom < obstacle.height:
                    self.death_type = "late_jump" if not self.is_jumping else "early_jump"
                    return True
            else:
                bird_bottom = obstacle.y
                bird_top = obstacle.y + obstacle.height
                overlap_y = trex_bottom < bird_top and trex_top > bird_bottom
                if overlap_y and not self.is_ducking:
                    self.death_type = "bird_no_duck"
                    return True
                if overlap_y and self.is_ducking and obstacle.y < 45:
                    self.death_type = "duck_collision"
                    return True
        self.death_type = "none"
        return False

    def _state_vector(self) -> np.ndarray:
        first = self.obstacles[0] if self.obstacles else Obstacle(600.0, 0.0, 0.0, 0)
        second = self.obstacles[1] if len(self.obstacles) > 1 else Obstacle(900.0, 0.0, 0.0, 0)
        values = np.array(
            [
                self.trex_y / 120.0,
                self.trex_vy / 20.0,
                float(self.is_jumping),
                float(self.is_ducking),
                self.game_speed / 13.0,
                first.x / 600.0,
                first.width / 60.0,
                first.height / 80.0,
                first.type_id / 2.0,
                second.x / 900.0,
                self.score / 1000.0,
            ],
            dtype=np.float32,
        )
        return values

    def _observe(self) -> np.ndarray | dict[str, np.ndarray]:
        state = self._state_vector()
        if self.obs_mode == "state":
            return state
        image = np.stack(tuple(self._frames), axis=0).astype(np.uint8)
        if self.obs_mode == "vision":
            return image
        return {"image": image, "state": state}

    def _render_frame(self) -> np.ndarray:
        height, width = self.frame_size
        img = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(img)
        ground_y = height - 14
        draw.line((0, ground_y, width, ground_y), fill=90, width=1)
        trex_x = 8
        trex_h = 12 if self.is_ducking else 20
        trex_y = ground_y - trex_h - int(self.trex_y * 0.45)
        draw.rectangle((trex_x, trex_y, trex_x + 8, trex_y + trex_h), fill=30)
        for obstacle in self.obstacles:
            ox = int(obstacle.x / 600.0 * width)
            ow = max(2, int(obstacle.width / 600.0 * width))
            if obstacle.type_id == 1:
                oh = max(4, int(obstacle.height / 80.0 * 28))
                draw.rectangle((ox, ground_y - oh, ox + ow, ground_y), fill=20)
            else:
                oy = ground_y - int(obstacle.y * 0.45) - 18
                draw.rectangle((ox, oy, ox + ow, oy + 8), fill=45)
        return np.asarray(img, dtype=np.uint8)

    def _info(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        first = self.obstacles[0] if self.obstacles else None
        return {
            "score": float(self.score),
            "raw_score": float(self.score),
            "survival_time_s": float(self.elapsed_s),
            "steps": int(self.steps),
            "game_speed": float(self.game_speed),
            "obstacles_cleared": int(self.obstacles_cleared),
            "next_obstacle_type": "bird" if first and first.type_id == 2 else "cactus",
            "death_type": self.death_type,
            "backend": self.backend,
            "obs_mode": self.obs_mode,
            "action_mode": self.action_mode,
            "seed": self._seed,
            "reward_terms": reward_terms,
        }

    def _ensure_browser(self) -> None:
        if self._browser_objects:
            return
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=self.headless)
        context = browser.new_context(viewport={"width": 800, "height": 450})
        page = context.new_page()
        url = self.game_url
        if url is None and self.backend == "browser":
            url = Path("assets/dino_game.html").resolve().as_uri()
        elif url is None:
            url = "chrome://dino"
        page.goto(url)
        self._browser_objects = {
            "playwright": playwright,
            "browser": browser,
            "context": context,
            "page": page,
        }

    def _reset_browser(self) -> None:
        page = self._browser_objects.get("page")
        if page is None:
            return
        try:
            page.keyboard.press("Space")
            time.sleep(0.2)
        except Exception:
            pass

    def _send_browser_action(self, action: int) -> None:
        page = self._browser_objects.get("page")
        if page is None:
            return
        if self.action_names[action] in {"JUMP", "JUMP_PRESS"}:
            page.keyboard.press("Space")
        elif self.action_names[action] == "DUCK_PRESS":
            page.keyboard.down("ArrowDown")
        elif self.action_names[action] == "RELEASE":
            page.keyboard.up("ArrowDown")
            page.keyboard.up("Space")
