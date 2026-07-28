"""Local real-time simulator and HTTP server for the FlashRL demo."""

from __future__ import annotations

import json
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from flashrl.agents.baselines import RandomAgent, RuleBasedDinoAgent
from flashrl.envs import DinoEnv
from flashrl.schemas import ENVIRONMENT_ID


class DemoSession:
    """Own one simulator and expose bounded, JSON-safe telemetry."""

    def __init__(
        self,
        *,
        policy_name: str = "rule",
        checkpoint: str | Path | None = None,
        seed: int = 0,
        max_episode_steps: int = 2_000,
    ) -> None:
        if policy_name not in {"rule", "random", "dqn"}:
            raise ValueError(f"Unsupported demo policy: {policy_name}")
        self.policy_name = policy_name
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.seed = int(seed)
        self.episode = 0
        self.paused = False
        self.speed = 1.0
        self._lock = Lock()
        self._stop = Event()
        self._worker: Thread | None = None
        self._last_reward = 0.0
        self._last_action = 0
        self._terminated = False
        self._truncated = False
        self._reward_history: deque[float] = deque(maxlen=180)
        self._score_history: deque[float] = deque(maxlen=180)

        if policy_name == "dqn":
            if self.checkpoint is None:
                raise ValueError("A local checkpoint is required for policy_name='dqn'")
            from flashrl.agents.dqn.train import (
                DQNConfig,
                DQNPolicy,
                load_checkpoint,
            )

            payload = load_checkpoint(self.checkpoint)
            cfg = DQNConfig(**payload["config"])
            self.env = DinoEnv(
                obs_mode=cfg.obs_mode,
                action_mode=cfg.action_mode,
                max_episode_steps=max_episode_steps,
                seed=self.seed,
            )
            self.policy = DQNPolicy(str(self.checkpoint))
            self.policy.bind_env(self.env)
            self.policy_identity = {
                "name": "dqn",
                "algorithm_id": payload["experiment_id"].rsplit("-", 1)[0],
                "training_run_id": payload["run_id"],
                "checkpoint_role": payload["role"],
                "train_frames": payload["train_frames"],
            }
        else:
            self.env = DinoEnv(
                obs_mode="state",
                action_mode="full",
                max_episode_steps=max_episode_steps,
                seed=self.seed,
            )
            if policy_name == "rule":
                self.policy = RuleBasedDinoAgent(action_mode="full")
            else:
                self.policy = RandomAgent(self.env.action_space, seed=self.seed)
            self.policy_identity = {"name": policy_name}

        self._obs, _ = self.env.reset(seed=self.seed)

    def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = Thread(
            target=self._run,
            name="flashrl-demo-simulator",
            daemon=True,
        )
        self._worker.start()

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2)
        with self._lock:
            self.env.close()

    def reset(self, seed: int) -> dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not -(2**31) <= seed < 2**31:
            raise ValueError("seed must fit a signed 32-bit integer")
        with self._lock:
            self.seed = seed
            self.episode = 0
            self._reward_history.clear()
            self._score_history.clear()
            self._reset_episode()
            return {"seed": self.seed, "episode": self.episode}

    def control(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {"paused", "speed"}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("unsupported control fields: " + ", ".join(sorted(unknown)))
        with self._lock:
            if "paused" in payload:
                if not isinstance(payload["paused"], bool):
                    raise ValueError("paused must be a boolean")
                self.paused = payload["paused"]
            if "speed" in payload:
                speed = payload["speed"]
                if isinstance(speed, bool) or not isinstance(speed, (int, float)):
                    raise ValueError("speed must be numeric")
                if not 0.25 <= float(speed) <= 8.0:
                    raise ValueError("speed must be between 0.25 and 8")
                self.speed = float(speed)
            return {"paused": self.paused, "speed": self.speed}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "environment_id": ENVIRONMENT_ID,
                "policy": self.policy_identity,
                "seed": self.seed,
                "episode": self.episode,
                "paused": self.paused,
                "speed": self.speed,
                "action_names": self.env.action_names,
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            q_values = None
            if hasattr(self.policy, "q_values"):
                q_values = self.policy.q_values(self._obs)
            return {
                "episode": self.episode,
                "seed": self.seed + self.episode,
                "step": self.env.steps,
                "score": round(float(self.env.score), 3),
                "speed": round(float(self.env.game_speed), 3),
                "survival_time_s": round(float(self.env.elapsed_s), 3),
                "dino": {
                    "x": 50.0,
                    "y": float(self.env.trex_y),
                    "width": 28.0,
                    "height": 28.0 if self.env.is_ducking else 44.0,
                    "jumping": self.env.is_jumping,
                    "ducking": self.env.is_ducking,
                },
                "obstacles": [
                    {
                        "x": round(float(obstacle.x), 3),
                        "y": float(obstacle.y),
                        "width": float(obstacle.width),
                        "height": float(obstacle.height),
                        "type": "bird" if obstacle.type_id == 2 else "cactus",
                    }
                    for obstacle in self.env.obstacles
                ],
                "action": {
                    "id": self._last_action,
                    "name": self.env.action_names[self._last_action],
                },
                "reward": round(float(self._last_reward), 6),
                "q_values": q_values,
                "reward_history": list(self._reward_history),
                "score_history": list(self._score_history),
                "terminated": self._terminated,
                "truncated": self._truncated,
                "ending_reason": (
                    self.env.death_type
                    if self._terminated
                    else "time_limit"
                    if self._truncated
                    else "running"
                ),
                "policy": self.policy_identity,
            }

    def _reset_episode(self) -> None:
        self._obs, _ = self.env.reset(seed=self.seed + self.episode)
        self._last_reward = 0.0
        self._last_action = 0
        self._terminated = False
        self._truncated = False

    def _advance(self) -> None:
        if self._terminated or self._truncated:
            self.episode += 1
            self._reset_episode()
        self._last_action = int(self.policy.act(self._obs))
        (
            self._obs,
            self._last_reward,
            self._terminated,
            self._truncated,
            _,
        ) = self.env.step(self._last_action)
        self._reward_history.append(round(float(self._last_reward), 6))
        self._score_history.append(round(float(self.env.score), 3))

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if not self.paused:
                    self._advance()
                delay = 1.0 / (30.0 * self.speed)
            self._stop.wait(delay)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, session: DemoSession):
        self.session = session
        super().__init__(address, handler)


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._json(self.server.session.status())
            return
        if self.path == "/api/frame":
            self._json(self.server.session.snapshot())
            return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        asset = assets.get(self.path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        name, content_type = asset
        body = files("flashrl.demo.static").joinpath(name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/reset":
                response = self.server.session.reset(payload.get("seed"))
            elif self.path == "/api/control":
                response = self.server.session.control(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._json(response)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 4096:
            raise ValueError("request body must contain 1 to 4096 bytes")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(
    session: DemoSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> DemoServer:
    return DemoServer((host, port), DemoRequestHandler, session)


def run_demo(
    *,
    policy_name: str,
    checkpoint: str | Path | None,
    seed: int,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    session = DemoSession(
        policy_name=policy_name,
        checkpoint=checkpoint,
        seed=seed,
    )
    server = create_server(session, host=host, port=port)
    url = f"http://{host}:{server.server_port}"
    print(f"FlashRL live demo: {url}")
    session.start()
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        session.close()
