"""DQN training and evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from flashrl.agents.dqn.networks import build_q_network
from flashrl.agents.dqn.replay import NStepBuffer, PrioritizedReplayBuffer, ReplayBuffer, Transition
from flashrl.benchmark.evaluate import evaluate_policy
from flashrl.envs import DinoEnv


@dataclass
class DQNConfig:
    episodes: int = 50
    max_episode_steps: int = 1000
    seed: int = 0
    obs_mode: str = "state"
    action_mode: str = "full"
    backend: str = "sim"
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_size: int = 50_000
    warmup_steps: int = 500
    target_update_steps: int = 500
    train_every: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    double_dqn: bool = True
    dueling: bool = True
    prioritized_replay: bool = False
    n_step: int = 1
    device: str = "auto"
    output_dir: str = "runs"
    run_id: str | None = None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_to_torch(obs: Any, device: torch.device) -> Any:
    if isinstance(obs, dict):
        return {k: obs_to_torch(v, device) for k, v in obs.items()}
    tensor = torch.as_tensor(obs, device=device)
    if tensor.ndim == 1:
        return tensor.float().unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def batch_obs(observations: list[Any], device: torch.device) -> Any:
    first = observations[0]
    if isinstance(first, dict):
        return {k: batch_obs([obs[k] for obs in observations], device) for k in first}
    return torch.as_tensor(np.stack(observations), device=device)


def epsilon_by_step(step: int, cfg: DQNConfig) -> float:
    frac = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(model, obs, env, epsilon: float, device: torch.device) -> int:
    if random.random() < epsilon:
        return int(env.action_space.sample())
    with torch.no_grad():
        q_values = model(obs_to_torch(obs, device))
    return int(q_values.argmax(dim=1).item())


def optimize(
    policy_net,
    target_net,
    optimizer,
    replay,
    cfg: DQNConfig,
    device: torch.device,
) -> float | None:
    if len(replay) < max(cfg.batch_size, cfg.warmup_steps):
        return None
    transitions, indices, weights = replay.sample(cfg.batch_size)
    obs = batch_obs([t.obs for t in transitions], device)
    next_obs = batch_obs([t.next_obs for t in transitions], device)
    actions = torch.tensor([t.action for t in transitions], device=device, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor([t.reward for t in transitions], device=device, dtype=torch.float32)
    dones = torch.tensor([t.done for t in transitions], device=device, dtype=torch.float32)
    weights_t = torch.tensor(weights, device=device, dtype=torch.float32)

    q = policy_net(obs).gather(1, actions).squeeze(1)
    with torch.no_grad():
        if cfg.double_dqn:
            next_actions = policy_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = target_net(next_obs).gather(1, next_actions).squeeze(1)
        else:
            next_q = target_net(next_obs).max(dim=1).values
        target = rewards + (cfg.gamma**cfg.n_step) * next_q * (1.0 - dones)
    td_error = target - q
    loss = (F.smooth_l1_loss(q, target, reduction="none") * weights_t).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    replay.update_priorities(indices, td_error.detach().abs().cpu().numpy())
    return float(loss.item())


def train_dqn(cfg: DQNConfig) -> dict[str, Any]:
    set_seed(cfg.seed)
    run_id = cfg.run_id or datetime.now(timezone.utc).strftime("dqn_%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "train_metrics.csv"
    checkpoint_path = run_dir / "checkpoint.pt"
    config_path.write_text(json.dumps(asdict(cfg) | {"run_id": run_id}, indent=2), encoding="utf-8")

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    env = DinoEnv(
        obs_mode=cfg.obs_mode,
        action_mode=cfg.action_mode,
        backend=cfg.backend,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed,
    )
    policy_net = build_q_network(env.observation_space, env.action_space.n, cfg.obs_mode, cfg.dueling).to(device)
    target_net = build_q_network(env.observation_space, env.action_space.n, cfg.obs_mode, cfg.dueling).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.learning_rate)
    replay = (
        PrioritizedReplayBuffer(cfg.replay_size, seed=cfg.seed)
        if cfg.prioritized_replay
        else ReplayBuffer(cfg.replay_size, seed=cfg.seed)
    )
    n_step = NStepBuffer(cfg.n_step, cfg.gamma)

    total_steps = 0
    best_score = -float("inf")
    started = time.time()
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["episode", "reward", "score", "steps", "epsilon", "loss", "wall_clock_s"],
        )
        writer.writeheader()
        for episode in range(cfg.episodes):
            obs, _ = env.reset(seed=cfg.seed + episode)
            episode_reward = 0.0
            losses: list[float] = []
            done = False
            while not done:
                epsilon = epsilon_by_step(total_steps, cfg)
                action = select_action(policy_net, obs, env, epsilon, device)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                transition = Transition(obs, action, reward, next_obs, done)
                ready_transition = n_step.append(transition)
                if ready_transition is not None:
                    replay.push(ready_transition)
                if total_steps % cfg.train_every == 0:
                    loss = optimize(policy_net, target_net, optimizer, replay, cfg, device)
                    if loss is not None:
                        losses.append(loss)
                if total_steps % cfg.target_update_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                obs = next_obs
                episode_reward += reward
                total_steps += 1
            for flushed in n_step.flush():
                replay.push(flushed)
            row = {
                "episode": episode,
                "reward": episode_reward,
                "score": info.get("score", 0.0),
                "steps": info.get("steps", 0),
                "epsilon": epsilon_by_step(total_steps, cfg),
                "loss": float(np.mean(losses)) if losses else "",
                "wall_clock_s": time.time() - started,
            }
            writer.writerow(row)
            fh.flush()
            if row["score"] > best_score:
                best_score = float(row["score"])
                save_checkpoint(checkpoint_path, policy_net, optimizer, cfg, run_id, total_steps, best_score)
    env.close()
    save_checkpoint(checkpoint_path, policy_net, optimizer, cfg, run_id, total_steps, best_score)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "train_frames": total_steps,
        "best_score": best_score,
    }


def save_checkpoint(path, model, optimizer, cfg: DQNConfig, run_id: str, train_frames: int, best_score: float) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(cfg),
            "run_id": run_id,
            "git_commit": git_commit(),
            "train_frames": train_frames,
            "best_score": best_score,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


class DQNPolicy:
    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.model = None
        self.cfg = None

    def bind_env(self, env: DinoEnv) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.cfg = DQNConfig(**checkpoint["config"])
        self.model = build_q_network(
            env.observation_space,
            env.action_space.n,
            self.cfg.obs_mode,
            self.cfg.dueling,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def act(self, obs) -> int:
        if self.model is None:
            raise RuntimeError("DQNPolicy must be bound to an environment before use")
        with torch.no_grad():
            q_values = self.model(obs_to_torch(obs, self.device))
        return int(q_values.argmax(dim=1).item())
