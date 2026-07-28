"""DQN training and evaluation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from flashrl.artifacts import RunManifest, atomic_write_json, sha256_file
from flashrl.agents.dqn.networks import build_q_network
from flashrl.agents.dqn.replay import NStepBuffer, PrioritizedReplayBuffer, ReplayBuffer, Transition
from flashrl.benchmark.evaluate import evaluate_policy
from flashrl.envs import DinoEnv
from flashrl.identity import algorithm_id, hyperparameter_hash
from flashrl.schemas import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
)

CHECKPOINT_FORMAT_VERSION = 2


class CheckpointCompatibilityError(ValueError):
    """Raised before an incompatible checkpoint can construct a policy."""


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
    selection_interval_episodes: int = 10
    selection_episodes: int = 5
    selection_seed: int = 50_000
    device: str = "auto"
    output_dir: str = "runs"
    run_id: str | None = None


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception:
        return False


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


def compute_td_target(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    next_q: torch.Tensor,
    terminated: torch.Tensor,
) -> torch.Tensor:
    """Compute a target that bootstraps through time-limit truncations."""

    return rewards + discounts * next_q * (~terminated.bool()).float()


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    """Save a checkpoint without exposing a partially written canonical file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temporary_path = Path(fh.name)
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    expected = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
    }
    for field, expected_value in expected.items():
        actual = checkpoint.get(field)
        if actual != expected_value:
            name = field.replace("_schema_version", "").replace("_version", "")
            raise CheckpointCompatibilityError(
                f"Incompatible checkpoint {name}: expected {expected_value}, "
                f"found {actual!r}"
            )
    return checkpoint


class BestCheckpointTracker:
    def __init__(self, path: str | Path, best_score: float = -float("inf")) -> None:
        self.path = Path(path)
        self.best_score = float(best_score)

    def consider(self, score: float, payload: dict[str, Any]) -> bool:
        if score <= self.best_score:
            return False
        atomic_torch_save(self.path, payload)
        self.best_score = float(score)
        return True


def _checkpoint_payload(
    *,
    role: str,
    model,
    target_model,
    optimizer,
    cfg: DQNConfig,
    run_id: str,
    experiment_id: str,
    train_frames: int,
    episode: int,
    best_training_score: float,
    selection_score: float,
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "role": role,
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "hyperparameter_hash": experiment_id.rsplit("-", 1)[-1],
        "git_commit": git_commit(),
        "train_frames": train_frames,
        "episode": episode,
        "best_training_score": best_training_score,
        "selection_score": selection_score,
        "selection_seed": cfg.selection_seed,
        "selection_episodes": cfg.selection_episodes,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class _GreedyModelPolicy:
    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.device = device

    def act(self, obs) -> int:
        with torch.no_grad():
            q_values = self.model(obs_to_torch(obs, self.device))
        return int(q_values.argmax(dim=1).item())


def _selection_score(model, cfg: DQNConfig, device: torch.device) -> float:
    env = DinoEnv(
        obs_mode=cfg.obs_mode,
        action_mode=cfg.action_mode,
        backend="sim",
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.selection_seed,
    )
    policy = _GreedyModelPolicy(model, device)
    scores: list[float] = []
    model.eval()
    try:
        for offset in range(cfg.selection_episodes):
            obs, _ = env.reset(seed=cfg.selection_seed + offset)
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action = policy.act(obs)
                obs, _, terminated, truncated, info = env.step(action)
            scores.append(float(info["score"]))
    finally:
        env.close()
        model.train()
    return float(np.mean(scores))


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
    terminated = torch.tensor(
        [t.terminated for t in transitions], device=device, dtype=torch.bool
    )
    discounts = torch.tensor(
        [t.discount for t in transitions], device=device, dtype=torch.float32
    )
    weights_t = torch.tensor(weights, device=device, dtype=torch.float32)

    q = policy_net(obs).gather(1, actions).squeeze(1)
    with torch.no_grad():
        if cfg.double_dqn:
            next_actions = policy_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = target_net(next_obs).gather(1, next_actions).squeeze(1)
        else:
            next_q = target_net(next_obs).max(dim=1).values
        target = compute_td_target(rewards, discounts, next_q, terminated)
    td_error = target - q
    loss = (F.smooth_l1_loss(q, target, reduction="none") * weights_t).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    replay.update_priorities(indices, td_error.detach().abs().cpu().numpy())
    return float(loss.item())


def train_dqn(
    cfg: DQNConfig, resume_path: str | Path | None = None
) -> dict[str, Any]:
    set_seed(cfg.seed)
    run_id = cfg.run_id or datetime.now(timezone.utc).strftime("dqn_%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "train_metrics.csv"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"
    manifest_path = run_dir / "manifest.json"
    config_payload = asdict(cfg) | {"run_id": run_id}
    atomic_write_json(config_path, config_payload)
    identity_config = {
        key: value
        for key, value in config_payload.items()
        if key not in {"episodes", "output_dir", "run_id", "device"}
    }
    variant_id = algorithm_id(
        cfg.double_dqn,
        cfg.dueling,
        cfg.prioritized_replay,
        cfg.n_step,
    )
    config_hash = hyperparameter_hash(identity_config)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = RunManifest(
        run_id=run_id,
        experiment_id=f"{variant_id}-{config_hash}",
        algorithm_id=variant_id,
        hyperparameter_hash=config_hash,
        training_seed=cfg.seed,
        training_git_commit=git_commit(),
        git_dirty=git_dirty(),
        started_at=started_at,
        config=config_payload,
    )
    atomic_write_json(manifest_path, manifest.to_dict())

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
    env.action_space.seed(cfg.seed)
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
    start_episode = 0
    best_training_score = -float("inf")
    best_selection_score = -float("inf")
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        previous_cfg = DQNConfig(**checkpoint["config"])
        ignored_fields = {"episodes", "output_dir", "run_id", "device"}
        incompatible = [
            field
            for field in asdict(cfg)
            if field not in ignored_fields
            and getattr(cfg, field) != getattr(previous_cfg, field)
        ]
        if incompatible:
            raise CheckpointCompatibilityError(
                "Resume configuration differs in: " + ", ".join(incompatible)
            )
        policy_net.load_state_dict(checkpoint["model_state_dict"])
        target_net.load_state_dict(
            checkpoint.get("target_model_state_dict", checkpoint["model_state_dict"])
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        total_steps = int(checkpoint["train_frames"])
        start_episode = int(checkpoint["episode"]) + 1
        best_training_score = float(
            checkpoint.get("best_training_score", -float("inf"))
        )
        best_selection_score = float(
            checkpoint.get("selection_score", -float("inf"))
        )
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])

    best_tracker = BestCheckpointTracker(
        best_checkpoint_path, best_score=best_selection_score
    )
    started = time.time()
    metrics_mode = "a" if resume_path is not None else "w"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "episode",
                "reward",
                "score",
                "steps",
                "epsilon",
                "loss",
                "terminated",
                "truncated",
                "ending_reason",
                "wall_clock_s",
            ],
        )
        if metrics_mode == "w":
            writer.writeheader()
        for episode in range(start_episode, cfg.episodes):
            obs, _ = env.reset(seed=cfg.seed + episode)
            episode_reward = 0.0
            losses: list[float] = []
            done = False
            while not done:
                epsilon = epsilon_by_step(total_steps, cfg)
                action = select_action(policy_net, obs, env, epsilon, device)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                transition = Transition(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                    discount=cfg.gamma,
                )
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
                "terminated": terminated,
                "truncated": truncated,
                "ending_reason": (
                    info.get("death_type", "terminated")
                    if terminated
                    else "time_limit"
                ),
                "wall_clock_s": time.time() - started,
            }
            writer.writerow(row)
            fh.flush()
            best_training_score = max(best_training_score, float(row["score"]))
            should_select = (
                (episode + 1) % max(1, cfg.selection_interval_episodes) == 0
                or episode + 1 == cfg.episodes
            )
            if should_select:
                candidate_score = _selection_score(policy_net, cfg, device)
                candidate = _checkpoint_payload(
                    role="best",
                    model=policy_net,
                    target_model=target_net,
                    optimizer=optimizer,
                    cfg=cfg,
                    run_id=run_id,
                    experiment_id=manifest.experiment_id,
                    train_frames=total_steps,
                    episode=episode,
                    best_training_score=best_training_score,
                    selection_score=candidate_score,
                )
                best_tracker.consider(candidate_score, candidate)
            last_payload = _checkpoint_payload(
                role="last",
                model=policy_net,
                target_model=target_net,
                optimizer=optimizer,
                cfg=cfg,
                run_id=run_id,
                experiment_id=manifest.experiment_id,
                train_frames=total_steps,
                episode=episode,
                best_training_score=best_training_score,
                selection_score=best_tracker.best_score,
            )
            atomic_torch_save(last_checkpoint_path, last_payload)
    env.close()
    manifest.status = "completed"
    manifest.train_frames = total_steps
    manifest.wall_clock_train_s = time.time() - started
    manifest.completed_at = datetime.now(timezone.utc).isoformat()
    manifest.artifacts = {
        path.name: {"path": path.name, "sha256": sha256_file(path)}
        for path in (
            config_path,
            metrics_path,
            best_checkpoint_path,
            last_checkpoint_path,
        )
    }
    atomic_write_json(manifest_path, manifest.to_dict())
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "checkpoint_path": str(best_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "algorithm_id": variant_id,
        "train_frames": total_steps,
        "best_score": best_training_score,
        "best_selection_score": best_tracker.best_score,
    }


class DQNPolicy:
    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.model = None
        self.cfg = None

    def bind_env(self, env: DinoEnv) -> None:
        checkpoint = load_checkpoint(self.checkpoint_path, map_location=self.device)
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
