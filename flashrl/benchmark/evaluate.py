"""Evaluation harness for FlashRL policies."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flashrl.agents.baselines import RandomAgent, RuleBasedDinoAgent
from flashrl.artifacts import sha256_file
from flashrl.envs import DinoEnv
from flashrl.results import RESULT_FIELDS, RESULT_SCHEMA_VERSION
from flashrl.schemas import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_ID,
    ENVIRONMENT_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    SIMULATOR_VERSION,
)


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_agent(name: str, env: DinoEnv, seed: int | None = None, checkpoint: str | None = None):
    if name == "random":
        return RandomAgent(env.action_space, seed=seed)
    if name == "rule":
        return RuleBasedDinoAgent(action_mode=env.action_mode)
    if name == "dqn":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for --agent dqn")
        from flashrl.agents.dqn.train import DQNPolicy

        policy = DQNPolicy(checkpoint)
        policy.bind_env(env)
        return policy
    raise ValueError(f"Unsupported agent: {name}")


def evaluate_policy(
    agent,
    env: DinoEnv,
    episodes: int,
    eval_seed: int,
    run_id: str,
    agent_name: str,
    algorithm: str,
    phase: str,
    checkpoint_path: str = "",
    config_path: str = "",
    train_frames: int = 0,
    wall_clock_train_s: float = 0.0,
    identity: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity = identity or {}
    evaluation_started = time.perf_counter()
    for episode in range(episodes):
        episode_seed = eval_seed + episode
        episode_started = time.perf_counter()
        obs, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
        score = float(info.get("score", 0.0))
        rows.append(
            {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "evaluation_run_id": run_id,
                "training_run_id": identity.get("training_run_id", ""),
                "experiment_id": identity.get("experiment_id", f"{algorithm}-baseline"),
                "algorithm_id": identity.get("algorithm_id", algorithm),
                "hyperparameter_hash": identity.get("hyperparameter_hash", ""),
                "training_seed": identity.get("training_seed", ""),
                "evaluation_seed_base": eval_seed,
                "episode_seed": episode_seed,
                "training_git_commit": identity.get("training_git_commit", ""),
                "evaluation_git_commit": _git_commit(),
                "environment_id": ENVIRONMENT_ID,
                "environment_version": ENVIRONMENT_VERSION,
                "simulator_version": SIMULATOR_VERSION,
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "reward_schema_version": REWARD_SCHEMA_VERSION,
                "backend": env.backend,
                "obs_mode": env.obs_mode,
                "action_mode": env.action_mode,
                "agent": agent_name,
                "phase": phase,
                "episode": episode,
                "score": score,
                "survival_time_s": info.get("survival_time_s", 0.0),
                "steps": info.get("steps", 0),
                "obstacles_cleared": info.get("obstacles_cleared", 0),
                "death_type": info.get("death_type", "unknown"),
                "terminated": terminated,
                "truncated": truncated,
                "ending_reason": (
                    info.get("death_type", "terminated") if terminated else "time_limit"
                ),
                "train_frames": identity.get("train_frames", train_frames),
                "wall_clock_train_s": identity.get("wall_clock_train_s", wall_clock_train_s),
                "wall_clock_episode_s": time.perf_counter() - episode_started,
                "wall_clock_evaluation_total_s": 0.0,
                "checkpoint_role": identity.get("checkpoint_role", ""),
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": identity.get("checkpoint_sha256", ""),
                "manifest_path": identity.get("manifest_path", config_path),
            }
        )
    evaluation_total = time.perf_counter() - evaluation_started
    for row in rows:
        row["wall_clock_evaluation_total_s"] = evaluation_total
    return rows


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    episodes: int,
    eval_seed: int,
    phase: str = "eval",
) -> list[dict[str, Any]]:
    from flashrl.agents.dqn.train import DQNConfig, DQNPolicy, load_checkpoint

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = DQNConfig(**checkpoint["config"])
    env = DinoEnv(
        obs_mode=cfg.obs_mode,
        action_mode=cfg.action_mode,
        backend="sim",
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed,
    )
    policy = DQNPolicy(str(checkpoint_path))
    policy.bind_env(env)
    evaluation_run_id = datetime.now(timezone.utc).strftime("dqn_eval_%Y%m%dT%H%M%SZ")
    manifest_path = checkpoint_path.parent / "manifest.json"
    try:
        return evaluate_policy(
            policy,
            env,
            episodes=episodes,
            eval_seed=eval_seed,
            run_id=evaluation_run_id,
            agent_name="dqn",
            algorithm=checkpoint.get("experiment_id", "dqn"),
            phase=phase,
            checkpoint_path=str(checkpoint_path),
            identity={
                "training_run_id": checkpoint["run_id"],
                "experiment_id": checkpoint["experiment_id"],
                "algorithm_id": checkpoint["experiment_id"].rsplit("-", 1)[0],
                "hyperparameter_hash": checkpoint["hyperparameter_hash"],
                "training_seed": cfg.seed,
                "training_git_commit": checkpoint["git_commit"],
                "train_frames": checkpoint["train_frames"],
                "checkpoint_role": checkpoint["role"],
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "manifest_path": str(manifest_path),
            },
        )
    finally:
        env.close()


def write_results(rows: list[dict[str, Any]], csv_path: Path, jsonl_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = [float(row["score"]) for row in rows]
    steps = [float(row["steps"]) for row in rows]
    return {
        "episodes": len(rows),
        "mean_score": statistics.fmean(scores) if scores else 0.0,
        "median_score": statistics.median(scores) if scores else 0.0,
        "best_score": max(scores) if scores else 0.0,
        "mean_steps": statistics.fmean(steps) if steps else 0.0,
    }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent", choices=["random", "rule", "dqn"], default="rule")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--obs-mode", choices=["state", "vision", "hybrid"], default="state")
    parser.add_argument("--action-mode", choices=["minimal", "full"], default="full")
    parser.add_argument("--backend", choices=["sim"], default="sim")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--phase", default="eval")
    parser.add_argument("--out", default="results/eval.csv")
    parser.add_argument("--jsonl-out", default="")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FlashRL agents")
    configure_parser(parser)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.agent == "dqn":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for --agent dqn")
        rows = evaluate_checkpoint(
            args.checkpoint,
            episodes=args.episodes,
            eval_seed=args.eval_seed,
            phase=args.phase,
        )
    else:
        run_id = datetime.now(timezone.utc).strftime(f"{args.agent}_%Y%m%dT%H%M%SZ")
        env = DinoEnv(
            obs_mode=args.obs_mode,
            action_mode=args.action_mode,
            backend=args.backend,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        )
        agent = make_agent(args.agent, env, seed=args.seed)
        try:
            rows = evaluate_policy(
                agent,
                env,
                episodes=args.episodes,
                eval_seed=args.eval_seed,
                run_id=run_id,
                agent_name=args.agent,
                algorithm=args.agent,
                phase=args.phase,
            )
        finally:
            env.close()
    csv_path = Path(args.out)
    jsonl_path = Path(args.jsonl_out) if args.jsonl_out else csv_path.with_suffix(".jsonl")
    write_results(rows, csv_path, jsonl_path)
    print(json.dumps(summarize(rows), indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {jsonl_path}")
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
