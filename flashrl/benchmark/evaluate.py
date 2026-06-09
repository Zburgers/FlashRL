"""Evaluation harness for FlashRL policies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any

from flashrl.agents.baselines import RandomAgent, RuleBasedDinoAgent
from flashrl.envs import DinoEnv


RESULT_FIELDS = [
    "run_id",
    "git_commit",
    "seed",
    "eval_seed",
    "game",
    "backend",
    "obs_mode",
    "action_mode",
    "agent",
    "algorithm",
    "phase",
    "episode",
    "score",
    "median_score_so_far",
    "survival_time_s",
    "steps",
    "obstacles_cleared",
    "death_type",
    "terminated",
    "truncated",
    "train_frames",
    "wall_clock_train_s",
    "wall_clock_eval_s",
    "checkpoint_path",
    "config_path",
]


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    started = time.time()
    for episode in range(episodes):
        obs, _ = env.reset(seed=eval_seed + episode)
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
        score = float(info.get("score", 0.0))
        scores.append(score)
        rows.append(
            {
                "run_id": run_id,
                "git_commit": _git_commit(),
                "seed": env._seed,
                "eval_seed": eval_seed,
                "game": "dino",
                "backend": env.backend,
                "obs_mode": env.obs_mode,
                "action_mode": env.action_mode,
                "agent": agent_name,
                "algorithm": algorithm,
                "phase": phase,
                "episode": episode,
                "score": score,
                "median_score_so_far": statistics.median(scores),
                "survival_time_s": info.get("survival_time_s", 0.0),
                "steps": info.get("steps", 0),
                "obstacles_cleared": info.get("obstacles_cleared", 0),
                "death_type": info.get("death_type", "unknown"),
                "terminated": terminated,
                "truncated": truncated,
                "train_frames": train_frames,
                "wall_clock_train_s": wall_clock_train_s,
                "wall_clock_eval_s": time.time() - started,
                "checkpoint_path": checkpoint_path,
                "config_path": config_path,
            }
        )
    return rows


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FlashRL agents")
    parser.add_argument("--agent", choices=["random", "rule", "dqn"], default="rule")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--obs-mode", choices=["state", "vision", "hybrid"], default="state")
    parser.add_argument("--action-mode", choices=["minimal", "full"], default="full")
    parser.add_argument("--backend", choices=["sim", "browser", "chrome"], default="sim")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--phase", default="eval")
    parser.add_argument("--out", default="results/eval.csv")
    parser.add_argument("--jsonl-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime(f"{args.agent}_%Y%m%dT%H%M%SZ")
    env = DinoEnv(
        obs_mode=args.obs_mode,
        action_mode=args.action_mode,
        backend=args.backend,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    agent = make_agent(args.agent, env, seed=args.seed, checkpoint=args.checkpoint or None)
    rows = evaluate_policy(
        agent,
        env,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        run_id=run_id,
        agent_name=args.agent,
        algorithm=args.agent,
        phase=args.phase,
        checkpoint_path=args.checkpoint,
    )
    env.close()
    csv_path = Path(args.out)
    jsonl_path = Path(args.jsonl_out) if args.jsonl_out else csv_path.with_suffix(".jsonl")
    write_results(rows, csv_path, jsonl_path)
    print(json.dumps(summarize(rows), indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {jsonl_path}")


if __name__ == "__main__":
    main()
