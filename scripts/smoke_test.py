"""Fast smoke checks for clean-clone verification."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gymnasium.utils.env_checker import check_env

from flashrl.agents.baselines import RuleBasedDinoAgent
from flashrl.benchmark.evaluate import evaluate_policy, summarize
from flashrl.envs import DinoEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlashRL smoke checks")
    parser.add_argument("--train-smoke", action="store_true", help="run a tiny DQN train loop")
    args = parser.parse_args()

    for obs_mode in ("state", "vision", "hybrid"):
        env = DinoEnv(obs_mode=obs_mode, backend="sim", max_episode_steps=5, seed=0)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)
        env.close()

    env = DinoEnv(obs_mode="state", backend="sim", max_episode_steps=20, seed=0)
    check_env(env, skip_render_check=True)
    agent = RuleBasedDinoAgent(action_mode=env.action_mode)
    rows = evaluate_policy(
        agent,
        env,
        episodes=2,
        eval_seed=100,
        run_id="smoke",
        agent_name="rule",
        algorithm="rule",
        phase="smoke",
    )
    env.close()
    print(summarize(rows))

    if args.train_smoke:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "flashrl.benchmark.train",
                "--episodes",
                "1",
                "--max-episode-steps",
                "20",
                "--warmup-steps",
                "1000",
                "--output-dir",
                "runs/smoke",
            ]
        )


if __name__ == "__main__":
    main()
