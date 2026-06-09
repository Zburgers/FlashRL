"""Stable-Baselines3 PPO training for state-mode comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from flashrl.envs import DinoEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on FlashRL Dino")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-mode", choices=["minimal", "full"], default="full")
    parser.add_argument("--backend", choices=["sim", "browser", "chrome"], default="sim")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--output-dir", default="runs/ppo")
    return parser.parse_args()


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for PPO. Install requirements.txt first.") from exc

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = DinoEnv(
        obs_mode="state",
        action_mode=args.action_mode,
        backend=args.backend,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(out_dir / "tensorboard"),
    )
    model.learn(total_timesteps=args.timesteps)
    model_path = out_dir / "ppo_dino_state.zip"
    model.save(model_path)
    env.close()
    print(f"wrote {model_path}")


if __name__ == "__main__":
    main()
