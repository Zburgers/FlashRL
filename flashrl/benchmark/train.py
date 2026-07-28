"""Training CLI wrapper."""

from __future__ import annotations

import argparse
import json

from flashrl.agents.dqn.train import DQNConfig, train_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FlashRL agents")
    parser.add_argument("--algorithm", choices=["dqn"], default="dqn")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", choices=["state", "vision", "hybrid"], default="state")
    parser.add_argument("--action-mode", choices=["minimal", "full"], default="full")
    parser.add_argument("--backend", choices=["sim"], default="sim")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-steps", type=int, default=500)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20000)
    parser.add_argument("--no-double-dqn", action="store_false", dest="double_dqn")
    parser.add_argument("--no-dueling", action="store_false", dest="dueling")
    parser.add_argument("--prioritized-replay", action="store_true")
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--selection-interval-episodes", type=int, default=10)
    parser.add_argument("--selection-episodes", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=50000)
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DQNConfig(
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        obs_mode=args.obs_mode,
        action_mode=args.action_mode,
        backend=args.backend,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        replay_size=args.replay_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        target_update_steps=args.target_update_steps,
        epsilon_decay_steps=args.epsilon_decay_steps,
        double_dqn=args.double_dqn,
        dueling=args.dueling,
        prioritized_replay=args.prioritized_replay,
        n_step=args.n_step,
        selection_interval_episodes=args.selection_interval_episodes,
        selection_episodes=args.selection_episodes,
        selection_seed=args.selection_seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    result = train_dqn(cfg, resume_path=args.resume or None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
