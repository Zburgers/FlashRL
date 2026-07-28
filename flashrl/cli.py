"""Canonical command-line interface for FlashRL."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from flashrl import __version__
from flashrl.benchmark import aggregate, evaluate, train
from flashrl.doctor import collect_diagnostics


def _run_doctor(args: argparse.Namespace) -> dict:
    diagnostics = collect_diagnostics(args.artifact_dir)
    print(json.dumps(diagnostics, indent=2))
    return diagnostics


def _run_demo(args: argparse.Namespace) -> None:
    from flashrl.demo.server import record_demo, run_demo

    if args.record:
        record_demo(
            policy_name=args.policy,
            checkpoint=args.checkpoint or None,
            seed=args.seed,
            output=args.record,
            max_frames=args.max_record_frames,
        )
        return

    run_demo(
        policy_name=args.policy,
        checkpoint=args.checkpoint or None,
        seed=args.seed,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )


def _run_experiment(args: argparse.Namespace) -> list[dict]:
    from flashrl.experiments import execute_experiment, load_experiment

    records = execute_experiment(
        load_experiment(args.configuration),
        dry_run=args.dry_run,
        resume=args.resume,
        workers=args.workers,
    )
    print(json.dumps(records, indent=2))
    return records


def _run_analysis(args: argparse.Namespace):
    from flashrl.analysis import generate_report

    return generate_report(
        args.run_dir,
        out=args.out,
        publish_data=args.publish_data or None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashrl",
        description="Reproducible reinforcement learning for the FlashRL Dino simulator.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command")

    train_parser = commands.add_parser(
        "train", description="Train a versioned FlashRL DQN experiment."
    )
    train.configure_parser(train_parser)
    train_parser.set_defaults(handler=train.run)

    evaluate_parser = commands.add_parser(
        "evaluate", description="Evaluate baselines or a trained checkpoint."
    )
    evaluate.configure_parser(evaluate_parser)
    evaluate_parser.set_defaults(handler=evaluate.run)

    aggregate_parser = commands.add_parser(
        "aggregate", description="Aggregate compatible benchmark results safely."
    )
    aggregate.configure_parser(aggregate_parser)
    aggregate_parser.set_defaults(handler=aggregate.run)

    doctor_parser = commands.add_parser(
        "doctor", description="Inspect the local FlashRL experiment environment."
    )
    doctor_parser.add_argument("--artifact-dir", type=Path, default=Path("runs"))
    doctor_parser.set_defaults(handler=_run_doctor)

    demo_parser = commands.add_parser(
        "demo", description="Launch the live learned-policy simulator demo."
    )
    demo_parser.add_argument("--checkpoint", default="")
    demo_parser.add_argument("--policy", choices=["rule", "random", "dqn"], default="rule")
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.add_argument("--host", default="127.0.0.1")
    demo_parser.add_argument("--port", type=int, default=8765)
    demo_parser.add_argument("--no-open", action="store_true")
    demo_parser.add_argument("--record", type=Path, default=None)
    demo_parser.add_argument("--max-record-frames", type=int, default=500)
    demo_parser.set_defaults(handler=_run_demo)

    experiment_parser = commands.add_parser(
        "experiment", description="Run a reproducible RL experiment matrix."
    )
    experiment_parser.add_argument("configuration", type=Path)
    experiment_parser.add_argument("--dry-run", action="store_true")
    experiment_parser.add_argument("--resume", action="store_true")
    experiment_parser.add_argument("--workers", type=int, default=1)
    experiment_parser.set_defaults(handler=_run_experiment)

    analyze_parser = commands.add_parser(
        "analyze", description="Generate a traceable RL research report."
    )
    analyze_parser.add_argument("run_dir", type=Path)
    analyze_parser.add_argument("--out", type=Path, default=Path("reports/experiment.md"))
    analyze_parser.add_argument("--publish-data", type=Path, default=None)
    analyze_parser.set_defaults(handler=_run_analysis)
    return parser


def main(argv: Sequence[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    args.handler(args)
    return 0


if __name__ == "__main__":
    main()
