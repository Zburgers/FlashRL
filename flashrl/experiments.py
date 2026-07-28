"""Declarative, reproducible FlashRL experiment matrices."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from flashrl.agents.dqn.train import DQNConfig, train_dqn
from flashrl.artifacts import atomic_write_json, sha256_file
from flashrl.benchmark.evaluate import evaluate_checkpoint, write_results


class ExperimentConfigurationError(ValueError):
    """Raised when an experiment matrix is ambiguous or invalid."""


@dataclass(frozen=True)
class ExperimentJob:
    """One fully expanded training and evaluation job."""

    run_id: str
    variant_name: str
    config: DQNConfig
    evaluation_episodes: int
    evaluation_seed_base: int


def load_experiment(path: str | Path) -> dict[str, Any]:
    """Load the JSON-compatible subset of YAML used by FlashRL.

    JSON is valid YAML, keeps the core package dependency-free beyond the RL
    stack, and provides one deterministic representation for experiment files.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigurationError(f"Cannot load experiment: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentConfigurationError("Experiment root must be an object")
    return payload


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentConfigurationError(f"{field} must be an object")
    return value


def expand_jobs(configuration: dict[str, Any]) -> list[ExperimentJob]:
    """Expand a variant-by-seed matrix in stable declaration order."""

    name = configuration.get("name")
    output_dir = configuration.get("output_dir")
    seeds = configuration.get("seeds")
    variants = configuration.get("variants")
    evaluation = _require_mapping(configuration.get("evaluation", {}), "evaluation")
    base = _require_mapping(configuration.get("base", {}), "base")
    if not isinstance(name, str) or not name:
        raise ExperimentConfigurationError("name must be a non-empty string")
    if not isinstance(output_dir, str) or not output_dir:
        raise ExperimentConfigurationError("output_dir must be a non-empty string")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
        raise ExperimentConfigurationError("seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise ExperimentConfigurationError("Duplicate training seed")
    if not isinstance(variants, list) or not variants:
        raise ExperimentConfigurationError("variants must be a non-empty list")

    valid_fields = {field.name for field in fields(DQNConfig)}
    unknown_base = sorted(set(base) - valid_fields)
    if unknown_base:
        raise ExperimentConfigurationError(
            "Unknown DQN configuration field: " + ", ".join(unknown_base)
        )
    evaluation_episodes = evaluation.get("episodes", 100)
    evaluation_seed_base = evaluation.get("seed_base", 100_000)
    if not isinstance(evaluation_episodes, int) or evaluation_episodes <= 0:
        raise ExperimentConfigurationError("evaluation.episodes must be positive")
    if not isinstance(evaluation_seed_base, int) or evaluation_seed_base < 0:
        raise ExperimentConfigurationError("evaluation.seed_base must be non-negative")

    jobs: list[ExperimentJob] = []
    seen_variants: set[str] = set()
    for variant in variants:
        variant = _require_mapping(variant, "variant")
        variant_name = variant.get("name")
        if not isinstance(variant_name, str) or not variant_name:
            raise ExperimentConfigurationError("variant name must be a non-empty string")
        if variant_name in seen_variants:
            raise ExperimentConfigurationError(f"Duplicate variant: {variant_name}")
        seen_variants.add(variant_name)
        overrides = _require_mapping(variant.get("overrides", {}), "variant overrides")
        unknown = sorted(set(overrides) - valid_fields)
        if unknown:
            raise ExperimentConfigurationError(
                "Unknown DQN configuration field: " + ", ".join(unknown)
            )
        for seed in seeds:
            run_id = f"{name}-{variant_name}-seed{seed}"
            config_values = (
                base
                | overrides
                | {
                    "seed": seed,
                    "output_dir": output_dir,
                    "run_id": run_id,
                }
            )
            try:
                config = DQNConfig(**config_values)
            except TypeError as exc:
                raise ExperimentConfigurationError(str(exc)) from exc
            jobs.append(
                ExperimentJob(
                    run_id=run_id,
                    variant_name=variant_name,
                    config=config,
                    evaluation_episodes=evaluation_episodes,
                    evaluation_seed_base=evaluation_seed_base,
                )
            )
    return jobs


def _completed_evaluation(run_dir: Path) -> bool:
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "eval_results.jsonl"
    if not manifest_path.is_file() or not results_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def _execute_job(job: ExperimentJob, resume: bool) -> dict[str, Any]:
    run_dir = Path(job.config.output_dir) / job.run_id
    if resume and _completed_evaluation(run_dir):
        return {
            "run_id": job.run_id,
            "variant": job.variant_name,
            "status": "skipped",
            "reason": "completed",
        }

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    manifest_path = run_dir / "manifest.json"
    if resume and best_path.is_file() and manifest_path.is_file():
        training = {"best_checkpoint_path": str(best_path), "manifest_path": str(manifest_path)}
    else:
        resume_path = last_path if resume and last_path.is_file() else None
        training = train_dqn(job.config, resume_path=resume_path)

    rows = evaluate_checkpoint(
        training["best_checkpoint_path"],
        episodes=job.evaluation_episodes,
        eval_seed=job.evaluation_seed_base,
        phase="heldout",
    )
    csv_path = run_dir / "eval_results.csv"
    jsonl_path = run_dir / "eval_results.jsonl"
    write_results(rows, csv_path, jsonl_path)

    manifest_path = Path(training["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.setdefault("artifacts", {})
    for path in (csv_path, jsonl_path):
        artifacts[path.name] = {
            "path": path.name,
            "sha256": sha256_file(path),
        }
    atomic_write_json(manifest_path, manifest)
    return {
        "run_id": job.run_id,
        "variant": job.variant_name,
        "status": "completed",
        "train_frames": manifest.get("train_frames", 0),
        "evaluation_episodes": len(rows),
        "run_dir": str(run_dir),
    }


def execute_experiment(
    configuration: dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Execute or describe a reproducible local experiment matrix."""

    jobs = expand_jobs(configuration)
    maximum_workers = max(1, os.cpu_count() or 1)
    if not isinstance(workers, int) or not 1 <= workers <= maximum_workers:
        raise ExperimentConfigurationError(f"workers must be between 1 and {maximum_workers}")
    if dry_run:
        return [
            {
                "run_id": job.run_id,
                "variant": job.variant_name,
                "seed": job.config.seed,
                "status": "planned",
                "episodes": job.config.episodes,
                "evaluation_episodes": job.evaluation_episodes,
            }
            for job in jobs
        ]
    if workers == 1:
        return [_execute_job(job, resume) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_execute_job, jobs, [resume] * len(jobs)))
