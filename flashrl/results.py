"""Schema-v2 benchmark result validation and two-level statistics."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

RESULT_SCHEMA_VERSION = 2

RESULT_FIELDS = [
    "result_schema_version",
    "evaluation_run_id",
    "training_run_id",
    "experiment_id",
    "algorithm_id",
    "hyperparameter_hash",
    "training_seed",
    "evaluation_seed_base",
    "episode_seed",
    "episode",
    "training_git_commit",
    "evaluation_git_commit",
    "environment_id",
    "environment_version",
    "simulator_version",
    "observation_schema_version",
    "action_schema_version",
    "reward_schema_version",
    "backend",
    "obs_mode",
    "action_mode",
    "agent",
    "phase",
    "score",
    "survival_time_s",
    "steps",
    "obstacles_cleared",
    "death_type",
    "terminated",
    "truncated",
    "ending_reason",
    "train_frames",
    "wall_clock_train_s",
    "wall_clock_episode_s",
    "checkpoint_role",
    "checkpoint_path",
    "checkpoint_sha256",
    "manifest_path",
]

IDENTITY_FIELDS = [
    "result_schema_version",
    "experiment_id",
    "algorithm_id",
    "hyperparameter_hash",
    "environment_id",
    "environment_version",
    "simulator_version",
    "observation_schema_version",
    "action_schema_version",
    "reward_schema_version",
    "backend",
    "obs_mode",
    "action_mode",
]


class ResultIdentityError(ValueError):
    """Raised when rows cannot support one valid experiment comparison."""


def validate_compatible_results(rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ResultIdentityError("No result rows supplied")
    reference = materialized[0]
    for field in IDENTITY_FIELDS:
        expected = reference.get(field)
        conflicting = sorted(
            {str(row.get(field)) for row in materialized if row.get(field) != expected}
        )
        if conflicting:
            raise ResultIdentityError(
                f"Incompatible {field}: expected {expected!r}, found {', '.join(conflicting)}"
            )


def _bootstrap_ci(values: list[float], samples: int, seed: int = 0) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    source = np.asarray(values, dtype=np.float64)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(source, size=len(source), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_results(
    rows: list[dict[str, Any]], bootstrap_samples: int = 10_000
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_compatible_results(rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["training_run_id"]), str(row["checkpoint_sha256"]))
        grouped[key].append(float(row["score"]))
        metadata[key] = row

    run_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        scores = grouped[key]
        source = metadata[key]
        run_rows.append(
            {
                "experiment_id": source["experiment_id"],
                "training_run_id": key[0],
                "training_seed": int(source["training_seed"]),
                "checkpoint_sha256": key[1],
                "episodes": len(scores),
                "mean_score": float(statistics.fmean(scores)),
                "median_score": float(statistics.median(scores)),
                "standard_deviation": (float(statistics.stdev(scores)) if len(scores) > 1 else 0.0),
                "best_score": float(max(scores)),
            }
        )

    run_means = [float(row["mean_score"]) for row in run_rows]
    ci_low, ci_high = _bootstrap_ci(run_means, samples=max(1, bootstrap_samples))
    reference = rows[0]
    experiment_rows = [
        {
            "experiment_id": reference["experiment_id"],
            "algorithm_id": reference["algorithm_id"],
            "hyperparameter_hash": reference["hyperparameter_hash"],
            "runs": len(run_rows),
            "episodes": len(rows),
            "mean_score": float(statistics.fmean(run_means)),
            "median_score": float(statistics.median(run_means)),
            "standard_deviation": (
                float(statistics.stdev(run_means)) if len(run_means) > 1 else 0.0
            ),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
    ]
    return run_rows, experiment_rows
