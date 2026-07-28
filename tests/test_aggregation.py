import pytest

from flashrl.results import (
    ResultIdentityError,
    summarize_results,
    validate_compatible_results,
)


def row(run_id, seed, score, *, reward_version=2):
    return {
        "result_schema_version": 2,
        "experiment_id": "double-dqn-config",
        "evaluation_run_id": f"eval-{run_id}",
        "training_run_id": run_id,
        "training_seed": seed,
        "algorithm_id": "double_dqn",
        "hyperparameter_hash": "abc123",
        "environment_id": "FlashRL-DinoSim-v2",
        "environment_version": 2,
        "simulator_version": 2,
        "observation_schema_version": 2,
        "action_schema_version": 2,
        "reward_schema_version": reward_version,
        "backend": "sim",
        "obs_mode": "state",
        "action_mode": "full",
        "checkpoint_sha256": f"checkpoint-{run_id}",
        "episode": 0,
        "score": score,
    }


def test_aggregation_rejects_incompatible_reward_versions():
    rows = [row("run-1", 1, 10), row("run-2", 2, 20, reward_version=3)]
    with pytest.raises(ResultIdentityError, match="reward_schema_version"):
        validate_compatible_results(rows)


def test_aggregation_summarizes_runs_before_training_seeds():
    rows = [
        row("run-1", 1, 10),
        row("run-1", 1, 30),
        row("run-2", 2, 40),
        row("run-2", 2, 60),
    ]
    run_rows, experiment_rows = summarize_results(rows, bootstrap_samples=200)
    assert len(run_rows) == 2
    assert [item["mean_score"] for item in run_rows] == [20.0, 50.0]
    summary = experiment_rows[0]
    assert summary["runs"] == 2
    assert summary["episodes"] == 4
    assert summary["mean_score"] == 35.0
    assert summary["ci95_low"] <= summary["mean_score"] <= summary["ci95_high"]


@pytest.mark.parametrize(
    "field",
    [
        "result_schema_version",
        "environment_version",
        "simulator_version",
        "observation_schema_version",
        "action_schema_version",
        "reward_schema_version",
        "backend",
        "hyperparameter_hash",
    ],
)
def test_aggregation_rejects_every_identity_conflict(field):
    first = row("run-1", 1, 10)
    second = row("run-2", 2, 20)
    second[field] = "conflict"
    with pytest.raises(ResultIdentityError, match=field):
        validate_compatible_results([first, second])
