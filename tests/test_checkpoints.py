import pytest
import torch

from flashrl.agents.dqn.train import (
    CHECKPOINT_FORMAT_VERSION,
    BestCheckpointTracker,
    CheckpointCompatibilityError,
    atomic_torch_save,
    load_checkpoint,
)
from flashrl.schemas import OBSERVATION_SCHEMA_VERSION


def payload(*, role="last", score=1.0):
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": 2,
        "reward_schema_version": 2,
        "environment_version": 2,
        "role": role,
        "selection_score": score,
        "model_state_dict": {"weight": torch.tensor([score])},
        "optimizer_state_dict": {},
        "config": {},
        "run_id": "run-1",
    }


def test_atomic_torch_save_replaces_complete_file(tmp_path):
    path = tmp_path / "last.pt"
    atomic_torch_save(path, payload(score=1))
    atomic_torch_save(path, payload(score=2))
    assert load_checkpoint(path)["selection_score"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_best_tracker_does_not_replace_best_with_worse_score(tmp_path):
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path)
    assert tracker.consider(10.0, payload(role="best", score=10))
    assert not tracker.consider(3.0, payload(role="best", score=3))
    assert load_checkpoint(path)["selection_score"] == 10


def test_checkpoint_rejects_unknown_format(tmp_path):
    path = tmp_path / "future.pt"
    future = payload()
    future["checkpoint_format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    torch.save(future, path)
    with pytest.raises(CheckpointCompatibilityError, match="format"):
        load_checkpoint(path)


def test_checkpoint_rejects_observation_schema_mismatch(tmp_path):
    path = tmp_path / "old-observation.pt"
    old = payload()
    old["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION - 1
    torch.save(old, path)
    with pytest.raises(CheckpointCompatibilityError, match="observation"):
        load_checkpoint(path)
