"""Versioned manifests and safe filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flashrl.schemas import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_ID,
    ENVIRONMENT_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    REWARD_SCHEMA_VERSION,
    SIMULATOR_VERSION,
)

MANIFEST_SCHEMA_VERSION = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temporary_path = Path(fh.name)
            json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@dataclass
class RunManifest:
    run_id: str
    experiment_id: str
    algorithm_id: str
    hyperparameter_hash: str
    training_seed: int
    training_git_commit: str
    git_dirty: bool
    started_at: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    environment_id: str = ENVIRONMENT_ID
    environment_version: int = ENVIRONMENT_VERSION
    simulator_version: int = SIMULATOR_VERSION
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    action_schema_version: int = ACTION_SCHEMA_VERSION
    reward_schema_version: int = REWARD_SCHEMA_VERSION
    status: str = "running"
    train_frames: int = 0
    wall_clock_train_s: float = 0.0
    completed_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        return cls(**dict(value))
