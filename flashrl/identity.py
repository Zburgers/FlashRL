"""Canonical identities for algorithms, configurations, and artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hyperparameter_hash(config: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(dict(config)).encode("utf-8"))[:16]


def algorithm_id(
    double_dqn: bool,
    dueling: bool,
    prioritized_replay: bool,
    n_step: int,
) -> str:
    parts: list[str] = []
    if dueling:
        parts.append("dueling")
    if double_dqn:
        parts.append("double")
    parts.append("dqn")
    if prioritized_replay:
        parts.append("per")
    if n_step > 1:
        parts.append(f"n{n_step}")
    return "_".join(parts)
