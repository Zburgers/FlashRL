"""Runtime diagnostics for reproducible FlashRL experiments."""

from __future__ import annotations

import os
import platform
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import gymnasium
import torch

from flashrl import __version__


def _directory_status(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    writable = False
    try:
        fd, temporary = tempfile.mkstemp(prefix=".flashrl-doctor-", dir=path)
        os.close(fd)
        Path(temporary).unlink()
        writable = True
    except OSError:
        pass
    return {"path": str(path.resolve()), "writable": writable}


def collect_diagnostics(artifact_dir: str | Path = "runs") -> dict[str, Any]:
    return {
        "flashrl_version": __version__,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "compute_device": "cuda" if torch.cuda.is_available() else "cpu",
        "artifact_directory": _directory_status(Path(artifact_dir)),
        "optional_features": {
            "browser": find_spec("playwright") is not None,
            "ppo": find_spec("stable_baselines3") is not None,
        },
    }
