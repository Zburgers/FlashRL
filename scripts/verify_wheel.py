#!/usr/bin/env python3
"""Install and smoke-test a FlashRL wheel outside the source checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def executable(environment: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return environment / directory / f"{name}{suffix}"


def verify(
    wheel: Path,
    *,
    no_deps: bool = False,
    system_site_packages: bool = False,
) -> None:
    wheel = wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise SystemExit(f"Not a wheel: {wheel}")
    with tempfile.TemporaryDirectory(prefix="flashrl-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(
            with_pip=True,
            clear=True,
            system_site_packages=system_site_packages,
        ).create(environment)
        python = executable(environment, "python")
        flashrl = executable(environment, "flashrl")
        install = [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
        ]
        if no_deps:
            install.append("--no-deps")
        install.append(str(wheel))
        subprocess.run(install, cwd=root, check=True)
        subprocess.run([str(flashrl), "--help"], cwd=root, check=True)
        subprocess.run(
            [str(flashrl), "doctor", "--artifact-dir", str(root / "runs")],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                (
                    "from flashrl.envs import DinoEnv; "
                    "env=DinoEnv(max_episode_steps=2, seed=3); "
                    "obs,_=env.reset(seed=3); "
                    "assert env.observation_space.contains(obs); "
                    "env.step(0); env.close()"
                ),
            ],
            cwd=root,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--no-deps", action="store_true")
    parser.add_argument("--system-site-packages", action="store_true")
    args = parser.parse_args()
    verify(
        args.wheel,
        no_deps=args.no_deps,
        system_site_packages=args.system_site_packages,
    )


if __name__ == "__main__":
    main()
