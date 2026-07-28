from pathlib import Path

import tomllib


def test_package_metadata_defines_release_surface():
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["requires-python"] == ">=3.10"
    assert project["license"] == "MIT"
    assert project["scripts"]["flashrl"] == "flashrl.cli:main"
    assert set(project["optional-dependencies"]) == {"dev", "browser", "ppo"}
    assert all(
        dependency.split(">=")[0] not in {"playwright", "stable-baselines3", "pytest"}
        for dependency in project["dependencies"]
    )


def test_ci_gates_pushes_pull_requests_wheels_and_supported_python():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "master" in workflow
    for version in ["3.10", "3.11", "3.12", "3.13"]:
        assert version in workflow
    assert "python -m build" in workflow
    assert "verify_wheel.py" in workflow
    assert "pytest" in workflow
    assert "ruff" in workflow


def test_wheel_verifier_exists_and_uses_isolated_environment():
    verifier = Path("scripts/verify_wheel.py").read_text(encoding="utf-8")
    assert "venv.EnvBuilder" in verifier
    assert "--force-reinstall" in verifier
    assert '"flashrl"' in verifier
    assert '"--help"' in verifier
    assert "scripts/smoke_test.py" not in verifier
