import subprocess
from pathlib import Path

LEGACY_ROOT_FILES = {
    "config.py",
    "dino_env.py",
    "dqn_eval.py",
    "dqn_train.py",
    "menu.py",
    "test_script.py",
    "utils.py",
}

LEGACY_PREFIXES = (
    "__pycache__/",
    "assets/",
    "chromedriver-win64/",
    "data/debug_frames/",
    "data/models/",
    "logs/",
)


def tracked_files():
    output = subprocess.check_output(["git", "ls-files"], text=True)
    return set(output.splitlines())


def test_git_tree_contains_no_generated_or_legacy_runtime_artifacts():
    tracked = tracked_files()
    violations = sorted(
        path
        for path in tracked
        if path in LEGACY_ROOT_FILES
        or path.startswith(LEGACY_PREFIXES)
        or path.endswith((".pyc", ".pth", ".exe"))
        or (path.startswith("data/") and path.endswith(".log"))
    )
    assert violations == []


def test_supported_tree_has_one_cli_and_no_ppo_or_schema_v1_results():
    tracked = tracked_files()
    superseded_plans = {
        "issues/backlog.md",
        "reports/benchmark_smoke_summary.md",
        "reports/flashrl_repo_audit.md",
        "reports/flashrl_research_review.md",
        "reports/flashrl_v2_implementation_plan.md",
        "reports/next_agent_handoff_prompt.md",
    }
    violations = sorted(
        path
        for path in tracked
        if path == "flashrl/benchmark/train_ppo.py"
        or path == "requirements.txt"
        or path in superseded_plans
        or path.startswith("results/")
        and path != "results/.gitkeep"
    )
    assert violations == []


def test_generated_directories_are_ignored():
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    for entry in ["runs/", "logs/", "dist/", "*.pt", "results/*.csv"]:
        assert entry in ignore


def test_v2_documentation_has_tested_surface_and_no_retired_claims():
    readme = Path("README.md").read_text(encoding="utf-8")
    report = Path("REPORT.md").read_text(encoding="utf-8")
    combined = readme + report
    for command in [
        'pip install -e ".[dev]"',
        "flashrl demo",
        "flashrl experiment",
        "flashrl analyze",
        "python scripts/smoke_test.py",
    ]:
        assert command in combined
    assert "301.90" in combined
    for retired in [
        "--backend browser",
        "--backend chrome",
        "train_ppo",
        "requirements.txt",
        "checkpoint.pt",
    ]:
        assert retired not in combined


def test_contributor_and_github_templates_exist():
    for path in [
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        ".github/ISSUE_TEMPLATE/bug.yml",
        ".github/ISSUE_TEMPLATE/experiment.yml",
        ".github/pull_request_template.md",
    ]:
        assert Path(path).is_file()
