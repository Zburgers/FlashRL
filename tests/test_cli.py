import pytest

from flashrl import __version__
from flashrl.cli import build_parser, main


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as ending:
        main(["--version"])
    assert ending.value.code == 0
    assert capsys.readouterr().out.strip() == f"flashrl {__version__}"


@pytest.mark.parametrize(
    ("command", "description"),
    [
        ("train", "Train"),
        ("evaluate", "Evaluate"),
        ("aggregate", "Aggregate"),
        ("doctor", "environment"),
        ("demo", "live"),
    ],
)
def test_cli_exposes_canonical_commands(command, description):
    parser = build_parser()
    subparser = parser._subparsers._group_actions[0].choices[command]
    assert description.lower() in subparser.description.lower()


def test_cli_without_command_prints_help(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "train" in output
    assert "evaluate" in output
    assert "demo" in output


def test_successful_command_returns_zero_exit_status(tmp_path, capsys):
    assert main(["doctor", "--artifact-dir", str(tmp_path)]) == 0
    assert '"compute_device"' in capsys.readouterr().out
