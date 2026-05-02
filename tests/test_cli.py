"""Tests for the dlmf Typer CLI."""
from typer.testing import CliRunner

from core.cli import app

runner = CliRunner()


def test_cli_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render" in result.stdout
    assert "cvat-push" in result.stdout
    assert "cvat-pull" in result.stdout


def test_cli_render_help_mentions_project_flag():
    result = runner.invoke(app, ["render", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout


def test_cli_cvat_push_help_mentions_project_flag():
    result = runner.invoke(app, ["cvat-push", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout


def test_cli_cvat_pull_help_mentions_project_flag():
    result = runner.invoke(app, ["cvat-pull", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
