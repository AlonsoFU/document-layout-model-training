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
    assert "predict" in result.stdout


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


def test_cli_predict_help_mentions_required_flags():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
    assert "--pre-annotate" in result.stdout
    assert "--threshold" in result.stdout
    assert "--limit" in result.stdout
