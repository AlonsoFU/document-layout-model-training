"""Typer CLI for the Document Layout Model Factory.

Entry point declared in pyproject.toml as `dlmf = "core.cli:app"`.
Subcommands are wired in this file but their implementations live
in core/render.py, core/cvat_sync.py, etc.
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="dlmf",
    help="Document Layout Model Factory — train layout models per document type.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="render")
def render_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug (e.g. 'eaf')."),
) -> None:
    """Render the project's PDFs to PNGs at the configured DPI."""
    from core.render import render

    render(project)


@app.command(name="cvat-push")
def cvat_push_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    coco: str = typer.Option(
        None, "--coco", help="Optional path to a COCO JSON to pre-load as annotations."
    ),
) -> None:
    """Create the CVAT project + tasks and upload images (and optional pre-labels)."""
    from core.cvat_sync import push

    push(project, coco_path=coco)


@app.command(name="cvat-pull")
def cvat_pull_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    version: str = typer.Option(
        None,
        "--version",
        help="Version label (e.g. v3). Defaults to next sequential version with today's date.",
    ),
) -> None:
    """Export the CVAT project as COCO and write to projects/<slug>/cvat/exports/v<N>_<date>/."""
    from core.cvat_sync import pull

    pull(project, version=version)


if __name__ == "__main__":
    app()
