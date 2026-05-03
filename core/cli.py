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


@app.command(name="train")
def train_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    run: str = typer.Option(..., "--run", "-r", help="Run name (becomes the dir under projects/<slug>/runs/)."),
    override: list[str] = typer.Option(
        None, "--override", "-o", help="Hyperparameter override KEY=VALUE (e.g. training.lora.rank=64). Repeatable."
    ),
) -> None:
    """Fine-tune the project's layout model (LoRA on Heron) and log to MLflow."""
    from core.train import train

    train(project, run_name=run, overrides=list(override or []))


@app.command(name="predict")
def predict_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    pre_annotate: bool = typer.Option(
        False,
        "--pre-annotate",
        help="Generate COCO predictions to projects/<slug>/cvat/pre_annotations/<timestamp>.json.",
    ),
    threshold: float = typer.Option(
        None, "--threshold", help="Override the default confidence threshold."
    ),
    limit: int = typer.Option(
        None, "--limit", help="Cap the number of images (smoke testing)."
    ),
) -> None:
    """Run the Heron baseline (or, in later plans, a fine-tuned model) over the project's images."""
    if not pre_annotate:
        raise typer.BadParameter("must pass --pre-annotate (other modes added in later plans)")
    from core.predict import predict

    predict(project, mode="pre-annotate", threshold=threshold, limit=limit)


if __name__ == "__main__":
    app()
