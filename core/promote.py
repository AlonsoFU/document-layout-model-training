"""Promote a saved run to production: update the symlink and register in MLflow."""
from __future__ import annotations

import os
from pathlib import Path

PROJECTS_ROOT = Path("projects")


def promote(project_slug: str, run_name: str) -> Path:
    project_dir = PROJECTS_ROOT / project_slug
    run_dir = project_dir / "runs" / run_name
    src = run_dir / "best_model.pt"
    if not src.exists():
        raise FileNotFoundError(f"no best_model.pt in {run_dir}")

    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    link = models_dir / "production.pt"

    # Atomic replace: remove existing symlink/file, create new.
    if link.is_symlink() or link.exists():
        link.unlink()
    # Use a relative path so the symlink survives if the repo is moved.
    rel_target = os.path.relpath(src.resolve(), link.parent)
    link.symlink_to(rel_target)
    print(f"[promote] {link} -> {rel_target}")

    _register_in_mlflow(project_slug, run_name, src)
    return link


def _register_in_mlflow(project_slug: str, run_name: str, model_path: Path) -> None:
    """Best-effort: register in MLflow Model Registry. Skip silently if unavailable."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        exp = client.get_experiment_by_name(f"dlmf-{project_slug}")
        if exp is None:
            print(f"[promote] (skip MLflow) experiment 'dlmf-{project_slug}' not found")
            return
        runs = client.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'")
        if not runs:
            print(f"[promote] (skip MLflow) run '{run_name}' not found in MLflow")
            return
        run = runs[0]
        model_uri = f"runs:/{run.info.run_id}/best_model.pt"
        result = mlflow.register_model(model_uri=model_uri, name=f"dlmf-{project_slug}")
        client.transition_model_version_stage(
            name=f"dlmf-{project_slug}",
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"[promote] MLflow Model Registry: dlmf-{project_slug} v{result.version} -> Production")
    except Exception as e:
        print(f"[promote] (MLflow registry skipped: {type(e).__name__}: {e})")
