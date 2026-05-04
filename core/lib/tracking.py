"""MLflow tracking wrapper — context manager that owns one run."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class MlflowRun:
    """Context manager for an MLflow run.

    Usage:
        with MlflowRun(experiment="dlmf-eaf", run_name="P_repeat_factor", params={...}) as run:
            for epoch in ...:
                run.log_metric("train_loss", loss, step=epoch)
            run.log_artifact("history.json")
    """

    def __init__(
        self,
        experiment: str,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ):
        self.experiment = experiment
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self._run = None

    def __enter__(self):
        mlflow.set_experiment(self.experiment)
        self._run = mlflow.start_run(run_name=self.run_name)
        if self.params:
            mlflow.log_params(_flatten_params(self.params))
        if self.tags:
            mlflow.set_tags(self.tags)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
        mlflow.end_run()
        return False  # don't suppress exceptions

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(name, float(value), step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def log_dict(self, d: dict, artifact_path: str) -> None:
        mlflow.log_dict(d, artifact_path)


def _flatten_params(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested dicts into dotted-key strings (MLflow param values must be strings)."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        else:
            out[key] = str(v)
    return out
