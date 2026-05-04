"""Tests for core.lib.tracking — MLflow wrapper."""
from pathlib import Path
import json

import mlflow

from core.lib.tracking import MlflowRun


def test_mlflow_run_logs_params_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    with MlflowRun(experiment="test-exp", run_name="t1", params={"a": 1, "b": "x"}) as run:
        run.log_metric("loss", 0.5, step=0)
        run.log_metric("loss", 0.4, step=1)
        run.log_metric("mAP", 0.85, step=1)

    # Verify via MLflow client
    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}/mlruns")
    exp = client.get_experiment_by_name("test-exp")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    r = runs[0]
    assert r.data.params["a"] == "1"
    assert r.data.params["b"] == "x"
    assert r.data.metrics["mAP"] == 0.85


def test_mlflow_run_logs_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    artifact = tmp_path / "history.json"
    artifact.write_text(json.dumps([{"epoch": 0, "loss": 0.5}]))

    with MlflowRun(experiment="test-exp", run_name="t2", params={}) as run:
        run.log_artifact(artifact)

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}/mlruns")
    exp = client.get_experiment_by_name("test-exp")
    runs = client.search_runs([exp.experiment_id])
    artifact_list = client.list_artifacts(runs[0].info.run_id)
    assert any(a.path == "history.json" for a in artifact_list)
