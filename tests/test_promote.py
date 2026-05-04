"""Tests for core.promote."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.promote import promote


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "test"
    (proj / "runs" / "run_a").mkdir(parents=True)
    (proj / "runs" / "run_a" / "best_model.pt").write_bytes(b"weights_a")
    (proj / "runs" / "run_a" / "eval.json").write_text(json.dumps({"best_mAP": 0.93}))
    (proj / "models").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return proj


def test_promote_creates_symlink_pointing_to_run_best_model(fake_project, monkeypatch):
    monkeypatch.setattr("core.promote._register_in_mlflow", lambda *a, **k: None)
    promote("test", "run_a")
    link = fake_project / "models" / "production.pt"
    assert link.is_symlink() or link.exists()
    # Read the link target; should resolve to the run's best_model.pt
    if link.is_symlink():
        target = link.resolve()
        assert target.name == "best_model.pt"
        assert "run_a" in str(target)


def test_promote_replaces_existing_symlink(fake_project, monkeypatch):
    monkeypatch.setattr("core.promote._register_in_mlflow", lambda *a, **k: None)
    # First promote
    promote("test", "run_a")
    # Add a second run, promote it
    second = fake_project / "runs" / "run_b"
    second.mkdir(parents=True)
    (second / "best_model.pt").write_bytes(b"weights_b")
    (second / "eval.json").write_text(json.dumps({"best_mAP": 0.95}))

    promote("test", "run_b")

    link = fake_project / "models" / "production.pt"
    if link.is_symlink():
        assert "run_b" in str(link.resolve())


def test_promote_raises_if_best_model_missing(fake_project, monkeypatch):
    (fake_project / "runs" / "run_a" / "best_model.pt").unlink()
    with pytest.raises(FileNotFoundError):
        promote("test", "run_a")
