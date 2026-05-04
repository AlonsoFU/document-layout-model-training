"""Tests for core.init_project."""
from pathlib import Path

import pytest

from core.init_project import init_project


def test_init_project_creates_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = init_project("testdoc")
    assert (out / "config.yaml").exists()
    assert (out / "data" / "pdfs").is_dir()
    assert (out / "cvat" / "exports").is_dir()
    assert (out / "runs").is_dir()
    assert (out / "models").is_dir()
    assert (out / "EXPERIMENTS.md").exists()


def test_init_project_config_uses_slug(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = init_project("contracts")
    text = (out / "config.yaml").read_text()
    assert "slug: contracts" in text
    assert "CONTRACTS" in text  # display_name etc


def test_init_project_raises_on_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project("x")
    with pytest.raises(FileExistsError):
        init_project("x")


def test_init_project_config_loads_with_core_loader(tmp_path, monkeypatch):
    """Sanity: the templated config.yaml must be loadable by core.lib.config (after copying labels file)."""
    monkeypatch.chdir(tmp_path)
    # The init template uses !include ../../core/labels/doclaynet_17.yaml, so
    # we need that path to exist relative to projects/<slug>/.
    (tmp_path / "core" / "labels").mkdir(parents=True)
    (tmp_path / "core" / "labels" / "doclaynet_17.yaml").write_text("- A\n- B\n- C\n")
    init_project("z")
    from core.lib.config import load_config
    cfg = load_config(tmp_path / "projects" / "z" / "config.yaml")
    assert cfg["project"]["slug"] == "z"
    assert cfg["labels"] == ["A", "B", "C"]
