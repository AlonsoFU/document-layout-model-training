"""Tests for core.classify_doctype."""
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from core.classify_doctype import _classify_by_regex, classify


@pytest.fixture()
def fake_projects(tmp_path, monkeypatch):
    """Two project configs: 'eaf' (matches EAF-...) and 'contracts' (matches Contrato_...)."""
    p1 = tmp_path / "projects" / "eaf"
    p1.mkdir(parents=True)
    (p1 / "config.yaml").write_text(yaml.safe_dump({
        "project": {"slug": "eaf"},
        "classification": {
            "filename_regex": "^EAF[-_]\\d+[-_]\\d{4}\\.pdf$",
            "ollama_fallback": {"enabled": True, "model": "qwen2.5:7b", "prompt": "Is this an EAF? yes/no"},
        },
    }))
    p2 = tmp_path / "projects" / "contracts"
    p2.mkdir(parents=True)
    (p2 / "config.yaml").write_text(yaml.safe_dump({
        "project": {"slug": "contracts"},
        "classification": {
            "filename_regex": "^Contrato_\\d+\\.pdf$",
            "ollama_fallback": {"enabled": False},
        },
    }))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_classify_by_regex_matches_eaf(fake_projects):
    from core.classify_doctype import _list_project_configs
    configs = _list_project_configs()
    assert _classify_by_regex("EAF-477-2025.pdf", configs) == "eaf"


def test_classify_by_regex_matches_contracts(fake_projects):
    from core.classify_doctype import _list_project_configs
    configs = _list_project_configs()
    assert _classify_by_regex("Contrato_42.pdf", configs) == "contracts"


def test_classify_by_regex_no_match(fake_projects):
    from core.classify_doctype import _list_project_configs
    configs = _list_project_configs()
    assert _classify_by_regex("random_doc.pdf", configs) is None


def test_classify_full_flow_regex_path(fake_projects, tmp_path):
    pdf = tmp_path / "EAF-089-2025.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert classify(pdf) == "eaf"


def test_classify_raises_on_missing_pdf(fake_projects, tmp_path):
    with pytest.raises(FileNotFoundError):
        classify(tmp_path / "does_not_exist.pdf")


def test_classify_ollama_fallback_invoked_when_no_regex(fake_projects, tmp_path, monkeypatch):
    """If no regex matches and Ollama returns 'yes', return that slug."""
    pdf = tmp_path / "weird_name.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    # Mock _render_first_page_png to skip pdftoppm.
    fake_png = tmp_path / "page.png"
    fake_png.write_bytes(b"\x89PNG")
    monkeypatch.setattr("core.classify_doctype._render_first_page_png", lambda _p, dpi=150: fake_png)

    # Mock _ollama_classify: True for 'eaf', False otherwise.
    monkeypatch.setattr(
        "core.classify_doctype._ollama_classify",
        lambda image_path, model, prompt, timeout=30.0: "EAF" in prompt,
    )

    assert classify(pdf) == "eaf"
