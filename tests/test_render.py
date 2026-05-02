"""Tests for core.render — generic PDF→PNG batch renderer.

Uses monkeypatching to replace `subprocess.run` so tests don't depend on
a real `pdftoppm` binary or actual PDFs.
"""
from pathlib import Path

import pytest
import yaml

from core.render import render, _expected_page_count


@pytest.fixture()
def fake_project(tmp_path: Path, monkeypatch) -> Path:
    """Create a minimal projects/test/ tree with a config.yaml and 2 placeholder PDFs."""
    proj = tmp_path / "projects" / "test"
    (proj / "data" / "pdfs").mkdir(parents=True)
    (proj / "data" / "images").mkdir(parents=True)
    (proj / "data" / "pdfs" / "doc-A.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    (proj / "data" / "pdfs" / "doc-B.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    (proj / "config.yaml").write_text(
        yaml.safe_dump({"render": {"dpi": 200}, "project": {"slug": "test"}})
    )
    monkeypatch.chdir(tmp_path)
    return proj


def test_render_calls_pdftoppm_for_each_pdf(fake_project, monkeypatch):
    calls = []

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        # Simulate creating the expected PNGs
        out_prefix = cmd[-1]  # last arg is "<dir>/pagina"
        out_dir = Path(out_prefix).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # Create 3 fake PNGs
        for i in range(1, 4):
            (out_dir / f"pagina-{i:03d}.png").write_bytes(b"\x89PNG\r\n")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def fake_count(_pdf):
        return 3

    monkeypatch.setattr("core.render.subprocess.run", fake_run)
    monkeypatch.setattr("core.render._expected_page_count", fake_count)

    render("test")

    assert len(calls) == 2
    assert all(c[0] == "pdftoppm" for c in calls)
    assert all("-r" in c and "200" in c for c in calls)  # DPI from config
    # Output paths under data/images/<stem>/
    assert any("doc-A" in str(c) for c in calls)
    assert any("doc-B" in str(c) for c in calls)


def test_render_skips_when_pngs_already_present(fake_project, monkeypatch):
    """If target dir has the right number of PNGs, pdftoppm is not invoked."""
    out_dir = fake_project / "data" / "images" / "doc-A"
    out_dir.mkdir(parents=True)
    for i in range(1, 4):
        (out_dir / f"pagina-{i:03d}.png").write_bytes(b"\x89PNG")
    out_dir2 = fake_project / "data" / "images" / "doc-B"
    out_dir2.mkdir(parents=True)
    for i in range(1, 4):
        (out_dir2 / f"pagina-{i:03d}.png").write_bytes(b"\x89PNG")

    calls = []
    monkeypatch.setattr("core.render.subprocess.run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("core.render._expected_page_count", lambda _: 3)

    render("test")

    assert calls == []  # nothing rendered, fully cached


def test_render_raises_if_no_pdfs(fake_project, monkeypatch):
    for p in (fake_project / "data" / "pdfs").glob("*.pdf"):
        p.unlink()
    with pytest.raises(FileNotFoundError, match="no PDFs"):
        render("test")


def test_render_raises_if_project_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        render("nonexistent")
