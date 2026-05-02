"""Render a project's PDFs to PNGs using pdftoppm.

Uses pdftoppm directly (not pdf2image) because pdf2image loads all pages
into RAM at once — for 400-page PDFs at 300 DPI that exceeds typical
laptop RAM. pdftoppm streams to disk one page at a time.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from core.lib.config import load_config

PROJECTS_ROOT = Path("projects")


def render(project_slug: str) -> None:
    """Render all PDFs in projects/<slug>/data/pdfs/ to PNGs.

    Output: projects/<slug>/data/images/<pdf_stem>/pagina-NNN.png
    Idempotent: skips a PDF if its image dir already has the expected
    number of PNGs.
    """
    project_dir = PROJECTS_ROOT / project_slug
    config_path = project_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    cfg = load_config(config_path)
    dpi = int(cfg.get("render", {}).get("dpi", 300))

    pdf_dir = project_dir / "data" / "pdfs"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {pdf_dir}")

    images_root = project_dir / "data" / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        out_dir = images_root / pdf.stem
        expected = _expected_page_count(pdf)
        existing = sorted(out_dir.glob("pagina-*.png")) if out_dir.exists() else []
        if out_dir.exists() and len(existing) == expected:
            print(f"[skip] {pdf.name}: {expected} PNGs already present")
            continue
        if out_dir.exists():
            for f in existing:
                f.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix = str(out_dir / "pagina")
        cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf), prefix]
        print(f"[render] {pdf.name} → {out_dir} (dpi={dpi})")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        produced = sorted(out_dir.glob("pagina-*.png"))
        if len(produced) != expected:
            raise RuntimeError(
                f"{pdf.name}: expected {expected} PNGs, got {len(produced)}"
            )


def _expected_page_count(pdf: Path) -> int:
    """Get page count using pdfinfo (poppler ships with pdftoppm)."""
    out = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    )
    for line in out.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not determine page count for {pdf}")
