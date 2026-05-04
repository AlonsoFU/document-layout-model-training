"""Auto-classify a PDF to one of the projects/<slug>/ configs.

Strategy (Pregunta 4a:D del spec):
1. Try filename_regex of each project; return slug if any matches.
2. Fall back to Ollama vision on the first page using each project's
   classification.ollama_fallback.prompt.
"""
from __future__ import annotations

import base64
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import requests

from core.lib.config import load_config

PROJECTS_ROOT = Path("projects")
OLLAMA_URL = "http://localhost:11434/api/generate"


def _list_project_configs() -> list[tuple[str, dict]]:
    """Return [(slug, config_dict), ...] sorted by slug."""
    out = []
    if not PROJECTS_ROOT.exists():
        return out
    for d in sorted(PROJECTS_ROOT.iterdir()):
        cfg_path = d / "config.yaml"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
            slug = cfg.get("project", {}).get("slug", d.name)
            out.append((slug, cfg))
    return out


def _classify_by_regex(pdf_filename: str, configs: list[tuple[str, dict]]) -> Optional[str]:
    matches = []
    for slug, cfg in configs:
        regex = cfg.get("classification", {}).get("filename_regex")
        if not regex:
            continue
        if re.match(regex, pdf_filename):
            matches.append(slug)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[classify] WARNING: multiple regex matches: {matches}; using {matches[0]}")
    return matches[0]


def _render_first_page_png(pdf_path: Path, dpi: int = 150) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="dlmf_classify_"))
    out_prefix = tmp_dir / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1", str(pdf_path), str(out_prefix)],
        check=True, capture_output=True, text=True,
    )
    pngs = list(tmp_dir.glob("page-*.png"))
    if not pngs:
        raise RuntimeError(f"pdftoppm did not produce a PNG for {pdf_path}")
    return pngs[0]


def _ollama_classify(image_path: Path, model: str, prompt: str, timeout: float = 30.0) -> bool:
    """Send image+prompt to Ollama; return True if response contains 'yes'."""
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {"model": model, "prompt": prompt, "images": [img_b64], "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    response = r.json().get("response", "").lower().strip()
    return "yes" in response


def classify(pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    configs = _list_project_configs()
    if not configs:
        raise RuntimeError("no projects/<slug>/config.yaml files found")

    # Phase 1: regex on filename.
    by_regex = _classify_by_regex(pdf_path.name, configs)
    if by_regex:
        print(f"[classify] '{pdf_path.name}' -> {by_regex} (filename regex)")
        return by_regex

    # Phase 2: Ollama fallback.
    print(f"[classify] no regex matched '{pdf_path.name}'; trying Ollama vision...")
    img = _render_first_page_png(pdf_path)
    try:
        for slug, cfg in configs:
            ofb = cfg.get("classification", {}).get("ollama_fallback", {})
            if not ofb.get("enabled"):
                continue
            try:
                if _ollama_classify(img, model=ofb["model"], prompt=ofb["prompt"]):
                    print(f"[classify] '{pdf_path.name}' -> {slug} (Ollama)")
                    return slug
            except Exception as e:
                print(f"[classify] Ollama check for {slug} failed: {e}")
        raise RuntimeError(f"could not classify '{pdf_path.name}' (no regex match, no Ollama match)")
    finally:
        img.unlink(missing_ok=True)
        try:
            img.parent.rmdir()
        except OSError:
            pass
