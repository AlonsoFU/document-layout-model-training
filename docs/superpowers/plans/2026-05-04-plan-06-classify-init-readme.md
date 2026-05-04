# Plan 06 — Classify + Init-Project + README final

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Cerrar la migración con los dos comandos auxiliares (`dlmf classify` y `dlmf init-project`) y un README reescrito que documente el estado real del repo (no el POC viejo).

**Architecture:**
- `core/classify_doctype.py` — `classify(pdf_path) -> str` que itera todos los `projects/<slug>/config.yaml`, prueba `classification.filename_regex` contra el filename; si ninguno matchea, renderiza la primera página y consulta Ollama con `classification.ollama_fallback.prompt` por cada tipo (vision-capable model).
- `core/init_project.py` — `init_project(slug)` crea `projects/<slug>/` con un `config.yaml` plantilla (regex placeholder, prompt placeholder, hyperparams iguales al EAF como punto de partida).
- README rewrite: el header migration notice + estructura ya quedaron al día en Plan 01; ahora hay que reemplazar las secciones obsoletas (CVAT, workflow legacy, "training/EXPERIMENTS.md" links rotos) con el flujo `dlmf` real.

**Tech Stack:** existente + `requests` para Ollama API local.

**Salida verificable:**
- `dlmf classify --pdf=projects/eaf/data/pdfs/EAF-477-2025.pdf` imprime `eaf` (vía regex).
- `dlmf classify --pdf=/tmp/random.pdf` cae al fallback Ollama (o error claro si Ollama no responde / no se reconoce).
- `dlmf init-project --slug=test_doc` deja `projects/test_doc/{config.yaml, data/pdfs/, cvat/exports/, runs/, models/, EXPERIMENTS.md}` listos.
- README ya no menciona `training/`, `cvat_projects/`, ni los scripts legacy.
- Tag `plan-06-classify-init-readme`.

---

## Tasks

### Task 1: `core/classify_doctype.py` con TDD

**Files:** Create `core/classify_doctype.py`, `tests/test_classify.py`.

The classifier:
1. Discovers all `projects/*/config.yaml` files.
2. Loads each config; reads `project.slug` + `classification.filename_regex` + `classification.ollama_fallback`.
3. **Phase 1 — regex on filename**: tries each project's regex against the PDF's filename. If exactly one matches, return its slug. If multiple match, return the first (alphabetical) and warn.
4. **Phase 2 — Ollama fallback**: if no regex matches:
   - Render the PDF's first page to a temporary PNG (DPI 150 — fast for classifier).
   - For each project (alphabetical), if `classification.ollama_fallback.enabled` is true, send POST to `http://localhost:11434/api/generate` with the model from config + the prompt + the image. Look for `"yes"` in the lowercased response.
   - Return the first matching slug. If none match, raise `RuntimeError("could not classify; no project config matched")`.

```python
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
```

Tests (with `monkeypatch` for the Ollama HTTP call and pdftoppm subprocess):

```python
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
```

Steps: failing tests → implement → all pass → commit.

---

### Task 2: `core/init_project.py` con TDD

`init_project(slug)`:
1. If `projects/<slug>/` already exists, raise.
2. Create the directory tree: `data/pdfs/`, `cvat/exports/`, `runs/`, `models/`.
3. Write a `config.yaml` template based on the EAF config but with placeholders for:
   - `display_name`: f"TODO: {slug.upper()} description"
   - `classification.filename_regex`: f"^{slug.upper()}_.*\\.pdf$"  (placeholder; user edits)
   - `classification.ollama_fallback.prompt`: f"Is this a {slug} document? yes/no"
   - `cvat.project_name`: f"{slug.upper()} - layout"
4. Copy `core/labels/doclaynet_17.yaml` reference (via `!include`).
5. Print next steps.

```python
"""Scaffolding command for a new document type."""
from __future__ import annotations

from pathlib import Path

PROJECTS_ROOT = Path("projects")

_CONFIG_TEMPLATE = """\
project:
  slug: {slug}
  display_name: "TODO: {slug_upper} description"

# Identificación automática (Plan 06: dlmf classify usa esto)
classification:
  filename_regex: "^{slug_upper}[-_].*\\\\.pdf$"
  ollama_fallback:
    enabled: true
    model: "qwen2.5:7b"
    prompt: "Is this a {slug} document? yes/no"

render:
  dpi: 300

cvat:
  project_name: "{slug_upper} - layout"
  url: "http://localhost:8080"

# 17 labels DocLayNet (compartidos). Override aquí si el tipo necesita labels custom.
labels: !include ../../core/labels/doclaynet_17.yaml

training:
  base_model: "docling-project/docling-layout-heron"
  lora:
    rank: 32
    alpha: 64
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj]
  optimizer: AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-4
  lr_schedule: cosine
  warmup_epochs: 5
  batch_size: 1
  gradient_accumulation: 4
  gradient_clip: 0.1
  max_epochs: 50
  early_stop_patience: 10
  sampling:
    method: repeat_factor
    threshold: 0.5
  augmentation:
    color_jitter: true
    rotation_degrees: 3
    gaussian_blur: true

postprocess:
  thresholds:
    default: 0.5
    Section-header: 0.45
    Title: 0.45
    Code: 0.45
  nms_iou: 0.5
  cross_cat_iou: 0.3
  full_page_picture_filter: 0.9

evaluation:
  metric: "mAP@[0.5:0.95]"
  val_split: 0.15
  random_seed: 42
"""


def init_project(slug: str) -> Path:
    project_dir = PROJECTS_ROOT / slug
    if project_dir.exists():
        raise FileExistsError(f"{project_dir} already exists")
    (project_dir / "data" / "pdfs").mkdir(parents=True)
    (project_dir / "cvat" / "exports").mkdir(parents=True)
    (project_dir / "runs").mkdir(parents=True)
    (project_dir / "models").mkdir(parents=True)
    # Add .gitkeep markers
    for sub in ("data/pdfs", "cvat/exports", "runs", "models"):
        (project_dir / sub / ".gitkeep").touch()
    # Write config.yaml
    cfg = _CONFIG_TEMPLATE.format(slug=slug, slug_upper=slug.upper())
    (project_dir / "config.yaml").write_text(cfg)
    # Stub EXPERIMENTS.md
    (project_dir / "EXPERIMENTS.md").write_text(
        f"# Experimentos — {slug.upper()}\n\n"
        "_Aún no se ha entrenado ningún modelo para este tipo._\n\n"
        "Cuando `dlmf train` produzca runs, documentar acá los hyperparams y resultados.\n"
    )
    print(f"[init-project] created {project_dir}")
    print("[init-project] next steps:")
    print(f"  1. Edit {project_dir}/config.yaml — set filename_regex and Ollama prompt for {slug}.")
    print(f"  2. Drop your PDFs into {project_dir}/data/pdfs/.")
    print(f"  3. Run: dlmf render --project={slug}")
    print(f"  4. Run: dlmf predict --project={slug} --pre-annotate")
    print(f"  5. Run: dlmf cvat-push --project={slug} --coco=<the pre_annotation file>")
    print(f"  6. Annotate in CVAT, then: dlmf cvat-pull --project={slug}")
    print(f"  7. Run: dlmf train --project={slug} --run=baseline")
    return project_dir
```

Tests:

```python
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
```

Steps: failing tests → implement → all pass → commit.

---

### Task 3: CLI wiring (classify + init-project)

Add to `core/cli.py`:

```python
@app.command(name="classify")
def classify_cmd(
    pdf: str = typer.Option(..., "--pdf", help="Path to a PDF to classify."),
) -> None:
    """Auto-detect which project a PDF belongs to (regex + Ollama fallback)."""
    from core.classify_doctype import classify
    slug = classify(pdf)
    print(slug)


@app.command(name="init-project")
def init_project_cmd(
    slug: str = typer.Option(..., "--slug", help="Slug for the new project (e.g. 'contracts')."),
) -> None:
    """Scaffold a new projects/<slug>/ directory with a config.yaml template."""
    from core.init_project import init_project
    init_project(slug)
```

Update `tests/test_cli.py`:
- Add `"classify"` and `"init-project"` to `test_cli_help_lists_subcommands`.
- Add `test_cli_classify_help_mentions_pdf_flag` and `test_cli_init_project_help_mentions_slug_flag`.

Commit.

---

### Task 4: Smoke tests of `classify` and `init-project`

```bash
# 4a — classify EAF-477 should return 'eaf' via regex
uv run dlmf classify --pdf=projects/eaf/data/pdfs/EAF-477-2025.pdf
# Expected: 'eaf'

# 4b — init-project creates a new tree (then we delete it because we don't
# want to carry a 'demo' project in the repo)
uv run dlmf init-project --slug=demo_test
ls projects/demo_test/
cat projects/demo_test/config.yaml | head -20
rm -rf projects/demo_test
```

No commit needed (smoke is informational).

---

### Task 5: README final rewrite

Replace the body of `README.md` with content reflecting the dlmf CLI, removing all references to `training/`, `cvat_projects/`, the legacy scripts, etc.

Keep the migration banner (Plan 01 added it) but update it to "Migration complete — Plans 01-06 shipped". Or remove it entirely since the migration is done.

Key sections:
- Header + tagline
- Pipeline diagram (already in spec, simplify)
- Quick start (commands needed to add a new tipo and train)
- Project structure (already updated in Plan 01)
- Hardware / dependencies (note GTX 1080, PyTorch 2.5+cu118, uv)
- Reference to spec + plans

Commit.

---

### Task 6: Final verification + tag

```bash
uv run pytest -q
uv run dlmf --help
git tag -a plan-06-classify-init-readme -m "Plan 06 complete: classify + init-project + README finalizado. Migración POC -> factory completa (planes 01-06)."
git tag -l "plan-*"
git log master..HEAD --oneline | wc -l
```

Mark the migration complete.
