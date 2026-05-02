# Plan 01 — Repo Foundation & EAF Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrar el repo del POC actual a la estructura `projects/<tipo>/ + core/` definida en el spec, sin perder ningún dato y dejando un loader de config con tests que validen el contrato.

**Architecture:** Reorganización de archivos + skeleton de `core/` con un único módulo funcional (`core/lib/config.py`). Los scripts de ML existentes (`train_*.py`, `predict`, etc.) se quedan **donde están y NO se migran en este plan** — se migran en los planes 2-6. Este plan deja los cimientos.

**Tech Stack:** Python 3.11+, uv (gestor de deps), PyYAML, pytest, Typer (placeholder, sin comandos aún).

**Salida verificable:** `pytest` pasa todos los tests de `core/lib/config.py`. `projects/eaf/config.yaml` carga correctamente y resuelve `!include`. La estructura de carpetas refleja el spec.

**Out of scope (planes futuros):**
- Plan 2: `core/render.py` + `core/cvat_sync.py` + Typer CLI con primeros comandos.
- Plan 3: `core/predict.py --pre-annotate` + `core/lib/postproc.py`.
- Plan 4: `core/train.py` + `core/lib/{data,model,tracking}.py` + MLflow.
- Plan 5: `core/evaluate.py` + `core/promote.py` + `core/predict.py --output=anotado.pdf`.
- Plan 6: `core/classify_doctype.py` + `core/init_project.py` + parity gate (mAP >= 0.86).

---

## File Structure

### Files to CREATE

| Path | Responsabilidad |
|---|---|
| `pyproject.toml` | Project metadata + deps gestionadas con uv |
| `.python-version` | Pin de Python (3.11) |
| `core/__init__.py` | Marker de package |
| `core/lib/__init__.py` | Marker de subpackage |
| `core/lib/config.py` | Loader YAML con `!include` resolver y aplicación de overrides |
| `core/labels/__init__.py` | Marker (necesario para que `!include` resuelva paths relativos) |
| `core/labels/doclaynet_17.yaml` | Lista de los 17 labels DocLayNet (compartida entre tipos) |
| `projects/eaf/config.yaml` | Config del tipo EAF (slug, regex, labels, hyperparams) |
| `projects/eaf/.gitkeep` | Mantiene la carpeta en git |
| `projects/eaf/data/.gitkeep` | id |
| `projects/eaf/cvat/.gitkeep` | id |
| `projects/eaf/runs/.gitkeep` | id |
| `projects/eaf/models/.gitkeep` | id |
| `tests/__init__.py` | Marker |
| `tests/test_config.py` | Tests de `core/lib/config.py` |
| `tests/fixtures/sample_config.yaml` | Fixture para tests del loader |
| `tests/fixtures/sample_labels.yaml` | Fixture incluida vía `!include` |

### Files to MOVE (preservando contenido)

| De | A |
|---|---|
| `cvat_projects/project4_docling_heron_clean/annotations/instances_default.json` | `projects/eaf/cvat/exports/v1_2026-03-20/instances_default.json` |
| `cvat_projects/project4_docling_heron_clean/README.md` | `projects/eaf/cvat/exports/v1_2026-03-20/README.md` |
| `data/pdfs/EAF-089-2025.pdf` | `projects/eaf/data/pdfs/EAF-089-2025.pdf` |
| `data/pdfs/EAF-477-2025.pdf` | `projects/eaf/data/pdfs/EAF-477-2025.pdf` |
| `training/models/P_repeat_factor/history.json` | `projects/eaf/runs/P_repeat_factor/history.json` |
| `training/EXPERIMENTS.md` | `projects/eaf/EXPERIMENTS.md` |

### Files to DELETE

| Path | Razón |
|---|---|
| `clean_overlaps.py` | v1, sólo se usa v3 |
| `clean_overlaps_v2.py` | v2, sólo se usa v3 |
| `https:/` (carpeta) | Clone accidental |
| `training/EAF-477-2025_ground_truth.json` | Duplica el CVAT export |
| `training/models/data_split.json` | Vivirá dentro de `runs/<run>/` en futuro |
| `training/models/results.json` | Reemplazado por per-run `eval.json` |
| `training/models/results_real_map.json` | id |
| `training/models/results_round3.json` | id |
| `training/models/results_round4.json` | id |

### Files to MODIFY

| Path | Cambio |
|---|---|
| `.gitignore` | Añadir paths nuevos (`projects/*/data/images/`, `projects/*/runs/*/best_model.pt`, `mlruns/`, `tests/__pycache__/`, `.venv/`); eliminar reglas obsoletas (`training/models/*/`, `cvat_projects/*/images/`) |
| `README.md` | Nota al inicio: "El repo está en migración a Layout Model Factory. Ver `docs/superpowers/specs/2026-05-02-layout-model-factory-design.md`" |

### Files NOT touched in this plan (migrated in plans 2-6)

`generate_heron_coco.py`, `upload_to_cvat.py`, `clean_overlaps_v3.py`, `reimport_annotations.py`, `scripts/render_pdf_to_png.py`, `scripts/restore_cvat_project4.py`, `training/train_round*.py`, `training/draw_boxes_on_pdf*.py`, `training/run_docling_pipeline.py`, `training/test_best_model.py`, `training/reevaluate_real_map.py`, `training/train_strategies.py`, `training/generate_comparison_pdf*.py`, `conversacion_claude.md`.

---

## Tasks

### Task 1: Cleanup obsolete files

**Files:**
- Delete: `clean_overlaps.py`, `clean_overlaps_v2.py`, `https:/`, `training/EAF-477-2025_ground_truth.json`, `training/models/data_split.json`, `training/models/results.json`, `training/models/results_real_map.json`, `training/models/results_round3.json`, `training/models/results_round4.json`

- [ ] **Step 1: Verify each deletion target exists before removing**

```bash
ls clean_overlaps.py clean_overlaps_v2.py training/EAF-477-2025_ground_truth.json
ls training/models/data_split.json training/models/results.json training/models/results_real_map.json training/models/results_round3.json training/models/results_round4.json
ls -la https:/
```
Expected: All listed.

- [ ] **Step 2: Verify `clean_overlaps_v3.py` still exists (do NOT delete it)**

```bash
ls clean_overlaps_v3.py
```
Expected: file present (it stays — refactored in Plan 3).

- [ ] **Step 3: Remove the unused files**

```bash
git rm clean_overlaps.py clean_overlaps_v2.py
git rm training/EAF-477-2025_ground_truth.json
git rm training/models/data_split.json training/models/results.json training/models/results_real_map.json training/models/results_round3.json training/models/results_round4.json
rm -rf https:/
```

- [ ] **Step 4: Verify deletions**

```bash
git status --short
```
Expected: 8 deletions staged + `https:/` gone (was untracked, doesn't appear).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove obsolete v1/v2 overlap scripts and per-round result files

The v1 and v2 overlap-cleaning scripts were superseded by v3.
Per-round results (results_round3.json, results_round4.json, etc.)
will be replaced by per-run eval.json in the new factory structure.
The accidental https:/ clone is also removed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create `pyproject.toml` with uv

**Files:**
- Create: `pyproject.toml`, `.python-version`

- [ ] **Step 1: Verify `uv` is installed**

```bash
uv --version
```
Expected: version string. If not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`.

- [ ] **Step 2: Create `.python-version`**

Write file `/home/alonso/Documentos/Github/document-layout-model-training/.python-version`:
```
3.11
```

- [ ] **Step 3: Create `pyproject.toml`**

Write file `/home/alonso/Documentos/Github/document-layout-model-training/pyproject.toml`:
```toml
[project]
name = "dlmf"
version = "0.1.0"
description = "Document Layout Model Factory"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "transformers>=4.40",
    "Pillow>=10.0",
    "PyMuPDF>=1.23",
    "pdf2image>=1.17",
    "pikepdf>=8.0",
    "requests>=2.31",
    "pyyaml>=6.0",
    "typer>=0.12",
    "rich>=13.0",
    "mlflow>=2.10",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]

[project.scripts]
dlmf = "core.cli:app"  # placeholder; CLI implemented in Plan 2

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["core"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 4: Run `uv sync` to verify deps resolve**

```bash
uv sync --extra dev
```
Expected: `Resolved N packages` and `.venv/` updated. If pytest not installed yet, this installs it.

- [ ] **Step 5: Verify `pytest` is callable**

```bash
uv run pytest --version
```
Expected: `pytest <version>`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version
git commit -m "chore: add pyproject.toml managed with uv

Defines dlmf package metadata, dependencies (torch, transformers,
mlflow, typer, etc.) and pytest config. The CLI script entry point
references core.cli:app which is implemented in Plan 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update `.gitignore` for new structure

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Replace `.gitignore` content**

Replace entire contents of `/home/alonso/Documentos/Github/document-layout-model-training/.gitignore` with:
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
htmlcov/

# Virtualenv
.venv/

# Rendered PNGs (regenerable from data/pdfs/ via `dlmf render`)
projects/*/data/images/

# CVAT pre-annotations (regenerable via `dlmf predict --pre-annotate`)
projects/*/cvat/pre_annotations/

# Model weights and optimizer states (large; tracked via MLflow Model Registry)
projects/*/runs/*/best_model.pt
projects/*/runs/*/*.pt
projects/*/runs/*/checkpoints/

# MLflow tracking (reconstructible from runs metadata)
mlruns/

# uv lock (optional; generated)
# Comment out next line if you want a committed lockfile
# uv.lock

# IDE
.idea/
.vscode/
*.swp
.DS_Store

# Legacy POC artifacts (will be removed in plans 2-6 as scripts are migrated)
coco_output/
inputs/imagenes/
training/resultados/

# Old cvat_projects layout (kept for reference until fully migrated)
cvat_projects/*/images/
```

- [ ] **Step 2: Verify ignore works for the future paths**

```bash
mkdir -p projects/eaf/data/images projects/eaf/cvat/pre_annotations projects/eaf/runs/test_run mlruns
touch projects/eaf/data/images/test.png projects/eaf/cvat/pre_annotations/test.json projects/eaf/runs/test_run/best_model.pt mlruns/test
git status --short --ignored | grep -E "(images|pre_annotations|best_model|mlruns)"
```
Expected: All four appear under `!!` (ignored). If they show as untracked, the gitignore is wrong — fix and re-test.

- [ ] **Step 3: Cleanup test artifacts**

```bash
rm -rf projects/eaf/data/images projects/eaf/cvat/pre_annotations projects/eaf/runs/test_run mlruns
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: rewrite .gitignore for projects/<type>/ layout

Adds patterns for the new factory structure: rendered PNGs,
pre-annotations, model weights, and mlruns/ are gitignored
(all regenerable). Legacy POC paths (training/resultados,
coco_output, inputs/imagenes) remain ignored until migrated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Create `projects/eaf/` directory structure

**Files:**
- Create: `projects/eaf/{data,cvat/exports,runs,models}/.gitkeep`

- [ ] **Step 1: Create directories**

```bash
mkdir -p projects/eaf/data/pdfs
mkdir -p projects/eaf/cvat/exports
mkdir -p projects/eaf/runs
mkdir -p projects/eaf/models
```

- [ ] **Step 2: Add `.gitkeep` files for empty dirs**

Create empty file at each of:
- `projects/eaf/data/pdfs/.gitkeep`
- `projects/eaf/cvat/exports/.gitkeep` (will be removed once exports land)
- `projects/eaf/runs/.gitkeep`
- `projects/eaf/models/.gitkeep`

```bash
touch projects/eaf/data/pdfs/.gitkeep projects/eaf/cvat/exports/.gitkeep projects/eaf/runs/.gitkeep projects/eaf/models/.gitkeep
```

- [ ] **Step 3: Verify structure**

```bash
find projects/eaf -type d
```
Expected output:
```
projects/eaf
projects/eaf/data
projects/eaf/data/pdfs
projects/eaf/cvat
projects/eaf/cvat/exports
projects/eaf/runs
projects/eaf/models
```

- [ ] **Step 4: Commit (skeleton; files added in next tasks)**

```bash
git add projects/eaf/
git commit -m "feat: scaffold projects/eaf/ directory tree

Creates the per-type layout for the EAF document type as
defined in the design spec (data/, cvat/exports/, runs/, models/).
Files migrated from cvat_projects/, data/, training/ in subsequent
tasks of this plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Move CVAT export to `projects/eaf/cvat/exports/v1_2026-03-20/`

**Files:**
- Move: `cvat_projects/project4_docling_heron_clean/annotations/instances_default.json` → `projects/eaf/cvat/exports/v1_2026-03-20/instances_default.json`
- Move: `cvat_projects/project4_docling_heron_clean/README.md` → `projects/eaf/cvat/exports/v1_2026-03-20/README.md`

- [ ] **Step 1: Create version directory**

```bash
mkdir -p projects/eaf/cvat/exports/v1_2026-03-20
```

- [ ] **Step 2: Move files via git mv**

```bash
git mv cvat_projects/project4_docling_heron_clean/annotations/instances_default.json projects/eaf/cvat/exports/v1_2026-03-20/instances_default.json
git mv cvat_projects/project4_docling_heron_clean/README.md projects/eaf/cvat/exports/v1_2026-03-20/README.md
```

- [ ] **Step 3: Remove the now-empty `cvat_projects/` tree**

```bash
rmdir cvat_projects/project4_docling_heron_clean/annotations
rmdir cvat_projects/project4_docling_heron_clean
rmdir cvat_projects
```
Expected: All `rmdir` succeed (dirs are empty). If `cvat_projects/project4_docling_heron_clean/images/` still exists with files, that's the rendered PNGs — keep them in place for now (they're already gitignored), they'll be regenerated under the new structure in Plan 2.

If `images/` exists and blocks `rmdir`, run instead:
```bash
mkdir -p projects/eaf/data/images
git mv cvat_projects/project4_docling_heron_clean/images/EAF-089-2025 projects/eaf/data/images/EAF-089-2025 2>/dev/null || mv cvat_projects/project4_docling_heron_clean/images/EAF-089-2025 projects/eaf/data/images/EAF-089-2025
git mv cvat_projects/project4_docling_heron_clean/images/EAF-477-2025 projects/eaf/data/images/EAF-477-2025 2>/dev/null || mv cvat_projects/project4_docling_heron_clean/images/EAF-477-2025 projects/eaf/data/images/EAF-477-2025
rm -rf cvat_projects
```
(The PNGs are gitignored so the `mv` is filesystem-only, no git impact.)

- [ ] **Step 4: Drop the old `.gitkeep` from `projects/eaf/cvat/exports/` since real content now exists**

```bash
git rm projects/eaf/cvat/exports/.gitkeep
```

- [ ] **Step 5: Verify integrity of moved JSON**

```bash
python3 -c "import json; d=json.load(open('projects/eaf/cvat/exports/v1_2026-03-20/instances_default.json')); print(f'images: {len(d[\"images\"])}, annotations: {len(d[\"annotations\"])}, categories: {len(d[\"categories\"])}')"
```
Expected: `images: 561, annotations: 3105, categories: 17`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: migrate CVAT export to projects/eaf/cvat/exports/v1_2026-03-20/

Moves the project-level COCO export (561 images, 3105 annotations,
17 categories) from the legacy cvat_projects/ path to the new
versioned exports/<v>_<date>/ layout. The version v1_2026-03-20
matches the date recorded in the export's README.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Move PDFs to `projects/eaf/data/pdfs/`

**Files:**
- Move: `data/pdfs/EAF-089-2025.pdf`, `data/pdfs/EAF-477-2025.pdf` → `projects/eaf/data/pdfs/`

- [ ] **Step 1: Move PDFs via git mv**

```bash
git mv data/pdfs/EAF-089-2025.pdf projects/eaf/data/pdfs/EAF-089-2025.pdf
git mv data/pdfs/EAF-477-2025.pdf projects/eaf/data/pdfs/EAF-477-2025.pdf
```

- [ ] **Step 2: Remove the empty `data/` tree**

```bash
rmdir data/pdfs data
```
Expected: succeed.

- [ ] **Step 3: Drop the placeholder `.gitkeep` for pdfs/**

```bash
git rm projects/eaf/data/pdfs/.gitkeep
```

- [ ] **Step 4: Verify both PDFs are present and readable**

```bash
ls -lh projects/eaf/data/pdfs/
file projects/eaf/data/pdfs/*.pdf
```
Expected: both `.pdf` listed, `file` reports `PDF document`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: move EAF PDFs to projects/eaf/data/pdfs/

Source PDFs (EAF-089-2025.pdf, EAF-477-2025.pdf) live with the
rest of the EAF project under the new factory structure. Total
size ~19MB — fits comfortably in git without LFS for now.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Move winner run `P_repeat_factor` to `projects/eaf/runs/`

**Files:**
- Move: `training/models/P_repeat_factor/history.json` → `projects/eaf/runs/P_repeat_factor/history.json`

- [ ] **Step 1: Create run directory**

```bash
mkdir -p projects/eaf/runs/P_repeat_factor
```

- [ ] **Step 2: Move via git mv**

```bash
git mv training/models/P_repeat_factor/history.json projects/eaf/runs/P_repeat_factor/history.json
```

- [ ] **Step 3: Remove empty `training/models/` directories left behind**

```bash
rmdir training/models/P_repeat_factor
rmdir training/models
```
Expected: succeed (other rounds' subdirs were already either deleted or never tracked).

- [ ] **Step 4: Drop the `.gitkeep` from `projects/eaf/runs/`**

```bash
git rm projects/eaf/runs/.gitkeep
```

- [ ] **Step 5: Verify history.json is readable**

```bash
python3 -c "import json; h=json.load(open('projects/eaf/runs/P_repeat_factor/history.json')); print(f'epochs logged: {len(h)}'); print(f'first epoch: {list(h[0].keys()) if isinstance(h, list) and h else \"object\"}')"
```
Expected: prints epoch count and key names. Won't fail.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: migrate winner run P_repeat_factor to projects/eaf/runs/

Moves history.json of the winning experiment (mAP@[.5:.95]=0.8700)
to the per-type runs/ layout. Best_model.pt is gitignored and
remains on local disk only — to be re-promoted via dlmf promote
in Plan 5. Other rounds (A..O) live in EXPERIMENTS.md as
historical narrative.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Move EXPERIMENTS.md to `projects/eaf/`

**Files:**
- Move: `training/EXPERIMENTS.md` → `projects/eaf/EXPERIMENTS.md`

- [ ] **Step 1: Move via git mv**

```bash
git mv training/EXPERIMENTS.md projects/eaf/EXPERIMENTS.md
```

- [ ] **Step 2: Verify other training/ files still exist (NOT touched in this plan)**

```bash
ls training/
```
Expected: at minimum `train_round2.py`, `train_round3.py`, `train_round4.py`, `train_strategies.py`, `draw_boxes_on_pdf.py`, `draw_boxes_on_pdf_v2.py`, `generate_comparison_pdf.py`, `generate_comparison_pdf_v2.py`, `reevaluate_real_map.py`, `run_docling_pipeline.py`, `test_best_model.py`. These migrate in plans 2-5.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "docs: move EXPERIMENTS.md to projects/eaf/

Per-type experiment narrative lives with the EAF project.
Documents the 16 experiments (A..P plus E_v2) and the winner
selection rationale.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Create `core/labels/doclaynet_17.yaml`

**Files:**
- Create: `core/__init__.py`, `core/labels/__init__.py`, `core/labels/doclaynet_17.yaml`

- [ ] **Step 1: Create package markers**

```bash
mkdir -p core/labels core/lib
```

Write empty file `/home/alonso/Documentos/Github/document-layout-model-training/core/__init__.py`:
```python
"""Document Layout Model Factory — shared code across document types."""
```

Write empty file `/home/alonso/Documentos/Github/document-layout-model-training/core/labels/__init__.py`:
```python
```

Write empty file `/home/alonso/Documentos/Github/document-layout-model-training/core/lib/__init__.py`:
```python
```

- [ ] **Step 2: Create the labels file**

The 17 labels in the order CVAT produced them (1-based IDs preserved). Write file `/home/alonso/Documentos/Github/document-layout-model-training/core/labels/doclaynet_17.yaml`:
```yaml
# DocLayNet 17 categories, in the 1-based ID order produced by CVAT.
# Used by all document types unless their config.yaml overrides `labels`.
# See projects/eaf/cvat/exports/v1_2026-03-20/README.md for the source mapping.
- Document Index
- List-item
- Footnote
- Checkbox-selected
- Page-header
- Checkbox-unselected
- Code
- Table
- Text
- Section-header
- Formula
- Picture
- Key-value-region
- Caption
- Title
- Form
- Page-footer
```

- [ ] **Step 3: Verify it parses correctly**

```bash
uv run python -c "import yaml; labels=yaml.safe_load(open('core/labels/doclaynet_17.yaml')); print(f'count={len(labels)}'); [print(f'  {i+1}: {n}') for i,n in enumerate(labels)]"
```
Expected: `count=17` and all 17 names listed.

- [ ] **Step 4: Commit**

```bash
git add core/__init__.py core/labels/__init__.py core/lib/__init__.py core/labels/doclaynet_17.yaml
git commit -m "feat: add core/ package skeleton with shared DocLayNet labels

The 17 labels in 1-based ID order match the CVAT export. Per-type
configs reference this file via YAML !include — see Task 11.
Empty __init__.py markers keep core/, core/labels/, and core/lib/
importable as packages.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Implement `core/lib/config.py` with TDD

**Files:**
- Create: `core/lib/config.py`, `tests/__init__.py`, `tests/test_config.py`, `tests/fixtures/sample_config.yaml`, `tests/fixtures/sample_labels.yaml`, `tests/fixtures/__init__.py`

The loader must:
1. Read a YAML file.
2. Resolve `!include <relative-path>` directives by reading the referenced file and substituting its content (path resolved relative to the including file).
3. Apply CLI-style overrides of dotted keys (`training.lora.rank=64`) producing a mutated copy.
4. Raise on unknown keys when overriding (avoid silent typos).

- [ ] **Step 1: Set up test fixtures and the failing test file**

Create `tests/__init__.py` (empty) and `tests/fixtures/__init__.py` (empty) so they are importable.

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/fixtures/sample_labels.yaml`:
```yaml
- LabelOne
- LabelTwo
- LabelThree
```

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/fixtures/sample_config.yaml`:
```yaml
project:
  slug: testdoc
  display_name: "Test Doc"
labels: !include sample_labels.yaml
training:
  lora:
    rank: 32
    alpha: 64
  lr: 1.0e-4
```

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_config.py`:
```python
from pathlib import Path

import pytest

from core.lib.config import apply_overrides, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_config_reads_basic_yaml():
    cfg = load_config(FIXTURES / "sample_config.yaml")
    assert cfg["project"]["slug"] == "testdoc"
    assert cfg["training"]["lr"] == 1.0e-4


def test_load_config_resolves_include_relative_to_file():
    cfg = load_config(FIXTURES / "sample_config.yaml")
    assert cfg["labels"] == ["LabelOne", "LabelTwo", "LabelThree"]


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(FIXTURES / "does_not_exist.yaml")


def test_apply_overrides_sets_nested_key():
    cfg = {"training": {"lora": {"rank": 32}}}
    out = apply_overrides(cfg, ["training.lora.rank=64"])
    assert out["training"]["lora"]["rank"] == 64
    # Original is untouched (returns a copy).
    assert cfg["training"]["lora"]["rank"] == 32


def test_apply_overrides_parses_floats_and_ints():
    cfg = {"training": {"lr": 1e-4, "batch_size": 1}}
    out = apply_overrides(cfg, ["training.lr=5e-5", "training.batch_size=2"])
    assert out["training"]["lr"] == 5e-5
    assert out["training"]["batch_size"] == 2


def test_apply_overrides_parses_booleans():
    cfg = {"augmentation": {"enabled": False}}
    out = apply_overrides(cfg, ["augmentation.enabled=true"])
    assert out["augmentation"]["enabled"] is True


def test_apply_overrides_unknown_key_raises():
    cfg = {"training": {"lr": 1e-4}}
    with pytest.raises(KeyError, match="training.typo"):
        apply_overrides(cfg, ["training.typo=5"])


def test_apply_overrides_no_overrides_returns_equal_dict():
    cfg = {"a": 1, "b": {"c": 2}}
    out = apply_overrides(cfg, [])
    assert out == cfg
```

- [ ] **Step 2: Run tests to confirm they fail (no implementation yet)**

```bash
uv run pytest tests/test_config.py -v
```
Expected: collection error (`ModuleNotFoundError: No module named 'core.lib.config'`) or all tests fail. This confirms the suite is wired correctly.

- [ ] **Step 3: Implement `core/lib/config.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/lib/config.py`:
```python
"""YAML config loader with !include resolution and CLI override application."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader extension that resolves `!include <path>` relative to the YAML file."""


def _construct_include(loader: _IncludeLoader, node: yaml.Node) -> Any:
    rel = loader.construct_scalar(node)
    base = Path(loader.name).parent if loader.name else Path.cwd()
    target = (base / rel).resolve()
    with target.open("r", encoding="utf-8") as f:
        sub_loader = _IncludeLoader(f)
        sub_loader.name = str(target)
        try:
            return sub_loader.get_single_data()
        finally:
            sub_loader.dispose()


_IncludeLoader.add_constructor("!include", _construct_include)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config from `path`, resolving any `!include` directives."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        loader = _IncludeLoader(f)
        loader.name = str(p)
        try:
            data = loader.get_single_data()
        finally:
            loader.dispose()
    return data or {}


def _coerce(value: str) -> Any:
    """Parse override value as YAML scalar (handles ints, floats, bools, strings)."""
    return yaml.safe_load(value)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Return a deep copy of `cfg` with dotted-key overrides applied.

    Overrides are strings of the form `a.b.c=value`. The leaf must already
    exist in `cfg`; missing keys raise KeyError to catch typos.
    """
    out = copy.deepcopy(cfg)
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"override missing '=': {raw!r}")
        dotted, value_str = raw.split("=", 1)
        keys = dotted.split(".")
        node = out
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(dotted)
            node = node[k]
        leaf = keys[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(dotted)
        node[leaf] = _coerce(value_str)
    return out
```

- [ ] **Step 4: Run tests and confirm all pass**

```bash
uv run pytest tests/test_config.py -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/lib/config.py tests/__init__.py tests/test_config.py tests/fixtures/__init__.py tests/fixtures/sample_config.yaml tests/fixtures/sample_labels.yaml
git commit -m "feat(core): add YAML config loader with !include and override support

core.lib.config provides:
- load_config(path): reads YAML, resolves !include directives
  relative to the including file's directory.
- apply_overrides(cfg, ['a.b=value']): returns a copy with dotted-
  key overrides applied. Raises KeyError on unknown keys to catch
  typos. Values are coerced via yaml.safe_load (ints, floats,
  booleans, strings).

8 unit tests in tests/test_config.py cover the contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Create `projects/eaf/config.yaml`

**Files:**
- Create: `projects/eaf/config.yaml`

- [ ] **Step 1: Write the EAF config**

Write file `/home/alonso/Documentos/Github/document-layout-model-training/projects/eaf/config.yaml`:
```yaml
project:
  slug: eaf
  display_name: "Estudios de Análisis de Falla (CEN)"

# Identificación automática del tipo de documento (regex + Ollama fallback)
classification:
  filename_regex: "^EAF[-_]\\d+[-_]\\d{4}\\.pdf$"
  ollama_fallback:
    enabled: true
    model: "qwen2.5:7b"
    prompt: "¿Es un Estudio de Análisis de Falla del CEN? Responde solo 'yes' o 'no'."

render:
  dpi: 300

cvat:
  project_name: "Docling Heron CLEAN - EAF"
  url: "http://localhost:8080"

# Labels: 17 fijos de DocLayNet (Opción A del spec)
labels: !include ../../core/labels/doclaynet_17.yaml

# Hyperparámetros DEFAULT — cada run puede overridear via CLI
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

# Post-procesamiento de inferencia
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
```

- [ ] **Step 2: Verify it loads with the new loader**

```bash
uv run python -c "
from core.lib.config import load_config
cfg = load_config('projects/eaf/config.yaml')
assert cfg['project']['slug'] == 'eaf'
assert len(cfg['labels']) == 17
assert cfg['labels'][0] == 'Document Index'
assert cfg['training']['lora']['rank'] == 32
assert cfg['postprocess']['thresholds']['default'] == 0.5
print('OK: projects/eaf/config.yaml loads correctly with 17 labels.')
"
```
Expected: prints `OK: projects/eaf/config.yaml loads correctly with 17 labels.`

- [ ] **Step 3: Verify override mechanism on the real config**

```bash
uv run python -c "
from core.lib.config import load_config, apply_overrides
cfg = load_config('projects/eaf/config.yaml')
out = apply_overrides(cfg, ['training.lora.rank=64', 'training.lr=5e-5'])
assert out['training']['lora']['rank'] == 64
assert out['training']['lr'] == 5e-5
assert cfg['training']['lora']['rank'] == 32  # original unchanged
print('OK: overrides apply correctly to real EAF config.')
"
```
Expected: prints `OK: overrides apply correctly to real EAF config.`

- [ ] **Step 4: Commit**

```bash
git add projects/eaf/config.yaml
git commit -m "feat(eaf): add config.yaml for the EAF document type

Captures all per-type knobs: classification (regex + Ollama
fallback), CVAT project name, labels (via !include of the shared
DocLayNet 17), training hyperparams (LoRA r=32, repeat-factor
sampling — the params that produced the winning P run), and
post-processing thresholds. This is the file every dlmf command
in plans 2-6 will load via core.lib.config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Update `README.md` with migration notice

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README.md to preserve content**

```bash
head -3 README.md
```

- [ ] **Step 2: Prepend migration notice to README.md**

Insert this block at the **top** of the existing `README.md` (before the first `# docling-layout-fine-tuning` heading):

```markdown
> ⚠️ **Repo en migración a Layout Model Factory.** El diseño objetivo está en
> [`docs/superpowers/specs/2026-05-02-layout-model-factory-design.md`](docs/superpowers/specs/2026-05-02-layout-model-factory-design.md).
> El plan de migración por fases está en [`docs/superpowers/plans/`](docs/superpowers/plans/).
>
> **Estado actual (Plan 01 completado):** estructura `projects/eaf/` y `core/lib/config.py` listos.
> Los scripts originales (`generate_heron_coco.py`, `upload_to_cvat.py`,
> `clean_overlaps_v3.py`, `training/train_round*.py`, etc.) **siguen funcionando**
> y se migrarán en planes 02-06.

---

```

(Use the `Edit` tool with `old_string` = the current first line of README and `new_string` = the migration block + the original first line.)

- [ ] **Step 3: Verify**

```bash
head -15 README.md
```
Expected: the migration notice at top, followed by the original heading.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add migration notice to README

Flags the repo as in-flight migration to the Layout Model Factory
design and points to the spec + plans. The original POC content
follows untouched until later plans migrate the scripts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Final verification & end-of-plan checklist

- [ ] **Step 1: Verify directory structure matches the spec**

```bash
find projects core tests -type f | sort
```
Expected at minimum:
```
core/__init__.py
core/labels/__init__.py
core/labels/doclaynet_17.yaml
core/lib/__init__.py
core/lib/config.py
projects/eaf/EXPERIMENTS.md
projects/eaf/config.yaml
projects/eaf/cvat/exports/v1_2026-03-20/README.md
projects/eaf/cvat/exports/v1_2026-03-20/instances_default.json
projects/eaf/data/pdfs/EAF-089-2025.pdf
projects/eaf/data/pdfs/EAF-477-2025.pdf
projects/eaf/runs/P_repeat_factor/history.json
tests/__init__.py
tests/fixtures/__init__.py
tests/fixtures/sample_config.yaml
tests/fixtures/sample_labels.yaml
tests/test_config.py
```
(Plus `.gitkeep` files in `projects/eaf/runs/` and `projects/eaf/models/` if they remained.)

- [ ] **Step 2: Verify deletions stuck**

```bash
ls clean_overlaps.py clean_overlaps_v2.py 2>&1 | grep -E "No such|cannot"
ls https:/ 2>&1 | grep -E "No such|cannot"
ls cvat_projects/ 2>&1 | grep -E "No such|cannot"
ls data/ 2>&1 | grep -E "No such|cannot"
```
Expected: each command reports the path does not exist.

- [ ] **Step 3: Run all tests**

```bash
uv run pytest -v
```
Expected: 8 tests passing in `tests/test_config.py`. No collection errors.

- [ ] **Step 4: Verify git is clean**

```bash
git status --short
```
Expected: empty (all changes committed).

- [ ] **Step 5: Verify the legacy scripts still exist (they migrate later)**

```bash
ls clean_overlaps_v3.py generate_heron_coco.py upload_to_cvat.py reimport_annotations.py
ls training/train_round{2,3,4}.py training/draw_boxes_on_pdf{,_v2}.py training/reevaluate_real_map.py training/run_docling_pipeline.py
ls scripts/render_pdf_to_png.py scripts/restore_cvat_project4.py
```
Expected: all listed (no errors).

- [ ] **Step 6: Tag the milestone**

```bash
git tag -a plan-01-foundation -m "Plan 01 complete: repo restructured to projects/<type>/ + core/ skeleton with config loader."
```

---

## Self-Review

**Spec coverage:**
- ✅ Section 3 of spec (estructura del repo): Tasks 4-9 create it.
- ✅ Section 4 of spec (`config.yaml` por tipo): Task 11.
- ✅ Section 7 of spec (plan migración fases 1-3): Tasks 1, 2, 3, 4-8.
- 🔜 Section 5 of spec (CLI commands): only `pyproject.toml` script entry registered as placeholder; commands implemented in plans 2-6.
- 🔜 Sections 6 (data flow), 8 (escalabilidad): no code in this plan, design captured in spec.
- 🔜 Section 10 of spec (definition of done MVP): partial (structure present); full DoD met after plan 6.

**Placeholder scan:** No `TBD` / `TODO` / "implement later" left. All overrides have concrete examples. All steps have either commands or full code.

**Type/name consistency:**
- `core.lib.config.load_config` and `core.lib.config.apply_overrides` used identically in tests, in Task 11 verification, and in the plan body.
- `projects/eaf/cvat/exports/v1_2026-03-20/instances_default.json` referenced consistently (Tasks 5, 13).
- `core/labels/doclaynet_17.yaml` referenced as `!include ../../core/labels/doclaynet_17.yaml` from `projects/eaf/config.yaml` — relative path verified (`projects/eaf/` → up two → `core/labels/`).

**Out-of-scope items deferred to later plans (acknowledged in header):**
- `core/render.py`, `core/cvat_sync.py`, `core/predict.py`, `core/train.py`, `core/evaluate.py`, `core/promote.py`, `core/classify_doctype.py`, `core/init_project.py`, `core/cli.py`.
- `core/lib/{data,model,postproc,tracking}.py`.
- MLflow integration.
- Parity gate (mAP >= 0.86).
- Migration of all `training/train_*.py`, `clean_overlaps_v3.py`, `draw_boxes_on_pdf*.py`, etc.
