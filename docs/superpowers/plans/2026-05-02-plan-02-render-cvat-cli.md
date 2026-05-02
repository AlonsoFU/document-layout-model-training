# Plan 02 — Render + CVAT Sync + Typer CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar los primeros 3 comandos del CLI `dlmf` (`render`, `cvat-push`, `cvat-pull`) generalizando los scripts existentes (`scripts/render_pdf_to_png.py`, `scripts/restore_cvat_project4.py`, `upload_to_cvat.py`) en módulos parametrizados por `--project=<slug>`.

**Architecture:** Typer app en `core/cli.py` que despacha a módulos en `core/`. Cliente CVAT compartido en `core/lib/cvat.py` (auth, request-id polling, COCO import/export). Render usa `pdftoppm` directamente (evita OOM de pdf2image).

**Tech Stack:** Typer, requests, PyYAML, pytest + responses (HTTP mocking), pdftoppm (binario del sistema).

**Salida verificable:**
- `dlmf render --project=eaf` regenera los 561 PNGs (limpiando primero).
- `dlmf cvat-push --project=eaf` crea el proyecto + 2 tasks + sube imágenes + (opcional) anotaciones desde el último export.
- `dlmf cvat-pull --project=eaf` exporta COCO desde CVAT a `projects/eaf/cvat/exports/v<N+1>_<YYYY-MM-DD>/`.
- `pytest` pasa los tests de Plan 01 (8) más los nuevos (~12).
- Tag `plan-02-render-cvat-cli` apunta al último commit.

**Out of scope (planes futuros):**
- Plan 03: `core/predict.py --pre-annotate` (pre-anotaciones generadas por Heron) + `core/lib/postproc.py`. Por ahora `cvat-push` solo sube imágenes; las pre-anotaciones se cargan desde un COCO pre-existente o se omiten.
- Plan 04: training + MLflow.
- Plan 05: evaluate, promote, predict --output=PDF.
- Plan 06: classify, init-project, parity gate.

---

## File Structure

### Files to CREATE

| Path | Responsabilidad |
|---|---|
| `core/cli.py` | Typer app principal; registra subcomandos `render`, `cvat-push`, `cvat-pull`. Punto de entrada `dlmf`. |
| `core/render.py` | Función `render(project_slug)` que renderiza PDFs en `projects/<slug>/data/pdfs/` a PNGs en `projects/<slug>/data/images/<pdf_stem>/pagina-NNN.png` usando `pdftoppm`. |
| `core/cvat_sync.py` | Funciones `push(project_slug, coco_path=None)` y `pull(project_slug, version=None)`. Usa `core/lib/cvat.py`. |
| `core/lib/cvat.py` | Cliente HTTP minimal: `CvatClient(url, auth)` con métodos `get_project_by_name`, `create_project`, `create_task`, `upload_data`, `import_annotations`, `export_dataset`, `wait_request`. |
| `tests/test_render.py` | Test que `render` produce el número correcto de PNGs por PDF (usa monkeypatch para no llamar `pdftoppm` real; valida solo la lógica de wiring + paths). |
| `tests/test_cvat_client.py` | Tests del cliente CVAT con `responses` library (mock HTTP). Cubre auth, polling de request-id (HTTP 410 deprecation handling), error paths. |
| `tests/test_cvat_sync.py` | Tests de `push`/`pull` con cliente mockeado. Cubre: filtrado COCO por task, reasignación de IDs, normalización de nombres `_1` suffix, naming de versiones (`v2_<fecha>`). |
| `tests/test_cli.py` | Tests con `typer.testing.CliRunner`: comandos invocables, flags presentes, errores claros. |

### Files to MODIFY

| Path | Cambio |
|---|---|
| `pyproject.toml` | Añadir `responses>=0.25` a `dev` deps. |
| `README.md` | Actualizar la nota de migración del header: "Estado actual (Plan 02 completado)" + listar los 3 comandos disponibles. |

### Files to DELETE (al final, después de validar paridad)

| Path | Razón |
|---|---|
| `scripts/render_pdf_to_png.py` | Reemplazado por `core/render.py` (mismo comportamiento). |
| `scripts/restore_cvat_project4.py` | Reemplazado por `core/cvat_sync.py push`. |
| `upload_to_cvat.py` | Idem (es el "padre" del restore script, ya estaba obsoleto). |
| `reimport_annotations.py` | Reemplazado por `cvat-push --coco=<file>`. |

### Files NOT touched in this plan

`generate_heron_coco.py` (Plan 03), `clean_overlaps_v3.py` (Plan 03), `training/*.py` (Plans 04-05).

---

## Tasks

### Task 1: Add `responses` to dev deps + skeleton Typer app

**Files:**
- Modify: `pyproject.toml`
- Create: `core/cli.py`, `tests/test_cli.py`

- [ ] **Step 1: Add `responses` to dev deps**

Edit `pyproject.toml`. Find:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
```

Replace with:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "responses>=0.25",
]
```

- [ ] **Step 2: Sync deps**

```bash
uv sync --extra dev
```
Expected: `responses` installed.

- [ ] **Step 3: Write failing CLI test**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_cli.py`:
```python
"""Tests for the dlmf Typer CLI."""
from typer.testing import CliRunner

from core.cli import app

runner = CliRunner()


def test_cli_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render" in result.stdout
    assert "cvat-push" in result.stdout
    assert "cvat-pull" in result.stdout


def test_cli_render_help_mentions_project_flag():
    result = runner.invoke(app, ["render", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout


def test_cli_cvat_push_help_mentions_project_flag():
    result = runner.invoke(app, ["cvat-push", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout


def test_cli_cvat_pull_help_mentions_project_flag():
    result = runner.invoke(app, ["cvat-pull", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
```

- [ ] **Step 4: Run test, confirm import error**

```bash
uv run pytest tests/test_cli.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.cli'`.

- [ ] **Step 5: Implement skeleton `core/cli.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/cli.py`:
```python
"""Typer CLI for the Document Layout Model Factory.

Entry point declared in pyproject.toml as `dlmf = "core.cli:app"`.
Subcommands are wired in this file but their implementations live
in core/render.py, core/cvat_sync.py, etc.
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="dlmf",
    help="Document Layout Model Factory — train layout models per document type.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="render")
def render_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug (e.g. 'eaf')."),
) -> None:
    """Render the project's PDFs to PNGs at the configured DPI."""
    from core.render import render

    render(project)


@app.command(name="cvat-push")
def cvat_push_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    coco: str = typer.Option(
        None, "--coco", help="Optional path to a COCO JSON to pre-load as annotations."
    ),
) -> None:
    """Create the CVAT project + tasks and upload images (and optional pre-labels)."""
    from core.cvat_sync import push

    push(project, coco_path=coco)


@app.command(name="cvat-pull")
def cvat_pull_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project slug."),
    version: str = typer.Option(
        None,
        "--version",
        help="Version label (e.g. v3). Defaults to next sequential version with today's date.",
    ),
) -> None:
    """Export the CVAT project as COCO and write to projects/<slug>/cvat/exports/v<N>_<date>/."""
    from core.cvat_sync import pull

    pull(project, version=version)


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Run tests; collection should fail because `core.render` and `core.cvat_sync` don't exist yet**

```bash
uv run pytest tests/test_cli.py -v
```
Expected: All 4 CLI tests PASS — they only check `--help` output, which doesn't import the underlying modules. The lazy `from core.render import render` inside the command body avoids the import until the command actually runs. If they FAIL with import errors at test collection, the lazy imports are not lazy enough — verify the imports are inside the function bodies, not at module top.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml core/cli.py tests/test_cli.py
git commit -m "feat(cli): scaffold dlmf Typer app with render, cvat-push, cvat-pull commands

The CLI shell registers all three subcommands with --project required.
Implementations are lazily imported inside the command bodies so help
output works even before render.py / cvat_sync.py exist (next tasks).

Adds 'responses>=0.25' to dev deps for HTTP mocking in upcoming
CVAT client tests.

4 tests in tests/test_cli.py verify --help output for each command.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Implement `core/render.py` with TDD

**Files:**
- Create: `core/render.py`, `tests/test_render.py`

The renderer must:
1. Load the project's `config.yaml` to get `render.dpi`.
2. Find all PDFs in `projects/<slug>/data/pdfs/*.pdf`.
3. For each PDF, render to `projects/<slug>/data/images/<pdf_stem>/pagina-NNN.png` using `pdftoppm`.
4. Skip PDFs whose target dir already has the right number of PNGs (idempotent re-runs).
5. Validate the output count matches the PDF page count.

- [ ] **Step 1: Write failing tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_render.py`:
```python
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
```

- [ ] **Step 2: Run tests; expect failure (no module)**

```bash
uv run pytest tests/test_render.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.render'`.

- [ ] **Step 3: Implement `core/render.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/render.py`:
```python
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
        existing = sorted(out_dir.glob("pagina-*.png"))
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
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_render.py -v
```
Expected: 4 tests pass.

- [ ] **Step 5: Smoke-test against the real EAF project**

```bash
ls projects/eaf/data/images/ 2>&1 | head -5
uv run dlmf render --project=eaf 2>&1 | tail -10
ls projects/eaf/data/images/EAF-089-2025/ | wc -l
ls projects/eaf/data/images/EAF-477-2025/ | wc -l
```
Expected outputs: if PNGs already present (from prior session), `[skip]` messages and counts of 399 and 162. If not present, render generates them and reports counts.

- [ ] **Step 6: Commit**

```bash
git add core/render.py tests/test_render.py
git commit -m "feat(core): add render command — pdftoppm-based PDF→PNG batch

core.render.render(project_slug) reads projects/<slug>/config.yaml for
DPI, finds PDFs in data/pdfs/, and renders each to data/images/<stem>/
pagina-NNN.png. Idempotent: skips PDFs whose image dir already matches
the expected page count (via pdfinfo).

Uses pdftoppm directly to stream pages to disk (avoids the ~10GB RAM
spike of pdf2image.convert_from_path on 400-page PDFs at 300 DPI).

4 tests in tests/test_render.py cover the dispatch logic with
monkeypatched subprocess.

Smoke-tested on projects/eaf/ — both EAF PDFs (561 pages total)
render correctly via 'dlmf render --project=eaf'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Implement `core/lib/cvat.py` HTTP client with TDD

**Files:**
- Create: `core/lib/cvat.py`, `tests/test_cvat_client.py`

The client must:
1. Authenticate via HTTP basic (admin/admin by default, or from env `CVAT_USER`/`CVAT_PASSWORD`).
2. List projects, create projects with labels, list tasks, create tasks.
3. Upload data (zipped images) to a task.
4. Import annotations (zipped COCO).
5. Export dataset (returns JSON content).
6. Poll `/api/requests/{rq_id}` for async operations (CVAT deprecated `action=import_status`).

- [ ] **Step 1: Write failing tests**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_cvat_client.py`:
```python
"""Tests for core.lib.cvat — HTTP client for CVAT REST API.

Uses the `responses` library to mock HTTP. No real CVAT server is contacted.
"""
import io
import json
import zipfile
from pathlib import Path

import pytest
import responses

from core.lib.cvat import CvatClient


@pytest.fixture()
def client():
    return CvatClient("http://localhost:8080", auth=("admin", "admin"))


@responses.activate
def test_get_project_by_name_returns_match(client):
    responses.get(
        "http://localhost:8080/api/projects",
        match=[responses.matchers.query_param_matcher({"search": "Foo"})],
        json={
            "count": 1,
            "results": [{"id": 7, "name": "Foo", "tasks": {"count": 0}}],
        },
    )
    p = client.get_project_by_name("Foo")
    assert p == {"id": 7, "name": "Foo", "tasks": {"count": 0}}


@responses.activate
def test_get_project_by_name_returns_none_when_no_exact_match(client):
    responses.get(
        "http://localhost:8080/api/projects",
        match=[responses.matchers.query_param_matcher({"search": "Foo"})],
        json={"count": 1, "results": [{"id": 7, "name": "Foobar"}]},
    )
    assert client.get_project_by_name("Foo") is None


@responses.activate
def test_create_project_with_labels(client):
    responses.post(
        "http://localhost:8080/api/projects",
        json={"id": 42, "name": "Test"},
        status=201,
    )
    pid = client.create_project("Test", ["A", "B"])
    assert pid == 42
    body = json.loads(responses.calls[0].request.body)
    assert body["name"] == "Test"
    assert body["labels"] == [{"name": "A"}, {"name": "B"}]


@responses.activate
def test_create_task(client):
    responses.post(
        "http://localhost:8080/api/tasks",
        json={"id": 99},
        status=201,
    )
    tid = client.create_task("EAF-089-2025", project_id=42)
    assert tid == 99


@responses.activate
def test_upload_data_zip_polls_until_finished(client):
    responses.post(
        "http://localhost:8080/api/tasks/99/data",
        status=202,
    )
    responses.get(
        "http://localhost:8080/api/tasks/99/status",
        json={"state": "Started"},
    )
    responses.get(
        "http://localhost:8080/api/tasks/99/status",
        json={"state": "Finished"},
    )
    buf = io.BytesIO(b"zipcontent")
    client.upload_data(99, buf, poll_interval=0)
    # 3 calls: POST data + 2 GET status
    assert len(responses.calls) == 3


@responses.activate
def test_import_annotations_uses_request_id_polling(client):
    """CVAT current API: POST returns 202 with rq_id; GET /api/requests/{rq_id}."""
    responses.post(
        "http://localhost:8080/api/tasks/99/annotations",
        json={"rq_id": "rq_xyz"},
        status=202,
    )
    responses.get(
        "http://localhost:8080/api/requests/rq_xyz",
        json={"status": "started"},
    )
    responses.get(
        "http://localhost:8080/api/requests/rq_xyz",
        json={"status": "finished"},
    )
    buf = io.BytesIO(b"cocozip")
    client.import_annotations(99, buf, fmt="COCO 1.0", poll_interval=0)
    assert len(responses.calls) == 3


@responses.activate
def test_import_annotations_raises_on_failure(client):
    responses.post(
        "http://localhost:8080/api/tasks/99/annotations",
        json={"rq_id": "rq_bad"},
        status=202,
    )
    responses.get(
        "http://localhost:8080/api/requests/rq_bad",
        json={"status": "failed", "message": "bad coco"},
    )
    with pytest.raises(RuntimeError, match="bad coco"):
        client.import_annotations(99, io.BytesIO(b"x"), fmt="COCO 1.0", poll_interval=0)


@responses.activate
def test_get_task_annotations_count(client):
    responses.get(
        "http://localhost:8080/api/tasks/99/annotations",
        json={"shapes": [{}, {}, {}], "tracks": [], "tags": []},
    )
    n = client.count_task_annotations(99)
    assert n == 3
```

- [ ] **Step 2: Run tests; expect import error**

```bash
uv run pytest tests/test_cvat_client.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.lib.cvat'`.

- [ ] **Step 3: Implement `core/lib/cvat.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/lib/cvat.py`:
```python
"""HTTP client for the CVAT REST API.

Handles auth (basic), project/task CRUD, data upload, COCO import/export,
and the async request-id polling pattern that replaced the deprecated
`action=import_status` endpoint in recent CVAT versions.
"""
from __future__ import annotations

import time
from typing import Any, BinaryIO

import requests


class CvatClient:
    def __init__(self, url: str, auth: tuple[str, str]) -> None:
        self.url = url.rstrip("/")
        self.auth = auth
        self._session = requests.Session()
        self._session.auth = auth

    # ---------- Projects ----------

    def get_project_by_name(self, name: str) -> dict[str, Any] | None:
        r = self._session.get(f"{self.url}/api/projects", params={"search": name})
        r.raise_for_status()
        for p in r.json().get("results", []):
            if p["name"] == name:
                return p
        return None

    def create_project(self, name: str, labels: list[str]) -> int:
        payload = {"name": name, "labels": [{"name": n} for n in labels]}
        r = self._session.post(f"{self.url}/api/projects", json=payload)
        r.raise_for_status()
        return int(r.json()["id"])

    # ---------- Tasks ----------

    def get_task_by_name(self, project_id: int, name: str) -> dict[str, Any] | None:
        r = self._session.get(
            f"{self.url}/api/tasks",
            params={"project_id": project_id, "search": name},
        )
        r.raise_for_status()
        for t in r.json().get("results", []):
            if t["name"] == name and t.get("project_id") == project_id:
                return t
        return None

    def create_task(self, name: str, project_id: int) -> int:
        r = self._session.post(
            f"{self.url}/api/tasks",
            json={"name": name, "project_id": project_id},
        )
        r.raise_for_status()
        return int(r.json()["id"])

    # ---------- Data upload ----------

    def upload_data(
        self,
        task_id: int,
        zip_stream: BinaryIO,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
    ) -> None:
        r = self._session.post(
            f"{self.url}/api/tasks/{task_id}/data",
            files={"client_files[0]": ("images.zip", zip_stream, "application/zip")},
            data={
                "image_quality": 100,
                "use_zip_chunks": "true",
                "sorting_method": "lexicographical",
            },
        )
        r.raise_for_status()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self._session.get(f"{self.url}/api/tasks/{task_id}/status").json()
            state = s.get("state", "")
            if state == "Finished":
                return
            if state == "Failed":
                raise RuntimeError(f"task {task_id} data processing failed: {s}")
            time.sleep(poll_interval)
        raise TimeoutError(f"task {task_id} data upload did not finish in {timeout}s")

    # ---------- Annotations ----------

    def import_annotations(
        self,
        task_id: int,
        zip_stream: BinaryIO,
        fmt: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> None:
        r = self._session.post(
            f"{self.url}/api/tasks/{task_id}/annotations",
            params={"format": fmt},
            files={"annotation_file": ("annotations.zip", zip_stream, "application/zip")},
        )
        if r.status_code not in (200, 201, 202):
            raise RuntimeError(f"annotation import HTTP {r.status_code}: {r.text[:500]}")
        rq_id = (r.json() or {}).get("rq_id")
        if not rq_id:
            return  # synchronous response
        self._wait_request(rq_id, poll_interval=poll_interval, timeout=timeout)

    def _wait_request(self, rq_id: str, poll_interval: float, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self._session.get(f"{self.url}/api/requests/{rq_id}")
            r.raise_for_status()
            s = r.json()
            status = s.get("status")
            if status == "finished":
                return
            if status == "failed":
                raise RuntimeError(s.get("message") or f"request {rq_id} failed")
            time.sleep(poll_interval)
        raise TimeoutError(f"request {rq_id} did not finish in {timeout}s")

    def count_task_annotations(self, task_id: int) -> int:
        r = self._session.get(f"{self.url}/api/tasks/{task_id}/annotations")
        r.raise_for_status()
        d = r.json()
        return len(d.get("shapes", [])) + len(d.get("tracks", [])) + len(d.get("tags", []))

    # ---------- Dataset export ----------

    def export_dataset(
        self,
        project_id: int,
        fmt: str = "COCO 1.0",
        save_images: bool = False,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
    ) -> bytes:
        """Trigger export and return the downloaded ZIP bytes."""
        r = self._session.post(
            f"{self.url}/api/projects/{project_id}/dataset/export",
            params={"format": fmt, "save_images": str(save_images).lower()},
        )
        r.raise_for_status()
        rq_id = r.json().get("rq_id")
        if not rq_id:
            raise RuntimeError("CVAT did not return rq_id for export")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sr = self._session.get(f"{self.url}/api/requests/{rq_id}")
            sr.raise_for_status()
            s = sr.json()
            status = s.get("status")
            if status == "finished":
                url = s.get("result_url")
                if not url:
                    raise RuntimeError(f"export {rq_id} finished but no result_url")
                dr = self._session.get(url)
                dr.raise_for_status()
                return dr.content
            if status == "failed":
                raise RuntimeError(s.get("message") or f"export {rq_id} failed")
            time.sleep(poll_interval)
        raise TimeoutError(f"export {rq_id} did not finish in {timeout}s")
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_cvat_client.py -v
```
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/lib/cvat.py tests/test_cvat_client.py
git commit -m "feat(core): add CvatClient — HTTP wrapper for CVAT REST API

core.lib.cvat.CvatClient covers the operations cvat_sync needs:
- Project CRUD: get_project_by_name, create_project (with labels).
- Task CRUD: get_task_by_name, create_task.
- Data upload: upload_data (zipped images, polls task status).
- Annotations import: import_annotations (uses /api/requests/{rq_id}
  polling — the deprecated action=import_status returns HTTP 410 in
  current CVAT).
- Dataset export: export_dataset (project-level COCO with rq polling
  and result_url download).
- Helpers: count_task_annotations.

8 tests in tests/test_cvat_client.py mock all HTTP via the responses
library — no live CVAT server needed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Implement `core/cvat_sync.py push` with TDD

**Files:**
- Create: `core/cvat_sync.py`, `tests/test_cvat_sync.py`

The `push` function must:
1. Load `projects/<slug>/config.yaml` for cvat.project_name and cvat.url.
2. Get-or-create the CVAT project with the 17 labels from config.
3. For each PDF in `projects/<slug>/data/pdfs/`:
   - Get-or-create a task named `<pdf_stem>`.
   - If task has 0 images, zip the PNGs from `data/images/<pdf_stem>/` and upload.
   - If `--coco=<path>` was given, filter the COCO for this task's images and import annotations.
4. Idempotent: skip image upload if task already has images, skip annotation import if task already has annotations.

Note: this task ONLY implements `push`. `pull` is Task 5.

- [ ] **Step 1: Write failing tests for `push`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_cvat_sync.py`:
```python
"""Tests for core.cvat_sync — push (creates project + tasks + uploads) and pull (export)."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from core.cvat_sync import _filter_coco_for_task, push


@pytest.fixture()
def fake_project(tmp_path, monkeypatch):
    proj = tmp_path / "projects" / "test"
    (proj / "data" / "pdfs").mkdir(parents=True)
    (proj / "data" / "images" / "doc-A").mkdir(parents=True)
    (proj / "data" / "images" / "doc-B").mkdir(parents=True)
    # Create 2 fake PNGs per PDF
    for i in (1, 2):
        (proj / "data" / "images" / "doc-A" / f"pagina-{i:03d}.png").write_bytes(b"\x89PNG")
        (proj / "data" / "images" / "doc-B" / f"pagina-{i:03d}.png").write_bytes(b"\x89PNG")
    # Empty PDF placeholders so push iterates them
    (proj / "data" / "pdfs" / "doc-A.pdf").write_bytes(b"%PDF")
    (proj / "data" / "pdfs" / "doc-B.pdf").write_bytes(b"%PDF")
    (proj / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "project": {"slug": "test"},
                "cvat": {"project_name": "Test Project", "url": "http://localhost:8080"},
                "labels": ["A", "B", "C"],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    return proj


def test_push_creates_project_and_tasks_and_uploads(fake_project, monkeypatch):
    client = MagicMock()
    client.get_project_by_name.return_value = None
    client.create_project.return_value = 42
    client.get_task_by_name.return_value = None
    client.create_task.side_effect = [101, 102]

    monkeypatch.setattr("core.cvat_sync._build_client", lambda cfg: client)

    push("test", coco_path=None)

    client.create_project.assert_called_once_with("Test Project", ["A", "B", "C"])
    assert client.create_task.call_count == 2
    assert client.upload_data.call_count == 2
    client.import_annotations.assert_not_called()


def test_push_skips_existing_project_and_tasks_with_images(fake_project, monkeypatch):
    client = MagicMock()
    client.get_project_by_name.return_value = {"id": 42, "name": "Test Project"}
    client.get_task_by_name.side_effect = [
        {"id": 101, "size": 2},  # already has 2 images
        {"id": 102, "size": 0},  # empty, needs upload
    ]
    client.count_task_annotations.return_value = 0

    monkeypatch.setattr("core.cvat_sync._build_client", lambda cfg: client)

    push("test", coco_path=None)

    client.create_project.assert_not_called()
    client.create_task.assert_not_called()
    assert client.upload_data.call_count == 1  # only doc-B uploaded


def test_filter_coco_for_task_renumbers_ids_and_strips_suffix():
    coco = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "A"}],
        "images": [
            {"id": 1, "file_name": "pagina-001.png", "width": 10, "height": 20},
            {"id": 2, "file_name": "pagina-002.png", "width": 10, "height": 20},
            {"id": 3, "file_name": "pagina-001_1.png", "width": 10, "height": 20},
            {"id": 4, "file_name": "pagina-002_1.png", "width": 10, "height": 20},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},
            {"id": 2, "image_id": 3, "category_id": 1, "bbox": [0, 0, 5, 5]},
        ],
    }
    out = _filter_coco_for_task(
        coco, image_filenames={"pagina-001.png", "pagina-002.png"}, strip_suffix=False
    )
    assert len(out["images"]) == 2
    assert {i["file_name"] for i in out["images"]} == {"pagina-001.png", "pagina-002.png"}
    assert out["images"][0]["id"] == 1
    assert out["images"][1]["id"] == 2

    out2 = _filter_coco_for_task(
        coco, image_filenames={"pagina-001.png", "pagina-002.png"}, strip_suffix=True
    )
    # strip_suffix removes _1 from source filenames before matching
    assert len(out2["images"]) == 2
    names = {i["file_name"] for i in out2["images"]}
    assert names == {"pagina-001.png", "pagina-002.png"}
```

- [ ] **Step 2: Run tests; expect import error**

```bash
uv run pytest tests/test_cvat_sync.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.cvat_sync'`.

- [ ] **Step 3: Implement `core/cvat_sync.py`**

Write `/home/alonso/Documentos/Github/document-layout-model-training/core/cvat_sync.py`:
```python
"""Sync between CVAT and projects/<slug>/cvat/.

push: creates the CVAT project + tasks (one per PDF), uploads PNGs from
      data/images/<pdf_stem>/, and optionally imports COCO annotations
      filtered per task.

pull: exports the CVAT project as COCO and writes to
      projects/<slug>/cvat/exports/v<N+1>_<YYYY-MM-DD>/instances_default.json.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from core.lib.config import load_config
from core.lib.cvat import CvatClient

PROJECTS_ROOT = Path("projects")


def _build_client(cfg: dict[str, Any]) -> CvatClient:
    url = cfg.get("cvat", {}).get("url", "http://localhost:8080")
    user = os.environ.get("CVAT_USER", "admin")
    password = os.environ.get("CVAT_PASSWORD", "admin")
    return CvatClient(url, auth=(user, password))


def push(project_slug: str, coco_path: str | None = None) -> None:
    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")
    client = _build_client(cfg)

    project_name = cfg["cvat"]["project_name"]
    labels = list(cfg["labels"])

    # Project (idempotent)
    p = client.get_project_by_name(project_name)
    if p is None:
        project_id = client.create_project(project_name, labels)
        print(f"[push] created project '{project_name}' id={project_id}")
    else:
        project_id = int(p["id"])
        print(f"[push] reusing project '{project_name}' id={project_id}")

    # Optional COCO file for pre-labels
    coco = None
    if coco_path:
        with open(coco_path) as f:
            coco = json.load(f)

    # Tasks: one per PDF, named after the PDF stem
    pdf_dir = project_dir / "data" / "pdfs"
    images_root = project_dir / "data" / "images"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs in {pdf_dir}")

    for pdf in pdfs:
        task_name = pdf.stem
        img_dir = images_root / task_name
        pngs = sorted(img_dir.glob("pagina-*.png")) if img_dir.exists() else []
        if not pngs:
            raise FileNotFoundError(
                f"no PNGs in {img_dir} — run 'dlmf render --project={project_slug}' first"
            )

        # Task (idempotent)
        t = client.get_task_by_name(project_id, task_name)
        if t is None:
            task_id = client.create_task(task_name, project_id=project_id)
            current_size = 0
            print(f"[push] created task '{task_name}' id={task_id}")
        else:
            task_id = int(t["id"])
            current_size = int(t.get("size") or 0)
            print(f"[push] reusing task '{task_name}' id={task_id} size={current_size}")

        # Images
        if current_size == 0:
            buf = _zip_files(pngs)
            print(f"[push] uploading {len(pngs)} images ({buf.getbuffer().nbytes/1e6:.1f} MB)")
            client.upload_data(task_id, buf)
        elif current_size != len(pngs):
            raise RuntimeError(
                f"task '{task_name}' has {current_size} images, expected {len(pngs)}"
            )
        else:
            print(f"[push] images already present ({current_size})")

        # Annotations
        if coco is None:
            continue
        if client.count_task_annotations(task_id) > 0:
            print(f"[push] annotations already present, skip")
            continue
        task_filenames = {p.name for p in pngs}
        # Detect if this task uses the _1 suffix in the COCO (project-level merge artefact)
        suffix_pattern = re.compile(r"^.*_1\.png$")
        task_has_suffix_in_coco = any(
            suffix_pattern.match(img["file_name"])
            and img["file_name"].rsplit("_1", 1)[0] + ".png" in task_filenames
            for img in coco["images"]
        )
        task_coco = _filter_coco_for_task(
            coco, image_filenames=task_filenames, strip_suffix=task_has_suffix_in_coco
        )
        if not task_coco["images"]:
            print(f"[push] no matching COCO images for task '{task_name}', skip")
            continue
        print(f"[push] importing {len(task_coco['annotations'])} annotations")
        ann_buf = _zip_coco(task_coco)
        client.import_annotations(task_id, ann_buf, fmt="COCO 1.0")


def _zip_files(paths: list[Path]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, p.name)
    buf.seek(0)
    return buf


def _zip_coco(coco: dict) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("annotations/instances_default.json", json.dumps(coco))
    buf.seek(0)
    return buf


def _filter_coco_for_task(
    coco: dict, image_filenames: set[str], strip_suffix: bool
) -> dict:
    """Slice a project-level COCO to one task; renumber image_ids 1..N.

    If `strip_suffix`, normalize filenames `pagina-NNN_1.png` → `pagina-NNN.png`
    before matching against `image_filenames`.
    """
    def normalize(name: str) -> str:
        if strip_suffix:
            return re.sub(r"_1(\.png)$", r"\1", name)
        return name

    selected = [img for img in coco["images"] if normalize(img["file_name"]) in image_filenames]
    selected.sort(key=lambda i: int(i["id"]))
    id_map = {}
    new_images = []
    for new_id, img in enumerate(selected, start=1):
        id_map[img["id"]] = new_id
        new_images.append({**img, "id": new_id, "file_name": normalize(img["file_name"])})
    new_anns = []
    for ann in coco["annotations"]:
        if ann["image_id"] not in id_map:
            continue
        new_anns.append({**ann, "image_id": id_map[ann["image_id"]]})
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_anns,
    }


def pull(project_slug: str, version: str | None = None) -> None:
    """Export CVAT project to COCO and save to projects/<slug>/cvat/exports/v<N>_<YYYY-MM-DD>/."""
    project_dir = PROJECTS_ROOT / project_slug
    cfg = load_config(project_dir / "config.yaml")
    client = _build_client(cfg)

    project_name = cfg["cvat"]["project_name"]
    p = client.get_project_by_name(project_name)
    if p is None:
        raise RuntimeError(f"CVAT project '{project_name}' not found — push first")
    project_id = int(p["id"])

    exports_dir = project_dir / "cvat" / "exports"
    if version is None:
        version = _next_version(exports_dir)
    out_dir = exports_dir / version
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; pass --version=v<N>_<date> explicitly")
    out_dir.mkdir(parents=True)

    print(f"[pull] exporting project {project_id} → {out_dir}")
    zip_bytes = client.export_dataset(project_id, fmt="COCO 1.0", save_images=False)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("instances_default.json"):
                target = out_dir / "instances_default.json"
                target.write_bytes(zf.read(name))
                # Validate
                d = json.loads(target.read_text())
                print(
                    f"[pull] saved {target} — "
                    f"{len(d['images'])} images, "
                    f"{len(d['annotations'])} annotations, "
                    f"{len(d['categories'])} categories"
                )
                return
    raise RuntimeError("export ZIP did not contain annotations/instances_default.json")


def _next_version(exports_dir: Path) -> str:
    """Compute next sequential version label like v3_2026-05-02."""
    if not exports_dir.exists():
        n = 1
    else:
        existing = [d.name for d in exports_dir.iterdir() if d.is_dir()]
        nums = []
        for name in existing:
            m = re.match(r"v(\d+)_", name)
            if m:
                nums.append(int(m.group(1)))
        n = (max(nums) + 1) if nums else 1
    today = dt.date.today().isoformat()
    return f"v{n}_{today}"
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
uv run pytest tests/test_cvat_sync.py -v
```
Expected: 3 tests pass (the COCO filter test + 2 push tests).

- [ ] **Step 5: Smoke-test push against the real CVAT (project should already exist from earlier session)**

```bash
uv run dlmf cvat-push --project=eaf 2>&1 | tail -20
```
Expected: detects the existing CVAT project (id=1), reuses both tasks, reports "images already present" and "annotations already present, skip" for both. No errors. (If the existing CVAT state was cleared, it will re-create — that's OK too.)

- [ ] **Step 6: Commit**

```bash
git add core/cvat_sync.py tests/test_cvat_sync.py
git commit -m "feat(core): add cvat_sync.push — idempotent CVAT project/task creation

core.cvat_sync.push(project_slug, coco_path=None) does:
- Get-or-create the CVAT project from cvat.project_name + labels
  (from config.yaml).
- Per PDF in data/pdfs/: get-or-create a task; zip PNGs from
  data/images/<stem>/ and upload if task has no images.
- If coco_path is given: filter the project-level COCO per task
  (auto-detects the _1 suffix from project-level merges and
  normalizes filenames), then import via the new request-id
  polling path.

3 tests in tests/test_cvat_sync.py cover the dispatch logic with
a MagicMock CvatClient + a fake projects/test/ tree.

The 'pull' function is also stubbed in this file but tested in
Task 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Test `core/cvat_sync.py pull` and version naming

**Files:**
- Modify: `tests/test_cvat_sync.py` (add tests)

The pull function is already implemented in Task 4. This task adds test coverage and a smoke test.

- [ ] **Step 1: Append failing tests for pull and version naming**

Append to `/home/alonso/Documentos/Github/document-layout-model-training/tests/test_cvat_sync.py`:
```python


# --- Pull / version tests --------------------------------------------------

import io as _io
import json as _json
import zipfile as _zipfile

from core.cvat_sync import _next_version, pull


def test_next_version_first_export(tmp_path):
    assert _next_version(tmp_path / "no_such_dir").startswith("v1_")


def test_next_version_increments(tmp_path):
    (tmp_path / "v1_2026-03-20").mkdir()
    (tmp_path / "v2_2026-04-15").mkdir()
    out = _next_version(tmp_path)
    assert out.startswith("v3_")


def test_next_version_ignores_non_v_dirs(tmp_path):
    (tmp_path / "v1_2026-03-20").mkdir()
    (tmp_path / "scratch").mkdir()
    out = _next_version(tmp_path)
    assert out.startswith("v2_")


def test_pull_writes_instances_default_json(fake_project, monkeypatch):
    # Build a fake export ZIP containing annotations/instances_default.json
    fake_coco = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "A"}],
        "images": [{"id": 1, "file_name": "p.png", "width": 1, "height": 1}],
        "annotations": [],
    }
    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("annotations/instances_default.json", _json.dumps(fake_coco))
    zip_bytes = buf.getvalue()

    client = MagicMock()
    client.get_project_by_name.return_value = {"id": 42, "name": "Test Project"}
    client.export_dataset.return_value = zip_bytes
    monkeypatch.setattr("core.cvat_sync._build_client", lambda cfg: client)

    pull("test", version="v1_2026-05-02")

    out = fake_project / "cvat" / "exports" / "v1_2026-05-02" / "instances_default.json"
    assert out.exists()
    loaded = _json.loads(out.read_text())
    assert loaded["categories"][0]["name"] == "A"


def test_pull_raises_if_project_not_found(fake_project, monkeypatch):
    client = MagicMock()
    client.get_project_by_name.return_value = None
    monkeypatch.setattr("core.cvat_sync._build_client", lambda cfg: client)
    with pytest.raises(RuntimeError, match="not found"):
        pull("test")


def test_pull_raises_if_version_dir_exists(fake_project, monkeypatch):
    out_dir = fake_project / "cvat" / "exports" / "v1_2026-05-02"
    out_dir.mkdir(parents=True)
    client = MagicMock()
    client.get_project_by_name.return_value = {"id": 42}
    monkeypatch.setattr("core.cvat_sync._build_client", lambda cfg: client)
    with pytest.raises(FileExistsError):
        pull("test", version="v1_2026-05-02")
```

- [ ] **Step 2: Run tests, confirm pass**

```bash
uv run pytest tests/test_cvat_sync.py -v
```
Expected: 9 tests pass (3 from Task 4 + 6 new).

- [ ] **Step 3: Smoke-test pull against the real CVAT**

```bash
uv run dlmf cvat-pull --project=eaf 2>&1 | tail -10
```
Expected: detects existing CVAT project, exports it, writes to `projects/eaf/cvat/exports/v2_<today>/instances_default.json`, reports counts (561 images, 3105 annotations, 17 categories — matches the v1 export since no edits were made in CVAT).

- [ ] **Step 4: Verify the new export is valid**

```bash
python3 -c "
import json
d = json.load(open('projects/eaf/cvat/exports/v2_2026-05-02/instances_default.json'))
assert len(d['images']) == 561, f'images={len(d[\"images\"])}'
assert len(d['annotations']) == 3105, f'anns={len(d[\"annotations\"])}'
assert len(d['categories']) == 17, f'cats={len(d[\"categories\"])}'
print('v2 export validated: 561/3105/17')
"
```
Expected: `v2 export validated: 561/3105/17`

- [ ] **Step 5: Commit**

```bash
git add tests/test_cvat_sync.py projects/eaf/cvat/exports/v2_*/instances_default.json
git commit -m "feat(core): cvat_sync.pull + version naming, with tests

pull(project_slug, version=None) exports the CVAT project to a fresh
versioned dir. _next_version computes 'v<N+1>_<YYYY-MM-DD>' from the
existing exports dir.

6 new tests cover next_version (first run, increments, ignores
non-v dirs) and pull (writes JSON correctly, errors clear when
project is missing or version dir exists).

Smoke test against the live CVAT: pulls the existing project to
v2_<today>/ — 561 images / 3105 annotations / 17 categories,
matching the original v1_2026-03-20 export (no edits made
in CVAT between).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Delete superseded scripts

**Files:**
- Delete: `scripts/render_pdf_to_png.py`, `scripts/restore_cvat_project4.py`, `upload_to_cvat.py`, `reimport_annotations.py`

These four scripts are now fully replaced by `dlmf render` and `dlmf cvat-push/pull`.

- [ ] **Step 1: Verify each is fully replaced**

```bash
# render_pdf_to_png.py: replaced by core/render.py
# restore_cvat_project4.py: replaced by core/cvat_sync.py push (idempotent)
# upload_to_cvat.py: replaced by core/cvat_sync.py push
# reimport_annotations.py: replaced by core/cvat_sync.py push --coco=<file>

ls scripts/render_pdf_to_png.py scripts/restore_cvat_project4.py upload_to_cvat.py reimport_annotations.py
```
Expected: all four exist.

- [ ] **Step 2: Search for any remaining references in active code (not in docs)**

```bash
grep -rn "render_pdf_to_png\|restore_cvat_project4\|upload_to_cvat\|reimport_annotations" \
  --include="*.py" --include="*.toml" --include="*.yaml" --include="*.md" \
  --exclude-dir=.venv --exclude-dir=.git --exclude-dir=docs/superpowers \
  2>/dev/null | head -20
```
Expected: hits should ONLY be in `README.md` (legacy doc) and possibly in `conversacion_claude.md`. No hits in `.py` or `.toml`. If you see hits in active scripts, STOP and report BLOCKED.

- [ ] **Step 3: Delete the four scripts**

```bash
git rm scripts/render_pdf_to_png.py scripts/restore_cvat_project4.py upload_to_cvat.py reimport_annotations.py
```

If `scripts/restore_cvat_project4.py` is untracked (it might still be — check `git status`), use `rm` instead:
```bash
rm -f scripts/restore_cvat_project4.py
```

- [ ] **Step 4: If `scripts/` is now empty, leave a `.gitkeep` (or remove if you prefer)**

```bash
ls scripts/
```
If empty: `touch scripts/.gitkeep && git add scripts/.gitkeep`

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: remove superseded one-off scripts

These are now fully replaced by the dlmf CLI:
- scripts/render_pdf_to_png.py        → dlmf render
- scripts/restore_cvat_project4.py    → dlmf cvat-push
- upload_to_cvat.py                   → dlmf cvat-push
- reimport_annotations.py             → dlmf cvat-push --coco=<file>

The README still references these by name in narrative sections;
that will be cleaned up when the README is fully rewritten in Plan 06.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final verification + tag

- [ ] **Step 1: Verify all tests pass**

```bash
uv run pytest -v
```
Expected: 8 (Plan 01) + 4 (CLI) + 4 (render) + 8 (cvat client) + 9 (cvat sync) = **33 tests**, all green.

- [ ] **Step 2: Verify CLI is invokable**

```bash
uv run dlmf --help
uv run dlmf render --help
uv run dlmf cvat-push --help
uv run dlmf cvat-pull --help
```
Expected: each prints usage with `--project` flag.

- [ ] **Step 3: Verify directory structure**

```bash
find core tests -type f | sort
```
Expected (additions over Plan 01):
```
core/cli.py
core/cvat_sync.py
core/lib/cvat.py
core/render.py
tests/test_cli.py
tests/test_cvat_client.py
tests/test_cvat_sync.py
tests/test_render.py
```
plus everything from Plan 01.

- [ ] **Step 4: Verify the deletions stuck**

```bash
ls scripts/render_pdf_to_png.py upload_to_cvat.py reimport_annotations.py 2>&1 | grep -E "No such|cannot|no se puede"
```
Expected: each missing.

- [ ] **Step 5: Verify the v2 CVAT export exists**

```bash
ls projects/eaf/cvat/exports/
```
Expected: `v1_2026-03-20/` (Plan 01) and `v2_<today>/` (Plan 02 smoke test).

- [ ] **Step 6: Tag the milestone**

```bash
git tag -a plan-02-render-cvat-cli -m "Plan 02 complete: dlmf render + cvat-push + cvat-pull working end-to-end against live CVAT."
git tag -l "plan-*"
```

- [ ] **Step 7: Summary report**

Print to stdout:
```
git log master..HEAD --oneline | wc -l   # commit count for plan 02
git log $(git rev-list -n 1 plan-01-foundation)..HEAD --oneline
```

---

## Self-Review

**Spec coverage:**
- ✅ Spec section 5 (CLI): `dlmf render`, `dlmf cvat-push`, `dlmf cvat-pull` shipped. `dlmf train`, `dlmf evaluate`, `dlmf promote`, `dlmf predict`, `dlmf classify`, `dlmf init-project` deferred to plans 03-06.
- ✅ Spec section 6 (Data flow): version-naming for `cvat/exports/v<N>_<date>/` implemented in `_next_version`.
- ✅ Spec section 4 (config.yaml drives everything): `render` reads `render.dpi`; `cvat-push/pull` read `cvat.project_name`, `cvat.url`, `labels`.
- ✅ MLflow + parity gate: out of scope (plans 04 + 06).

**Placeholder scan:** No `TBD`/`TODO` left in the plan. All commands have full code.

**Type/name consistency:**
- `CvatClient` method names match between `cvat.py` and `cvat_sync.py` callers.
- `_filter_coco_for_task` signature matches between Task 4's tests and the implementation in `cvat_sync.py`.
- `_next_version` returns `v<N>_<YYYY-MM-DD>` consistently across tests and use.

**Deferred to next plans:**
- Plan 03: `core/predict.py --pre-annotate` will produce a COCO that `dlmf cvat-push --coco=<file>` can consume. This is why `cvat-push` already accepts the `--coco` flag.
- Plan 04: `core/train.py` will load configs the same way.
- Plan 06: README full rewrite (Plan 02 only updates the migration banner).
