# Document Layout Model Factory (dlmf)

Herramienta para producir modelos de detección de **layout de documentos** especializados, uno por tipo de documento. Fine-tunea [Docling Heron](https://huggingface.co/docling-project/docling-layout-heron) (RT-DETR v2) con LoRA, usa CVAT para anotación humana, MLflow para tracking, y todo se orquesta vía CLI `dlmf`.

> **Visión:** Esta es la primera de dos piezas de un proyecto de extracción de **dark data**. Este repo entrena los modelos de layout. Otro repo (futuro) consume esos modelos `.pt` + LLMs locales (Ollama) para extraer datos estructurados de los PDFs.

---

## Resultados actuales (tipo `eaf`)

| Modelo | mAP@[0.5:0.95] |
|---|---|
| Heron base (sin fine-tune) | 0.8482 |
| **P_repeat_factor_v2** (LoRA r=32 + RFS, 561 imgs) | **0.8890 overall**<br>0.9961 EAF-089-2025<br>0.8493 EAF-477-2025 |

Ver [`projects/eaf/EXPERIMENTS.md`](projects/eaf/EXPERIMENTS.md) para el detalle de los 16 experimentos del POC original (A–P) y [`projects/eaf/runs/P_repeat_factor_v2/eval_detailed.json`](projects/eaf/runs/P_repeat_factor_v2/eval_detailed.json) para el breakdown per-PDF del modelo en producción.

---

## Pipeline

```
PDFs source
    │
    ▼
[1. Render]  pdftoppm 300 DPI ─────────────────► PNGs (gitignored)
    │
    ▼
[2. Pre-anotar]  Heron baseline ───────────────► COCO predictions
    │
    ▼
[3. CVAT push]  Crea proyecto + tasks + sube imágenes y pre-labels
    │
    ▼
[4. Revisión humana]  en http://localhost:8080
    │
    ▼
[5. CVAT pull]  Export COCO versionado v<N>_<fecha>/
    │
    ▼
[6. Train]  LoRA fine-tune + Repeat Factor Sampling + MLflow tracking
    │
    ▼
[7. Evaluate]  mAP global + per-PDF + per-class
    │
    ▼
[8. Promote]  production.pt symlink + MLflow Model Registry
    │
    ▼
[9. Predict]  Inferencia con boxes dibujados en PDF anotado
```

---

## Instalación

Requiere Python ≥ 3.11, `uv`, y CVAT corriendo localmente para los pasos de anotación.

```bash
# Clonar
git clone <repo-url> && cd document-layout-model-training

# Instalar deps con uv
uv sync --extra dev

# Verificar
uv run pytest -q
uv run dlmf --help
```

**GPU:** PyTorch viene fijado a `2.5.1+cu118` para soportar la GTX 1080 (sm_6.1). Para otras GPUs (RTX 30xx+, Ampere, Hopper) cambiar el index URL en `[tool.uv.sources]` de `pyproject.toml` a `cu121` o `cu124` y `uv sync` de nuevo.

**CVAT:**
```bash
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
# Acceder en http://localhost:8080
```

`dlmf cvat-push/pull` usan `admin/admin` por defecto. Override con `CVAT_USER` y `CVAT_PASSWORD` en el ambiente.

---

## Comandos CLI

```bash
dlmf render        --project=<slug>                              # PDFs → PNGs
dlmf predict       --project=<slug> --pre-annotate               # Heron baseline → COCO
dlmf cvat-push     --project=<slug> [--coco=<file>]              # Crea proyecto + sube
dlmf cvat-pull     --project=<slug> [--version=v<N>_<fecha>]     # Exporta de CVAT
dlmf train         --project=<slug> --run=<nombre> [--override KEY=VAL]+
dlmf evaluate      --project=<slug> --run=<nombre>               # Per-PDF mAP breakdown
dlmf promote       --project=<slug> --run=<nombre>               # Symlink production.pt
dlmf predict       --project=<slug> --pdf=<file> --output=<file>.pdf  # PDF anotado
dlmf classify      --pdf=<file>                                  # Auto-detecta tipo
dlmf init-project  --slug=<nuevo>                                # Scaffolding
```

Ver `dlmf <comando> --help` para detalles de cada uno.

---

## Quick start: añadir un tipo nuevo

```bash
# 1. Scaffolding (crea projects/contratos/ con config.yaml plantilla)
dlmf init-project --slug=contratos

# 2. Editar projects/contratos/config.yaml:
#    - filename_regex (cómo se llaman tus PDFs)
#    - ollama_fallback.prompt (descripción para que el LLM clasifique)
#    - hyperparams si querés desviarte del default

# 3. Copiar tus PDFs a projects/contratos/data/pdfs/

# 4. Renderizar
dlmf render --project=contratos

# 5. Generar pre-anotaciones con Heron baseline
dlmf predict --project=contratos --pre-annotate
# → projects/contratos/cvat/pre_annotations/<timestamp>.json

# 6. Subir a CVAT con pre-labels
dlmf cvat-push --project=contratos \
    --coco=projects/contratos/cvat/pre_annotations/<timestamp>.json

# 7. Revisar/corregir manualmente en http://localhost:8080

# 8. Bajar el export limpio
dlmf cvat-pull --project=contratos
# → projects/contratos/cvat/exports/v1_<fecha>/instances_default.json

# 9. Entrenar
dlmf train --project=contratos --run=baseline
# → projects/contratos/runs/baseline/{best_model.pt,history.json,...}

# 10. Evaluar y promover
dlmf evaluate --project=contratos --run=baseline
dlmf promote  --project=contratos --run=baseline
# → projects/contratos/models/production.pt

# 11. Visualizar
dlmf predict --project=contratos --pdf=algún_doc.pdf --output=resultado.pdf

# 12. Inspeccionar experimentos en MLflow
mlflow ui --backend-store-uri ./mlruns
# http://localhost:5000
```

---

## Estructura del proyecto

```
.
├── projects/                          # 1 carpeta por tipo de documento
│   └── eaf/                           # tipo EAF (Estudios Análisis de Falla, CEN)
│       ├── config.yaml                # ADN del tipo (regex, hyperparams, thresholds)
│       ├── EXPERIMENTS.md             # narrativa de experimentos
│       ├── data/
│       │   ├── pdfs/                  # PDFs source (en git ≤ 100MB; LFS sino)
│       │   └── images/                # PNGs renderizados (gitignored)
│       ├── cvat/
│       │   ├── pre_annotations/       # gitignored (regenerable)
│       │   └── exports/v<N>_<fecha>/  # COCO versionado, en git
│       ├── runs/<run_name>/
│       │   ├── best_model.pt          # LoRA weights (gitignored, ~5MB)
│       │   ├── history.json           # per-epoch metrics
│       │   ├── eval.json              # mejor epoch
│       │   ├── eval_detailed.json     # per-PDF breakdown (post `dlmf evaluate`)
│       │   ├── config_resolved.yaml   # config + overrides aplicados
│       │   └── data_split.json        # train/val ids
│       └── models/
│           └── production.pt          # symlink → ../runs/<ganador>/best_model.pt
│
├── core/                              # código compartido entre todos los tipos
│   ├── cli.py                         # Typer app (entry point: dlmf)
│   ├── render.py                      # PDF → PNG (pdftoppm)
│   ├── cvat_sync.py                   # push/pull CVAT
│   ├── predict.py                     # inferencia (--pre-annotate / --output=PDF)
│   ├── train.py                       # training loop
│   ├── evaluate.py                    # per-PDF/per-class breakdown
│   ├── promote.py                     # symlink + MLflow registry
│   ├── classify_doctype.py            # regex + Ollama vision fallback
│   ├── init_project.py                # scaffolding nuevo tipo
│   ├── labels/doclaynet_17.yaml       # 17 labels DocLayNet (compartidos)
│   └── lib/
│       ├── config.py                  # YAML loader con !include + overrides
│       ├── cvat.py                    # cliente HTTP CVAT
│       ├── data.py                    # Dataset, RFS, Augmenter
│       ├── eval.py                    # mAP@[.5:.95]
│       ├── model.py                   # Heron + LoRA + label index map
│       ├── postproc.py                # NMS, kill_wrappers, resolve_overlaps
│       └── tracking.py                # MLflow context manager
│
├── tests/                             # 100+ tests pytest
├── mlruns/                            # MLflow tracking (gitignored)
├── docs/superpowers/
│   ├── specs/2026-05-02-layout-model-factory-design.md   # diseño completo
│   └── plans/                         # planes de migración 01-06
├── pyproject.toml                     # uv + deps
└── .gitignore
```

---

## Decisiones arquitectónicas clave

- **1 modelo por tipo de documento.** No sub-modelos por contenido — la variación dentro de un tipo se absorbe con dataset diverso + Repeat Factor Sampling.
- **17 labels fijos** (DocLayNet base). Cada `config.yaml` puede override la lista si un tipo necesita labels custom.
- **LoRA r=32** en `q_proj/k_proj/v_proj` del decoder de RT-DETR v2. ~5.2M params trainables (12% del modelo).
- **Repeat Factor Sampling** con threshold 0.5 — fue lo que más subió la métrica del POC original (+2.6% mAP).
- **Label remap**: el COCO usa orden CVAT (1=Document Index, 14=Caption); el modelo Heron usa orden DocLayNet (0=Caption, 11=Document Index). El dataset hace el remap automáticamente — sin esto el modelo tarda 15+ epochs en converger.
- **Tracking centralizado en MLflow** (`mlruns/` local). Plan futuro: migrar a un MLflow tracking server cuando haya más de 1 desarrollador.
- **Versionado de datasets**: cada `dlmf cvat-pull` produce `cvat/exports/v<N>_<fecha>/`. Los runs anclan al export que usaron via `config_resolved.yaml`.

---

## Hardware / costos

Probado en:
- NVIDIA GTX 1080 (sm_6.1, 8 GB VRAM)
- Ubuntu 24.04, Python 3.11
- ~5 min/epoch de training; un experimento típico (P_repeat_factor) tarda ~50 min

VRAM real para LoRA r=32 + batch_size=1 + grad_accum=4: ~3 GB. Cabe holgado en 8 GB. Para batch_size=2 se necesitan ~5 GB.

---

## Documentación

- **Diseño completo**: [`docs/superpowers/specs/2026-05-02-layout-model-factory-design.md`](docs/superpowers/specs/2026-05-02-layout-model-factory-design.md)
- **Planes de migración** (POC original → factory):
  - [Plan 01 — Foundation](docs/superpowers/plans/2026-05-02-plan-01-foundation.md)
  - [Plan 02 — Render+CVAT+CLI](docs/superpowers/plans/2026-05-02-plan-02-render-cvat-cli.md)
  - [Plan 03 — Predict+Postproc](docs/superpowers/plans/2026-05-02-plan-03-predict-postproc.md)
  - [Plan 04 — Training+MLflow](docs/superpowers/plans/2026-05-03-plan-04-training-mlflow.md)
  - [Plan 05 — Eval+Promote+Predict-PDF](docs/superpowers/plans/2026-05-03-plan-05-eval-promote-predict-pdf.md)
  - [Plan 06 — Classify+Init+README](docs/superpowers/plans/2026-05-04-plan-06-classify-init-readme.md)
- **Experimentos del POC** (16 estrategias A–P): [`projects/eaf/EXPERIMENTS.md`](projects/eaf/EXPERIMENTS.md)
