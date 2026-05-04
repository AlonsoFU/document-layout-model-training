# Layout Model Factory — Design Spec

**Fecha**: 2026-05-02
**Autor**: Alonso (con Claude)
**Estado**: Borrador para review

---

## 1. Contexto y propósito

Hoy el repo `document-layout-model-training` es un POC de fine-tuning de Docling Heron sobre un único tipo de documento (EAFs del Coordinador Eléctrico Nacional). La arquitectura (paths hardcodeados, scripts one-shot por experimento, archivos JSON sueltos) no escala a múltiples tipos de documentos.

Este diseño transforma el repo en una **Layout Model Factory**: una herramienta para producir modelos de detección de layout especializados, uno por tipo de documento, con un flujo estandarizado y trackeable. Es el primero de dos repos en la visión a largo plazo del usuario:

- **Repo 1 (este)** — Layout Model Factory: produce modelos `.pt` versionados por tipo.
- **Repo 2 (futuro)** — Dark Data Extraction Pipeline: consume esos modelos + LLMs (Ollama) + revisión humana para extraer datos estructurados.

Este spec cubre solo el Repo 1.

### Objetivos

1. **Escalar a N tipos de documentos** sin duplicar código (un comando, N proyectos).
2. **Trackeo experimental** con MLflow (comparar runs, versionar modelos).
3. **Reproducibilidad**: cada run debe ser regenerable desde commits de git.
4. **Eficiencia en hardware modesto**: GTX 1650 4GB, mantener LoRA + Repeat Factor Sampling.
5. **Camino claro a producción**: el diseño actual debe escalar a Airflow/cloud sin rewrite.

### No-objetivos

- Servir modelos por HTTP (eso vive en Repo 2).
- Extraer datos estructurados desde el layout (eso vive en Repo 2).
- Multi-GPU / cluster training (futuro, no MVP).
- Soporte de modelos no-Docling (YOLO, Surya, etc.). Docling Heron es la elección consciente — ver sección 2.

---

## 2. Decisiones arquitectónicas

| Decisión | Elección | Justificación |
|---|---|---|
| Modelo base | **Docling Heron (RT-DETR v2)** | SOTA para layout fino, fine-tuneable con LoRA en 4GB, 17 clases pre-entrenadas en DocLayNet, evidencia empírica de +2.6% mAP en este repo |
| Granularidad de tipo | **1 modelo por tipo semántico** (EAF, contrato, factura...) | NO sub-modelos por "cantidad de contenido" dentro del mismo tipo: la variación se absorbe con dataset diverso + Repeat Factor Sampling |
| Labels | **17 fijos de DocLayNet (Opción A)** | Override por tipo permitido en config pero no usado en MVP |
| Identificación de tipo | **Regex de filename + fallback Ollama vision** | Rápido y barato en el caso común; LLM solo cuando filename es ambiguo |
| Tracking | **MLflow local** | Open source, sin vendor lock-in, skill transferible a producción enterprise; W&B alternativo descartado por requerir cuenta y vendor lock-in |
| CLI | **Typer** sobre `python -m core.<cmd>` | Idiomático Python, autocompletado, ayuda formateada |
| Storage | **Git para metadata, gitignore para binarios** | PNGs y `.pt` regenerables o reemplazables; JSONs versionados |
| Serving | **Fuera de scope** — vive en Repo 2 | Separación training/serving es estándar en MLOps |

---

## 3. Estructura del repo

```
document-layout-model-training/
├── projects/                          # 1 carpeta por tipo de documento
│   └── eaf/
│       ├── config.yaml                # ADN del tipo
│       ├── EXPERIMENTS.md             # narrativa histórica de experimentos
│       ├── data/
│       │   ├── pdfs/                  # PDFs source (git si <100MB; LFS sino)
│       │   └── images/                # PNGs renderizados (gitignored)
│       ├── cvat/
│       │   ├── pre_annotations/       # gitignored, regenerable
│       │   └── exports/
│       │       └── v<N>_<YYYY-MM-DD>/instances_default.json
│       ├── runs/
│       │   └── <run_name>/
│       │       ├── best_model.pt      # gitignored
│       │       ├── history.json       # per-epoch metrics
│       │       ├── eval.json          # final metrics (mAP global + per-PDF + per-class)
│       │       ├── config_resolved.yaml  # config + overrides aplicados
│       │       └── data_split.json    # qué páginas en train/val
│       └── models/
│           └── production.pt          # symlink → mejor run
│
├── core/                              # código compartido entre todos los tipos
│   ├── train.py                       # entrena un run
│   ├── evaluate.py                    # mAP@[.5:.95] global + per-PDF + per-class
│   ├── predict.py                     # inferencia (CLI de validación)
│   ├── classify_doctype.py            # regex + Ollama fallback
│   ├── cvat_sync.py                   # push (crear proyecto + tasks + pre-labels) y pull (export)
│   ├── render.py                      # PDF → PNG con pdftoppm
│   ├── init_project.py                # scaffolding interactivo de un tipo nuevo
│   ├── promote.py                     # actualiza production.pt + MLflow Model Registry
│   ├── labels/
│   │   └── doclaynet_17.yaml          # los 17 labels estándar reusables
│   └── lib/
│       ├── data.py                    # COCO dataset, Repeat Factor Sampling, augmentation
│       ├── model.py                   # carga Heron + apply_lora() + key mapping
│       ├── postproc.py                # NMS per-cat, cross-cat cleanup, full-page filter
│       ├── tracking.py                # wrapper MLflow
│       └── config.py                  # carga + resolve overrides de YAML
│
├── scripts/
│   └── render_pdf_to_png.py           # ya existe, se mantiene como CLI auxiliar
│
├── mlruns/                            # MLflow tracking local (gitignored)
├── docs/superpowers/specs/            # specs de diseño
├── pyproject.toml                     # dependencias gestionadas con uv
└── README.md                          # workflow + setup
```

---

## 4. `config.yaml` por tipo de documento

```yaml
project:
  slug: eaf
  display_name: "Estudios de Análisis de Falla (CEN)"

# Identificación automática del tipo de documento
classification:
  filename_regex: "^EAF[-_]\\d+[-_]\\d{4}\\.pdf$"
  ollama_fallback:
    enabled: true
    model: "qwen2.5:7b"
    prompt: "¿Es un Estudio de Análisis de Falla del CEN? yes/no"

render:
  dpi: 300

cvat:
  project_name: "Docling Heron CLEAN - EAF"

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
```

**Override por run** desde CLI:
```bash
dlmf train --project=eaf --run=B_lora64 \
    --override training.lora.rank=64 \
    --override training.lr=5e-5
```

Cada override queda registrado en MLflow como hyperparam del run.

**Lo que va en config (varía por tipo)**: filename_regex, prompt Ollama, nombre CVAT, labels, hyperparams default, thresholds.
**Lo que NO va en config (vive en core/)**: lógica de LoRA, training loop, NMS, mAP, MLflow, COCO export.

---

## 5. CLI — comandos día a día

Empaquetado con Typer, instalable como `dlmf` (document-layout-model-factory).

### Setup de un tipo nuevo

```bash
dlmf init-project --slug=contratos                  # scaffolding interactivo
dlmf render --project=contratos                     # PDFs → PNGs (pdftoppm)
```

### Ciclo de anotación (CVAT)

```bash
dlmf predict --project=contratos --pre-annotate     # Heron baseline → COCO predictions
dlmf cvat-push --project=contratos                  # crea proyecto + tasks + sube imgs + pre-labels
# (revisas en http://localhost:8080)
dlmf cvat-pull --project=contratos                  # export → cvat/exports/v<N>_<fecha>/
```

### Entrenamiento + experimentación

```bash
dlmf train --project=contratos --run=A_baseline
dlmf train --project=contratos --run=B_lora64 --override training.lora.rank=64
dlmf evaluate --project=contratos --run=B_lora64    # mAP global + per-PDF + per-class
mlflow ui --backend-store-uri ./mlruns              # http://localhost:5000
```

### Promoción a producción

```bash
dlmf promote --project=contratos --run=B_lora64     # actualiza production.pt + Model Registry
dlmf predict --project=contratos --pdf=doc.pdf --output=anotado.pdf  # validación visual
```

### Auto-clasificación

```bash
dlmf classify --pdf=mystery.pdf                                # → "eaf" o "contrato"
dlmf predict --pdf=mystery.pdf --auto-classify --output=anotado.pdf
```

---

## 6. Data flow + integración MLflow

### Flujo end-to-end

```
PDFs                                          [git si <100MB; LFS sino]
  ↓ render
PNGs (gitignored, regenerables)
  ↓ predict --pre-annotate
Pre-anotaciones COCO (gitignored, regenerables)
  ↓ cvat-push
CVAT (revisión humana en http://localhost:8080)
  ↓ cvat-pull
Export COCO versionado: cvat/exports/v<N>_<fecha>/   [git, ~600KB]
  ↓ train --run=<nombre>
Run dir: runs/<nombre>/
  - best_model.pt                                    [gitignored, ~30MB]
  - history.json, config_resolved.yaml, data_split.json  [git, livianos]
MLflow tracking: mlruns/<exp>/<run>/                 [gitignored]
  - params/, metrics/, artifacts/, tags
  ↓ evaluate
eval.json (mAP global + per-PDF + per-class)         [git]
  ↓ promote --run=<ganador>
production.pt → ../runs/<ganador>/best_model.pt      [symlink]
MLflow Model Registry: dlmf-<tipo>, stage=Production
```

### Versionado del dataset

Cada `cvat-pull` genera un export con timestamp (`v<N>_<YYYY-MM-DD>/`). Los runs apuntan a una versión específica en su `config_resolved.yaml`:

```yaml
dataset:
  cvat_export: "projects/eaf/cvat/exports/v2_2026-05-02/instances_default.json"
  num_images: 561
  num_annotations: 3105
  hash: "sha256:abc123..."
```

Reproducibilidad: dado un commit + un run name → el dataset exacto y los hyperparams exactos están determinados.

### Qué se commitea

| Cosa | Tamaño típico | En git |
|---|---|---|
| `config.yaml`, `EXPERIMENTS.md` | KB | ✅ |
| `cvat/exports/v*/instances_default.json` | ~600 KB cada uno | ✅ |
| `runs/*/{history,eval,data_split}.json`, `config_resolved.yaml` | KB | ✅ |
| `data/pdfs/*.pdf` | ~10-50 MB cada uno | ✅ hasta ~100MB total; sino Git LFS |
| `data/images/*.png` | ~700 KB cada uno × 561 imgs ≈ 387 MB | ❌ |
| `runs/*/best_model.pt` | ~30 MB | ❌ |
| `mlruns/` | crece con runs | ❌ |
| `.venv/` | GB | ❌ |

---

## 7. Plan de migración del repo actual

Migración incremental, validable por fase, sin pérdida de datos.

| Fase | Acción | Validación | Tiempo |
|---|---|---|---|
| **1. Limpieza** | Borrar `https:/github.com/...` (clone accidental). Borrar `clean_overlaps_v1.py` y `_v2.py` (solo v3 se usa). | `git status` limpio | 10 min |
| **2. Scaffolding** | Crear `core/`, `core/lib/`, `pyproject.toml` con deps. Setup `uv`. | `uv sync` ok | 30 min |
| **3. Mover EAF a `projects/eaf/`** | `cvat_projects/project4_*/` → `projects/eaf/cvat/exports/v1_2026-03-20/`. `data/pdfs/` → `projects/eaf/data/pdfs/`. `training/models/P_repeat_factor/` → `projects/eaf/runs/P_repeat_factor/`. Crear `projects/eaf/config.yaml`. | Estructura visible, exports válidos | 30 min |
| **4. Generalizar scripts → `core/`** | Refactor de scripts existentes a módulos parametrizados por `--project=<slug>`: `upload_to_cvat.py` + `restore_cvat_project4.py` → `core/cvat_sync.py`; `generate_heron_coco.py` → `core/predict.py --pre-annotate`; `train_round{1..4}.py` → `core/train.py`; `reevaluate_real_map.py` → `core/evaluate.py`; `draw_boxes_on_pdf_v2.py` → `core/predict.py --output=anotado.pdf`; `clean_overlaps_v3.py` → `core/lib/postproc.py`. | Cada comando corre en EAF | 2-3 días |
| **5. Integrar MLflow + Typer** | Wrapper `core/lib/tracking.py` con `mlflow.start_run`. CLI Typer `dlmf <cmd>`. | `mlflow ui` muestra el run | 1 día (paralelo a 4) |
| **6. Validación de paridad** ⚠️ | Reentrenar el ganador P_repeat_factor con la nueva pipeline. Confirmar mAP final >= 0.86. Si cae >2%, revertir y depurar. | `dlmf evaluate --project=eaf --run=P_v2` | 1-2 días |
| **7. Documentar + commit** | Actualizar `README.md`. PR en branch `migrate-to-factory`. | PR review | 1/2 día |

**Total estimado**: ~5-6 días de trabajo focalizado.

### Decisiones explícitas

- Mantener `EXPERIMENTS.md` (copiado a `projects/eaf/EXPERIMENTS.md`).
- NO re-correr los 16 experimentos — solo el ganador P para validar paridad.
- Eliminar `EAF-477-2025_ground_truth.json` (duplica el CVAT export).
- Eliminar `data_split.json` y `results*.json` sueltos (ahora viven dentro de cada `runs/<run>/`).
- `conversacion_claude.md` → mover a `docs/` o eliminar (decisión del usuario).

### Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Refactor rompe algo sutil del entrenamiento | Fase 6: gate de paridad (mAP >= 0.86) antes de mergear |
| Pérdida de runs históricos (P, E, M, O...) | Solo el ganador P se rescata. Resto vive en `EXPERIMENTS.md` como narrativa |
| MLflow logging falla y rompe el train | `try/except` alrededor de los hooks; el train no depende del tracker |
| API CVAT cambió (descubierto hoy: `import_status` 410) | `cvat_sync.py` usa `/api/requests/{rq_id}` desde el principio |

---

## 8. Escalabilidad futura

El diseño es deliberadamente evolutivo. Cada salto es aditivo, no rewrite:

| Salto | Trabajo |
|---|---|
| **Hydra configs** (configs jerárquicos avanzados) | Decorar `train()` con `@hydra.main`, YAMLs ya compatibles | ~1 día |
| **Airflow / Prefect** (orquestación) | Envolver cada CLI en un task. NO tocar `core/` | ~2-3 días |
| **Docker** (cloud) | Dockerfile alrededor del CLI | ~½ día |
| **SageMaker / Vertex AI** | El mismo Docker corre allá. Storage local → S3 | ~3-5 días |
| **Multi-GPU** | Cambiar training loop a `torchrun`/`accelerate` | ~1-2 días |
| **MLflow tracking server** (multi-usuario) | Cambiar `MLFLOW_TRACKING_URI` apuntando a un server. Cero cambios en código | ~½ día |
| **Git LFS** (PDFs >100 MB total) | `git lfs track "*.pdf"`, retroactivo | ~10 min |

Lo que escala automáticamente al añadir un tipo nuevo: **0 líneas de código nuevas**, solo crear `projects/<tipo>/config.yaml`.

Lo que NO escala automáticamente: storage (local → S3), secrets (`.env` → vault), concurrencia (1 GPU → queue). Todos son cambios localizados cuando hagan falta.

---

## 9. Open questions / decisiones diferidas

1. **PDFs sensibles**: si en algún tipo los PDFs son confidenciales, ese tipo migra a S3 antes de commit. Política a definir cuando aparezca el primer caso.
2. **Sub-tipos**: si en algún tipo `per-document mAP` muestra >15% diferencia consistente entre sub-grupos, considerar split. Decisión guiada por datos, no a priori.
3. **Modelo base alternativo**: re-evaluar Surya o PaddleOCR si Docling se estanca <0.85 con varios runs. No antes.
4. **Versionado de `core/`**: hoy `git tag` simple. Si N tipos dependen de versiones distintas de `core/`, considerar empaquetar `core` como wheel versionado (sobra para MVP).

---

## 10. Definición de "done" para el MVP

- [ ] `projects/eaf/` tiene la estructura nueva con `config.yaml` válido.
- [ ] `core/` reemplaza todos los scripts hardcodeados del repo actual.
- [ ] `dlmf train --project=eaf --run=P_v2` reproduce mAP >= 0.86.
- [ ] MLflow UI muestra el run con todos sus params/metrics/artifacts.
- [ ] `dlmf promote --run=P_v2` actualiza `projects/eaf/models/production.pt` y registra en Model Registry.
- [ ] `dlmf predict --pdf=doc.pdf --auto-classify --output=anotado.pdf` genera un PDF anotado.
- [ ] `README.md` actualizado documentando el workflow nuevo.
- [ ] PR mergeado a master.
