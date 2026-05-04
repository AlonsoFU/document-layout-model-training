# Resume / Estado del proyecto

> Este documento es para retomar el trabajo después de un compact / nueva sesión. Lee `README.md` para qué hace el proyecto. Lee este archivo para saber **dónde quedamos**, qué decisiones se tomaron, y qué hacer cuando vuelvas.

**Última sesión:** 2026-05-04
**Branch activa:** `migrate-to-factory` (52 commits ahead de `master`)
**Último commit:** `8e5be3a` — `docs: rewrite README to reflect dlmf CLI`

---

## TL;DR

La migración POC → Factory **está completa**. Los 6 planes corrieron, todos los tags están creados, 99 tests pasan, el modelo `P_repeat_factor_v2` produce mAP 0.8890 (overall) y vive en `projects/eaf/models/production.pt`. El branch `migrate-to-factory` está listo para mergearse a `master`.

---

## Estado de los 6 planes

| Plan | Tag | Resultado |
|---|---|---|
| 01 — Foundation | `plan-01-foundation` | ✅ Estructura + config loader |
| 02 — Render+CVAT+CLI | `plan-02-render-cvat-cli` | ✅ 3 comandos CLI base |
| 03 — Predict+Postproc | `plan-03-predict-postproc` | ✅ Pre-anotación con Heron baseline |
| 04 — Training+MLflow | `plan-04-training-mlflow` | ✅ mAP 0.9316 (training) → 0.8890 (re-eval) |
| 05 — Eval+Promote+Predict-PDF | `plan-05-eval-promote-predict-pdf` | ✅ Per-PDF breakdown + production symlink |
| 06 — Classify+Init+README | `plan-06-classify-init-readme` | ✅ classify + init-project + README final |

Cada plan vive en `docs/superpowers/plans/`. Spec maestro en `docs/superpowers/specs/2026-05-02-layout-model-factory-design.md`.

---

## Lo que hace cada comando del CLI

```
dlmf render        --project=<slug>                              # PDFs → PNGs
dlmf predict       --project=<slug> --pre-annotate               # Heron baseline → COCO
dlmf cvat-push     --project=<slug> [--coco=<file>]              # Crea proyecto + sube
dlmf cvat-pull     --project=<slug> [--version=v<N>_<fecha>]     # Exporta de CVAT
dlmf train         --project=<slug> --run=<nombre> [--override KEY=VAL]+
dlmf evaluate      --project=<slug> --run=<nombre>               # Per-PDF mAP breakdown
dlmf promote       --project=<slug> --run=<nombre>               # Symlink production.pt
dlmf predict       --project=<slug> --pdf=<file> --output=<file>.pdf
dlmf classify      --pdf=<file>                                  # Auto-detecta tipo
dlmf init-project  --slug=<nuevo>                                # Scaffolding
```

---

## Modelo en producción

```
projects/eaf/models/production.pt -> ../runs/P_repeat_factor_v2/best_model.pt
```

| Métrica | Valor |
|---|---|
| Overall mAP@[.5:.95] | 0.8890 |
| EAF-089-2025 | 0.9961 (57 val imgs) |
| EAF-477-2025 | 0.8493 (27 val imgs) |
| Best epoch | 4 (early-stop epoch 14) |
| Trainable params | 5,211,027 (12.10% de 43M) |

Hyperparams en `projects/eaf/runs/P_repeat_factor_v2/config_resolved.yaml`.

---

## 6 bugs no obvios resueltos durante la migración

Estos están enterrados en commits y memoria; documentándolos acá para no re-pisar:

1. **PyYAML 6 no parsea `5e-5` como float** (devuelve string). Fix en `core/lib/config.py::_coerce` con fallback `float()` cuando `safe_load` retorna str.
2. **`crop_box` necesita 4 candidatos (top/bottom/left/right)**, no 2. Multi-column docs requieren strips verticales también. Fix commit `7da0e2f`.
3. **CVAT API `action=import_status` deprecado** → devuelve HTTP 410. Hay que usar `/api/requests/{rq_id}` para polling. Implementado en `core/lib/cvat.py`.
4. **PyTorch wheels default (sm_7.5+) no soportan GTX 1080 (sm_6.1)**. CUDA opera silenciosamente en CPU. Fix: pin `torch==2.5.1+cu118` con `[tool.uv.sources]` en `pyproject.toml`.
5. **COCO `_N` suffix en file_names** (ej `pagina-001_1.png`). Hay que mapearlo a "Nth occurrence" del mismo nombre en subdirs alfabéticos. Implementado en `core/train.py` y `core/evaluate.py`.
6. **Label order mismatch (el más grande)** — COCO usa orden CVAT (1=Document Index, 14=Caption); Heron espera orden DocLayNet (0=Caption, 11=Document Index). Sin remap el modelo plateauaba en mAP 0.77 (15+ epochs). Con remap llegó a 0.93 en 4 epochs. Implementado vía `category_remap` parameter en `CocoDocDataset` + construcción automática en `core/train.py` usando `MODEL_INDEX_TO_LABEL_NAME`.

---

## Estado de servicios externos en este equipo

- **CVAT**: corriendo en `http://localhost:8080`, admin/admin. Proyecto "Docling Heron CLEAN - EAF" (id 2) tiene los 561 imgs + 3105 anotaciones. (Hay un id=1 "Docling Heron CLEAN (sin overlaps v3)" huérfano de una sesión anterior — ignorable o borrable.)
- **MLflow**: tracking local en `mlruns/` (gitignored). Experiment `dlmf-eaf` tiene los runs `smoke` y `P_repeat_factor_v2`. Levantar UI con `mlflow ui --backend-store-uri ./mlruns`.
- **Ollama**: instalado con varios modelos disponibles (`qwen2.5:7b`, `qwen3.5:9b`, `llama3.1:8b`, etc.). Plan 06's `dlmf classify` lo usa como fallback en `http://localhost:11434/api/generate`.
- **GPU**: GTX 1080 sm_6.1, 8GB VRAM, driver 580.126.20. PyTorch 2.5.1+cu118.

---

## Qué hacer al volver

### Opción A — Mergear a master (recomendado si la sesión está cerrada)

```bash
git checkout master
git merge migrate-to-factory  # fast-forward, son 52 commits limpios
git tag -l "plan-*"            # los 6 tags siguen apuntando a sus commits
```

Dado que `master` no se tocó durante la migración, el merge es trivial.

### Opción B — Seguir mejorando antes de mergear

Áreas sugeridas (en orden de valor):

1. **Reentrenar P_repeat_factor con más data de EAF-477** (subir el 0.8493).
   ```bash
   # Anotar más PDFs EAF-477 en CVAT, luego:
   dlmf cvat-pull --project=eaf  # genera v3_<fecha>/
   dlmf train --project=eaf --run=P_v3 --override training.lora.rank=64
   dlmf evaluate --project=eaf --run=P_v3
   # Si supera 0.89 promote
   dlmf promote --project=eaf --run=P_v3
   ```

2. **Añadir un segundo tipo** (ej: contratos) para validar el flujo end-to-end con datos nuevos:
   ```bash
   dlmf init-project --slug=contratos
   # editar projects/contratos/config.yaml con regex y prompt reales
   # cargar PDFs en projects/contratos/data/pdfs/
   dlmf render --project=contratos
   dlmf predict --project=contratos --pre-annotate
   dlmf cvat-push --project=contratos --coco=projects/contratos/cvat/pre_annotations/<ts>.json
   # anotar manualmente en CVAT
   dlmf cvat-pull --project=contratos
   dlmf train --project=contratos --run=baseline
   ```

3. **Visualizar el modelo nuevo en PDF** (la preview commiteada en Plan 05 usó Heron baseline porque promote vino después):
   ```bash
   dlmf predict --project=eaf --pdf=projects/eaf/data/pdfs/EAF-477-2025.pdf \
                --output=/tmp/eaf477_lora.pdf --limit=20
   # comparar visualmente con projects/eaf/runs/P_repeat_factor_v2/preview_EAF-477_first10.pdf
   ```

4. **Otro repo (futuro)**: Dark Data Extraction Pipeline. Consume `projects/<slug>/models/production.pt` + Ollama + schemas por tipo. Está fuera del scope de este repo.

### Opción C — Mejoras infra (cuando crezca el equipo)

- MLflow tracking server centralizado (en lugar de file-based local).
- Git LFS para PDFs (si superas ~100 MB total).
- Migración a Hydra para configs jerárquicos.
- Airflow/Prefect para orquestación.

Detalle en `docs/superpowers/specs/2026-05-02-layout-model-factory-design.md` sección 8 "Escalabilidad futura".

---

## Comandos sanity check al retomar

```bash
# 1. Verificar branch + tests
git branch --show-current   # → migrate-to-factory
uv run pytest -q            # → 99 passed, 1 skipped

# 2. Verificar que el CLI carga
uv run dlmf --help          # → 9 commands listed

# 3. Verificar que la GPU funciona
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
# → cuda: True NVIDIA GeForce GTX 1080

# 4. Verificar que el modelo de producción está
ls -la projects/eaf/models/production.pt
# → symlink to ../runs/P_repeat_factor_v2/best_model.pt

# 5. Verificar que CVAT está vivo
curl -sf -u admin:admin http://localhost:8080/api/projects | python3 -c "import sys,json; print('CVAT projects:', json.load(sys.stdin).get('count'))"
```

---

## Archivos clave para entender el repo (en orden de lectura)

1. `README.md` — overview, quick start, structure.
2. `docs/superpowers/specs/2026-05-02-layout-model-factory-design.md` — el diseño completo.
3. `core/cli.py` — qué hace cada comando (delgada, despacha a módulos).
4. `core/train.py` — el corazón ML.
5. `projects/eaf/config.yaml` — ejemplo de configuración por tipo.
6. `projects/eaf/EXPERIMENTS.md` — historial de experimentos del POC original.
7. `docs/superpowers/plans/2026-05-04-plan-06-classify-init-readme.md` — el último plan (más representativo del estilo final).
