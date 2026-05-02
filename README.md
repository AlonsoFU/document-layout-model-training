> ⚠️ **Repo en migración a Layout Model Factory.** El diseño objetivo está en
> [`docs/superpowers/specs/2026-05-02-layout-model-factory-design.md`](docs/superpowers/specs/2026-05-02-layout-model-factory-design.md).
> El plan de migración por fases está en [`docs/superpowers/plans/`](docs/superpowers/plans/).
>
> **Estado actual (Plan 01 completado):** estructura `projects/eaf/` y `core/lib/config.py` listos.
> Los scripts originales (`generate_heron_coco.py`, `upload_to_cvat.py`,
> `clean_overlaps_v3.py`, `training/train_round*.py`, etc.) **siguen funcionando**
> y se migrarán en planes 02-06.

---

# docling-layout-fine-tuning

Fine-tuning del modelo [Docling Layout Heron](https://huggingface.co/docling-project/docling-layout-heron) (RT-DETR v2) para detección de layout en documentos tipo EAF (Estudios de Áreas de Influencia), usando CVAT para anotación y LoRA para entrenamiento eficiente.

## Resultados

| Modelo | mAP@[0.5:0.95] | Descripción |
|--------|----------------|-------------|
| Heron original | 0.8482 | Modelo base sin fine-tuning |
| **P (ganador)** | **0.8700** | LoRA r=32 + Repeat Factor Sampling |

El modelo fine-tuned supera al baseline en +2.6% mAP usando solo 162 páginas de entrenamiento y una GTX 1650 (4GB VRAM).

## Arquitectura

```
Modelo base:     docling-project/docling-layout-heron (RT-DETR v2, ResNet101, 42.7M params)
Fine-tuning:     LoRA rank=32, alpha=64, dropout=0.05
Capas adaptadas: q_proj, k_proj, v_proj en decoder attention (18 capas, 5.2M params trainables)
Backbone:        Frozen
Encoder:         Frozen
```

## Pipeline completo

```
1. PDF → Imágenes PNG (300 DPI)
2. Imágenes → CVAT (anotación/corrección humana)
3. CVAT → COCO JSON (export de anotaciones)
4. COCO JSON → Fine-tuning con LoRA + Repeat Factor Sampling
5. Modelo → Inferencia + Post-procesamiento → PDF anotado
```

## Requisitos

### Hardware
- GPU NVIDIA con >= 4GB VRAM (testeado en GTX 1650 Max-Q)
- 16GB RAM

### Software
```bash
pip install torch torchvision transformers Pillow PyMuPDF requests
```

### CVAT (para anotación)
```bash
# Clonar CVAT
git clone https://github.com/cvat-ai/cvat.git
cd cvat

# Levantar con docker compose
docker compose up -d

# Crear usuario admin
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

# Acceder en http://localhost:8080
```

## Estructura del proyecto

```
.
├── README.md                          # Este archivo
├── training/
│   ├── EXPERIMENTS.md                 # Detalle de todos los experimentos
│   ├── train_strategies.py            # Round 1: A, B, C, D
│   ├── train_round2.py               # Round 2: E, F, G
│   ├── train_round3.py               # Round 3: H, I, J, K
│   ├── train_round4.py               # Round 4: L, M, N, O
│   ├── reevaluate_real_map.py         # Evaluación con mAP@[0.5:0.95]
│   ├── draw_boxes_on_pdf_v2.py        # Genera PDFs anotados sobre PDF original
│   ├── run_docling_pipeline.py        # Pipeline Docling con monkey-patch LoRA
│   ├── EAF-477-2025_ground_truth.json # Anotaciones COCO del dataset
│   └── models/
│       └── P_repeat_factor/           # Modelo ganador
│           ├── best_model.pt          # Pesos del modelo
│           └── history.json           # Historia de entrenamiento
├── generate_heron_coco.py             # Inferencia Heron → COCO JSON
├── upload_to_cvat.py                  # Sube imágenes + anotaciones a CVAT
├── clean_overlaps_v3.py               # Limpieza de overlaps para CVAT
└── coco_output/                       # COCO JSONs generados por inferencia
```

## Configuración de CVAT

### Instalación
1. Clonar el repo de CVAT: `git clone https://github.com/cvat-ai/cvat.git`
2. Levantar con Docker: `docker compose up -d` (levanta ~18 contenedores)
3. Crear superusuario: `docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'`
4. Acceder en `http://localhost:8080`

### Proyectos creados
| # | Proyecto | Descripción |
|---|----------|-------------|
| 1 | Docling Layout PDF | Anotaciones manuales originales |
| 2 | Docling Heron Predictions | Predicciones del modelo (obsoleto) |
| 3 | Docling Heron RAW (con overlaps) | Predicciones crudas sin filtrar |
| 4 | Docling Heron CLEAN (sin overlaps v3) | **Datos de entrenamiento** |

### Labels (17 categorías)
Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title, Document Index, Code, Checkbox-selected, Checkbox-unselected, Form, Key-value-region

### Workflow de anotación
1. Correr `generate_heron_coco.py` para generar predicciones del modelo
2. Subir a CVAT con `upload_to_cvat.py`
3. Limpiar overlaps con `clean_overlaps_v3.py`
4. Corregir manualmente en CVAT
5. Exportar COCO JSON actualizado para entrenamiento

## Entrenamiento

### Modelo ganador: P (Repeat Factor Sampling)

```python
# Parámetros
LoRA rank:        32
LoRA alpha:       64
LoRA dropout:     0.05
LoRA target:      q_proj, k_proj, v_proj (18 capas del decoder)
LR:               1e-4
LR schedule:      Cosine annealing + warmup 5 epochs
Optimizer:        AdamW (weight_decay=1e-4)
Batch size:       1 (grad accum = 4, effective = 4)
Gradient clip:    0.1
Max epochs:       50 (early stop patience=10)
Augmentation:     Color jitter, rotación ±3°, Gaussian blur
Sampling:         Repeat Factor Sampling (todas las clases)
```

### Repeat Factor Sampling
Cada imagen se repite proporcionalmente a la rareza de sus clases:
- Imágenes con Footnote (2 ejemplos) → se repiten 5x
- Imágenes con Caption (4 ejemplos) → se repiten 4x
- Imágenes con Picture (26 ejemplos) → se repiten 2x
- Imágenes con Table (194 ejemplos) → se repiten 1x

### Cómo entrenar
```bash
# 1. Exportar anotaciones de CVAT a COCO JSON
# 2. Validar datos (sin overlaps, bboxes válidos)
# 3. Entrenar
python training/train_round4.py
```

## Post-procesamiento

Las predicciones crudas del modelo necesitan post-procesamiento:

1. **NMS per-category**: Elimina duplicados de la misma categoría (IoU > 0.5)
2. **Threshold per-category**: Caption/Table/Text/Picture >= 0.5, Section-header/Title/Code >= 0.45
3. **Cross-category cleanup**: Solapamientos entre categorías distintas → elimina el de menor confianza
4. **Full-page picture filter**: Pictures >90% de la página = falsos positivos

Docling aplica todo esto automáticamente via `LayoutPostprocessor`.

### Estrategia de limpieza de overlaps

#### Paso 1: Matar wrappers (boxes con 2+ hijos)
Un box que contiene 2+ boxes adentro (>70% containment) es un wrapper falso.

```
ANTES:                          DESPUÉS:
┌───────────────────────┐
│ TEXT (wrapper falso)   │
│  ┌─────┐  ┌────────┐  │       ┌─────┐  ┌────────┐
│  │TABLE│  │TEXT real│  │  →    │TABLE│  │TEXT real│
│  └─────┘  └────────┘  │       └─────┘  └────────┘
└───────────────────────┘
```

#### Paso 2: Resolver solapamientos
| Situación | Acción |
|-----------|--------|
| IoU > 0.8 (casi idénticos) | Eliminar el de menor confianza |
| Containment > 70% (uno dentro de otro) | Eliminar el exterior |
| IoU 0.3-0.8 (parcial) | Eliminar el de menor confianza |
| IoU < 0.3 | Mantener ambos |

## Generar PDF anotado

```bash
# Con el modelo ganador P
python training/draw_boxes_on_pdf_v2.py
# Output: resultados/EAF-477_pdf_repeat_factor.pdf
```

## Integración con Docling

El modelo LoRA se puede inyectar en la pipeline de Docling via monkey-patch:

```python
# Ver run_docling_pipeline.py para el ejemplo completo
predictor = layout_model.layout_predictor
apply_lora(predictor._model, rank=32, alpha=64)
predictor._model.load_state_dict(mapped_state)
```

Requiere mapear keys entre versiones de transformers (4.x vs 5.x).

## Experimentos detallados

Ver [training/EXPERIMENTS.md](training/EXPERIMENTS.md) para el detalle completo de los 16 experimentos realizados.
