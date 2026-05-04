# Experimentos de Fine-tuning Docling Heron

## Modelo base
- **Modelo**: docling-project/docling-layout-heron (RT-DETR v2, ResNet101)
- **Parámetros totales**: 42.7M
- **Dataset**: EAF-477-2025, 162 páginas (137 train / 25 val)
- **Datos de entrenamiento**: Proyecto CVAT 4 "Docling Heron CLEAN (sin overlaps v3)" - 646 anotaciones limpias
- **GPU**: NVIDIA GeForce GTX 1650 Max-Q (4GB VRAM)
- **Métrica**: mAP@[0.5:0.95] (estándar COCO/DocLayNet)

## Resultados completos

### Round 1: Estrategias base

| # | Estrategia | Params | mAP@[.5:.95] | mAP@.50 | Epochs | Estado |
|---|-----------|--------|-------------|---------|--------|--------|
| - | **Baseline (original)** | 0 | **0.8482** | 0.8694 | - | referencia |
| A | Frozen backbone+encoder | 6.1M (14.3%) | 0.8069 | 0.8463 | 25 | completado |
| B | LoRA r=8, lr=1e-4 | 5.0M (11.6%) | 0.8102 | 0.9244 | 23 | completado |
| C | Progressive unfreezing | 6.1M→100M | - | 0.7607 | 15 | FAILED (param groups) |
| D | Full fine-tune, lr=2e-6 | 42.8M (100%) | - | - | 1 | FAILED (NaN/OOM) |

### Round 2: Técnicas avanzadas

| # | Estrategia | mAP@[.5:.95] | mAP@.50 | Epochs | Estado |
|---|-----------|-------------|---------|--------|--------|
| E | LoRA r=32 + cosine warmup 5ep + aug | 0.8117 | 0.9562 | 35 | completado |
| F | LoRA r=32 + EMA + aug (sin cosine) | 0.6784 | 0.8263 | 25 | completado |
| G | LoRA r=32 + EMA(CPU) + cosine + aug | 0.7570 | 0.9137 | 50 | completado |

### Round 3: Variaciones de LoRA

| # | Estrategia | mAP@[.5:.95] | mAP@.50 | Epochs | Estado |
|---|-----------|-------------|---------|--------|--------|
| H | LoRA r=64 + cosine + aug | 0.8135 | 0.9442 | 41 | completado |
| I | LoRA r=32 + cosine + aug, lr=5e-5 | 0.8162 | 0.8520 | 18 | completado |
| J | LoRA r=32 ALL linears + cosine + aug | 0.7743 | 0.8454 | - | completado |
| K | LoRA r=32 + cosine + aug, warmup 10ep | 0.7710 | 0.9429 | 26 | completado |

### Round 4: Desbalance de clases

| # | Estrategia | mAP@[.5:.95] | Epochs | Estado |
|---|-----------|-------------|--------|--------|
| L | E + oversampling 3x pictures | 0.8129 | - | completado |
| M | E + oversampling 5x pictures | 0.8676 | 24 | completado |
| N | E + class-weighted loss (3x pictures) | 0.8112 | - | completado |
| O | E + oversampling 3x + weighted loss | 0.8583 | 40 | completado |
| **P** | **E + Repeat Factor Sampling (todas las clases)** | **0.8700** | **19** | **GANADOR** |
| E_v2 | E sin oversample (datos actualizados) | 0.8126 | 32 | completado |

## Ganador: Estrategia P - Repeat Factor Sampling

```
Técnica:          LoRA (Low-Rank Adaptation)
LoRA rank:        32
LoRA alpha:       64
LoRA dropout:     0.05
LoRA target:      q_proj, k_proj, v_proj en decoder attention (18 capas)
Backbone:         Frozen
Encoder:          Frozen
LR:               1e-4
LR schedule:      Cosine annealing con warmup 5 epochs
Optimizer:        AdamW (weight_decay=1e-4)
Batch size:       1 (gradient accumulation = 4, effective batch = 4)
Gradient clip:    0.1
Max epochs:       50 (early stop patience=10)
Best epoch:       19
Data augmentation: Color jitter, rotación ±3°, Gaussian blur
Denoising:        Deshabilitado (num_denoising=0)
EOS coefficient:  0.0001
Sampling:         Repeat Factor Sampling (todas las clases)
                  - threshold=0.5
                  - 77 imgs x1, 22 imgs x2, 16 imgs x4, 22 imgs x5
                  - Total: 295 samples por epoch (vs 137 original)
```

Modelo guardado en: `models/P_repeat_factor/best_model.pt`

## Distribución de clases

```
Table:           194  (30.0%)
Section-header:  150  (23.2%)
Text:            144  (22.3%)
Page-header:      51  (7.9%)
List-item:        42  (6.5%)
Picture:          26  (4.0%)
Page-footer:      23  (3.6%)
Title:            10  (1.5%)
Caption:           4  (0.6%)
Footnote:          2  (0.3%)
Document Index:    1  (0.2%)
```

## Lecciones aprendidas

1. **LoRA > Frozen decoder**: LoRA adapta las capas de atención de forma más eficiente
2. **Cosine warmup es clave**: Sin warmup el LR constante oscila sin converger
3. **EMA no ayuda en datasets pequeños**: Suaviza demasiado y frena el aprendizaje
4. **LoRA rank 32 es suficiente**: r=64 no mejoró, r=8 también fue competitivo
5. **LR 5e-5 es muy bajo**: Converge muy lento y hace early stop antes del óptimo
6. **Full fine-tune imposible en 4GB**: Necesita fp16 que causa NaN, o más VRAM
7. **Repeat Factor Sampling > Oversampling selectivo**: Balancear todas las clases funciona mejor que solo sobremuestrear una. Oversampling 5x de pictures causaba confusión en tablas
8. **Weighted loss solo no sirve**: Escalar el loss no compensa sin más ejemplos
9. **mAP@0.50 vs mAP@[0.5:0.95]**: El fine-tuning mejora detección pero puede perder precisión geométrica en IoU altos
