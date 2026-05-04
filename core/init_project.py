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
