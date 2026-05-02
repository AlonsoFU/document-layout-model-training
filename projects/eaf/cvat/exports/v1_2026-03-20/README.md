# Project 4 — Docling Heron CLEAN (sin overlaps v3)

CVAT project export. Last updated **2026-03-20**.

## Contents
- `annotations/instances_default.json` — COCO 1.0 export (project-level merge of both tasks)

## Stats
- **Images:** 561 (EAF-089-2025: 399, EAF-477-2025: 162)
- **Annotations:** 3105
- **Categories:** 17 (1-based IDs — see `feedback_cvat_ids.md` in memory)

## Source data
PDFs are in `../../data/pdfs/`. To regenerate the PNGs that CVAT references:

```bash
python ../../scripts/render_pdf_to_png.py ../../data/pdfs/EAF-089-2025.pdf --out images/EAF-089-2025
python ../../scripts/render_pdf_to_png.py ../../data/pdfs/EAF-477-2025.pdf --out images/EAF-477-2025
```

## Categories (1-based as exported by CVAT)
| ID | Name |
|----|------|
| 1 | Document Index |
| 2 | List-item |
| 3 | Footnote |
| 4 | Checkbox-selected |
| 5 | Page-header |
| 6 | Checkbox-unselected |
| 7 | Code |
| 8 | Table |
| 9 | Text |
| 10 | Section-header |
| 11 | Formula |
| 12 | Picture |
| 13 | Key-value-region |
| 14 | Caption |
| 15 | Title |
| 16 | Form |
| 17 | Page-footer |
