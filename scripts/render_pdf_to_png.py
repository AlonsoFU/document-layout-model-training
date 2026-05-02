"""Render PDF pages to PNG at 300 DPI (matches CVAT annotations)."""
import argparse
from pathlib import Path

import pikepdf  # noqa: F401  # ensure pdf libs available
from pdf2image import convert_from_path


def render(pdf_path: Path, out_dir: Path, dpi: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    for i, page in enumerate(pages, start=1):
        page.save(out_dir / f"pagina-{i:03d}.png", "PNG")
    print(f"{pdf_path.name}: {len(pages)} pages -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path, help="Path to source PDF")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    render(args.pdf, args.out, args.dpi)


if __name__ == "__main__":
    main()
