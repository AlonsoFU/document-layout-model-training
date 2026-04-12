"""
Corre la pipeline completa de Docling sobre EAF-477:
  1. Con el modelo original
  2. Con el modelo E (LoRA r=32) fine-tuned

Genera PDFs anotados con el post-procesamiento real de Docling.
"""
import os
import gc
import math
import torch
from pathlib import Path
from PIL import Image, ImageDraw

PDF_PATH = "/home/alonso/prueba_cvat/inputs/EAF-477-2025/EAF-477-2025.pdf"
OUTPUT_DIR = "/home/alonso/cvat_docling/training/resultados"
BEST_MODEL_PATH = "/home/alonso/cvat_docling/training/models/E_lora32_cosine_aug/best_model.pt"

COLORS = {
    "caption": (255, 0, 0), "footnote": (0, 0, 255), "formula": (0, 200, 0),
    "list_item": (255, 165, 0), "page_footer": (128, 0, 128),
    "page_header": (0, 200, 200), "picture": (255, 0, 255),
    "section_header": (200, 200, 0), "table": (0, 255, 0),
    "text": (255, 105, 180), "title": (139, 69, 19),
    "document_index": (128, 128, 128), "code": (0, 0, 128),
    "checkbox_selected": (0, 128, 128), "checkbox_unselected": (255, 127, 80),
    "form": (255, 215, 0), "key_value_region": (128, 128, 0),
}


class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank=32, alpha=64, dropout=0.05):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = torch.nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, original.out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora(model, rank=32, alpha=64):
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
    return model


def draw_clusters_on_image(page_image, clusters, title=""):
    """Draw clusters on page image."""
    img = page_image.copy()
    draw = ImageDraw.Draw(img)
    detected = set()

    for cluster in clusters:
        label_name = cluster.label.value
        color = COLORS.get(label_name, (0, 0, 0))
        detected.add(label_name)
        conf = cluster.confidence

        # Scale bbox to image coordinates
        bbox = cluster.bbox
        x1, y1, x2, y2 = bbox.l, bbox.t, bbox.r, bbox.b

        # Scale to image size
        sx = img.width / page_size_w if page_size_w else 1
        sy = img.height / page_size_h if page_size_h else 1
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        text = f"{label_name} {conf:.0%}"
        try:
            tb = draw.textbbox((x1, y1 - 32), text, font_size=22)
            draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill="white", outline=color, width=2)
            draw.text((x1, y1 - 32), text, fill=color, font_size=22)
        except:
            draw.text((x1, y1 - 15), text, fill=color)

    if title:
        try:
            tb = draw.textbbox((10, 10), title, font_size=36)
            draw.rectangle([5, 5, tb[2]+10, tb[3]+10], fill="white", outline="black", width=3)
            draw.text((10, 10), title, fill="black", font_size=36)
        except:
            draw.text((10, 10), title, fill="black")

    # Legend
    y = 60 if title else 10
    for name in sorted(detected):
        c = COLORS.get(name, (0, 0, 0))
        draw.rectangle([10, y, 35, y + 22], fill=c)
        try:
            draw.text((42, y), name, fill="black", font_size=18)
        except:
            draw.text((42, y), name, fill="black")
        y += 26

    return img


def run_docling_pipeline(pdf_path, output_pdf, title, patch_model_path=None):
    """Run Docling pipeline and generate annotated PDF."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorDevice

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = False
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_page_images = True
    # When patching, force OCR to CPU to leave GPU for layout
    if patch_model_path:
        from docling.datamodel.pipeline_options import OcrOptions, EasyOcrOptions
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
        pipeline_options.do_ocr = False  # Skip OCR entirely for speed + memory

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # If we need to patch the model, do it after converter is created
    if patch_model_path:
        pdf_pipeline = converter._get_pipeline(InputFormat.PDF)
        layout_model = pdf_pipeline.layout_model
        predictor = layout_model.layout_predictor

        print(f"  Patching layout model with {patch_model_path}")

        # The training venv (transformers 5.3) saves state_dict with different
        # key names than docling's venv (transformers 4.57). We need to:
        # 1. Apply LoRA to docling's model (same architecture, different key names)
        # 2. Map the saved state_dict keys to docling's naming convention
        # 3. Load the mapped weights

        # Apply LoRA to docling's model
        apply_lora(predictor._model, rank=32, alpha=64)

        # Load saved state_dict and map keys
        saved_state = torch.load(patch_model_path, map_location=predictor._device)

        # The saved state has keys from transformers 5.3 naming:
        #   encoder.aifi, self_attn.o_proj, mlp.fc1
        # Docling's transformers 4.57 uses:
        #   encoder.encoder, self_attn.out_proj, fc1
        # LoRA wrapping adds .original, .lora_A, .lora_B sub-keys
        #
        # In saved state (5.3 + LoRA on o_proj):
        #   self_attn.o_proj.weight -> this is the ORIGINAL weight (o_proj was not LoRA'd)
        #   Wait - LoRA matches "out_proj" in the name, but 5.3 calls it "o_proj"...
        #
        # Let me check: in training, apply_lora looks for "out_proj" in the name.
        # In transformers 5.3, the layer is called "o_proj", NOT "out_proj".
        # So LoRA was NOT applied to o_proj! It was only applied to q_proj, k_proj, v_proj.

        key_map = {}
        for key in saved_state.keys():
            new_key = key
            new_key = new_key.replace("encoder.aifi.", "encoder.encoder.")
            new_key = new_key.replace("self_attn.o_proj.", "self_attn.out_proj.")
            new_key = new_key.replace(".mlp.fc1.", ".fc1.")
            new_key = new_key.replace(".mlp.fc2.", ".fc2.")
            key_map[key] = new_key

        mapped_state = {key_map.get(k, k): v for k, v in saved_state.items()}

        # Don't apply LoRA to out_proj since it wasn't LoRA'd in training (it was o_proj there)
        # Only apply LoRA to q_proj, k_proj, v_proj
        # Re-do: apply LoRA only to the layers that actually have LoRA weights
        lora_layers = set()
        for key in mapped_state:
            if ".lora_A.weight" in key:
                # e.g. model.decoder.layers.0.self_attn.q_proj.lora_A.weight
                layer_name = key.rsplit(".lora_A.weight", 1)[0]
                lora_layers.add(layer_name)

        print(f"  LoRA layers found in checkpoint: {len(lora_layers)}")

        # Free old model from GPU first
        del predictor._model
        gc.collect()
        torch.cuda.empty_cache()

        # Reload clean model and apply LoRA only to correct layers
        from transformers import AutoModelForObjectDetection
        predictor._model = AutoModelForObjectDetection.from_pretrained(
            "docling-project/docling-layout-heron",
            device_map=predictor._device
        )

        # Apply LoRA only to layers that have LoRA weights in checkpoint
        replaced = 0
        for name, module in predictor._model.named_modules():
            full_name = "model." + name if not name.startswith("model.") else name
            if full_name in lora_layers and isinstance(module, torch.nn.Linear):
                parts = name.split(".")
                parent = predictor._model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank=32, alpha=64))
                replaced += 1

        print(f"  Applied LoRA to {replaced} layers")
        predictor._model.load_state_dict(mapped_state)
        predictor._model.to(predictor._device)
        predictor._model.eval()
        print("  Model patched!")

    print(f"  Converting {pdf_path}...")
    result = converter.convert(pdf_path)

    # Extract pages with clusters and generate annotated images
    print(f"  Generating annotated PDF...")
    import pikepdf
    tmp_pages = []
    global page_size_w, page_size_h

    for page_no, page in enumerate(result.pages):
        if page.predictions and page.predictions.layout:
            clusters = page.predictions.layout.clusters
            page_image = page.image
            if page.size:
                page_size_w = page.size.width
                page_size_h = page.size.height
            else:
                page_size_w = page_image.width
                page_size_h = page_image.height

            annotated = draw_clusters_on_image(page_image, clusters, title)

            # Resize to 50%
            w, h = annotated.size
            annotated = annotated.resize((w // 2, h // 2), Image.LANCZOS)

            tmp_path = f"/tmp/docling_page_{page_no:04d}.jpg"
            annotated.save(tmp_path, "JPEG", quality=95)
            tmp_pages.append(tmp_path)

            if (page_no + 1) % 50 == 0:
                print(f"    Page {page_no + 1}...")

    # Save as PDF
    CHUNK = 30
    partial_paths = []
    for c in range(0, len(tmp_pages), CHUNK):
        chunk = tmp_pages[c:c+CHUNK]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        pp = output_pdf + f".part{c}"
        imgs[0].save(pp, "PDF", save_all=True, append_images=imgs[1:], resolution=100)
        for img in imgs:
            img.close()
        partial_paths.append(pp)
        gc.collect()

    if len(partial_paths) == 1:
        os.rename(partial_paths[0], output_pdf)
    else:
        pdf = pikepdf.Pdf.open(partial_paths[0])
        for pp in partial_paths[1:]:
            src = pikepdf.Pdf.open(pp)
            pdf.pages.extend(src.pages)
        pdf.save(output_pdf)
        pdf.close()
        for pp in partial_paths:
            os.remove(pp)

    # Cleanup
    for p in tmp_pages:
        if os.path.exists(p):
            os.remove(p)

    print(f"  Saved: {output_pdf}")
    print(f"  Total clusters: {sum(len(p.predictions.layout.clusters) for p in result.pages if p.predictions and p.predictions.layout)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Original Docling pipeline
    print("\n=== DOCLING ORIGINAL ===")
    run_docling_pipeline(
        PDF_PATH,
        os.path.join(OUTPUT_DIR, "EAF-477_docling_original.pdf"),
        "DOCLING ORIGINAL"
    )

    gc.collect()
    torch.cuda.empty_cache()

    # 2. Docling with LoRA fine-tuned model
    print("\n=== DOCLING + LORA FINE-TUNED ===")
    run_docling_pipeline(
        PDF_PATH,
        os.path.join(OUTPUT_DIR, "EAF-477_docling_lora.pdf"),
        "DOCLING + LORA E",
        patch_model_path=BEST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
