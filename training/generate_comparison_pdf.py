"""
Genera PDFs con boxes dibujados: Original vs LoRA fine-tuned.
"""
import os
import gc
import math
import torch
import torchvision
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_NAME = "docling-project/docling-layout-heron"
BEST_MODEL_PATH = "/home/alonso/cvat_docling/training/models/B_lora/best_model.pt"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
OUTPUT_DIR = "/home/alonso/cvat_docling/training/resultados"
THRESHOLD = 0.3
NMS_IOU = 0.5

LABELS = {
    0: "caption", 1: "footnote", 2: "formula", 3: "list_item",
    4: "page_footer", 5: "page_header", 6: "picture", 7: "section_header",
    8: "table", 9: "text", 10: "title", 11: "document_index",
    12: "code", 13: "checkbox_selected", 14: "checkbox_unselected",
    15: "form", 16: "key_value_region"
}

COLORS = {
    0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 200, 0), 3: (255, 165, 0),
    4: (128, 0, 128), 5: (0, 200, 200), 6: (255, 0, 255), 7: (200, 200, 0),
    8: (0, 255, 0), 9: (255, 105, 180), 10: (139, 69, 19), 11: (128, 128, 128),
    12: (0, 0, 128), 13: (0, 128, 128), 14: (255, 127, 80), 15: (255, 215, 0),
    16: (128, 128, 0),
}


class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank=8, alpha=16):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        in_features = original.in_features
        out_features = original.out_features
        self.lora_A = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora(model, rank=8, alpha=16):
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    if p.isdigit():
                        parent = parent[int(p)]
                    else:
                        parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
    return model


def apply_nms(results, iou_threshold=0.5):
    if len(results["scores"]) == 0:
        return results
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    keep_indices = []
    for label in labels.unique():
        mask = labels == label
        nms_keep = torchvision.ops.nms(boxes[mask], scores[mask], iou_threshold)
        keep_indices.extend(torch.where(mask)[0][nms_keep].tolist())
    keep = torch.tensor(sorted(keep_indices), dtype=torch.long, device=boxes.device)
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def draw_detections(image, results, title=""):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    detected_labels = set()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        lid = label.item()
        conf = score.item()
        box = box.tolist()
        name = LABELS.get(lid, f"?{lid}")
        color = COLORS.get(lid, (0, 0, 0))
        detected_labels.add(lid)

        draw.rectangle(box, outline=color, width=4)
        text = f"{name} {conf:.0%}"
        tb = draw.textbbox((box[0], box[1] - 32), text, font_size=22)
        draw.rectangle([tb[0]-2, tb[1]-2, tb[2]+2, tb[3]+2], fill="white", outline=color, width=2)
        draw.text((box[0], box[1] - 32), text, fill=color, font_size=22)

    # Title
    if title:
        tb = draw.textbbox((10, 10), title, font_size=36)
        draw.rectangle([5, 5, tb[2]+10, tb[3]+10], fill="white", outline="black", width=3)
        draw.text((10, 10), title, fill="black", font_size=36)

    # Legend
    y = 60 if title else 10
    for lid in sorted(detected_labels):
        c = COLORS.get(lid, (0, 0, 0))
        draw.rectangle([10, y, 35, y + 22], fill=c)
        draw.text((42, y), LABELS.get(lid, "?"), fill="black", font_size=18)
        y += 26

    return img


def save_pdf_chunked(pages, output_path):
    """Save pages as PDF in chunks to avoid OOM."""
    import pikepdf
    CHUNK = 30
    partial_paths = []

    for c in range(0, len(pages), CHUNK):
        chunk = pages[c:c+CHUNK]
        first = chunk[0]
        rest = chunk[1:]
        pp = output_path + f".part{c}"
        first.save(pp, "PDF", save_all=True, append_images=rest, resolution=100)
        first.close()
        for img in rest:
            img.close()
        partial_paths.append(pp)
        gc.collect()

    if len(partial_paths) == 1:
        os.rename(partial_paths[0], output_path)
    else:
        pdf = pikepdf.Pdf.open(partial_paths[0])
        for pp in partial_paths[1:]:
            src = pikepdf.Pdf.open(pp)
            pdf.pages.extend(src.pages)
        pdf.save(output_path)
        pdf.close()
        for pp in partial_paths:
            os.remove(pp)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)

    # Load both models
    print("Cargando modelo original...")
    original_model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    original_model.to(device).eval()

    print("Cargando modelo LoRA...")
    lora_model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    lora_model = apply_lora(lora_model, rank=8, alpha=16)
    lora_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    lora_model.to(device).eval()

    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])
    print(f"Procesando {len(image_files)} páginas...")

    pages_original = []
    pages_lora = []
    pages_sidebyside = []

    for i, fname in enumerate(image_files):
        image = Image.open(os.path.join(IMAGES_DIR, fname)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_sizes = torch.tensor([image.size[::-1]]).to(device)

        with torch.no_grad():
            out_orig = original_model(**inputs)
            out_lora = lora_model(**inputs)

        res_orig = apply_nms(
            processor.post_process_object_detection(out_orig, target_sizes=target_sizes, threshold=THRESHOLD)[0],
            NMS_IOU
        )
        res_lora = apply_nms(
            processor.post_process_object_detection(out_lora, target_sizes=target_sizes, threshold=THRESHOLD)[0],
            NMS_IOU
        )

        img_orig = draw_detections(image, res_orig, "ORIGINAL")
        img_lora = draw_detections(image, res_lora, "LORA FINE-TUNED")

        # Reduce to 50% for PDF size
        w, h = img_orig.size
        half = (w // 2, h // 2)
        img_orig_small = img_orig.resize(half, Image.LANCZOS)
        img_lora_small = img_lora.resize(half, Image.LANCZOS)

        pages_original.append(img_orig_small)
        pages_lora.append(img_lora_small)

        # Side by side
        sw, sh = half
        combined = Image.new("RGB", (sw * 2, sh), "white")
        combined.paste(img_orig_small, (0, 0))
        combined.paste(img_lora_small, (sw, 0))
        pages_sidebyside.append(combined)

        del out_orig, out_lora, inputs, image, img_orig, img_lora
        torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(image_files)}...")
            gc.collect()

    # Save PDFs
    print("Generando PDFs...")

    save_pdf_chunked(pages_original, os.path.join(OUTPUT_DIR, "EAF-477_original.pdf"))
    print(f"  {OUTPUT_DIR}/EAF-477_original.pdf")

    save_pdf_chunked(pages_lora, os.path.join(OUTPUT_DIR, "EAF-477_lora_finetuned.pdf"))
    print(f"  {OUTPUT_DIR}/EAF-477_lora_finetuned.pdf")

    save_pdf_chunked(pages_sidebyside, os.path.join(OUTPUT_DIR, "EAF-477_comparacion.pdf"))
    print(f"  {OUTPUT_DIR}/EAF-477_comparacion.pdf")

    print("\nListo!")


if __name__ == "__main__":
    main()
