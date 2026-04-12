"""
Dibuja bounding boxes directamente sobre el PDF original usando PyMuPDF (fitz).
No necesita cargar modelos ni imágenes — trabaja directo con el PDF.
"""
import json
import fitz  # PyMuPDF
import torch
import torchvision
import math
import gc
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

PDF_PATH = "/home/alonso/prueba_cvat/inputs/EAF-477-2025/EAF-477-2025.pdf"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
OUTPUT_DIR = "/home/alonso/cvat_docling/training/resultados"
THRESHOLD = 0.3
NMS_IOU = 0.5
MODEL_NAME = "docling-project/docling-layout-heron"
BEST_MODEL_PATH = "/home/alonso/cvat_docling/training/models/E_lora32_cosine_aug/best_model.pt"

LABELS = {
    0: "caption", 1: "footnote", 2: "formula", 3: "list_item",
    4: "page_footer", 5: "page_header", 6: "picture", 7: "section_header",
    8: "table", 9: "text", 10: "title", 11: "document_index",
    12: "code", 13: "checkbox_selected", 14: "checkbox_unselected",
    15: "form", 16: "key_value_region"
}

# Colors as RGB tuples (0-1 range for fitz)
COLORS = {
    0: (1, 0, 0), 1: (0, 0, 1), 2: (0, 0.78, 0), 3: (1, 0.65, 0),
    4: (0.5, 0, 0.5), 5: (0, 0.78, 0.78), 6: (1, 0, 1), 7: (0.78, 0.78, 0),
    8: (0, 1, 0), 9: (1, 0.41, 0.71), 10: (0.55, 0.27, 0.07), 11: (0.5, 0.5, 0.5),
    12: (0, 0, 0.5), 13: (0, 0.5, 0.5), 14: (1, 0.5, 0.31), 15: (1, 0.84, 0),
    16: (0.5, 0.5, 0),
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


def apply_nms(results):
    if len(results["scores"]) == 0:
        return results
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    keep_indices = []
    for label in labels.unique():
        mask = labels == label
        nms_keep = torchvision.ops.nms(boxes[mask], scores[mask], NMS_IOU)
        keep_indices.extend(torch.where(mask)[0][nms_keep].tolist())
    keep = torch.tensor(sorted(keep_indices), dtype=torch.long, device=boxes.device)
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def run_inference(model, processor, device, image_files):
    """Run inference on all images and return detections per page."""
    import os
    all_detections = []
    for i, fname in enumerate(image_files):
        image = Image.open(os.path.join(IMAGES_DIR, fname)).convert("RGB")
        img_w, img_h = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_sizes = torch.tensor([[img_h, img_w]]).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=THRESHOLD
        )[0]
        results = apply_nms(results)

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": label.item(),
                "score": score.item(),
                "box": box.tolist(),  # [x1, y1, x2, y2] in pixel coords
                "img_w": img_w,
                "img_h": img_h,
            })
        all_detections.append(detections)

        del outputs, inputs, image
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(image_files)}...")
            gc.collect()

    return all_detections


def draw_on_pdf(pdf_path, output_path, all_detections, title):
    """Draw boxes directly on the PDF pages."""
    doc = fitz.open(pdf_path)

    for page_no in range(len(doc)):
        if page_no >= len(all_detections):
            break

        page = doc[page_no]
        detections = all_detections[page_no]
        if not detections:
            continue

        # PDF page dimensions
        pdf_w = page.rect.width
        pdf_h = page.rect.height

        # Image dimensions (from inference)
        if detections:
            img_w = detections[0]["img_w"]
            img_h = detections[0]["img_h"]
        else:
            continue

        # Scale factors: image coords -> PDF coords
        sx = pdf_w / img_w
        sy = pdf_h / img_h

        detected_labels = set()

        for det in detections:
            lid = det["label"]
            conf = det["score"]
            x1, y1, x2, y2 = det["box"]
            color = COLORS.get(lid, (0, 0, 0))
            name = LABELS.get(lid, f"?{lid}")
            detected_labels.add(lid)

            # Scale to PDF coordinates
            rect = fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)

            # Draw rectangle
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=color, width=1.5)
            shape.commit()

            # Draw label text
            label_text = f"{name} {conf:.0%}"
            text_point = fitz.Point(rect.x0, rect.y0 - 3)
            page.insert_text(text_point, label_text, fontsize=7, color=color)

        # Draw title
        if title:
            page.insert_text(fitz.Point(10, 15), title, fontsize=12, color=(0, 0, 0))

    doc.save(output_path)
    doc.close()
    print(f"  Saved: {output_path}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])

    # === ORIGINAL ===
    print("=== Modelo Original ===")
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME).to(device).eval()
    detections_orig = run_inference(model, processor, device, image_files)
    del model; gc.collect(); torch.cuda.empty_cache()
    draw_on_pdf(PDF_PATH, os.path.join(OUTPUT_DIR, "EAF-477_pdf_original.pdf"), detections_orig, "ORIGINAL")
    del detections_orig; gc.collect()

    # === LORA ===
    print("\n=== Modelo LoRA E ===")
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    apply_lora(model, rank=32, alpha=64)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device).eval()
    detections_lora = run_inference(model, processor, device, image_files)
    del model; gc.collect(); torch.cuda.empty_cache()
    draw_on_pdf(PDF_PATH, os.path.join(OUTPUT_DIR, "EAF-477_pdf_lora.pdf"), detections_lora, "LORA E")

    print("\nListo!")


if __name__ == "__main__":
    main()
