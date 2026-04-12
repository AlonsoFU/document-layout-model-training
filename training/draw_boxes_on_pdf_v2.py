"""
Dibuja boxes sobre PDF con:
  - Threshold 0.5 (como Docling)
  - Limpieza cross-category (elimina solapamientos entre categorías distintas)
"""
import os
import gc
import math
import fitz
import torch
import torchvision
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

PDF_PATH = "/home/alonso/prueba_cvat/inputs/EAF-477-2025/EAF-477-2025.pdf"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
OUTPUT_DIR = "/home/alonso/cvat_docling/training/resultados"
MODEL_NAME = "docling-project/docling-layout-heron"
BEST_MODEL_PATH = "/home/alonso/cvat_docling/training/models/M_oversample_5x/best_model.pt"
NMS_IOU = 0.5

# Per-category thresholds (Docling style)
CONF_THRESHOLDS = {
    0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5,
    7: 0.45, 8: 0.5, 9: 0.5, 10: 0.45, 11: 0.45,
    12: 0.45, 13: 0.45, 14: 0.45, 15: 0.45, 16: 0.45,
}

LABELS = {
    0: "caption", 1: "footnote", 2: "formula", 3: "list_item",
    4: "page_footer", 5: "page_header", 6: "picture", 7: "section_header",
    8: "table", 9: "text", 10: "title", 11: "document_index",
    12: "code", 13: "checkbox_selected", 14: "checkbox_unselected",
    15: "form", 16: "key_value_region"
}

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


def box_iou_single(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def containment(inner, outer):
    """Fraction of inner inside outer."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / (area_inner + 1e-6)


def clean_cross_category(detections):
    """Remove cross-category overlaps (Docling-style)."""
    if len(detections) <= 1:
        return detections

    to_remove = set()

    for i in range(len(detections)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(detections)):
            if j in to_remove:
                continue

            bi = detections[i]["box"]
            bj = detections[j]["box"]
            iou = box_iou_single(bi, bj)
            ci_in_j = containment(bi, bj)
            cj_in_i = containment(bj, bi)

            # Not overlapping enough
            if iou < 0.3 and ci_in_j < 0.7 and cj_in_i < 0.7:
                continue

            li = detections[i]["label"]
            lj = detections[j]["label"]
            si = detections[i]["score"]
            sj = detections[j]["score"]

            # Near identical (IoU > 0.8) -> keep higher confidence
            if iou > 0.8:
                if si >= sj:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break
                continue

            # One contains the other (>70%) -> keep the smaller (inner) one
            area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
            area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])

            if cj_in_i > 0.7 and area_i > area_j:
                to_remove.add(i)
                break
            if ci_in_j > 0.7 and area_j > area_i:
                to_remove.add(j)
                continue

            # Partial overlap -> keep higher confidence
            if iou > 0.5:
                if si >= sj:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

    return [d for idx, d in enumerate(detections) if idx not in to_remove]


def run_inference(model, processor, device, image_files):
    """Run inference with per-category thresholds and cross-category cleanup."""
    all_detections = []
    for i, fname in enumerate(image_files):
        image = Image.open(os.path.join(IMAGES_DIR, fname)).convert("RGB")
        img_w, img_h = image.size
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_sizes = torch.tensor([[img_h, img_w]]).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        # Use low base threshold, filter per-category after
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.3
        )[0]
        results = apply_nms(results)

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            lid = label.item()
            conf = score.item()
            # Per-category threshold
            if conf < CONF_THRESHOLDS.get(lid, 0.5):
                continue
            detections.append({
                "label": lid,
                "score": conf,
                "box": box.tolist(),
                "img_w": img_w,
                "img_h": img_h,
            })

        # Cross-category cleanup
        detections = clean_cross_category(detections)
        all_detections.append(detections)

        del outputs, inputs, image
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(image_files)}...")
            gc.collect()

    return all_detections


def draw_on_pdf(pdf_path, output_path, all_detections, title):
    doc = fitz.open(pdf_path)

    for page_no in range(min(len(doc), len(all_detections))):
        page = doc[page_no]
        detections = all_detections[page_no]
        if not detections:
            continue

        pdf_w = page.rect.width
        pdf_h = page.rect.height
        img_w = detections[0]["img_w"]
        img_h = detections[0]["img_h"]
        sx = pdf_w / img_w
        sy = pdf_h / img_h

        for det in detections:
            lid = det["label"]
            conf = det["score"]
            x1, y1, x2, y2 = det["box"]
            color = COLORS.get(lid, (0, 0, 0))
            name = LABELS.get(lid, f"?{lid}")

            rect = fitz.Rect(x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=color, width=1.5)
            shape.commit()

            label_text = f"{name} {conf:.0%}"
            text_point = fitz.Point(rect.x0, rect.y0 - 3)
            page.insert_text(text_point, label_text, fontsize=7, color=color)

        if title:
            page.insert_text(fitz.Point(10, 15), title, fontsize=12, color=(0, 0, 0))

    doc.save(output_path)
    doc.close()
    total = sum(len(d) for d in all_detections)
    print(f"  Saved: {output_path} ({total} detections)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])

    # === ORIGINAL ===
    print("=== Modelo Original (threshold 0.5 + cross-category cleanup) ===")
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME).to(device).eval()
    dets = run_inference(model, processor, device, image_files)
    del model; gc.collect(); torch.cuda.empty_cache()
    draw_on_pdf(PDF_PATH, os.path.join(OUTPUT_DIR, "EAF-477_pdf_original_v2.pdf"), dets, "ORIGINAL")
    del dets; gc.collect()

    # === LORA ===
    print("\n=== Modelo LoRA E (threshold 0.5 + cross-category cleanup) ===")
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    apply_lora(model, rank=32, alpha=64)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device).eval()
    dets = run_inference(model, processor, device, image_files)
    del model; gc.collect(); torch.cuda.empty_cache()
    draw_on_pdf(PDF_PATH, os.path.join(OUTPUT_DIR, "EAF-477_pdf_lora_v2.pdf"), dets, "LORA E")

    print("\nListo!")


if __name__ == "__main__":
    main()
