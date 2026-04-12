"""
Genera PDFs con boxes dibujados: Original vs LoRA fine-tuned.
Procesa un modelo a la vez para no reventar VRAM.
"""
import os
import gc
import math
import torch
import torchvision
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_NAME = "docling-project/docling-layout-heron"
BEST_MODEL_PATH = "/home/alonso/cvat_docling/training/models/E_lora32_cosine_aug/best_model.pt"
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
    if title:
        tb = draw.textbbox((10, 10), title, font_size=36)
        draw.rectangle([5, 5, tb[2]+10, tb[3]+10], fill="white", outline="black", width=3)
        draw.text((10, 10), title, fill="black", font_size=36)
    y = 60 if title else 10
    for lid in sorted(detected_labels):
        c = COLORS.get(lid, (0, 0, 0))
        draw.rectangle([10, y, 35, y + 22], fill=c)
        draw.text((42, y), LABELS.get(lid, "?"), fill="black", font_size=18)
        y += 26
    return img


def run_model_and_save(model, processor, device, image_files, title, tmp_dir):
    """Run inference and save annotated pages as JPEG to tmp."""
    os.makedirs(tmp_dir, exist_ok=True)
    for i, fname in enumerate(image_files):
        image = Image.open(os.path.join(IMAGES_DIR, fname)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = apply_nms(
            processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=THRESHOLD)[0]
        )
        annotated = draw_detections(image, results, title)
        w, h = annotated.size
        annotated = annotated.resize((w // 2, h // 2), Image.LANCZOS)
        annotated.save(os.path.join(tmp_dir, f"{i:04d}.jpg"), "JPEG", quality=80)
        del outputs, inputs, image, annotated
        torch.cuda.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(image_files)}...")
            gc.collect()


def jpgs_to_pdf(tmp_dir, output_path):
    import pikepdf
    files = sorted(os.listdir(tmp_dir))
    CHUNK = 30
    partial_paths = []
    for c in range(0, len(files), CHUNK):
        chunk = files[c:c+CHUNK]
        imgs = [Image.open(os.path.join(tmp_dir, f)).convert("RGB") for f in chunk]
        pp = output_path + f".part{c}"
        imgs[0].save(pp, "PDF", save_all=True, append_images=imgs[1:], resolution=100)
        for img in imgs:
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
    # Cleanup tmp
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".png")])

    # --- ORIGINAL ---
    print("=== Modelo original ===")
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME).to(device).eval()
    run_model_and_save(model, processor, device, image_files, "ORIGINAL", "/tmp/pdf_original")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    jpgs_to_pdf("/tmp/pdf_original", os.path.join(OUTPUT_DIR, "EAF-477_original.pdf"))
    print(f"  -> {OUTPUT_DIR}/EAF-477_original.pdf")

    # --- LORA ---
    print("=== Modelo LoRA fine-tuned ===")
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    model = apply_lora(model)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device).eval()
    run_model_and_save(model, processor, device, image_files, "LORA FINE-TUNED", "/tmp/pdf_lora")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    jpgs_to_pdf("/tmp/pdf_lora", os.path.join(OUTPUT_DIR, "EAF-477_lora_finetuned.pdf"))
    print(f"  -> {OUTPUT_DIR}/EAF-477_lora_finetuned.pdf")

    print("\nListo! PDFs en:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
