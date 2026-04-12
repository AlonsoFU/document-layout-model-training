"""
Round 2: Técnicas avanzadas para mejorar fine-tuning de Docling Heron.

Estrategias:
  E: LoRA rank=32 + cosine warmup + augmentation
  F: LoRA rank=32 + EMA + augmentation
  G: LoRA rank=32 + EMA + cosine warmup + augmentation (todo junto)
"""
import os
import gc
import json
import math
import random
import copy
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

MODEL_NAME = "docling-project/docling-layout-heron"
COCO_FILE = "/home/alonso/cvat_docling/training/EAF-477-2025_ground_truth.json"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
OUTPUT_BASE = "/home/alonso/cvat_docling/training/models"
SPLIT_FILE = "/home/alonso/cvat_docling/training/models/data_split.json"
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 10
GRAD_ACCUM = 4
GRAD_CLIP = 0.1

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATA AUGMENTATION
# ============================================================
class DocumentAugmenter:
    """Augmentaciones para documentos que preservan bboxes."""

    def __init__(self, strength=1.0):
        self.strength = strength

    def __call__(self, image, boxes):
        """
        image: PIL Image
        boxes: list of [cx, cy, w, h] normalized
        Returns: augmented image, augmented boxes
        """
        if random.random() < 0.5 * self.strength:
            # Color jitter
            image = self._color_jitter(image)

        if random.random() < 0.3 * self.strength:
            # Slight rotation (±3 degrees)
            image, boxes = self._rotate(image, boxes, max_angle=3)

        if random.random() < 0.4 * self.strength:
            # Scale jitter (±15%)
            image, boxes = self._scale_jitter(image, boxes)

        if random.random() < 0.3 * self.strength:
            # Gaussian blur
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < 0.2 * self.strength:
            # Grayscale
            image = ImageEnhance.Color(image).enhance(random.uniform(0.0, 0.3))

        return image, boxes

    def _color_jitter(self, image):
        # Brightness
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
        # Contrast
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
        # Saturation
        image = ImageEnhance.Color(image).enhance(random.uniform(0.7, 1.3))
        return image

    def _rotate(self, image, boxes, max_angle=3):
        angle = random.uniform(-max_angle, max_angle)
        # For small angles, bbox shift is negligible on document pages
        image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
        return image, boxes  # boxes stay same for tiny rotations

    def _scale_jitter(self, image, boxes):
        scale = random.uniform(0.85, 1.15)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        # Crop or pad back to original size
        if scale > 1.0:
            # Center crop
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            image = image.crop((left, top, left + w, top + h))
            # Adjust boxes
            new_boxes = []
            for cx, cy, bw, bh in boxes:
                ncx = (cx * new_w - left) / w
                ncy = (cy * new_h - top) / h
                nbw = bw * scale
                nbh = bh * scale
                # Clip to image bounds
                x1 = max(0, ncx - nbw / 2)
                y1 = max(0, ncy - nbh / 2)
                x2 = min(1, ncx + nbw / 2)
                y2 = min(1, ncy + nbh / 2)
                if x2 > x1 and y2 > y1:
                    new_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            boxes = new_boxes
        else:
            # Center pad
            padded = Image.new("RGB", (w, h), (255, 255, 255))
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            padded.paste(image, (left, top))
            image = padded
            new_boxes = []
            for cx, cy, bw, bh in boxes:
                ncx = (cx * new_w + left) / w
                ncy = (cy * new_h + top) / h
                nbw = bw * scale
                nbh = bh * scale
                x1 = max(0, ncx - nbw / 2)
                y1 = max(0, ncy - nbh / 2)
                x2 = min(1, ncx + nbw / 2)
                y2 = min(1, ncy + nbh / 2)
                if x2 > x1 and y2 > y1:
                    new_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
            boxes = new_boxes

        return image, boxes


# ============================================================
# DATASET
# ============================================================
class DocLayoutDataset(Dataset):
    def __init__(self, coco_data, image_ids, images_dir, processor, augmenter=None):
        self.images_dir = images_dir
        self.processor = processor
        self.image_ids = image_ids
        self.augmenter = augmenter
        self.id_to_img = {img["id"]: img for img in coco_data["images"]}
        self.img_to_anns = defaultdict(list)
        for ann in coco_data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.id_to_img[img_id]
        path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(path).convert("RGB")

        anns = self.img_to_anns[img_id]
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            img_w, img_h = img_info["width"], img_info["height"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            boxes.append([cx, cy, nw, nh])
            labels.append(ann["category_id"] - 1)

        # Apply augmentation
        if self.augmenter and boxes:
            image, boxes = self.augmenter(image, boxes)
            # Recompute labels (some boxes may have been removed by clipping)
            if len(boxes) < len(labels):
                labels = labels[:len(boxes)]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "class_labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
        }

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, target


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return pixel_values, targets


# ============================================================
# LORA
# ============================================================
class LoRALinear(torch.nn.Module):
    def __init__(self, original, rank=8, alpha=16, dropout=0.05):
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


def apply_lora(model, rank=32, alpha=64, dropout=0.05):
    replaced = 0
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))
                replaced += 1
    print(f"  LoRA rank={rank} aplicado a {replaced} capas")
    return model


# ============================================================
# EMA
# ============================================================
class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model):
        """Apply EMA weights to model (for evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================
# EVAL
# ============================================================
@torch.no_grad()
def evaluate(model, processor, val_dataset, device):
    model.eval()
    all_predictions = []
    all_targets = []

    for idx in range(len(val_dataset)):
        pixel_values, target = val_dataset[idx]
        pixel_values = pixel_values.unsqueeze(0).to(device)
        image_info = val_dataset.id_to_img[val_dataset.image_ids[idx]]
        orig_size = torch.tensor([[image_info["height"], image_info["width"]]]).to(device)

        with torch.amp.autocast("cuda", enabled=False):
            outputs = model(pixel_values=pixel_values.float())
        results = processor.post_process_object_detection(
            outputs, target_sizes=orig_size, threshold=0.3
        )[0]

        if len(results["scores"]) > 0:
            keep_indices = []
            for label in results["labels"].unique():
                mask = results["labels"] == label
                nms_keep = torchvision.ops.nms(results["boxes"][mask], results["scores"][mask], 0.5)
                keep_indices.extend(torch.where(mask)[0][nms_keep].tolist())
            keep = torch.tensor(sorted(keep_indices), dtype=torch.long, device=device)
            pred_boxes = results["boxes"][keep].cpu()
            pred_scores = results["scores"][keep].cpu()
            pred_labels = results["labels"][keep].cpu()
        else:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros(0)
            pred_labels = torch.zeros(0, dtype=torch.long)

        all_predictions.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})

        gt_boxes_abs = []
        for box in target["boxes"]:
            cx, cy, w, h = box.tolist()
            img_w, img_h = image_info["width"], image_info["height"]
            gt_boxes_abs.append([(cx - w/2)*img_w, (cy - h/2)*img_h, (cx + w/2)*img_w, (cy + h/2)*img_h])

        all_targets.append({
            "boxes": torch.tensor(gt_boxes_abs) if gt_boxes_abs else torch.zeros((0, 4)),
            "labels": target["class_labels"],
        })

    return compute_map(all_predictions, all_targets)


def compute_map(predictions, targets, iou_threshold=0.5):
    all_classes = set()
    for t in targets:
        all_classes.update(t["labels"].tolist())
    if not all_classes:
        return 0.0
    aps = []
    for cls in sorted(all_classes):
        ap = compute_ap_single_class(predictions, targets, cls, iou_threshold)
        aps.append(ap)
    return sum(aps) / len(aps)


def compute_ap_single_class(predictions, targets, cls, iou_threshold):
    all_scores, all_tp = [], []
    n_gt = 0
    for pred, tgt in zip(predictions, targets):
        gt_mask = tgt["labels"] == cls
        gt_boxes = tgt["boxes"][gt_mask]
        n_gt += len(gt_boxes)
        pred_mask = pred["labels"] == cls
        pred_boxes = pred["boxes"][pred_mask]
        pred_scores_cls = pred["scores"][pred_mask]
        if len(pred_boxes) == 0:
            continue
        order = pred_scores_cls.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores_cls = pred_scores_cls[order]
        matched = set()
        for i in range(len(pred_boxes)):
            all_scores.append(pred_scores_cls[i].item())
            if len(gt_boxes) == 0:
                all_tp.append(0)
                continue
            ious = box_iou(pred_boxes[i].unsqueeze(0), gt_boxes)[0]
            best_iou, best_idx = ious.max(0)
            if best_iou.item() >= iou_threshold and best_idx.item() not in matched:
                all_tp.append(1)
                matched.add(best_idx.item())
            else:
                all_tp.append(0)
    if n_gt == 0:
        return 0.0
    indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    tp_sorted = [all_tp[i] for i in indices]
    tp_sum, fp_sum = 0, 0
    precisions, recalls = [], []
    for tp in tp_sorted:
        tp_sum += tp
        fp_sum += 1 - tp
        precisions.append(tp_sum / (tp_sum + fp_sum))
        recalls.append(tp_sum / n_gt)
    ap = 0.0
    for t in [i / 10.0 for i in range(11)]:
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p_at_r) / 11.0 if p_at_r else 0.0
    return ap


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    return inter / (area1[:, None] + area2 - inter + 1e-6)


# ============================================================
# TRAINING
# ============================================================
STRATEGIES = {
    "E_lora32_cosine_aug": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 1e-4,
        "warmup_epochs": 5,
        "cosine": True,
        "augmentation": True,
        "ema": False,
        "description": "LoRA r=32 + cosine warmup + data augmentation",
    },
    "F_lora32_ema_aug": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 1e-4,
        "warmup_epochs": 0,
        "cosine": False,
        "augmentation": True,
        "ema": True,
        "description": "LoRA r=32 + EMA + data augmentation",
    },
    "G_lora32_all": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lr": 1e-4,
        "warmup_epochs": 5,
        "cosine": True,
        "augmentation": True,
        "ema": True,
        "description": "LoRA r=32 + EMA + cosine warmup + augmentation (FULL)",
    },
}


def get_lr(epoch, config, base_lr):
    warmup = config.get("warmup_epochs", 0)
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    if config.get("cosine"):
        remaining = MAX_EPOCHS - warmup
        progress = (epoch - warmup) / remaining
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr


def train_strategy(strategy_name, config, coco_data, train_ids, val_ids, processor):
    print(f"\n{'='*60}")
    print(f"ESTRATEGIA: {strategy_name}")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(OUTPUT_BASE, strategy_name)
    os.makedirs(output_dir, exist_ok=True)

    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    model.config.num_denoising = 0
    model.config.eos_coefficient = 0.0001

    # Freeze backbone + encoder
    for n, param in model.named_parameters():
        if "backbone" in n or "encoder" in n:
            param.requires_grad = False

    # Apply LoRA
    apply_lora(model, config["lora_rank"], config["lora_alpha"], dropout=0.05)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    model.to(device)

    # EMA
    ema = EMA(model, decay=0.9995) if config.get("ema") else None

    # Augmenter
    augmenter = DocumentAugmenter(strength=1.0) if config.get("augmentation") else None

    # Datasets
    train_ds = DocLayoutDataset(coco_data, train_ids, IMAGES_DIR, processor, augmenter=augmenter)
    val_ds = DocLayoutDataset(coco_data, val_ids, IMAGES_DIR, processor)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"], weight_decay=1e-4
    )

    best_map = -1
    patience_counter = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        # Update LR
        lr = get_lr(epoch, config, config["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, (pixel_values, targets) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels = [{
                "boxes": t["boxes"].to(device),
                "class_labels": t["class_labels"].to(device),
            } for t in targets]

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            epoch_loss += loss.item() * GRAD_ACCUM

        avg_loss = epoch_loss / len(train_loader)

        # Evaluate (with EMA if available)
        if ema:
            ema.apply(model)
        val_map = evaluate(model, processor, val_ds, device)
        if ema:
            ema.restore(model)

        history.append({"epoch": epoch, "loss": avg_loss, "mAP": val_map, "lr": lr})
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS} | loss={avg_loss:.4f} | mAP@0.5={val_map:.4f} | lr={lr:.2e}")

        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            # Save with EMA weights if available
            if ema:
                ema.apply(model)
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            if ema:
                ema.restore(model)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  RESULTADO: best mAP@0.5 = {best_map:.4f}")
    del model, optimizer
    if ema:
        del ema
    gc.collect()
    torch.cuda.empty_cache()
    return best_map, history


def main():
    print("Cargando datos...")
    with open(COCO_FILE) as f:
        coco_data = json.load(f)

    # Use same split as round 1
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    train_ids = split["train"]
    val_ids = split["val"]
    print(f"  Train: {len(train_ids)} | Val: {len(val_ids)} (mismo split que round 1)")

    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)

    results = {}
    # Load round 1 best for comparison
    r1_results_file = os.path.join(OUTPUT_BASE, "results.json")
    if os.path.exists(r1_results_file):
        with open(r1_results_file) as f:
            r1 = json.load(f)
        results["B_lora_r8 (round1)"] = r1.get("B_lora", "N/A")

    for name, config in STRATEGIES.items():
        try:
            best_map, history = train_strategy(name, config, coco_data, train_ids, val_ids, processor)
            results[name] = best_map
        except Exception as e:
            print(f"\n  ERROR en {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"FAILED: {e}"
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("RESULTADOS ROUND 2")
    print(f"{'='*60}")
    print(f"{'Estrategia':<35} {'mAP@0.5':>10}")
    print("-" * 47)
    for name, score in results.items():
        if isinstance(score, float):
            print(f"{name:<35} {score:>10.4f}")
        else:
            print(f"{name:<35} {str(score):>10}")

    with open(os.path.join(OUTPUT_BASE, "results_round2.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate PDF for best model
    best_name = max(
        [(k, v) for k, v in results.items() if isinstance(v, float)],
        key=lambda x: x[1]
    )[0]
    print(f"\nMejor modelo: {best_name}")
    print(f"Resultados guardados en: {OUTPUT_BASE}/results_round2.json")


if __name__ == "__main__":
    main()
