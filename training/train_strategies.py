"""
Fine-tuning de Docling Heron con 4 estrategias.
Evalúa cada una y reporta mAP al final.

Estrategias:
  A: Frozen backbone+encoder (solo decoder)
  B: LoRA en decoder (parameter-efficient)
  C: Progressive unfreezing
  D: Full fine-tune con LR muy bajo

GPU: GTX 1650 4GB -> batch_size=1, fp16, gradient accumulation
"""
import os
import sys
import gc
import json
import math
import random
import copy
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "docling-project/docling-layout-heron"
COCO_FILE = "/home/alonso/cvat_docling/training/EAF-477-2025_ground_truth.json"
IMAGES_DIR = "/home/alonso/prueba_cvat/inputs/imagenes/EAF-477-2025"
OUTPUT_BASE = "/home/alonso/cvat_docling/training/models"
VAL_SPLIT = 0.15
SEED = 42
MAX_EPOCHS = 25
PATIENCE = 7
GRAD_ACCUM = 4  # effective batch = 4
GRAD_CLIP = 0.1

STRATEGIES = {
    "A_frozen_decoder": {
        "freeze": ["backbone", "encoder"],
        "lr": 5e-6,
        "description": "Frozen backbone+encoder, solo decoder",
    },
    "B_lora": {
        "freeze": ["backbone", "encoder"],
        "lora": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lr": 1e-4,
        "description": "LoRA rank=8 en decoder attention",
    },
    "C_progressive": {
        "freeze": ["backbone", "encoder"],
        "progressive_unfreeze": True,
        "unfreeze_epoch_encoder": 8,
        "unfreeze_epoch_backbone": 15,
        "lr": 5e-6,
        "lr_backbone": 1e-6,
        "description": "Progressive unfreezing",
    },
    "D_full_finetune": {
        "freeze": [],
        "lr": 2e-6,
        "description": "Full fine-tune con LR bajo",
    },
}

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================
class DocLayoutDataset(Dataset):
    def __init__(self, coco_data, image_ids, images_dir, processor):
        self.images_dir = images_dir
        self.processor = processor
        self.image_ids = image_ids

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
            # COCO [x,y,w,h] -> [cx, cy, w, h] normalized
            img_w, img_h = img_info["width"], img_info["height"]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            boxes.append([cx, cy, nw, nh])
            labels.append(ann["category_id"] - 1)  # 0-indexed for model

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

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling


def apply_lora(model, rank=8, alpha=16):
    """Apply LoRA to decoder self-attention and cross-attention layers."""
    replaced = 0
    for name, module in model.named_modules():
        if "decoder" in name and isinstance(module, torch.nn.Linear):
            if any(k in name for k in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                # Get parent module and attribute name
                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    if p.isdigit():
                        parent = parent[int(p)]
                    else:
                        parent = getattr(parent, p)
                attr = parts[-1]
                setattr(parent, attr, LoRALinear(module, rank, alpha))
                replaced += 1
    print(f"  LoRA aplicado a {replaced} capas lineales del decoder")
    return model


# ============================================================
# EVAL (mAP simplificado)
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

        # Always eval in fp32 to avoid NaN
        with torch.amp.autocast("cuda", enabled=False):
            outputs = model(pixel_values=pixel_values.float())
        results = processor.post_process_object_detection(
            outputs, target_sizes=orig_size, threshold=0.3
        )[0]

        # Apply NMS
        if len(results["scores"]) > 0:
            keep_indices = []
            for label in results["labels"].unique():
                mask = results["labels"] == label
                nms_keep = torchvision.ops.nms(
                    results["boxes"][mask], results["scores"][mask], 0.5
                )
                keep_indices.extend(torch.where(mask)[0][nms_keep].tolist())
            keep = torch.tensor(sorted(keep_indices), dtype=torch.long, device=device)
            pred_boxes = results["boxes"][keep].cpu()
            pred_scores = results["scores"][keep].cpu()
            pred_labels = results["labels"][keep].cpu()
        else:
            pred_boxes = torch.zeros((0, 4))
            pred_scores = torch.zeros(0)
            pred_labels = torch.zeros(0, dtype=torch.long)

        all_predictions.append({
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels,
        })

        # Convert target boxes back to absolute coords
        gt_boxes_abs = []
        for box, label in zip(target["boxes"], target["class_labels"]):
            cx, cy, w, h = box.tolist()
            img_w, img_h = image_info["width"], image_info["height"]
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            gt_boxes_abs.append([x1, y1, x2, y2])

        all_targets.append({
            "boxes": torch.tensor(gt_boxes_abs) if gt_boxes_abs else torch.zeros((0, 4)),
            "labels": target["class_labels"],
        })

    # Calculate mAP@0.5
    return compute_map(all_predictions, all_targets, iou_threshold=0.5)


def compute_map(predictions, targets, iou_threshold=0.5):
    """Compute mAP@IoU for all classes."""
    all_classes = set()
    for t in targets:
        all_classes.update(t["labels"].tolist())
    for p in predictions:
        all_classes.update(p["labels"].tolist())

    if not all_classes:
        return 0.0

    aps = []
    for cls in sorted(all_classes):
        ap = compute_ap_single_class(predictions, targets, cls, iou_threshold)
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0


def compute_ap_single_class(predictions, targets, cls, iou_threshold):
    all_scores = []
    all_tp = []
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

        # Sort by score descending
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

    # Sort by score
    indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    tp_sorted = [all_tp[i] for i in indices]

    tp_cumsum = []
    fp_cumsum = []
    tp_sum = 0
    fp_sum = 0
    for tp in tp_sorted:
        tp_sum += tp
        fp_sum += 1 - tp
        tp_cumsum.append(tp_sum)
        fp_cumsum.append(fp_sum)

    precisions = [tp / (tp + fp) for tp, fp in zip(tp_cumsum, fp_cumsum)]
    recalls = [tp / n_gt for tp in tp_cumsum]

    # AP via 11-point interpolation
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
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-6)


# ============================================================
# TRAINING
# ============================================================
def train_strategy(strategy_name, config, coco_data, train_ids, val_ids, processor):
    print(f"\n{'='*60}")
    print(f"ESTRATEGIA: {strategy_name}")
    print(f"  {config['description']}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(OUTPUT_BASE, strategy_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load model fresh
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    model.config.num_denoising = 0  # Disable for small dataset
    model.config.eos_coefficient = 0.0001

    # Apply LoRA if needed
    if config.get("lora"):
        apply_lora(model, config["lora_rank"], config["lora_alpha"])

    # Freeze layers
    for freeze_part in config.get("freeze", []):
        for n, param in model.named_parameters():
            if freeze_part in n:
                param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    model.to(device)

    # Datasets
    train_ds = DocLayoutDataset(coco_data, train_ids, IMAGES_DIR, processor)
    val_ds = DocLayoutDataset(coco_data, val_ids, IMAGES_DIR, processor)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Optimizer with differential LR for progressive unfreezing
    if config.get("progressive_unfreeze"):
        param_groups = [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and "backbone" not in n and "encoder" not in n],
             "lr": config["lr"]},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config["lr"]}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Try fp16 only for full finetune (needs it for memory), otherwise fp32
    use_amp = "D_full" in strategy_name
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_map = -1
    patience_counter = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        # Progressive unfreezing
        if config.get("progressive_unfreeze"):
            if epoch == config.get("unfreeze_epoch_encoder", 999):
                print(f"  [Epoch {epoch}] Unfreezing encoder")
                for n, param in model.named_parameters():
                    if "encoder" in n:
                        param.requires_grad = True
                optimizer.add_param_group({
                    "params": [p for n, p in model.named_parameters()
                               if "encoder" in n and p.requires_grad],
                    "lr": config["lr"] * 0.5
                })

            if epoch == config.get("unfreeze_epoch_backbone", 999):
                print(f"  [Epoch {epoch}] Unfreezing backbone")
                for n, param in model.named_parameters():
                    if "backbone" in n:
                        param.requires_grad = True
                optimizer.add_param_group({
                    "params": [p for n, p in model.named_parameters()
                               if "backbone" in n and p.requires_grad],
                    "lr": config.get("lr_backbone", 1e-6)
                })

        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, (pixel_values, targets) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels = [{
                "boxes": t["boxes"].to(device),
                "class_labels": t["class_labels"].to(device),
            } for t in targets]

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM

        avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        val_map = evaluate(model, processor, val_ds, device)

        history.append({"epoch": epoch, "loss": avg_loss, "mAP": val_map})
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS} | loss={avg_loss:.4f} | mAP@0.5={val_map:.4f}")

        # Early stopping
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # Save history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  RESULTADO: best mAP@0.5 = {best_map:.4f}")
    print(f"  Modelo guardado en: {output_dir}/best_model.pt")

    # Cleanup
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return best_map, history


# ============================================================
# MAIN
# ============================================================
def main():
    print("Cargando datos...")
    with open(COCO_FILE) as f:
        coco_data = json.load(f)

    all_image_ids = [img["id"] for img in coco_data["images"]]
    random.shuffle(all_image_ids)
    split = int(len(all_image_ids) * (1 - VAL_SPLIT))
    train_ids = sorted(all_image_ids[:split])
    val_ids = sorted(all_image_ids[split:])
    print(f"  Train: {len(train_ids)} imágenes | Val: {len(val_ids)} imágenes")

    # Save split for reproducibility
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    with open(os.path.join(OUTPUT_BASE, "data_split.json"), "w") as f:
        json.dump({"train": train_ids, "val": val_ids}, f)

    processor = RTDetrImageProcessor.from_pretrained(MODEL_NAME)

    # Baseline: evaluate original model
    print("\n" + "="*60)
    print("BASELINE: Modelo original sin fine-tuning")
    print("="*60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_NAME)
    base_model.to(device).eval()
    val_ds_baseline = DocLayoutDataset(coco_data, val_ids, IMAGES_DIR, processor)
    baseline_map = evaluate(base_model, processor, val_ds_baseline, device)
    print(f"  Baseline mAP@0.5 = {baseline_map:.4f}")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # Train each strategy
    results = {"baseline": baseline_map}

    for name, config in STRATEGIES.items():
        try:
            best_map, history = train_strategy(name, config, coco_data, train_ids, val_ids, processor)
            results[name] = best_map
        except Exception as e:
            print(f"\n  ERROR en {name}: {e}")
            results[name] = f"FAILED: {e}"
            gc.collect()
            torch.cuda.empty_cache()

    # Final comparison
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"{'Estrategia':<30} {'mAP@0.5':>10}")
    print("-" * 42)
    for name, score in results.items():
        if isinstance(score, float):
            delta = ""
            if name != "baseline":
                diff = score - results["baseline"]
                delta = f" ({'+' if diff > 0 else ''}{diff:.4f})"
            print(f"{name:<30} {score:>10.4f}{delta}")
        else:
            print(f"{name:<30} {str(score):>10}")

    # Save results
    with open(os.path.join(OUTPUT_BASE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResultados guardados en: {OUTPUT_BASE}/results.json")


if __name__ == "__main__":
    main()
