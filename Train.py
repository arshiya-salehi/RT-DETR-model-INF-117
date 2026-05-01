"""
Train.py
========
RT-DETR training script for the INF-117 dental endodontic detection project.

RT-DETR (Real-Time Detection Transformer) is a transformer-based detection model
that replaces the two-stage Mask R-CNN pipeline with an end-to-end transformer
decoder. It works natively with COCO-format annotations (bounding boxes only —
no segmentation masks required).

Dataset:  327 train images | 109 val images | 11 classes
Model:    RT-DETR-R50 (rtdetr_r50vd) with COCO pretrained weights
Hardware: NVIDIA A100 GPU (GPU index 1 on shared server)

Usage:
    conda activate inf117_rtdetr
    CUDA_VISIBLE_DEVICES=1 python Train.py

    # or as a background job (recommended on shared server):
    CUDA_VISIBLE_DEVICES=1 nohup python Train.py > logs/train.log 2>&1 &

Key differences from Mask R-CNN:
  - Detection only (bounding boxes), no instance segmentation masks
  - Uses Hugging Face transformers library instead of Detectron2
  - Much faster inference; competitive accuracy on small datasets
  - Requires COCO-format JSON (same format already produced by convert_to_coco.py)
"""

import os
import sys
import json
import logging
import math
from datetime import datetime
from pathlib import Path

# ── GPU LOCK ──────────────────────────────────────────────────────────────────
# Lock to GPU 1 on the shared server. Must come before any torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
)
from pycocotools.coco import COCO

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
ANN_DIR     = DATASET_DIR / "annotations"
IMG_DIR     = DATASET_DIR / "images"
OUTPUT_DIR  = BASE_DIR / "output"
LOG_DIR     = BASE_DIR / "logs"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"

for d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── CLASSES (must match data.yaml / convert_to_coco.py order exactly) ─────────
CLASS_NAMES = [
    "Apical Lesion",       # 0
    "Main Root",           # 1
    "Main Canal",          # 2
    "Mesial Root",         # 3
    "Mesial Canal",        # 4
    "Distal Root",         # 5
    "Distal Canal",        # 6
    "Palatal Canal",       # 7  — rare (only 26 training instances)
    "Palatal Root",        # 8
    "Root Canal Filling",  # 9
    "decay",               # 10
]
NUM_CLASSES = len(CLASS_NAMES)  # 11

# ── HYPERPARAMETERS ───────────────────────────────────────────────────────────
# Tuned for 327 training images on an A100 with RT-DETR-L
BATCH_SIZE   = 16       # 48GB L40S can handle batch 16 at 800px
NUM_EPOCHS   = 500      # Higher epochs for Transformer convergence
BASE_LR      = 2e-4     # Slightly higher LR for larger batch size
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.1     # gradient clipping (standard for DETR-family)
IMG_SIZE     = 1024      # Higher resolution for small lesions/decay
NUM_WORKERS  = 32       # Maximize 128-core AMD EPYC CPU
SCORE_THRESH = 0.4      # inference confidence threshold
SAVE_EVERY   = 10       # save a checkpoint every N epochs

# Pretrained model — RT-DETR-R50 is the best accuracy/speed tradeoff
PRETRAINED_MODEL = "PekingU/rtdetr_r50vd"


# ── LOGGING ───────────────────────────────────────────────────────────────────
def setup_logging():
    log_path = LOG_DIR / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ── DATASET ───────────────────────────────────────────────────────────────────
class DentalCocoDataset(torch.utils.data.Dataset):
    """
    COCO-format dataset reader for RT-DETR.

    Returns:
        pixel_values  — (3, H, W) float32 tensor, normalized by processor
        labels        — dict with 'class_labels' (N,) and 'boxes' (N, 4) in
                        [cx, cy, w, h] normalized format (required by RT-DETR)
        image_id      — int, for COCO evaluation
    """

    def __init__(self, ann_path: Path, img_dir: Path, processor, augment: bool = False):
        self.coco      = COCO(str(ann_path))
        self.img_dir   = img_dir
        self.processor = processor
        self.augment   = augment
        self.img_ids   = sorted(self.coco.imgs.keys())

        # Build id2name from categories stored in the JSON
        self.id2label = {cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())}
        self.tv_tensors=tv_tensors
        
        # Augmentation pipeline (train only)
        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            T.SanitizeBoundingBoxes(), # Safely drops boxes if they are translated out of bounds
        ]) if augment else None

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.img_dir / img_info["file_name"]

        image = torchvision.io.read_image(str(img_path))          # (3, H, W) uint8
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)                          # grayscale → RGB

        W_orig = img_info["width"]
        H_orig = img_info["height"]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        # Convert COCO [x_min, y_min, w, h] → [cx, cy, w, h] normalised
        boxes_xyxy = []
        labels = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes_xyxy.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        # 1. Apply augmentations safely to BOTH image and boxes
        if self.aug is not None:
            image_tv = self.tv_tensors.Image(image)
            boxes_tv = self.tv_tensors.BoundingBoxes(
                boxes_tensor, format="XYXY", canvas_size=(H_orig, W_orig)
            )
            # The transforms will now properly move the boxes when the image flips/shifts
            # SanitizeBoundingBoxes requires a dict to properly drop labels if boxes are removed
            out = self.aug({"image": image_tv, "boxes": boxes_tv, "labels": labels_tensor})
            image = out["image"]
            boxes_tv = out["boxes"]
            labels_tensor = out["labels"]
            boxes_tensor = boxes_tv.as_subclass(torch.Tensor)
        # Update H and W in case spatial transforms changed them
        _, H_new, W_new = image.shape
        
        # 2. Convert from XYXY pixel coords to Normalized CXCYWH (required by RT-DETR)
        final_boxes = []
        for box in boxes_tensor:
            x1, y1, x2, y2 = box.tolist()
            bw, bh = (x2 - x1), (y2 - y1)
            cx = (x1 + bw / 2) / W_new
            cy = (y1 + bh / 2) / H_new
            nw = bw / W_new
            nh = bh / H_new
            
            # Clamp to [0, 1]
            cx, cy, nw, nh = (
                max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)),
                max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))
            )
            final_boxes.append([cx, cy, nw, nh])
        # 3. Processor handles resize + normalize → (3, IMG_SIZE, IMG_SIZE) float
        encoding = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": IMG_SIZE, "width": IMG_SIZE},
        )
        pixel_values = encoding["pixel_values"].squeeze(0)  # (3, H, W)
        target = {
            "class_labels": labels_tensor,
            "boxes":        torch.tensor(final_boxes, dtype=torch.float32)
                            if final_boxes else torch.zeros((0, 4), dtype=torch.float32),
        }
        return pixel_values, target, img_id


def collate_fn(batch):
    """Custom collate: stack pixel_values, keep targets as list."""
    pixel_values = torch.stack([item[0] for item in batch])
    targets      = [item[1] for item in batch]
    img_ids      = [item[2] for item in batch]
    return pixel_values, targets, img_ids


# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, logger):
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (pixel_values, targets, _) in enumerate(loader):
        pixel_values = pixel_values.to(device)

        # RT-DETR expects labels as a list of dicts, each on device
        labels = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch:3d} | Batch {batch_idx+1:3d}/{n_batches} "
                f"| Loss {loss.item():.4f} | LR {lr_now:.2e}"
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, processor, device, logger):
    """
    Quick validation: compute average loss and a rough mAP estimate using
    torchvision's detection mean_ap utility.
    """
    model.eval()
    total_loss = 0.0

    for pixel_values, targets, _ in loader:
        pixel_values = pixel_values.to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    logger.info(f"  Validation loss: {avg_loss:.4f}")
    return avg_loss


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("INF-117 Dental Detection — RT-DETR Training")
    logger.info("=" * 60)

    # GPU check
    if not torch.cuda.is_available():
        logger.error("No GPU found. Check CUDA_VISIBLE_DEVICES and driver.")
        sys.exit(1)
    device   = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Image processor (handles resize + pixel normalization)
    processor = RTDetrImageProcessor.from_pretrained(PRETRAINED_MODEL)

    # Datasets
    train_ds = DentalCocoDataset(
        ANN_DIR / "train.json", IMG_DIR / "train", processor, augment=True
    )
    val_ds = DentalCocoDataset(
        ANN_DIR / "val.json", IMG_DIR / "val", processor, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )

    logger.info(f"Train images: {len(train_ds)}")
    logger.info(f"Val images:   {len(val_ds)}")
    logger.info(f"Classes:      {NUM_CLASSES} — {CLASS_NAMES}")
    logger.info(f"Epochs:       {NUM_EPOCHS}  |  Batch size: {BATCH_SIZE}")
    logger.info(f"Base LR:      {BASE_LR}     |  Model: {PRETRAINED_MODEL}")
    logger.info("-" * 60)

    # Model — RT-DETR-L with COCO pretrained weights, adapted to NUM_CLASSES
    model = RTDetrForObjectDetection.from_pretrained(
        PRETRAINED_MODEL,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,   # replaces classification head for new class count
    )
    model.to(device)

    # Optimizer — lower LR for backbone, higher for the new detection head
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n],
         "lr": BASE_LR * 0.1},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": BASE_LR},
    ]
    optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    total_steps = NUM_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps, 
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler()

    # Training loop
    best_val_loss = float("inf")
    history       = []

    for epoch in range(1, NUM_EPOCHS + 1):
        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch, logger
        )
        val_loss = validate(model, val_loader, processor, device, logger)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path     = OUTPUT_DIR / "model_best.pth"
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ✓ New best model saved → {best_path}")

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"model_epoch_{epoch:04d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  Checkpoint saved → {ckpt_path}")

    # Save final model
    final_path = OUTPUT_DIR / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\nFinal model saved → {final_path}")

    # Save training history
    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved → {history_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
