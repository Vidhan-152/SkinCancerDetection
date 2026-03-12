"""
train.py  —  Refined training pipeline for skin cancer detection
Run in Google Colab after uploading this project folder to Drive.

Usage:
    python train.py --mode seg      # train segmentation only
    python train.py --mode cls      # train classification only
    python train.py --mode both     # train both sequentially
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# ── Adjust path so we can import from project root ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import UNet, SkinClassifier
from utils import (
    load_labels_csv, filter_existing,
    SkinClassifierDataset, SegmentationDataset,
    transform_cls_train, transform_cls_val,
    SegTrainer, ClsTrainer,
)

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, "data", "images")
CROP_DIR   = os.path.join(BASE_DIR, "data", "cropped")
MASK_DIR   = os.path.join(BASE_DIR, "data", "masks")
LABELS_CSV = os.path.join(BASE_DIR, "data", "labels.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

SEG_EPOCHS   = 30
CLS_EPOCHS   = 20
BATCH_SIZE   = 8
LR_SEG       = 1e-4
LR_CLS       = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Segmentation Training ───────────────────────────────────────────────────

def train_segmentation():
    print("\n" + "="*50)
    print("   SEGMENTATION  (U-Net + Attention Gates)")
    print("="*50)

    # Collect image/mask pairs
    image_paths, mask_paths = [], []
    for fname in os.listdir(IMAGE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(MASK_DIR, stem + "_segmentation.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(MASK_DIR, stem + ".png")
        if os.path.exists(mask_path):
            image_paths.append(os.path.join(IMAGE_DIR, fname))
            mask_paths.append(mask_path)

    print(f"Found {len(image_paths)} image/mask pairs")
    if len(image_paths) == 0:
        print("No mask files found in", MASK_DIR)
        print("Skipping segmentation training.")
        return

    idx = list(range(len(image_paths)))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=42)

    train_ds = SegmentationDataset(
        [image_paths[i] for i in train_idx],
        [mask_paths[i] for i in train_idx],
        augment=True
    )
    val_ds = SegmentationDataset(
        [image_paths[i] for i in val_idx],
        [mask_paths[i] for i in val_idx],
        augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=2)

    model = UNet(use_attention=True, dropout=0.1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR_SEG, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5, factor=0.5)

    trainer = SegTrainer(model, optimizer, scheduler, DEVICE)

    best_dice = 0.0
    for epoch in range(1, SEG_EPOCHS + 1):
        tr_loss, tr_dice = trainer.train_epoch(train_loader)
        vl_loss, vl_dice, vl_iou = trainer.val_epoch(val_loader)
        trainer.step_scheduler(metric=vl_dice)

        print(f"Epoch {epoch:02d}/{SEG_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_dice={tr_dice:.4f}  "
              f"val_loss={vl_loss:.4f}  val_dice={vl_dice:.4f}  val_iou={vl_iou:.4f}")

        if vl_dice > best_dice:
            best_dice = vl_dice
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet_best.pth"))
            print(f"  ✓ New best Dice={best_dice:.4f}  → saved unet_best.pth")

    print(f"\nSegmentation training done. Best Dice: {best_dice:.4f}")


# ─── Classification Training ─────────────────────────────────────────────────

def train_classification():
    print("\n" + "="*50)
    print("   CLASSIFICATION  (MobileNetV2 fine-tune)")
    print("="*50)

    df = load_labels_csv(LABELS_CSV)
    df = filter_existing(df, CROP_DIR)
    print(f"Samples after filtering: {len(df)}")
    print(df["label"].value_counts().to_string())

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    train_ds = SkinClassifierDataset(train_df, CROP_DIR, transform=transform_cls_train)
    val_ds   = SkinClassifierDataset(val_df,   CROP_DIR, transform=transform_cls_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=2)

    # Compute class imbalance weight
    n_benign    = (df["label"] == 0).sum()
    n_malignant = (df["label"] == 1).sum()
    pos_weight  = n_benign / max(n_malignant, 1)
    print(f"pos_weight (benign/malignant): {pos_weight:.2f}")

    model = SkinClassifier(dropout=0.3, pretrained=True).to(DEVICE)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_CLS)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLS_EPOCHS)
    trainer = ClsTrainer(model, optimizer, scheduler, DEVICE, pos_weight=pos_weight)

    best_acc = 0.0
    unfreeze_done = False

    for epoch in range(1, CLS_EPOCHS + 1):
        # Gradually unfreeze backbone at epoch 5
        if epoch == 5 and not unfreeze_done:
            model.unfreeze_top_n(n=3)
            for g in optimizer.param_groups:
                g["lr"] = LR_CLS / 10
            print("  → Unfroze top-3 backbone blocks (lr reduced to 1e-4)")
            unfreeze_done = True

        tr_loss, tr_acc = trainer.train_epoch(train_loader)
        vl_loss, vl_acc = trainer.val_epoch(val_loader)
        trainer.step_scheduler(metric=vl_acc)

        print(f"Epoch {epoch:02d}/{CLS_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "mobilenet_best.pth"))
            print(f"  ✓ New best Acc={best_acc:.4f}  → saved mobilenet_best.pth")

    print(f"\nClassification training done. Best Acc: {best_acc:.4f}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seg", "cls", "both"], default="both")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if args.mode in ("seg", "both"):
        train_segmentation()
    if args.mode in ("cls", "both"):
        train_classification()
