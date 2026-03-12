"""
Dataset classes and transforms for skin cancer detection pipeline.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─── Transforms ──────────────────────────────────────────────────────────────

SEG_SIZE = 256
CLS_SIZE = 224

transform_seg = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((SEG_SIZE, SEG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_cls_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLS_SIZE, CLS_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_cls_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLS_SIZE, CLS_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Segmentation Dataset ────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    Expects image_dir with ISIC-style images and mask_dir with binary .png masks.
    Mask filename must match image filename (with .png extension).
    """
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        # Resize
        img = cv2.resize(img, (SEG_SIZE, SEG_SIZE))
        mask = cv2.resize(mask, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST)

        img_t = transforms.ToTensor()(img)
        img_t = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(img_t)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img_t, mask_t

    def _augment(self, img, mask):
        h, w = img.shape[:2]
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        angle = np.random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
        return img, mask


# ─── Classification Dataset ──────────────────────────────────────────────────

class SkinClassifierDataset(Dataset):
    """
    Reads from a CSV with columns: image_id, label (benign/malignant).
    image_dir should contain <image_id>.jpg files.
    """
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.image_dir, row["image"])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return img, label


def load_labels_csv(csv_path):
    """Load and clean the labels CSV."""
    df = pd.read_csv(csv_path, header=None)
    df.columns = ["image_id", "label"]
    df["label"] = df["label"].str.strip().str.lower().map({"benign": 0, "malignant": 1})
    df["image"] = df["image_id"] + ".jpg"
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def filter_existing(df, image_dir):
    """Keep only rows where the image file actually exists."""
    mask = df["image"].apply(lambda f: os.path.exists(os.path.join(image_dir, f)))
    return df[mask].reset_index(drop=True)
