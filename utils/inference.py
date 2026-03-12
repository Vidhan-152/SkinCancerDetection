"""
Inference pipeline:
1. Segment the lesion with U-Net → binary mask
2. Crop the lesion using the mask bounding box
3. Classify the crop with MobileNetV2 → benign / malignant
4. Build overlay visualization
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


SEG_SIZE = 256
CLS_SIZE = 224

_seg_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((SEG_SIZE, SEG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_cls_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CLS_SIZE, CLS_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

LABEL_MAP = {0: "Benign", 1: "Malignant"}
COLOR_MAP = {
    "Benign": (0, 220, 120),       # green
    "Malignant": (255, 60, 60),    # red
}


@torch.no_grad()
def segment(image_rgb: np.ndarray, model_seg, device) -> np.ndarray:
    """
    Run U-Net segmentation.
    Returns: probability mask (H, W) float32 in [0, 1], resized to original dims.
    """
    h, w = image_rgb.shape[:2]
    x = _seg_transform(image_rgb).unsqueeze(0).to(device)
    model_seg.eval()
    logit = model_seg(x)
    prob = torch.sigmoid(logit)[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h))


def binarize_mask(prob_mask: np.ndarray, threshold: float = None) -> np.ndarray:
    """
    Convert probability mask to binary mask.
    Uses Otsu thresholding if threshold=None.
    """
    uint8 = (prob_mask * 255).astype(np.uint8)
    if threshold is None:
        _, binary = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary  # values 0 or 255


def crop_lesion(image_rgb: np.ndarray, binary_mask: np.ndarray, padding: int = 10):
    """
    Crop the bounding box of the lesion with optional padding.
    Returns: (crop, bbox) or (full_image, None) if no lesion found.
    """
    coords = np.where(binary_mask > 0)
    if len(coords[0]) < 100:
        return image_rgb, None
    h, w = image_rgb.shape[:2]
    y_min = max(0, coords[0].min() - padding)
    y_max = min(h, coords[0].max() + padding)
    x_min = max(0, coords[1].min() - padding)
    x_max = min(w, coords[1].max() + padding)
    bbox = (x_min, y_min, x_max, y_max)
    return image_rgb[y_min:y_max, x_min:x_max], bbox


@torch.no_grad()
def classify(crop_rgb: np.ndarray, model_cls, device) -> dict:
    """
    Classify a cropped lesion image.
    Returns: {"label": str, "confidence": float, "prob_malignant": float}
    """
    x = _cls_transform(crop_rgb).unsqueeze(0).to(device)
    model_cls.eval()
    logit = model_cls(x)
    prob_malignant = torch.sigmoid(logit).item()
    label = LABEL_MAP[int(prob_malignant > 0.5)]
    confidence = prob_malignant if label == "Malignant" else (1 - prob_malignant)
    return {
        "label": label,
        "confidence": confidence,
        "prob_malignant": prob_malignant,
        "prob_benign": 1 - prob_malignant,
    }


def build_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray,
                  label: str, bbox=None, alpha: float = 0.35) -> np.ndarray:
    """
    Create a colored overlay visualization showing:
    - Semi-transparent colored mask over lesion
    - Bounding box rectangle
    - Label text
    """
    overlay = image_rgb.copy()
    color = COLOR_MAP.get(label, (255, 255, 0))

    # Fill mask region
    mask_bool = binary_mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)

    # Draw mask contour
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    # Draw bounding box
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label badge
        label_text = label.upper()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        pad = 6
        bx1, by1 = x1, max(0, y1 - th - 2 * pad)
        bx2, by2 = x1 + tw + 2 * pad, y1
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(overlay, label_text, (bx1 + pad, by2 - pad),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay


def full_pipeline(image_rgb: np.ndarray, model_seg, model_cls, device,
                  seg_threshold: float = None) -> dict:
    """
    End-to-end inference.
    Returns dict with all artifacts for display.
    """
    # Step 1: Segmentation
    prob_mask = segment(image_rgb, model_seg, device)
    binary_mask = binarize_mask(prob_mask, threshold=seg_threshold)

    # Step 2: Crop
    crop, bbox = crop_lesion(image_rgb, binary_mask)

    # Step 3: Classification
    result = classify(crop, model_cls, device)

    # Step 4: Overlay
    overlay = build_overlay(image_rgb, binary_mask, result["label"], bbox)

    return {
        "original": image_rgb,
        "prob_mask": prob_mask,
        "binary_mask": binary_mask,
        "crop": crop,
        "overlay": overlay,
        "bbox": bbox,
        **result,
    }
