# 🔬 DermaScan AI — Skin Cancer Detection

A complete skin lesion **segmentation + classification** pipeline using:
- **U-Net** (with Attention Gates) for lesion masking
- **MobileNetV2** for binary classification (Benign / Malignant)
- **Streamlit** for the interactive web UI

---

## 📁 Project Structure

```
skin_cancer_project/
├── app.py                    # Streamlit web application
├── train.py                  # Training script (run in Colab)
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── unet.py               # Refined U-Net with attention gates
│   └── classifier.py         # MobileNetV2 classifier head
└── utils/
    ├── __init__.py
    ├── datasets.py            # Dataset classes & transforms
    ├── training.py            # Trainers, losses (Dice + BCE), metrics
    └── inference.py           # End-to-end inference pipeline
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app.py
```

Set the model paths in the sidebar to point to your trained `.pth` files.

---

## 🏋️ Training (Google Colab)

### Expected directory layout on Google Drive:
```
/content/drive/MyDrive/SkinCancerProject/
├── data/
│   ├── images/          ← raw ISIC dermoscopy images (.jpg)
│   ├── masks/           ← binary segmentation masks (.png)
│   │                      naming: <image_id>_segmentation.png
│   ├── cropped/         ← lesion crops for classification
│   └── labels.csv       ← two-column CSV: image_id, label (benign/malignant)
└── models/
    ├── unet_best.pth     ← saved by train.py
    └── mobilenet_best.pth
```

### Run training in Colab:
```python
# Mount drive first, then:
!python train.py --mode both    # train both models
!python train.py --mode seg     # segmentation only
!python train.py --mode cls     # classification only
```

---


---

## 📊 Inference Pipeline

```
Input Image
    │
    ▼
[U-Net]  →  Probability mask (0–1)
    │
    ▼
Otsu Threshold + Morphological cleanup  →  Binary mask
    │
    ├── Overlay on original image  →  Display
    │
    ▼
Crop lesion (bounding box + padding)
    │
    ▼
[MobileNetV2]  →  P(Malignant)
    │
    ▼
Label: Benign / Malignant + Confidence
```

---

## ⚕️ Disclaimer

This tool is for research and educational purposes only. It does not constitute medical advice. Always consult a qualified dermatologist.
