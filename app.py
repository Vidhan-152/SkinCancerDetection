"""
app.py — Skin Cancer Detection Streamlit Application
Run: streamlit run app.py
"""

import os
import sys
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from download_models import download_models
download_models()

# ── Adjust path ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import UNet, SkinClassifier
from utils.inference import full_pipeline, COLOR_MAP

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0d14;
    --bg-card: #111520;
    --bg-elevated: #161b2e;
    --accent-cyan: #00d4ff;
    --accent-green: #00e676;
    --accent-red: #ff4444;
    --accent-amber: #ffaa00;
    --text-primary: #e8eaf2;
    --text-muted: #6b7394;
    --border: #1e2540;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

.main-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.main-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent-cyan), #7b5ea7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.main-header p {
    color: var(--text-muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

.result-card {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.diagnosis-badge {
    display: inline-block;
    padding: 0.5rem 1.4rem;
    border-radius: 50px;
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0.5rem 0;
}

.badge-benign {
    background: rgba(0, 230, 118, 0.15);
    color: #00e676;
    border: 1.5px solid rgba(0, 230, 118, 0.4);
}

.badge-malignant {
    background: rgba(255, 68, 68, 0.15);
    color: #ff4444;
    border: 1.5px solid rgba(255, 68, 68, 0.4);
}

.metric-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.metric-box .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}

.metric-box .value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.warning-box {
    background: rgba(255, 170, 0, 0.08);
    border: 1px solid rgba(255, 170, 0, 0.3);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--accent-amber);
    margin-top: 1rem;
}

.info-box {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--accent-cyan);
}

.step-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}

.sidebar-section {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.8rem 0;
}

/* Progress bar override */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent-cyan), #7b5ea7) !important;
}

/* Upload area */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-elevated) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 16px !important;
    transition: border-color 0.2s;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent-cyan) !important;
}

img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models(seg_path: str, cls_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, cls_model = None, None

    if seg_path and os.path.exists(seg_path):
        seg_model = UNet(use_attention=False).to(device)
        seg_model.load_state_dict(torch.load(seg_path, map_location=device))
        seg_model.eval()

    if cls_path and os.path.exists(cls_path):
        cls_model = SkinClassifier(pretrained=False).to(device)
        cls_model.load_state_dict(torch.load(cls_path, map_location=device))
        cls_model.eval()

    return seg_model, cls_model, device


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 DermaScan AI")
    st.markdown("---")

    st.markdown("### Model Paths")
    seg_path = st.text_input(
        "U-Net weights (.pth)",
        value="models/unet_trained.pth",
        help="Path to trained U-Net segmentation model"
    )
    cls_path = st.text_input(
        "Classifier weights (.pth)",
        value="models/mobilenet_balanced_2k.pth",
        help="Path to trained MobileNetV2 classifier"
    )

    st.markdown("### Settings")
    threshold = st.slider(
        "Segmentation threshold",
        min_value=0.0, max_value=1.0,
        value=0.0, step=0.05,
        help="0.0 = auto (Otsu). Higher = stricter mask boundary."
    )

    show_prob_map = st.checkbox("Show probability heatmap", value=True)
    show_crop = st.checkbox("Show cropped lesion", value=True)
    overlay_alpha = st.slider("Overlay opacity", 0.1, 0.6, 0.35, 0.05)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#6b7394; line-height:1.6;'>
    <b>Classes</b><br>
    🟢 Benign — non-cancerous lesion<br>
    🔴 Malignant — cancerous lesion<br><br>
    <b>Pipeline</b><br>
    U-Net → Lesion mask<br>
    Crop → MobileNetV2 → Label
    </div>
    """, unsafe_allow_html=True)


# ─── Load Models ─────────────────────────────────────────────────────────────

with st.spinner("Loading models…"):
    seg_model, cls_model, device = load_models(seg_path, cls_path)

models_ready = seg_model is not None and cls_model is not None


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>DermaScan AI</h1>
    <p>Skin lesion segmentation & classification powered by U-Net + MobileNetV2</p>
</div>
""", unsafe_allow_html=True)


# ─── Model Status ────────────────────────────────────────────────────────────

col_s, col_c, col_d = st.columns(3)
with col_s:
    status = "✅ Loaded" if seg_model else "⚠️ Not found"
    color = "#00e676" if seg_model else "#ffaa00"
    st.markdown(f"""<div class="metric-box">
        <div class="label">Segmentation Model</div>
        <div class="value" style="font-size:1rem; color:{color};">{status}</div>
    </div>""", unsafe_allow_html=True)

with col_c:
    status = "✅ Loaded" if cls_model else "⚠️ Not found"
    color = "#00e676" if cls_model else "#ffaa00"
    st.markdown(f"""<div class="metric-box">
        <div class="label">Classifier Model</div>
        <div class="value" style="font-size:1rem; color:{color};">{status}</div>
    </div>""", unsafe_allow_html=True)

with col_d:
    dev_label = "GPU 🚀" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""<div class="metric-box">
        <div class="label">Inference Device</div>
        <div class="value" style="font-size:1rem; color:#00d4ff;">{dev_label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not models_ready:
    st.markdown("""
    <div class="warning-box">
    ⚠️ One or both models could not be loaded. Check the sidebar paths.
    You can still test the UI — a demo mode will use random predictions.
    </div>
    """, unsafe_allow_html=True)


# ─── Upload ──────────────────────────────────────────────────────────────────

st.markdown("### Upload Dermoscopy Image")
uploaded = st.file_uploader(
    "Drag & drop or click to upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ─── Inference ───────────────────────────────────────────────────────────────

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("---")

    # Run inference
    with st.spinner("🔍 Analyzing lesion…"):
        if models_ready:
            seg_thresh = None if threshold == 0.0 else threshold
            results = full_pipeline(image_rgb, seg_model, cls_model, device,
                                    seg_threshold=seg_thresh)
        else:
            # Demo mode
            h, w = image_rgb.shape[:2]
            fake_prob = np.zeros((h, w), dtype=np.float32)
            cx, cy = w // 2, h // 2
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            fake_prob = np.clip(1 - dist / (min(h, w) * 0.35), 0, 1).astype(np.float32)
            from utils.inference import binarize_mask, crop_lesion, build_overlay
            binary = binarize_mask(fake_prob)
            crop, bbox = crop_lesion(image_rgb, binary)
            results = {
                "original": image_rgb,
                "prob_mask": fake_prob,
                "binary_mask": binary,
                "crop": crop,
                "overlay": build_overlay(image_rgb, binary, "Benign", bbox, overlay_alpha),
                "bbox": bbox,
                "label": "Benign",
                "confidence": 0.78,
                "prob_malignant": 0.22,
                "prob_benign": 0.78,
            }

    label = results["label"]
    conf = results["confidence"]
    prob_m = results["prob_malignant"]
    prob_b = results["prob_benign"]
    badge_cls = "badge-malignant" if label == "Malignant" else "badge-benign"

    # ── Diagnosis banner ──────────────────────────────────────────────────────
    icon = "🔴" if label == "Malignant" else "🟢"
    st.markdown(f"""
    <div class="result-card" style="text-align:center; border-color:{'rgba(255,68,68,0.4)' if label=='Malignant' else 'rgba(0,230,118,0.4)'};">
        <div style="font-size:0.8rem; font-weight:600; letter-spacing:0.1em; color:#6b7394; text-transform:uppercase; margin-bottom:0.6rem;">DIAGNOSIS</div>
        <span class="diagnosis-badge {badge_cls}">{icon} {label}</span>
        <div style="color:#6b7394; margin-top:0.6rem; font-size:0.9rem;">Confidence: <b style="color:#e8eaf2;">{conf*100:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability bars ──────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-box">
            <div class="label">Benign probability</div>
            <div class="value" style="color:#00e676;">{prob_b*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        st.progress(prob_b)
    with col2:
        st.markdown(f"""<div class="metric-box">
            <div class="label">Malignant probability</div>
            <div class="value" style="color:#ff4444;">{prob_m*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        st.progress(prob_m)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Image panels ──────────────────────────────────────────────────────────
    # Always show: Original + Overlay
    ncols = 2
    extra_cols = []
    if show_prob_map:
        extra_cols.append("heatmap")
    if show_crop:
        extra_cols.append("crop")
    ncols += len(extra_cols)

    cols = st.columns(ncols)

    with cols[0]:
        st.markdown('<div class="step-label">Original Image</div>', unsafe_allow_html=True)
        st.image(image_rgb, use_container_width=True)

    with cols[1]:
        st.markdown('<div class="step-label">Lesion Mask Overlay</div>', unsafe_allow_html=True)
        overlay = results["overlay"]
        if overlay_alpha != 0.35:
            from utils.inference import build_overlay
            overlay = build_overlay(image_rgb, results["binary_mask"],
                                    label, results["bbox"], overlay_alpha)
        st.image(overlay, use_container_width=True)

    idx = 2
    if show_prob_map and idx < len(cols):
        with cols[idx]:
            st.markdown('<div class="step-label">Probability Heatmap</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor("#161b2e")
            ax.set_facecolor("#161b2e")
            im = ax.imshow(results["prob_mask"], cmap="plasma", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white')
            ax.axis("off")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        idx += 1

    if show_crop and idx < len(cols) and results["crop"] is not None:
        with cols[idx]:
            st.markdown('<div class="step-label">Cropped Lesion</div>', unsafe_allow_html=True)
            st.image(results["crop"], use_container_width=True)

    # ── Mask stats ────────────────────────────────────────────────────────────
    binary = results["binary_mask"]
    mask_coverage = (binary > 0).sum() / binary.size * 100

    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(f"""<div class="metric-box">
            <div class="label">Lesion Coverage</div>
            <div class="value" style="color:#00d4ff;">{mask_coverage:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        h, w = image_rgb.shape[:2]
        st.markdown(f"""<div class="metric-box">
            <div class="label">Image Size</div>
            <div class="value" style="color:#00d4ff; font-size:1.1rem;">{w}×{h}</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        bbox = results["bbox"]
        if bbox:
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            st.markdown(f"""<div class="metric-box">
                <div class="label">Lesion Bounding Box</div>
                <div class="value" style="color:#00d4ff; font-size:1.1rem;">{lw}×{lh}px</div>
            </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="warning-box">
    ⚕️ <b>Medical Disclaimer:</b> This tool is for research and educational purposes only.
    It does not constitute medical advice. Always consult a qualified dermatologist for diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#6b7394;">
        <div style="font-size:4rem; margin-bottom:1rem;">🔬</div>
        <h3 style="color:#6b7394; font-weight:500;">Upload a dermoscopy image to begin</h3>
        <p style="font-size:0.9rem;">Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
