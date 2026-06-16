"""
LASA Pill Classifier — Streamlit Demo
Jalankan: streamlit run app.py
"""

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "MobileNetV3_S_state_dict.pth"
CLASS_NAMES = ["NSAID (Pereda Nyeri)", "Antibiotik"]
CLASS_EMOJI = ["💊", "💉"]
CLASS_COLOR = ["#1D9E75", "#378ADD"]   # hijau / biru
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    net = models.mobilenet_v3_small()
    in_f = net.classifier[3].in_features           # 1024
    net.classifier[3] = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 2)
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    net.load_state_dict(state)
    net.eval().to(DEVICE)
    return net


@torch.no_grad()
def predict(pil_img: Image.Image, model):
    tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LASA Pill Classifier",
    page_icon="💊",
    layout="wide",
)

st.markdown("""
<style>
    .result-box {
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
        text-align: center;
    }
    .label-text { font-size: 1.5rem; font-weight: 700; }
    .conf-text  { font-size: 1rem; opacity: 0.85; margin-top: 4px; }
    .bar-bg {
        background: #e9ecef;
        border-radius: 8px;
        height: 18px;
        margin: 6px 0;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.4s ease;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("💊 LASA Pill Classifier")
st.caption(
    "Klasifikasi gambar obat: **NSAID (Pereda Nyeri)** vs **Antibiotik** "
    "menggunakan MobileNetV3-Small."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.markdown("""
| | |
|---|---|
| **Arsitektur** | MobileNetV3-Small |
| **Input size** | 224 × 224 px |
| **Kelas** | 2 (NSAID / Antibiotik) |
| **Device** | `{}`|
""".format(str(DEVICE).upper()))
    st.divider()
    st.markdown("**Cara pakai:**")
    st.markdown("1. Upload foto obat (JPG/PNG)")
    st.markdown("2. Tunggu hasil prediksi")
    st.markdown("3. Lihat confidence tiap kelas")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
model = load_model()

uploaded = st.file_uploader(
    "Upload gambar obat",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("⬆️ Upload gambar obat untuk memulai klasifikasi.")
    st.stop()

# Load image
pil_img = Image.open(uploaded).convert("RGB")

col_img, col_result = st.columns([1, 1], gap="large")

# ── Kolom kiri: gambar ───────────────────────────────────────────────────────
with col_img:
    st.subheader("Gambar Input")
    st.image(pil_img, use_column_width=True)
    st.caption(f"Ukuran asli: {pil_img.width} × {pil_img.height} px")

# ── Kolom kanan: hasil ───────────────────────────────────────────────────────
with col_result:
    st.subheader("Hasil Prediksi")

    with st.spinner("Menganalisis…"):
        probs = predict(pil_img, model)

    pred_idx    = int(np.argmax(probs))
    pred_name   = CLASS_NAMES[pred_idx]
    pred_emoji  = CLASS_EMOJI[pred_idx]
    pred_color  = CLASS_COLOR[pred_idx]
    confidence  = probs[pred_idx] * 100

    # Result box
    st.markdown(f"""
<div class="result-box" style="background:{pred_color}22; border: 2px solid {pred_color};">
  <div style="font-size:2.5rem;">{pred_emoji}</div>
  <div class="label-text" style="color:{pred_color};">{pred_name}</div>
  <div class="conf-text">Confidence: <strong>{confidence:.1f}%</strong></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Probabilitas per Kelas")
    for i, (name, prob, color) in enumerate(zip(CLASS_NAMES, probs, CLASS_COLOR)):
        pct = prob * 100
        bold = "**" if i == pred_idx else ""
        st.markdown(
            f"{CLASS_EMOJI[i]} {bold}{name}{bold}",
        )
        st.markdown(f"""
<div class="bar-bg">
  <div class="bar-fill" style="width:{pct:.1f}%; background:{color};"></div>
</div>
<div style="text-align:right; font-size:0.85rem; margin-top:-4px; margin-bottom:8px;">
  {pct:.2f}%
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Sistem ini hanya untuk keperluan penelitian dan tidak menggantikan "
    "diagnosis farmasi profesional."
)