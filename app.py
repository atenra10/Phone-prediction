# app.py
import os, io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
import matplotlib.pyplot as plt

# loaders for TF 2.20 / Keras 3.10
try:
    import tf_keras as tfk
except Exception:
    tfk = None
try:
    import keras
except Exception:
    keras = None

st.set_page_config(page_title="Phone Model Detector", page_icon="📱", layout="centered")
st.title("📱 Phone Model Detector")
st.caption("Upload a phone image. The app loads the local .h5 model automatically and shows phone specs.")

IMG_SIZE = 224
MODEL_FILENAME = "phone_detector_model.h5"

LABELS = ["iPhone_13", "iPhone_14", "Samsung_Galaxy_S23", "Google_Pixel_7", "OnePlus_11"]

PHONE_SPECS = {
    "iPhone_13": {
        "pretty": "iPhone 13", "processor": "Apple A15 Bionic", "ram": "4 GB",
        "storage": "128/256/512 GB", "rear_camera": "Dual 12 MP", "front_camera": "12 MP",
        "display": "6.1\" OLED (Super Retina XDR)", "battery": "3227 mAh",
        "more_info": "https://support.apple.com/en-in/111872",
    },
    "iPhone_14": {
        "pretty": "iPhone 14", "processor": "Apple A15 Bionic (5-core GPU)", "ram": "6 GB",
        "storage": "128/256/512 GB", "rear_camera": "Dual 12 MP", "front_camera": "12 MP",
        "display": "6.1\" OLED (Super Retina XDR)", "battery": "3279 mAh",
        "more_info": "https://support.apple.com/en-in/111850",
    },
    "Samsung_Galaxy_S23": {
        "pretty": "Samsung Galaxy S23", "processor": "Snapdragon 8 Gen 2 for Galaxy", "ram": "8 GB",
        "storage": "128/256 GB", "rear_camera": "50 + 10 + 12 MP", "front_camera": "12 MP",
        "display": "6.1\" Dynamic AMOLED 2X, 120 Hz", "battery": "3900 mAh",
        "more_info": "https://www.samsung.com/in/smartphones/galaxy-s/galaxy-s23-phantom-black-256gb-sm-s911bzkcins/",
    },
    "Google_Pixel_7": {
        "pretty": "Google Pixel 7", "processor": "Google Tensor G2", "ram": "8 GB",
        "storage": "128/256 GB", "rear_camera": "50 + 12 MP", "front_camera": "10.8 MP",
        "display": "6.3\" OLED, 90 Hz", "battery": "4355 mAh",
        "more_info": "https://en.wikipedia.org/wiki/Pixel_7",
    },
    "OnePlus_11": {
        "pretty": "OnePlus 11", "processor": "Snapdragon 8 Gen 2", "ram": "8/16 GB (LPDDR5X)",
        "storage": "128 GB (UFS 3.1) / 256 GB (UFS 4.0)", "rear_camera": "50 + 48 + 32 MP",
        "front_camera": "16 MP", "display": "6.7\" 120 Hz LTPO3 (QHD+)", "battery": "5000 mAh",
        "more_info": "https://www.oneplus.in/11/specs",
    },
}

def format_label(s: str) -> str:
    return s.replace("_", " ")

@st.cache_resource(show_spinner=False)
def load_model_from_path(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    errs = []
    if tfk is not None:
        try:
            return tfk.models.load_model(model_path, compile=False)
        except Exception as e:
            errs.append(f"tf_keras error: {e!s}")
    if keras is not None:
        try:
            return keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e:
            errs.append(f"keras v3 error: {e!s}")
    raise RuntimeError("Failed to load model.\n" + "\n".join(errs))

def preprocess_image(pil_img, size=(IMG_SIZE, IMG_SIZE)):
    img = pil_img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# Sidebar: model status
st.sidebar.header("🧠 Model")
st.sidebar.code(os.path.abspath(MODEL_FILENAME))
status_box = st.sidebar.empty()
try:
    status_box.info("Loading model…")
    model = load_model_from_path(MODEL_FILENAME)
    status_box.success("Model loaded")
except Exception as e:
    status_box.error(f"Failed to load model:\n{e}")
    st.stop()
st.sidebar.header("🏷️ Phones that can be Classified")
st.sidebar.write([format_label(x) for x in LABELS])

# ---- Upload once → bytes → reuse safely ----
st.subheader("1) Upload an image")
uploaded = st.file_uploader("Choose a phone image (JPG/PNG)", type=["jpg", "jpeg", "png"])

file_bytes = None
pil_image = None
if uploaded:
    try:
        file_bytes = uploaded.read()  # read once
        pil_image = Image.open(io.BytesIO(file_bytes))
    except UnidentifiedImageError:
        st.error("Unsupported or corrupted image. Please upload a valid JPG/PNG.")
        uploaded = None
    except Exception as e:
        st.error(f"Could not read image: {e}")
        uploaded = None

col1, col2 = st.columns([1, 1], vertical_alignment="top")

with col1:
    st.subheader("2) Preview")
    if pil_image is not None:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)
    else:
        st.info("No image uploaded yet.")

with col2:
    st.subheader("3) Predict")
    if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
        if pil_image is None:
            st.error("Please upload an image first.")
        else:
            try:
                x = preprocess_image(pil_image, (IMG_SIZE, IMG_SIZE))

                # --- Predict & sanitize numeric types ---
                raw = model.predict(x, verbose=0)
                preds = np.asarray(raw, dtype=float).squeeze()  # ensure 1D float64

                # Handle unexpected shapes
                if preds.ndim != 1:
                    st.error(f"Unexpected model output shape: {preds.shape}")
                    st.stop()

                # Guard against NaN/inf; re-normalize if needed
                if not np.isfinite(preds).all():
                    # softmax fallback
                    e = np.exp(np.nan_to_num(preds, nan=0.0) - np.nanmax(preds))
                    preds = e / (e.sum() + 1e-12)

                # Guard against length mismatch
                if len(LABELS) != preds.shape[0]:
                    st.error(
                        f"Model outputs {int(preds.shape[0])} classes but label list has {len(LABELS)}."
                    )
                    st.stop()

                # --- Top prediction ---
                top_idx = int(np.argmax(preds))
                top_conf = float(preds[top_idx])          # cast to Python float
                key = str(LABELS[top_idx])               # cast to Python str
                spec = PHONE_SPECS.get(key, {})
                name = str(spec.get("pretty", key.replace("_", " ")))

                # Use markdown (robust) instead of st.success
                st.markdown(
                    f"✅ **Prediction:** {name}<br>"
                    f"**Confidence:** {top_conf*100:.2f}%",
                    unsafe_allow_html=True,
                )

                # --- Top-3 ---
                top3_idx = np.argsort(preds)[-3:][::-1]
                st.markdown("**Top-3 predictions**")
                for i in top3_idx:
                    lbl = str(LABELS[int(i)]).replace("_", " ")
                    prob = float(preds[int(i)]) * 100.0
                    st.write(f"- {lbl}: {prob:.2f}%")

                # --- Probability chart ---
                fig, ax = plt.subplots(figsize=(6.5, 3.6))
                names = [str(l).replace("_", " ") for l in LABELS]
                ax.bar(range(len(names)), preds.astype(float))
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=28, ha="right")
                ax.set_ylabel("Probability")
                ax.set_ylim(0.0, 1.0)
                ax.set_title("Class Probabilities")
                st.pyplot(fig)

                # --- Spec card ---
                st.markdown("### 📋 Phone details")
                cA, cB = st.columns(2)
                with cA:
                    st.write(f"**Processor:** {spec.get('processor','—')}")
                    st.write(f"**RAM:** {spec.get('ram','—')}")
                    st.write(f"**Storage:** {spec.get('storage','—')}")
                    st.write(f"**Display:** {spec.get('display','—')}")
                with cB:
                    st.write(f"**Rear camera:** {spec.get('rear_camera','—')}")
                    st.write(f"**Front camera:** {spec.get('front_camera','—')}")
                    st.write(f"**Battery:** {spec.get('battery','—')}")
                    mi = spec.get("more_info")
                    if isinstance(mi, str) and mi:
                        st.markdown(f"[More details]({mi})")

            except Exception as e:
                # Show exact type to help debug on Cloud
                st.error(f"Prediction failed ({type(e).__name__}).")
                st.exception(e)

