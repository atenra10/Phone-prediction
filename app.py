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

def to_1d_probs(raw):
    """
    Make model.predict(...) output a clean 1-D float array of class probabilities.
    Handles dict, list/tuple, extra batch dims, and non-softmax logits.
    """
    import numpy as np

    # 1) Pick the array out of dict/list/tuple
    arr = None
    if isinstance(raw, dict):
        # common key names people save in TF SavedModels
        for k in ("predictions", "logits", "output_0", "probs"):
            if k in raw:
                arr = raw[k]
                break
        if arr is None:
            # just take the first value
            arr = next(iter(raw.values()))
    elif isinstance(raw, (list, tuple)):
        # take the first array-like
        for v in raw:
            if hasattr(v, "shape"):
                arr = v
                break
        if arr is None:
            arr = raw[0]
    else:
        arr = raw

    # 2) Ensure numpy array, squeeze batch dims
    arr = np.asarray(arr, dtype=float).squeeze()

    # After squeeze: if still >1D, assume batch-first
    if arr.ndim > 1:
        arr = arr[0]

    if arr.ndim != 1:
        raise ValueError(f"Unexpected prediction shape after squeeze: {arr.shape}")

    # 3) Replace NaN/inf, then convert to probabilities if needed
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # If not a probability vector (sum not approx 1), softmax it
    s = float(arr.sum())
    if not (0.98 <= s <= 1.02):  # loose check
        e = np.exp(arr - np.max(arr))
        arr = e / (e.sum() + 1e-12)

    return arr


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

                # Robustly normalize predictions to a 1-D probability vector
                raw = model.predict(x, verbose=0)
                preds = to_1d_probs(raw)

                # Debug (expandable) to see shapes/types if something goes wrong
                with st.expander("Debug: raw output details", expanded=False):
                    st.write(
                        {
                            "raw_type": type(raw).__name__,
                            "preds_shape": getattr(preds, "shape", None),
                            "preds_sum": float(np.sum(preds)),
                            "preds_first5": preds[: min(5, preds.shape[0])].tolist(),
                        }
                    )

                # Guard against label length mismatch
                if preds.shape[0] != len(LABELS):
                    st.error(
                        f"Model outputs {int(preds.shape[0])} classes but LABELS has {len(LABELS)}.\n"
                        "Make sure the model was trained on these 5 classes, in this order:\n"
                        f"{[l.replace('_',' ') for l in LABELS]}"
                    )
                    st.stop()

                # Top-1
                top_idx = int(np.argmax(preds))
                top_conf = float(preds[top_idx])
                key = str(LABELS[top_idx])
                spec = PHONE_SPECS.get(key, {})
                name = str(spec.get("pretty", key.replace("_", " ")))

                st.markdown(
                    f"✅ **Prediction:** {name}<br>"
                    f"**Confidence:** {top_conf*100:.2f}%",
                    unsafe_allow_html=True,
                )

                # Top-3
                top3_idx = np.argsort(preds)[-3:][::-1]
                st.markdown("**Top-3 predictions**")
                for i in top3_idx:
                    lbl = str(LABELS[int(i)]).replace("_", " ")
                    prob = float(preds[int(i)]) * 100.0
                    st.write(f"- {lbl}: {prob:.2f}%")

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(6.5, 3.6))
                names = [l.replace("_", " ") for l in LABELS]
                ax.bar(range(len(names)), preds.astype(float))
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=28, ha="right")
                ax.set_ylabel("Probability")
                ax.set_ylim(0.0, 1.0)
                ax.set_title("Class Probabilities")
                st.pyplot(fig)

                # Specs card
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
                st.error(f"Prediction failed ({type(e).__name__}).")
                st.exception(e)


