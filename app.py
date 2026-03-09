from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# Auto-download helper for Google Drive folder
import gdown

APP_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = APP_DIR / "effb0_tta"

# Provided by user
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1A5QD3RfLkPEM7vU9LscPFXn-aIWQ9F0p?usp=sharing"


def _bundle_has_model() -> bool:
    keras_path = BUNDLE_DIR / "effb0_tta_model.keras"
    savedmodel_pb = BUNDLE_DIR / "effb0_tta_savedmodel" / "saved_model.pb"
    return keras_path.exists() or savedmodel_pb.exists()


@st.cache_resource
def ensure_bundle_available() -> bool:
    """Ensure required bundle exists locally. Download from Drive folder if missing."""
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    need_files = [
        BUNDLE_DIR / "effb0_tta_meta.json",
        BUNDLE_DIR / "effb0_tta_class_names.npy",
    ]

    if _bundle_has_model() and all(p.exists() for p in need_files):
        return True

    try:
        st.info("Model bundle not found locally. Downloading from Google Drive...")
        # Download all files inside the shared folder into BUNDLE_DIR
        gdown.download_folder(
            url=DRIVE_FOLDER_URL,
            output=str(BUNDLE_DIR),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except Exception as e:
        st.error(
            "Failed to download model bundle from Google Drive folder. "
            "Make sure link sharing is set to 'Anyone with the link'."
        )
        st.exception(e)
        return False

    if not _bundle_has_model():
        st.error(
            "Download completed but model file is still missing. "
            "Expected `effb0_tta_model.keras` or `effb0_tta_savedmodel/saved_model.pb` in bundle."
        )
        return False

    return True


def load_meta() -> dict:
    meta_path = BUNDLE_DIR / "effb0_tta_meta.json"
    if not meta_path.exists():
        st.error(f"Meta file not found: {meta_path}")
        st.stop()
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_class_names(meta: dict) -> list[str]:
    cname = meta.get("class_names_path", "effb0_tta_class_names.npy")
    cpath = BUNDLE_DIR / cname
    if not cpath.exists():
        # fallback fixed name
        cpath = BUNDLE_DIR / "effb0_tta_class_names.npy"
    if not cpath.exists():
        st.error(f"Class names file not found: {cpath}")
        st.stop()
    arr = np.load(cpath, allow_pickle=True)
    return [str(x) for x in arr.tolist()]


@st.cache_resource
def load_model(meta: dict):
    keras_name = meta.get("model_keras_path", "effb0_tta_model.keras")
    keras_path = BUNDLE_DIR / keras_name
    if keras_path.exists():
        try:
            return tf.keras.models.load_model(keras_path, compile=False)
        except Exception as e:
            print(f'Keras model load failed, fallback to SavedModel. Detail: {e}')

    # fallback fixed keras filename
    fallback_keras = BUNDLE_DIR / "effb0_tta_model.keras"
    if fallback_keras.exists():
        try:
            return tf.keras.models.load_model(fallback_keras, compile=False)
        except Exception as e:
            print(f'Fallback Keras load failed, trying SavedModel. Detail: {e}')

    savedmodel_name = meta.get("model_savedmodel_path", "effb0_tta_savedmodel")
    savedmodel_path = BUNDLE_DIR / savedmodel_name
    if savedmodel_path.exists():
        sm = tf.saved_model.load(str(savedmodel_path))
        if "serving_default" not in sm.signatures:
            raise RuntimeError("SavedModel has no serving_default signature")
        return sm.signatures["serving_default"]

    # fallback fixed savedmodel directory
    fallback_sm = BUNDLE_DIR / "effb0_tta_savedmodel"
    if fallback_sm.exists():
        sm = tf.saved_model.load(str(fallback_sm))
        if "serving_default" not in sm.signatures:
            raise RuntimeError("SavedModel has no serving_default signature")
        return sm.signatures["serving_default"]

    raise FileNotFoundError("No model artifact found in bundle")


def preprocess_image(file, img_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(file).convert("RGB").resize((img_size[1], img_size[0]))
    arr = np.asarray(image).astype("float32") / 255.0
    return arr


def tta_views(x: np.ndarray, transforms: list[str]) -> list[np.ndarray]:
    views = []
    for t in transforms:
        if t == "identity":
            views.append(x)
        elif t in ["flip_left_right", "hflip"]:
            views.append(np.flip(x, axis=2))
        elif t in ["flip_up_down", "vflip"]:
            views.append(np.flip(x, axis=1))
        elif t == "rot90":
            views.append(np.rot90(x, 1, axes=(1, 2)).copy())
        elif t == "rot270":
            views.append(np.rot90(x, 3, axes=(1, 2)).copy())
    return views if views else [x]


def predict_probs(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return model.predict(x, verbose=0)
    out = model(tf.convert_to_tensor(x, dtype=tf.float32))
    if isinstance(out, dict):
        return list(out.values())[0].numpy()
    return out.numpy()


def find_explainability_image() -> Path | None:
    candidates = [
        APP_DIR / "explainability_occlusion.png",
        APP_DIR / "fig05_gradcam_explanation_transfer_learning.png",
        APP_DIR.parent / "figure" / "fig05_gradcam_explanation_transfer_learning.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    st.set_page_config(page_title="Durian Leaf Disease Classifier", layout="wide")

    ok = ensure_bundle_available()
    if not ok:
        st.stop()

    meta = load_meta()
    class_names = load_class_names(meta)
    model = load_model(meta)

    model_name = str(meta.get("best_model_name", "effb0_tta"))
    acc = meta.get("best_model_accuracy", None)
    macro_f1 = meta.get("macro_f1", None)
    bal_acc = meta.get("balanced_accuracy", None)
    img_size = tuple(meta.get("img_size", [224, 224]))
    tta_enabled = bool(meta.get("tta_enabled", True))
    transforms = meta.get("tta_transforms", ["identity"])

    st.title("Durian Leaf Disease Classifier")
    if isinstance(acc, (float, int)):
        st.write(f"Selected algorithm: **{model_name}** | Test Accuracy: **{acc:.4f}**")
    else:
        st.write(f"Selected algorithm: **{model_name}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}" if isinstance(acc, (float, int)) else "-")
    c2.metric("Macro F1", f"{macro_f1:.4f}" if isinstance(macro_f1, (float, int)) else "-")
    c3.metric("Balanced Acc", f"{bal_acc:.4f}" if isinstance(bal_acc, (float, int)) else "-")

    st.caption("Inference policy: TTA probability averaging" if tta_enabled else "Inference policy: single-view")

    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        uploaded.seek(0)
        preview_img = Image.open(uploaded).convert("RGB")
        uploaded.seek(0)
        arr = preprocess_image(uploaded, img_size)
        x = np.expand_dims(arr, axis=0)

        views = tta_views(x, transforms if tta_enabled else ["identity"])
        probs = [predict_probs(model, v)[0] for v in views]
        p = np.mean(np.stack(probs, axis=0), axis=0)

        pred_idx = int(np.argmax(p))
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        conf = float(p[pred_idx])

        st.subheader("Prediction Results")
        left_col, right_col = st.columns([1, 1.1], gap="large")

        with left_col:
            # Keep a fixed, publication-style preview size for cleaner layout.
            preview_fixed = ImageOps.fit(preview_img, (320, 320), method=Image.Resampling.LANCZOS)
            st.image(preview_fixed, caption="Input Image (320x320)", width=320)
            st.success(f"Prediction: {pred_label}")
            st.write(f"Confidence: **{conf:.4f}**")

        with right_col:
            prob_df = pd.DataFrame({"Class": class_names, "Probability": p[: len(class_names)]})
            st.subheader("Class Probabilities")
            st.bar_chart(prob_df.set_index("Class"), height=320)

    st.subheader("Explainability")
    explain_path = find_explainability_image()
    if explain_path is not None:
        st.image(str(explain_path), caption="Explainability graph", use_column_width=True)
    else:
        st.info(
            "Explainability image not found. Add one of these files: "
            "`explainability_occlusion.png` in this folder, or `figure/fig05_gradcam_explanation_transfer_learning.png`."
        )


if __name__ == "__main__":
    main()

