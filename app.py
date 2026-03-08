from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = APP_DIR / "effb0_tta"


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
        st.error(f"Class names file not found: {cpath}")
        st.stop()
    arr = np.load(cpath, allow_pickle=True)
    return [str(x) for x in arr.tolist()]


@st.cache_resource
def load_model(meta: dict):
    keras_name = meta.get("model_keras_path", "effb0_tta_model.keras")
    keras_path = BUNDLE_DIR / keras_name
    if keras_path.exists():
        return tf.keras.models.load_model(keras_path, compile=False)

    savedmodel_name = meta.get("model_savedmodel_path", "effb0_tta_savedmodel")
    savedmodel_path = BUNDLE_DIR / savedmodel_name
    if savedmodel_path.exists():
        sm = tf.saved_model.load(str(savedmodel_path))
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
    st.set_page_config(page_title="Durian Leaf Disease Classifier", layout="centered")

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
    st.write(
        f"Selected algorithm: **{model_name}** | "
        f"Test Accuracy: **{acc:.4f}**" if isinstance(acc, (float, int)) else f"Selected algorithm: **{model_name}**"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}" if isinstance(acc, (float, int)) else "-")
    c2.metric("Macro F1", f"{macro_f1:.4f}" if isinstance(macro_f1, (float, int)) else "-")
    c3.metric("Balanced Acc", f"{bal_acc:.4f}" if isinstance(bal_acc, (float, int)) else "-")

    st.caption("Inference policy: TTA probability averaging" if tta_enabled else "Inference policy: single-view")

    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        arr = preprocess_image(uploaded, img_size)
        x = np.expand_dims(arr, axis=0)

        views = tta_views(x, transforms if tta_enabled else ["identity"])
        probs = [predict_probs(model, v)[0] for v in views]
        p = np.mean(np.stack(probs, axis=0), axis=0)

        pred_idx = int(np.argmax(p))
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        conf = float(p[pred_idx])

        st.image(arr, caption="Input Image", use_container_width=True)
        st.success(f"Prediction: {pred_label}")
        st.write(f"Confidence: **{conf:.4f}**")

        prob_df = pd.DataFrame({"Class": class_names, "Probability": p[: len(class_names)]})
        st.subheader("Class Probabilities")
        st.bar_chart(prob_df.set_index("Class"))

    st.subheader("Explainability")
    explain_path = find_explainability_image()
    if explain_path is not None:
        st.image(str(explain_path), caption="Explainability graph", use_container_width=True)
    else:
        st.info(
            "Explainability image not found. Add one of these files: "
            "`explainability_occlusion.png` in this folder, or `figure/fig05_gradcam_explanation_transfer_learning.png`."
        )


if __name__ == "__main__":
    main()
