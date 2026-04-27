import numpy as np
from PIL import Image as PILImage
import io
import tensorflow as tf
from tensorflow import keras

# ── Model cache ────────────────────────────────────────────────────────────────
_cnn_model = None

def load_cnn_model(path="weed_model.keras"):
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = keras.models.load_model(path)
    return _cnn_model


# ── Inference ──────────────────────────────────────────────────────────────────

def predict_classification(img_bytes, model_path="weed_model.keras"):
    """
    Binary CNN classifier.
    Trained with Pillow RGB on 128x128 .tif images.
    soybean → label 1 (Crop), broadleaf/grass/soil → label 0 (Weed/Other)
    Returns: { label, confidence, class_scores }
    """
    # Load exactly as training did: Pillow → RGB → resize → /255
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)

    model = load_cnn_model(model_path)
    prob = float(model.predict(arr, verbose=0)[0][0])

    # prob > 0.5 → soybean (crop), else weed/other
    is_crop = prob > 0.5
    label      = "Soybean (Crop)" if is_crop else "Weed / Other"
    confidence = round((prob if is_crop else 1 - prob) * 100, 2)

    return {
        "label":      label,
        "confidence": confidence,
        "is_crop":    is_crop,
        "raw_prob":   round(prob, 4)
    }
