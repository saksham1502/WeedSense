from flask import Flask, request, jsonify, render_template
import os, traceback
import numpy as np
from PIL import Image as PILImage
import io

app = Flask(__name__)

# Pre-load model at startup
MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "weed_model.keras")
_model = None

def load_model():
    global _model
    if _model is None:
        import tensorflow as tf
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return _model

# Load model immediately on import
print("Initializing model...")
load_model()
print("Model ready!")

# ── Pages ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/info")
def info():
    return render_template("info.html")

# ── API ────────────────────────────────────────────────────────────────────────

@app.route("/predict/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    try:
        print("Received prediction request")
        img_bytes = request.files["image"].read()
        print(f"Image size: {len(img_bytes)} bytes")
        
        # Load image exactly as training: Pillow → RGB → resize → /255
        print("Loading image with Pillow...")
        img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((128, 128))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)
        print(f"Image preprocessed: shape={arr.shape}")
        
        # Predict
        print("Running prediction...")
        model = load_model()
        prob = float(model.predict(arr, verbose=0)[0][0])
        print(f"Prediction complete: prob={prob}")
        
        # prob > 0.5 → soybean (crop), else weed/other
        is_crop = prob > 0.5
        label = "Soybean (Crop)" if is_crop else "Weed / Other"
        confidence = round((prob if is_crop else 1 - prob) * 100, 2)
        
        result = {
            "label": label,
            "confidence": confidence,
            "is_crop": is_crop,
            "raw_prob": round(prob, 4)
        }
        print(f"Returning result: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR in classify: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
