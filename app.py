from flask import Flask, request, jsonify, render_template
import os, traceback

app = Flask(__name__)

# Pre-load model at startup to avoid timeout on first request
MODEL_PATH = os.environ.get("CNN_MODEL_PATH", "weed_model.keras")
_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        import tensorflow as tf
        print(f"Loading model from {MODEL_PATH}...")
        _model_cache = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return _model_cache

# Load model on startup
try:
    get_model()
except Exception as e:
    print(f"Warning: Could not pre-load model: {e}")

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
    img_bytes  = request.files["image"].read()
    model_path = os.environ.get("CNN_MODEL_PATH", "weed_model.keras")
    try:
        from model_utils import predict_classification
        result = predict_classification(img_bytes, model_path)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Model file not found. Please train the model first (python train.py)"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
