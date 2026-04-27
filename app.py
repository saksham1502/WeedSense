from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, traceback

app = Flask(__name__)
CORS(app)

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
    app.run(debug=False, port=5000)
