from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
from image_model import MedicalBreastCancerImageModel
from risk_model import BreastCancerRiskModel

app = Flask(__name__)
CORS(app)

# Initialize models
image_model = MedicalBreastCancerImageModel()
risk_model = BreastCancerRiskModel()


@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        result = image_model.predict_medical_image(temp_path)
        return jsonify(result)
    finally:
        os.unlink(temp_path)


@app.route("/predict-tabular", methods=["POST"])
def predict_tabular():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        result = risk_model.predict_csv(temp_path, use_shap=True)
        return jsonify(result)
    finally:
        os.unlink(temp_path)


@app.route("/metrics", methods=["GET"])
def get_metrics():
    image_metrics = (
        image_model.get_metrics() if hasattr(image_model, "get_metrics") else {}
    )
    risk_metrics = risk_model.get_metrics()

    # Return combined or risk metrics
    return jsonify(risk_metrics)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
