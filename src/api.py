"""
Simple Flask API for InvasiveLens model serving.

Provides REST endpoints for single image prediction and batch inference.

Usage:
    python -m src.api --model resnet50 --port 5000 --threshold 0.6

Then test with:
    curl -X POST http://localhost:5000/predict \\
      -F "file=@image.jpg" \\
      -F "model=resnet50" \\
      -F "threshold=0.6"
"""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.exceptions import BadRequest

from src.inference import InvasiveLensPredictor


def create_app(model_name: str = "resnet50", fold: int = 0, threshold: float = 0.5):
    """Factory function to create Flask app with pre-loaded model."""
    app = Flask(__name__)

    # Load model at startup
    try:
        predictor = InvasiveLensPredictor(
            model_name=model_name,
            fold=fold,
            threshold=threshold,
        )
        app.logger.info(f"Loaded {model_name} model (fold {fold}, threshold {threshold})")
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        raise

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "model": model_name,
            "classes": predictor.classes,
            "threshold": threshold,
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Predict on uploaded image.

        Form parameters:
          - file: Image file (JPG/PNG)
          - threshold: Optional override for confidence threshold

        Returns:
            JSON with prediction, confidence, abstention flag, top 3
        """
        if "file" not in request.files:
            raise BadRequest("No file provided")

        file = request.files["file"]
        if file.filename == "":
            raise BadRequest("Empty filename")

        threshold_override = request.form.get("threshold", threshold)
        try:
            threshold_override = float(threshold_override)
        except ValueError:
            raise BadRequest(f"Invalid threshold: {threshold_override}")

        try:
            # Read image from upload
            image_data = file.read()
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Make prediction
            from src.data.augmentation import get_eval_transform
            from config import IMAGE_SIZE
            import torch

            transform = get_eval_transform(IMAGE_SIZE)
            tensor = transform(image).unsqueeze(0).to(predictor.device)
            pred = predictor.predict_from_tensor(tensor)

            # Apply per-request threshold without mutating the shared predictor
            abstain = pred.confidence < threshold_override

            return jsonify({
                "success": True,
                "predicted_class": pred.predicted_class,
                "confidence": pred.confidence,
                "abstain": abstain,
                "top_3": [{"class": c, "confidence": float(conf)} for c, conf in pred.top_3_classes],
                "threshold_used": threshold_override,
            })
        except Exception as e:
            app.logger.exception(f"Prediction error: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
            }), 400

    @app.route("/calibration", methods=["GET"])
    def calibration():
        """Get model calibration info."""
        try:
            calib = predictor.get_calibration_info()
            return jsonify({
                "success": True,
                "calibration": calib,
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
            }), 400

    @app.route("/classes", methods=["GET"])
    def classes():
        """List all supported classes."""
        return jsonify({
            "classes": predictor.classes,
            "n_classes": len(predictor.classes),
        })

    @app.errorhandler(BadRequest)
    def handle_bad_request(e):
        return jsonify({"success": False, "error": str(e)}), 400

    @app.errorhandler(500)
    def handle_internal_error(e):
        app.logger.exception("Unhandled exception")
        return jsonify({"success": False, "error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet50",
                        choices=["baseline", "resnet50", "efficientnet_v2_s"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(model_name=args.model, fold=args.fold, threshold=args.threshold)
    print(f"\nStarting InvasiveLens API on {args.host}:{args.port}")
    print(f"Model: {args.model} (fold {args.fold})")
    print(f"Threshold: {args.threshold}")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"Prediction: POST http://{args.host}:{args.port}/predict\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
