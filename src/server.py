import os
import re
import json
import random
from pathlib import Path
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy.signal import resample

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
SIGNS_DIR = BASE_DIR / "data" / "raw" / "low_quality" / "extracted" / "signs"
WEB_DIR = BASE_DIR / "web"
MODEL_CACHE = RESULTS_DIR / "svm_model.joblib"
CLASSES_CACHE = RESULTS_DIR / "classes.joblib"

COLS = [0, 1, 2, 3, 6, 7, 8, 9]
CHANNEL_NAMES = [
    "X Position (m)",
    "Y Position (m)",
    "Z Position (m)",
    "Wrist Roll (deg)",
    "Thumb Flexure",
    "Forefinger Flexure",
    "Middle Finger Flexure",
    "Ring Finger Flexure",
]

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)

# Global model state
MODEL = None
CLASSES = None
ALL_SIGN_FILES = []
SIGN_CATALOG = {}


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def parse_sign_text(text):
    """Parse raw .sign file content into a 2D numpy array (dropping hex columns)."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        row = [float(p) for p in parts if not p.lower().startswith("0x")]
        if row:
            rows.append(row)
    return np.array(rows)


def parse_sign_file(path):
    return parse_sign_text(Path(path).read_text())


def load_cached_model():
    global MODEL, CLASSES
    if MODEL is None or CLASSES is None:
        if not MODEL_CACHE.exists() or not CLASSES_CACHE.exists():
            raise FileNotFoundError(
                f"Model cache not found at {MODEL_CACHE}. Run src/demo.py first."
            )
        MODEL = joblib.load(MODEL_CACHE)
        CLASSES = list(joblib.load(CLASSES_CACHE))
    return MODEL, CLASSES


def index_dataset():
    global ALL_SIGN_FILES, SIGN_CATALOG
    if ALL_SIGN_FILES:
        return
    if not SIGNS_DIR.exists():
        return

    sign_files = list(SIGNS_DIR.rglob("*.sign"))
    valid = [p for p in sign_files if not p.stem.startswith("cal-")]
    ALL_SIGN_FILES = valid

    catalog = {}
    for p in valid:
        sign_name = re.sub(r"\d+$", "", p.stem)
        signer = p.parent.name
        rel_path = str(p.relative_to(BASE_DIR))
        if sign_name not in catalog:
            catalog[sign_name] = []
        catalog[sign_name].append({
            "signer": signer,
            "filename": p.name,
            "path": rel_path,
        })
    SIGN_CATALOG = catalog


def process_signal(signal, target_len=57):
    """
    Extracts the 8 channels, resamples to target_len, and normalizes.
    Returns:
      - feat_vector: (456,) flattened representation for SVM
      - resampled_signal: (57, 8) normalized array
      - raw_selected: (frames, 8) selected channels
    """
    if signal.ndim != 2 or signal.shape[0] < 2:
        raise ValueError("Signal must have at least 2 frames and multiple columns.")

    raw_selected = signal[:, COLS]
    resampled_arr = resample(raw_selected, target_len, axis=0)
    mn, mx = resampled_arr.min(axis=0), resampled_arr.max(axis=0)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    norm_resampled = (resampled_arr - mn) / denom
    feat_vector = norm_resampled.flatten()
    return feat_vector, norm_resampled, raw_selected


def run_prediction_pipeline(signal, true_label=None, source_desc=""):
    model, classes = load_cached_model()
    feat_vector, norm_resampled, raw_selected = process_signal(signal)

    # SVM decision function & probability estimation via softmax
    scores = model.decision_function([feat_vector])[0]
    probs = softmax(scores)

    # Top-5 rankings
    top_indices = np.argsort(probs)[::-1][:5]
    top_predictions = [
        {
            "rank": int(idx + 1),
            "label": str(classes[i]),
            "probability": float(probs[i]),
            "score": float(scores[i]),
        }
        for idx, i in enumerate(top_indices)
    ]

    predicted_label = classes[top_indices[0]]
    confidence = float(probs[top_indices[0]])

    # 3D spatial path from normalized positions (Cols: X=0, Y=1, Z=2, Roll=3)
    # Center and scale for 3D display
    pos_x = norm_resampled[:, 0].tolist()
    pos_y = norm_resampled[:, 1].tolist()
    pos_z = norm_resampled[:, 2].tolist()
    rot_roll = norm_resampled[:, 3].tolist()

    trajectory_3d = []
    for f in range(len(pos_x)):
        trajectory_3d.append({
            "frame": f,
            "x": round(float(pos_x[f]), 4),
            "y": round(float(pos_y[f]), 4),
            "z": round(float(pos_z[f]), 4),
            "roll": round(float(rot_roll[f]), 4),
        })

    # Prepare sensor channels for time-series charts
    channels_data = {
        "frames": list(range(len(norm_resampled))),
        "x": [round(float(v), 4) for v in norm_resampled[:, 0]],
        "y": [round(float(v), 4) for v in norm_resampled[:, 1]],
        "z": [round(float(v), 4) for v in norm_resampled[:, 2]],
        "roll": [round(float(v), 4) for v in norm_resampled[:, 3]],
        "thumb": [round(float(v), 4) for v in norm_resampled[:, 4]],
        "forefinger": [round(float(v), 4) for v in norm_resampled[:, 5]],
        "middle": [round(float(v), 4) for v in norm_resampled[:, 6]],
        "ring": [round(float(v), 4) for v in norm_resampled[:, 7]],
    }

    # Raw channel summary
    raw_frames = int(signal.shape[0])
    duration_sec = round(raw_frames / 50.0, 2)  # 50 Hz PowerGlove

    is_match = None
    if true_label:
        is_match = bool(true_label.strip().lower() == predicted_label.strip().lower())

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "true_label": true_label,
        "match": is_match,
        "source": source_desc,
        "raw_frames": raw_frames,
        "duration_sec": duration_sec,
        "sampling_rate": 50,
        "top_predictions": top_predictions,
        "trajectory_3d": trajectory_3d,
        "channels": channels_data,
    }


# =========================================================================
# API Endpoints
# =========================================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        model, classes = load_cached_model()
        index_dataset()
        return jsonify({
            "status": "online",
            "model_type": "Support Vector Classifier (RBF Kernel, C=10)",
            "accuracy": 0.602,
            "macro_f1": 0.601,
            "classes_count": len(classes),
            "total_samples": len(ALL_SIGN_FILES),
            "signers_count": len(set(p.parent.name for p in ALL_SIGN_FILES)) if ALL_SIGN_FILES else 18,
            "sampling_rate_hz": 50,
            "sensor_channels": 8,
            "target_frames": 57,
            "feature_dim": 456,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/signs", methods=["GET"])
def api_signs():
    index_dataset()
    sorted_signs = sorted(SIGN_CATALOG.keys())

    # Curate some prominent signs
    featured_candidates = [
        "hello", "thankyou", "danger", "read", "drink", "yes", "no",
        "father", "mother", "walk", "crazy", "God", "computer(PC)",
        "boy", "girl", "all", "alive", "answer", "money", "think"
    ]
    featured = [s for s in featured_candidates if s in SIGN_CATALOG]

    signers = sorted(list(set(item["signer"] for items in SIGN_CATALOG.values() for item in items)))

    return jsonify({
        "total_signs": len(sorted_signs),
        "signs": sorted_signs,
        "featured": featured,
        "signers": signers,
    })


@app.route("/api/sample-files", methods=["GET"])
def api_sample_files():
    sign = request.args.get("sign")
    signer = request.args.get("signer")
    index_dataset()

    if not sign or sign not in SIGN_CATALOG:
        return jsonify({"error": f"Sign '{sign}' not found"}), 404

    items = SIGN_CATALOG[sign]
    if signer:
        items = [it for it in items if it["signer"] == signer]

    return jsonify({"sign": sign, "samples": items})


@app.route("/api/predict-sample", methods=["GET"])
def api_predict_sample():
    file_rel = request.args.get("file")
    if not file_rel:
        return jsonify({"error": "Missing file parameter"}), 400

    target = BASE_DIR / file_rel
    if not target.exists():
        return jsonify({"error": f"File not found: {file_rel}"}), 404

    true_label = re.sub(r"\d+$", "", target.stem)
    try:
        signal = parse_sign_file(target)
        result = run_prediction_pipeline(
            signal, true_label=true_label, source_desc=str(file_rel)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/random", methods=["GET"])
def api_random():
    index_dataset()
    if not ALL_SIGN_FILES:
        return jsonify({"error": "No sign files found"}), 404

    target = random.choice(ALL_SIGN_FILES)
    true_label = re.sub(r"\d+$", "", target.stem)
    rel_path = str(target.relative_to(BASE_DIR))

    try:
        signal = parse_sign_file(target)
        result = run_prediction_pipeline(
            signal, true_label=true_label, source_desc=rel_path
        )
        result["sample_file"] = rel_path
        result["signer"] = target.parent.name
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        if "file" in request.files:
            uploaded_file = request.files["file"]
            content = uploaded_file.read().decode("utf-8", errors="replace")
            filename = uploaded_file.filename
            true_label = re.sub(r"\d+$", "", Path(filename).stem) if filename else None
            source_desc = f"Uploaded file: {filename}"
        elif request.is_json and "text" in request.json:
            content = request.json["text"]
            filename = request.json.get("filename", "custom.sign")
            true_label = request.json.get("true_label")
            source_desc = "Pasted text stream"
        else:
            return jsonify({"error": "No file or text payload provided"}), 400

        signal = parse_sign_text(content)
        result = run_prediction_pipeline(
            signal, true_label=true_label, source_desc=source_desc
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@app.route("/api/benchmarks", methods=["GET"])
def api_benchmarks():
    return jsonify({
        "classical_models": [
            {"model": "Linear SVM (C=1.0)", "scaling": "per-example", "accuracy": 0.306, "macro_f1": 0.308, "paper_f1": "—"},
            {"model": "Linear SVM (C=0.1)", "scaling": "per-example", "accuracy": 0.369, "macro_f1": 0.365, "paper_f1": 0.549},
            {"model": "Logistic Regression", "scaling": "per-example", "accuracy": 0.385, "macro_f1": 0.384, "paper_f1": 0.436},
            {"model": "RBF SVM (C=10.0)", "scaling": "global", "accuracy": 0.576, "macro_f1": 0.575, "paper_f1": "—"},
            {"model": "RBF SVM (C=10.0) [CHAMPION]", "scaling": "per-example", "accuracy": 0.602, "macro_f1": 0.601, "paper_f1": 0.549},
        ],
        "recurrent_models": [
            {"name": "Paper Reference LSTM (MSE step-by-step)", "arch": "LSTM", "units": 128, "dropout": 0.0, "stacked": False, "accuracy": 0.071, "macro_f1": 0.066, "note": "Paper's baseline failure"},
            {"name": "Baseline LSTM", "arch": "LSTM", "units": 128, "dropout": 0.3, "stacked": False, "accuracy": 0.383, "macro_f1": 0.370, "note": "5.6x improvement over paper"},
            {"name": "Hidden 64", "arch": "LSTM", "units": 64, "dropout": 0.3, "stacked": False, "accuracy": 0.350, "macro_f1": 0.339, "note": "Lower capacity"},
            {"name": "Hidden 256", "arch": "LSTM", "units": 256, "dropout": 0.3, "stacked": False, "accuracy": 0.409, "macro_f1": 0.396, "note": "Higher capacity"},
            {"name": "GRU (128)", "arch": "GRU", "units": 128, "dropout": 0.3, "stacked": False, "accuracy": 0.334, "macro_f1": 0.315, "note": "Gated Recurrent Unit"},
            {"name": "Bidirectional LSTM", "arch": "BiLSTM", "units": 128, "dropout": 0.3, "stacked": False, "accuracy": 0.437, "macro_f1": 0.431, "note": "Both forward and backward contexts"},
            {"name": "No Dropout (p=0.0)", "arch": "LSTM", "units": 128, "dropout": 0.0, "stacked": False, "accuracy": 0.347, "macro_f1": 0.332, "note": "Overfits early"},
            {"name": "Stacked LSTM (2x128) [BEST RNN]", "arch": "LSTM (Stacked)", "units": 128, "dropout": 0.3, "stacked": True, "accuracy": 0.447, "macro_f1": 0.440, "note": "6.7x improvement over paper"},
        ],
        "ablation_single": [
            {"removed": "None (Full 8 Features)", "test_error": 0.398, "delta_error": 0.0, "importance": "Baseline"},
            {"removed": "ROT (Wrist Roll)", "test_error": 0.418, "delta_error": 0.020, "importance": "Moderate (+2.0%)"},
            {"removed": "F4 (Ring Finger)", "test_error": 0.413, "delta_error": 0.015, "importance": "Minor (+1.5%)"},
            {"removed": "F3 (Middle Finger)", "test_error": 0.412, "delta_error": 0.014, "importance": "Minor (+1.4%)"},
            {"removed": "F1 (Thumb)", "test_error": 0.415, "delta_error": 0.017, "importance": "Minor (+1.7%)"},
            {"removed": "F2 (Forefinger)", "test_error": 0.421, "delta_error": 0.023, "importance": "Moderate (+2.3%)"},
            {"removed": "POS (X, Y, Z Coordinates)", "test_error": 0.756, "delta_error": 0.358, "importance": "CRITICAL (+35.8%)"},
        ],
        "ablation_cumulative": [
            {"stage": "All Features", "features_remaining": 8, "test_error": 0.398},
            {"stage": "- POS (X, Y, Z)", "features_remaining": 5, "test_error": 0.756},
            {"stage": "- ROT (Roll)", "features_remaining": 4, "test_error": 0.817},
            {"stage": "- F1 (Thumb)", "features_remaining": 3, "test_error": 0.828},
            {"stage": "- F2 (Forefinger)", "features_remaining": 2, "test_error": 0.900},
            {"stage": "- F3 (Middle)", "features_remaining": 1, "test_error": 0.956},
        ],
        "confused_pairs": [
            {"true": "man", "pred": "please", "count": 5, "reason": "Identical gross arm stroke, minimal difference in finger bend"},
            {"true": "which", "pred": "maybe", "count": 5, "reason": "Slight variation in wrist oscillation with similar spatial boundary"},
            {"true": "surprise", "pred": "more", "count": 5, "reason": "Fast burst trajectory overlapping in X-Z space"},
            {"true": "exit", "pred": "you", "count": 4, "reason": "Forward pointing motion sharing primary directional vector"},
            {"true": "hello", "pred": "forget", "count": 4, "reason": "Forehead/temple outward sweep motion"},
            {"true": "yes", "pred": "danger", "count": 4, "reason": "Vertical downward fist strike trajectory"},
        ]
    })


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


if __name__ == "__main__":
    # Ensure model is ready
    load_cached_model()
    index_dataset()
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Starting Auslan Sign Language Recognition Web Server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
