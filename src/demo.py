import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC

from load_data import load_low_quality, parse_sign_file
from preprocess import preprocess

MODEL_CACHE = Path("results/svm_model.joblib")
CLASSES_CACHE = Path("results/classes.joblib")
COLS = [0, 1, 2, 3, 6, 7, 8, 9]


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def get_trained_model():
    """Returns a trained RBF SVM model and list of class names, training and caching if needed."""
    if MODEL_CACHE.exists() and CLASSES_CACHE.exists():
        model = joblib.load(MODEL_CACHE)
        classes = joblib.load(CLASSES_CACHE)
        return model, classes

    print("Training SVM model for demo (this happens once and is cached)...")
    signals, labels, _ = load_low_quality()
    X, y, classes = preprocess(signals, labels, feature_cols=COLS)
    model = SVC(kernel="rbf", C=10, gamma="scale")
    model.fit(X, y)

    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_CACHE)
    joblib.dump(classes, CLASSES_CACHE)
    print("Model cached to", MODEL_CACHE)
    return model, classes


def predict_sign(file_path, model, classes):
    """Predict the sign from a single .sign file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Extract true label from filename (e.g. 'hello-1.sign' -> 'hello' or 'all-1.sign' -> 'all')
    raw_stem = path.stem
    import re
    true_label = re.sub(r"\d+$", "", raw_stem)

    signal = parse_sign_file(path)
    if signal.ndim != 2 or len(signal) == 0:
        raise ValueError(f"Invalid or empty signal in {file_path}")

    X, _, _ = preprocess([signal], [true_label], feature_cols=COLS)
    scores = model.decision_function(X)[0]
    probs = softmax(scores)
    top_indices = np.argsort(probs)[::-1][:3]

    pred_class = classes[top_indices[0]]
    confidence = probs[top_indices[0]]

    return {
        "file": str(path),
        "true_label": true_label,
        "predicted_label": pred_class,
        "confidence": confidence,
        "match": (true_label.lower() == pred_class.lower()),
        "top3": [(classes[i], probs[i]) for i in top_indices],
    }


def main():
    parser = argparse.ArgumentParser(description="Auslan Sign Language Recognition Single-Sample Demo")
    parser.add_argument("--file", type=str, default=None, help="Path to a .sign file to classify")
    parser.add_argument("--random", action="store_true", help="Pick a random sign from data/raw/")
    args = parser.parse_args()

    model, classes = get_trained_model()

    if args.file:
        target_path = Path(args.file)
    else:
        # Default or random pick
        all_signs = list(Path("data/raw/low_quality/extracted/signs").rglob("*.sign"))
        valid_signs = [p for p in all_signs if not p.stem.startswith("cal-")]
        if not valid_signs:
            print("No sign files found in data/raw/low_quality/extracted/signs")
            return
        if args.random:
            target_path = np.random.choice(valid_signs)
        else:
            # Pick a well-known sign like 'thankyou' or 'hello' or first available
            hello_signs = [p for p in valid_signs if "hello" in p.stem]
            target_path = hello_signs[0] if hello_signs else valid_signs[0]

    result = predict_sign(target_path, model, classes)

    print("\n" + "=" * 50)
    print("  AUSLAN SIGN RECOGNITION DEMO")
    print("=" * 50)
    print(f"File:            {result['file']}")
    print(f"True Label:      {result['true_label']}")
    print(f"Predicted Label: {result['predicted_label']} ({result['confidence']*100:.1f}%)")
    print(f"Result:          {'MATCH [SUCCESS]' if result['match'] else 'MISMATCH'}")
    print("\nTop 3 Predictions:")
    for rank, (cls, p) in enumerate(result['top3'], 1):
        print(f"  {rank}. {cls:<16} ({p*100:.1f}%)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
