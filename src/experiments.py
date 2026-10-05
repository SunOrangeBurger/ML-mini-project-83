import numpy as np
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score

from load_data import load_low_quality

COLS = [0, 1, 2, 3, 6, 7, 8, 9]
signals, labels, _ = load_low_quality()
classes, y = np.unique(labels, return_inverse=True)

# temporal scaling only
R = np.array([resample(s[:, COLS], 57, axis=0) for s in signals])  # (n, 57, 8)

# scaling variants
lo, hi = R.min(axis=(0, 1)), R.max(axis=(0, 1))
global_scaled = (R - lo) / np.where(hi - lo == 0, 1, hi - lo)
mn, mx = R.min(axis=1, keepdims=True), R.max(axis=1, keepdims=True)
per_example = (R - mn) / np.where(mx - mn == 0, 1, mx - mn)

for sname, data in [("per-example", per_example), ("global", global_scaled)]:
    X = data.reshape(len(data), -1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    models = {
        "LinearSVC C=1": LinearSVC(C=1, max_iter=5000),
        "LinearSVC C=0.1": LinearSVC(C=0.1, max_iter=5000),
        "RBF SVC C=10": SVC(kernel="rbf", C=10, gamma="scale"),
        "LogReg": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
    }
    for mname, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        print(f"{sname:12s} {mname:16s} acc={accuracy_score(y_te, p):.3f} "
              f"F1={f1_score(y_te, p, average='macro'):.3f} train_err={1 - m.score(X_tr, y_tr):.3f}")
