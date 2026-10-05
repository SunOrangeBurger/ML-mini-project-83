import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from load_data import load_low_quality
from preprocess import preprocess

FEATURE_COLS = [0, 1, 2, 3, 6, 7, 8, 9]  # x,y,z,roll + 4 fingers

signals, labels, _ = load_low_quality()
X, y, classes = preprocess(signals, labels, FEATURE_COLS)
print("X:", X.shape, "classes:", len(classes))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

models = {
    "SVM (linear)": OneVsRestClassifier(LinearSVC(max_iter=5000)),
    "LogReg": OneVsRestClassifier(LogisticRegression(max_iter=2000)),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    print(f"{name}: acc={accuracy_score(y_te, pred):.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} "
          f"train_err={1 - m.score(X_tr, y_tr):.3f}")
