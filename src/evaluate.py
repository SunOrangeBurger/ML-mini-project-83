import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from load_data import load_low_quality
from preprocess import preprocess

# Original column indices. Groupings follow the paper; verify against auslan.html.
GROUPS = {"POS": [0, 1, 2], "ROT": [3], "F1": [6], "F2": [7], "F3": [8], "F4": [9]}
ALL = [c for g in GROUPS.values() for c in g]

signals, labels, _ = load_low_quality()


def run(cols):
    X, y, classes = preprocess(signals, labels, sorted(cols))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    m = SVC(kernel="rbf", C=10, gamma="scale").fit(X_tr, y_tr)
    return X, y, classes, m.predict(X_te), y_te


# 1. Full model: metrics + confusion matrix
X, y, classes, pred, y_te = run(ALL)
print(f"FULL: acc={accuracy_score(y_te, pred):.3f} F1={f1_score(y_te, pred, average='macro'):.3f}")
cm = confusion_matrix(y_te, pred)
plt.figure(figsize=(7, 6)); plt.imshow(cm, cmap="Blues")
plt.title("Confusion matrix (RBF SVM, low quality)"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.colorbar(); plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

# most confused sign pairs, useful for the write-up
off = cm.copy(); np.fill_diagonal(off, 0)
print("Most confused (true -> predicted, count):")
for i, j in zip(*np.unravel_index(np.argsort(off, axis=None)[::-1][:8], off.shape)):
    print(f"  {classes[i]} -> {classes[j]}: {off[i, j]}")

# 2. PCA to 3D
Z = PCA(n_components=3).fit_transform(X)
fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap="nipy_spectral", s=4)
ax.set_title("PCA of flattened feature space (colour = sign)")
plt.tight_layout(); plt.savefig("results/pca_3d.png", dpi=150); plt.close()

# 3. Ablation round 1: remove each group on its own
print("\nAblation round 1 (remove one group):")
for name, cols in GROUPS.items():
    keep = [c for c in ALL if c not in cols]
    _, _, _, p, t = run(keep)
    print(f"  without {name:4s}: test_err={1 - accuracy_score(t, p):.3f}")

# 4. Ablation round 2: remove groups cumulatively
print("\nAblation round 2 (cumulative):")
removed, order = [], ["POS", "ROT", "F1", "F2", "F3"]
for name in order:
    removed += GROUPS[name]
    keep = [c for c in ALL if c not in removed]
    _, _, _, p, t = run(keep)
    print(f"  removed up to {name:3s}: test_err={1 - accuracy_score(t, p):.3f}")
