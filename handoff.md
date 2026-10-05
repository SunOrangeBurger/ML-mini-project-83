# Next Steps (handoff for the second push)

Status: data loading, preprocessing, baselines and evaluation are done and pushed. This document covers what remains.
Take the LSTM first, since it is the main missing technical piece and the paper's own suggested fix is easy to try.

## 0. Environment (do this first)

TensorFlow does not support the newest Python releases. Check `python --version`. If it is 3.13 or newer, recreate the env:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install numpy scipy pandas scikit-learn matplotlib seaborn tensorflow
uv pip freeze > requirements.txt
```

(If you would rather use PyTorch, that is fine. Keep the same input shape and evaluation.)

Get the data in place as described in `README.md`, then confirm `python src/load_data.py` prints 6648 files and 95 classes.

## 1. Task: LSTM classifier (`src/lstm_model.py`)

**Why the paper's LSTM failed (their hypothesis):** standard setups backpropagate at every time step, which suits next-value
prediction. Here the whole signal gets one label, so the loss should come from the **final time step only**.
In Keras, `LSTM(..., return_sequences=False)` does exactly that. The paper also used mean squared error;
categorical cross-entropy with softmax is the natural choice for 95-way classification.

### 1a. Add a sequence preprocessor

Add to `src/preprocess.py` (same steps as `preprocess`, but no flattening):

```python
def preprocess_sequences(signals, labels, feature_cols=None, target_len=TARGET_LEN):
    """Returns X (n, target_len, n_features), y (n,), class names."""
    X = []
    for s in signals:
        if feature_cols is not None:
            s = s[:, feature_cols]
        s = resample(s, target_len, axis=0)
        mn, mx = s.min(axis=0), s.max(axis=0)
        s = (s - mn) / np.where(mx - mn == 0, 1, mx - mn)
        X.append(s)
    classes, y = np.unique(labels, return_inverse=True)
    return np.array(X), y, classes
```

### 1b. Starter model

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

from load_data import load_low_quality
from preprocess import preprocess_sequences

COLS = [0, 1, 2, 3, 6, 7, 8, 9]
signals, labels, _ = load_low_quality()
X, y, classes = preprocess_sequences(signals, labels, COLS)   # (n, 57, 8)

# SAME split as the baselines so results are comparable
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

model = keras.Sequential([
    layers.Input(shape=X.shape[1:]),
    layers.LSTM(128, return_sequences=False),      # loss only from the final time step
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(classes), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

hist = model.fit(X_tr, y_tr, validation_split=0.15, epochs=60, batch_size=64,
                 callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])

pred = model.predict(X_te).argmax(axis=1)
print(f"LSTM: acc={accuracy_score(y_te, pred):.3f} F1={f1_score(y_te, pred, average='macro'):.3f}")
```

### 1c. Experiments to run and record

Keep a table of every run (hyperparameters and test accuracy / macro F1), including the ones that do not work.
Suggested:

- Hidden size 64 / 128 / 256
- LSTM vs GRU vs bidirectional LSTM
- With and without dropout
- Two stacked LSTM layers (`return_sequences=True` on the first only)
- Per-example vs global scaling of the input

Reference points: paper's LSTM F1 = 0.066 (poor), our RBF SVM F1 = 0.589. Any honest result is acceptable.
If the LSTM still trails the SVM, say so and discuss why (about 4,600 training examples across 95 classes is small for a neural network).
Save the training curve to `results/lstm_training.png` and the metrics to `results/lstm_results.md`.

## 2. Task: write-up (`docs/writeup.md`, then export to PDF)

The guidelines say both "one-page" and "two-page". Confirm with faculty, and aim for two pages at most.
Required sections: problem statement, dataset details, approach, brief implementation overview, conclusions.

Points worth including:
- High quality dataset unavailable (UCI `tctodd` files are broken symlinks / 403), so all results are on low quality data.
- Data cleaning: calibration recordings, empty file, constant and duplicate columns.
- RBF SVM F1 0.589 vs paper's 0.549, and RBF helps much more here than in the paper.
- Ablation findings from `src/evaluate.py` (fill in once run).
- LSTM results and discussion.

## 3. Task: slides and demo

- 6-8 slides: problem, data, preprocessing, models, results table, confusion matrix and PCA, ablation, LSTM, conclusions.
- Demo: `python src/baseline.py` is quick. Consider a small script that loads one `.sign` file, preprocesses it, and prints the predicted sign.
- Both members should be able to explain every file in `src/`. Read through `load_data.py` and `preprocess.py` even though you did not write them.

## 4. Optional stretch: Sequential Pattern Mining

Only if the above is done. See section 4.3 of the paper (discretize, Apriori-style pattern generation, chi-square ranking, binary features, then SVM).
The paper's own implementation scored about 0.065 F1, so treat it as exploratory.

## 5. Git workflow

Commit your own work under your own name so the history reflects individual contribution.

```bash
git config user.name  "Your Name"
git config user.email "you@example.com"
git pull
git checkout -b lstm
# ... work ...
git add src/lstm_model.py src/preprocess.py results/
git commit -m "Add LSTM classifier and experiment results"
git push -u origin lstm
```

Then merge into `main` (or open a pull request). Use small commits with clear messages rather than one big push at the end.

## 6. Deadlines

Reviews run Oct 5 to Oct 9. Final submission (repo, PDF write-up) is due Saturday Oct 10, 11:59 PM.