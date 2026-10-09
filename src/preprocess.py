import numpy as np
from scipy.signal import resample

TARGET_LEN = 57  # average sign length reported in the paper


def preprocess(signals, labels, feature_cols=None, target_len=TARGET_LEN):
    """signals: list of (frames x cols) arrays -> X (n, features*target_len), y (n,), class names."""
    X = []
    for s in signals:
        if feature_cols is not None:
            s = s[:, feature_cols]
        s = resample(s, target_len, axis=0)        # temporal scaling (FFT)
        mn, mx = s.min(axis=0), s.max(axis=0)
        s = (s - mn) / np.where(mx - mn == 0, 1, mx - mn)  # spatial scaling to 0-1, per feature
        X.append(s.flatten())                      # time-series flattening
    classes, y = np.unique(labels, return_inverse=True)
    return np.array(X), y, classes


def preprocess_sequences(signals, labels, feature_cols=None, target_len=TARGET_LEN, scaling="per-example"):
    """Returns X (n, target_len, n_features), y (n,), class names."""
    resampled = []
    for s in signals:
        if feature_cols is not None:
            s = s[:, feature_cols]
        s = resample(s, target_len, axis=0)
        resampled.append(s)
    R = np.array(resampled)
    if scaling == "global":
        lo, hi = R.min(axis=(0, 1), keepdims=True), R.max(axis=(0, 1), keepdims=True)
        X = (R - lo) / np.where(hi - lo == 0, 1, hi - lo)
    elif scaling == "per-example":
        mn, mx = R.min(axis=1, keepdims=True), R.max(axis=1, keepdims=True)
        X = (R - mn) / np.where(mx - mn == 0, 1, mx - mn)
    else:
        X = R
    classes, y = np.unique(labels, return_inverse=True)
    return X, y, classes

