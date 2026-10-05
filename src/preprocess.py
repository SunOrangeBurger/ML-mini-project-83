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
