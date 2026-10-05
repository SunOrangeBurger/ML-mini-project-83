import re
from collections import Counter
from pathlib import Path
import numpy as np


def parse_sign_file(path):
    """Return (frames x numeric_columns) array; hex columns (0x..) are dropped."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        rows.append([float(p) for p in parts if not p.lower().startswith("0x")])
    return np.array(rows)


def load_low_quality(root="data/raw/low_quality/extracted"):
    """Load all sign recordings, skipping empty files and calibration ('cal-*') recordings."""
    signals, labels, signers = [], [], []
    for f in sorted(Path(root).rglob("*.sign")):
        label = re.sub(r"\d+$", "", f.stem)
        if label.startswith("cal-"):
            continue
        s = parse_sign_file(f)
        if s.ndim != 2 or len(s) == 0:
            continue
        signals.append(s)
        labels.append(label)
        signers.append(f.parent.name)
    return signals, labels, signers


if __name__ == "__main__":
    signals, labels, signers = load_low_quality()
    counts = Counter(labels)
    lengths = [len(s) for s in signals]
    print("files:", len(signals))
    print("classes:", len(counts))
    print("examples per class min/max:", min(counts.values()), max(counts.values()))
    print("frames per sign mean/median:", np.mean(lengths), np.median(lengths))
    print("numeric columns:", Counter(s.shape[1] for s in signals))
