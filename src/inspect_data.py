from collections import Counter
import numpy as np
from load_data import load_low_quality

signals, labels, signers = load_low_quality()
lengths = np.array([len(s) for s in signals])

print("empty files:", int((lengths == 0).sum()))
print("files > 200 frames:", int((lengths > 200).sum()), "| longest:", sorted(lengths)[-5:])
print("files < 10 frames (non-empty):", int(((lengths > 0) & (lengths < 10)).sum()))

cols = Counter(s.shape[1] for s in signals if s.ndim == 2)
print("numeric columns:", cols)

counts = Counter(labels)
print("\nclasses with != 70 examples:")
for k, v in sorted(counts.items(), key=lambda kv: kv[1]):
    if v != 70:
        print(f"  {k!r}: {v}")

print("\nexamples per signer:", Counter(signers))
