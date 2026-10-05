import numpy as np
from load_data import load_low_quality

signals, _, _ = load_low_quality()
allrows = np.vstack(signals)
print("col  min      max      mean     std      unique")
for i in range(allrows.shape[1]):
    c = allrows[:, i]
    print(f"{i:>3}  {c.min():7.3f}  {c.max():7.3f}  {c.mean():7.3f}  {c.std():7.3f}  {len(np.unique(c))}")
