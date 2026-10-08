# Sequential Pattern Mining Results

Low-quality Auslan, 70/30 stratified split (seed 42). Paper's SPM: F1 = 0.065.

Config: `{'mode': 'both', 'n_levels': 3, 'window': 10, 'min_support': 0.2, 'max_len': 3, 'max_patterns': 1000}`, SVM: `linear`

| Model | Accuracy | Precision | Recall | Macro F1 | Train err | Test err |
|---|---|---|---|---|---|---|
| SPM + SVM | 0.518 | 0.522 | 0.518 | 0.516 | 0.000 | 0.482 |
| SPM + flattened, RBF SVM | 0.649 | 0.657 | 0.649 | 0.647 | 0.000 | 0.351 |

## Top 10 patterns by chi-square

State `H:3` = channel 3 high; suffix `d` = rate-of-change state (D/S/I); `-b->` before, `-o->` overlap.

| Pattern | chi2 |
|---|---|
| `Dd:5 -o-> L:5 -o-> Id:5` | 1165.6 |
| `L:5 -o-> Id:5` | 1077.1 |
| `L:5 -o-> Id:5 -o-> M:5` | 1036.9 |
| `L:5 -o-> Id:5 -o-> H:5` | 989.1 |
| `L:5 -o-> Sd:5 -b-> Id:5` | 935.3 |
| `L:5 -o-> Id:5 -b-> Sd:5` | 893.7 |
| `M:5 -b-> L:5 -o-> Id:5` | 881.7 |
| `H:1 -o-> Dd:5` | 856.5 |
| `L:1 -o-> H:2 -o-> Sd:5` | 856.0 |
| `H:1 -o-> Id:5 -b-> Dd:5` | 852.0 |
