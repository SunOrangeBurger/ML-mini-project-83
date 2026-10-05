# Sequential Pattern Mining Results

Low-quality Auslan, 70/30 stratified split (seed 42). Paper's SPM: F1 = 0.065.

Config: `{'mode': 'both', 'n_levels': 3, 'window': 20, 'min_support': 0.2, 'max_len': 3, 'max_patterns': 2000}`, SVM: `linear`

| Model | Accuracy | Precision | Recall | Macro F1 | Train err | Test err |
|---|---|---|---|---|---|---|
| SPM + SVM | 0.652 | 0.655 | 0.652 | 0.648 | 0.000 | 0.348 |
| SPM + flattened, RBF SVM | 0.655 | 0.663 | 0.655 | 0.653 | 0.000 | 0.345 |

## Top 10 patterns by chi-square

State `H:3` = channel 3 high; suffix `d` = rate-of-change state (D/S/I); `-b->` before, `-o->` overlap.

| Pattern | chi2 |
|---|---|
| `Dd:5 -o-> L:5 -o-> Id:5` | 1151.7 |
| `L:5 -o-> Id:5 -b-> Sd:5` | 1051.2 |
| `L:5 -o-> Id:5 -o-> H:5` | 1050.2 |
| `Dd:5 -o-> L:5 -b-> H:5` | 1045.4 |
| `H:5 -o-> H:1` | 1022.0 |
| `Dd:5 -b-> Sd:5 -b-> Id:5` | 1020.7 |
| `L:5 -o-> Sd:5 -b-> H:5` | 991.4 |
| `Sd:0 -b-> L:5 -o-> Id:5` | 986.0 |
| `L:5 -o-> Id:5 -b-> Sd:1` | 986.0 |
| `Dd:5 -b-> Id:5 -b-> M:0` | 930.3 |
