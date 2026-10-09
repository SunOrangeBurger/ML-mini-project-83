# Sign Language Recognition using Temporal Classification

UE24CS352A Machine Learning mini-project (Problem 118).
Reproduces and extends *"Sign Language Recognition using Temporal Classification"* (Cate, Dalvi, Hussain, 2015),
which classifies 95 Auslan (Australian Sign Language) signs from glove-sensor time series.

**Team:** Arun Hariharan (PES2UG24AM126), Pavan Kishore (PES2UG24AM111)

---

## 1. Problem

Given a multivariate time series of hand position, rotation and finger-bend readings, predict which of 95 signs was performed.
Camera-free, sensor-based sign recognition is the goal; the paper's baseline SVM/logistic regression models do well on
high-quality data but poorly on low-quality data, so the focus is on the low-quality dataset.

## 2. Dataset

Kadous's Auslan data from the UCI repository.

| | Low quality (used) | High quality (not used) |
|---|---|---|
| Hardware | Nintendo Powerglove, 1 hand | 5DT gloves + Flock of Birds, 2 hands |
| Rate | 50 Hz | 200 Hz |
| Signs | 95 | 95 |
| Examples per sign | ~70 (several signers) | 27 (one signer) |

**The high quality dataset is not used.** UCI's `tctodd.tar.gz` / `tctodd.tar.bz2` are broken symlinks in the
download, and the underlying files return `403 Forbidden`. All results here are on the low quality set.

Cleaning applied in `src/load_data.py`:
- Dropped 3 calibration recordings (`cal-full-forward/right/up`), which are not signs.
- Dropped 1 empty file.
- Dropped hex status columns. Of the remaining 11 numeric columns, columns 4 and 5 are constant and column 10
  duplicates column 9, leaving **8 features** (columns 0-3 and 6-9), matching the paper.
- Result: **6,648 recordings, 95 classes, 69-70 per class** (paper: 6,650).

## 3. Approach

1. **Preprocessing** (`src/preprocess.py`): temporal scaling (resample each sign to 57 frames via FFT),
   spatial scaling (each feature to 0-1), flatten to one vector per example (57 x 8 = 456 features).
2. **Baselines** (`src/baseline.py`, `src/experiments.py`): linear SVM, RBF SVM, logistic regression,
   70/30 stratified split.
3. **Evaluation** (`src/evaluate.py`): confusion matrix, 3D PCA, two rounds of feature-group ablation.
4. **Temporal model** (`src/lstm_model.py`): LSTM over the unflattened sequence. *(in progress, see `docs/NEXT_STEPS.md`)*

## 4. Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12 (TensorFlow does not support the newest Python releases).

```bash
git clone <repo-url> && cd sign-lang-rec
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Data

Download "Australian Sign Language signs" from the UCI repository and extract so that the layout is:

```
data/raw/low_quality/extracted/signs/<signer>/<sign><n>.sign
```

```bash
mkdir -p data/raw/low_quality/extracted
tar -xzf allsigns.tar.gz -C data/raw/low_quality/extracted
```

`data/raw/` is git-ignored.

## 5. Running

All commands run from the repository root.

```bash
python src/load_data.py       # sanity check: expect 6648 files, 95 classes
python src/baseline.py        # SVM + logistic regression baseline
python src/experiments.py     # scaling / kernel / regularisation comparison
python src/evaluate.py        # confusion matrix, PCA, ablation -> results/
python src/lstm_model.py      # run LSTM / RNN experiment suite -> results/lstm_results.md
python src/demo.py --random   # single-sample sign recognition demo
python app.py                 # interactive 3D motion lab & web app -> http://localhost:5001
python docs/generate_pdf.py   # build 2-page project PDF write-up -> docs/writeup.pdf
```

## 6. Experimental Results (Low quality, 70/30 stratified split, seed 42)

### Classical Baselines vs. Paper Reference

| Model | Scaling | Accuracy | Macro F1 | Paper F1 (Reference) |
|---|---|---|---|---|
| Linear SVM (C=1) | per-example | 0.306 | 0.308 | — |
| Linear SVM (C=0.1) | per-example | 0.369 | 0.365 | 0.549 |
| Logistic regression | per-example | 0.385 | 0.384 | 0.436 |
| RBF SVM (C=10) | global | 0.576 | 0.575 | — |
| **RBF SVM (C=10)** | **per-example** | **0.602** | **0.601** | **0.549** |

*Note:* The RBF kernel provides a dramatic +21.7% gain over linear classifiers on this dataset, outperforming the paper's reported SVM score.

### Recurrent Neural Networks (Resolving the Paper's LSTM Failure)

The paper reported an LSTM macro F1 of **0.066**, formulating training with frame-wise MSE loss. By shifting to sequence-to-label classification (`return_sequences=False`) and categorical cross-entropy with softmax, performance increases by an order of magnitude:

| Architecture / Config | Units | Dropout | Stacked | Test Accuracy | Macro F1 | vs. Paper F1 (0.066) |
|---|---|---|---|---|---|---|
| Baseline LSTM | 128 | 0.3 | No | 0.383 | 0.370 | +5.6x gain |
| Hidden 64 | 64 | 0.3 | No | 0.350 | 0.339 | +5.1x gain |
| Hidden 256 | 256 | 0.3 | No | 0.409 | 0.396 | +6.0x gain |
| GRU | 128 | 0.3 | No | 0.334 | 0.315 | +4.8x gain |
| Bidirectional LSTM | 128 | 0.3 | No | 0.437 | 0.431 | +6.5x gain |
| No Dropout | 128 | 0.0 | No | 0.347 | 0.332 | +5.0x gain (overfits) |
| **Stacked LSTM (2x128)** | **128** | **0.3** | **Yes** | **0.447** | **0.440** | **+6.7x gain** |

### Feature Modality Ablation (RBF SVM)
- **Full Model Error**: 39.8% (Accuracy: 60.2%)
- **Without POS (x, y, z coordinates)**: Test Error surges to **75.6%** (+35.8% error increase), proving global trajectory is the dominant modality.
- **Without ROT (wrist roll)**: Test Error = 41.8% (+2.0% change).
- **Cumulative Removal**: $\text{All} \xrightarrow{-\text{POS}} 75.6\% \xrightarrow{-\text{ROT}} 81.7\% \xrightarrow{-\text{F1}} 82.8\% \xrightarrow{-\text{F2}} 90.0\% \xrightarrow{-\text{F3}} 95.6\%$.

### Most Confused Sign Pairs
`man` $\rightarrow$ `please` (5), `which` $\rightarrow$ `maybe` (5), `surprise` $\rightarrow$ `more` (5), `exit` $\rightarrow$ `you` (4), `hello` $\rightarrow$ `forget` (4). These share similar arm movements and differ only in fine finger curl.

## 7. Repository Layout

```
data/            raw and processed data (git-ignored)
src/             load_data, preprocess, baseline, experiments, evaluate, lstm_model, demo
results/         figures (confusion_matrix, pca_3d, lstm_training), metrics, cached models
docs/            writeup.md, writeup.pdf, generate_pdf.py, slides.md
handoff.md       handoff documentation for subsequent development
```

## 8. Contributions

| Member | Work |
|---|---|
| Arun Hariharan (PES2UG24AM126) | Data loading and cleaning, preprocessing, baselines, evaluation |
| Pavan Kishore (PES2UG24AM111) | Sequence preprocessing, LSTM architecture & experiment suite, write-up (MD/PDF), slides, interactive demo |


## 9. Reference

Cate, Dalvi, Hussain. *Sign Language Recognition using Temporal Classification.* 2015.
Kadous, M. W. *Temporal Classification: Extending the Classification Paradigm to Multivariate Time Series.* PhD thesis, UNSW, 2002.