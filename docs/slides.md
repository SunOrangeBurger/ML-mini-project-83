---
marp: true
theme: default
paginate: true
header: "UE24CS352A Machine Learning Mini-Project | Problem 118"
footer: "Auslan Sign Language Recognition using Temporal Classification"
---

# Sign Language Recognition using Temporal Classification

### Auslan 95-Class Glove-Sensor Multivariate Time Series Recognition
**Problem 118 | UE24CS352A Machine Learning Mini-Project**

*Reproducing & Extending Cate, Dalvi, Hussain (2015)*

---

## 1. Problem Statement & Motivation

- **Goal**: Classify 95 distinct Australian Sign Language (Auslan) signs from multivariate glove sensor streams.
- **Why Sensor / Glove-Based?**
  - Robust against lighting changes, occlusions, complex backgrounds, and camera angles.
  - Directly captures fine-grained finger flexion and spatial hand trajectories.
- **The Challenge**:
  - Signs have varying temporal durations (lengths vary from ~10 to >200 frames; median: 55 frames).
  - High degree of intra-class variability across multiple human signers.
  - Previous literature (Cate et al., 2015) reported failure of standard LSTMs (F1 = 0.066) and modest performance on low-quality sensor data.

---

## 2. Dataset & Cleaning Pipeline

- **Source**: Kadous Auslan Dataset (UCI Machine Learning Repository #114).
- **Data Capture**: Nintendo PowerGlove (50 Hz capture rate, 1 hand).
- **Cleaning & Quality Assurance**:
  - Dropped non-sign calibration files (`cal-full-forward/right/up`).
  - Dropped 1 corrupted empty file.
  - Hex status flags discarded.
  - **Feature selection**: Analyzed 11 raw numeric channels:
    - Columns 4 & 5 are invariant/constant across all recordings (zero variance).
    - Column 10 is an exact duplicate of column 9.
    - Resulting 8 active channels: **POS (x, y, z)**, **ROT (roll)**, and **Finger Bends F1-F4**.
- **Final Cleaned Dataset**: **6,648 recordings**, **95 unique signs** (~70 examples per sign).

---

## 3. Feature Engineering & Preprocessing

1. **Temporal Resampling (FFT-based)**:
   - Signs have variable lengths; resampled along the time axis to uniform $T = 57$ frames (the empirical dataset average length).
2. **Spatial Feature Scaling**:
   - Evaluated both **per-example** min-max scaling ($[0, 1]$ per channel per sign) and **global** dataset min-max scaling.
3. **Representations**:
   - **Static / Flattened Representation**: $57 \times 8 = 456$-dimensional feature vector for traditional ML baselines.
   - **Temporal Sequential Representation**: $(N, 57, 8)$ tensor preserving time step order for Recurrent Neural Networks.

---

## 4. Baseline Models & Findings

- Evaluated using a strict **70/30 stratified train/test split** (random seed 42):

| Model | Representation / Scaling | Test Accuracy | Macro F1 | Paper F1 (Reference) |
|---|---|---|---|---|
| Linear SVM ($C=1.0$) | Flattened / Per-example | 0.306 | 0.308 | - |
| Linear SVM ($C=0.1$) | Flattened / Per-example | 0.369 | 0.365 | 0.549 |
| Logistic Regression | Flattened / Per-example | 0.385 | 0.384 | 0.436 |
| RBF SVM ($C=10$) | Flattened / Global | 0.576 | 0.575 | - |
| **RBF SVM ($C=10$)** | **Flattened / Per-example** | **0.602** | **0.601** | **0.549** |

- **Key Takeaway**: Non-linear RBF kernel outperforms linear baselines substantially (+21.7% accuracy over LR). Per-example scaling preserves local shape dynamics best.

---

## 5. Visualizations & Error Analysis

- **3D PCA Projection**:
  - First 3 principal components capture trajectory clusters; substantial overlap between kinematically adjacent signs.
- **Confusion Matrix Analysis**:
  - Model excels on signs with distinctive trajectory extrema.
  - Primary confusions occur between signs with identical hand position but subtle finger curl nuances:
    - `man` $\leftrightarrow$ `please`
    - `which` $\leftrightarrow$ `maybe`
    - `surprise` $\leftrightarrow$ `more`
    - `exit` $\leftrightarrow$ `you`

---

## 6. Feature Ablation Study

Systematic removal of sensor channels to determine predictive importance:

| Ablation Condition | Channels Retained | Impact on Classification |
|---|---|---|
| **Full Set (All 8 channels)** | POS (0,1,2), ROT (3), F1-F4 (6-9) | **Baseline Test Err: 39.8% (Acc: 60.2%)** |
| *Without Position (POS)* | ROT + Fingers F1-F4 | **Test Err jumps to 75.6% (+35.8% error)** |
| *Without Rotation (ROT)* | POS + Fingers F1-F4 | Minor error change (test error ~42%) |
| *Without Individual Fingers* | POS + ROT + 3 Fingers | Incremental degradation (~2-4%) |

- **Insight**: 3D spatial trajectory (`POS`) is by far the most informative modality for gross sign discrimination; finger sensors provide fine-grained separation.

---

## 7. Deep Learning: Resolving the LSTM Failure

- **Why the Paper's LSTM Failed ($F1 = 0.066$)**:
  - Original authors used Mean Squared Error (MSE) backpropagated at *every* frame.
  - Sign language recognition is a **sequence-to-label** task, not frame-wise regression!
- **Our Solution**:
  - Enforced `return_sequences=False` (loss computed only at final step $h_T$).
  - 95-way Softmax Sparse Categorical Cross-Entropy + Dropout ($p=0.3$).
- **Empirical Results**:
  | Model Config | Test Acc | Macro F1 | vs. Paper F1 (0.066) |
  |---|---|---|---|
  | Paper LSTM Baseline | — | 0.066 | Reference |
  | Baseline LSTM (128) | 0.383 | 0.370 | **+5.6x improvement** |
  | Bidirectional LSTM (128) | 0.437 | 0.431 | **+6.5x improvement** |
  | **Stacked LSTM (2x128)** | **0.447** | **0.440** | **+6.7x improvement** |
- **Conclusion**: Resolves paper's hypothesis; RBF SVM (0.601) remains superior due to high sample-efficiency on ~49 samples/class.


---

## 8. Conclusions & Summary

1. **Successful Reproduction & Outperformance**:
   - Replicated Cate et al.'s findings while outperforming their SVM benchmark (Macro F1: **0.601** vs **0.549**).
2. **Solved Temporal Recurrent Modeling**:
   - Identified and corrected the architectural flaw in the literature's LSTM approach.
3. **Data Constraint Insight**:
   - On low-sample multi-class regimes (~49 training signs/class), regularized RBF SVM provides exceptional parameter efficiency compared to deep neural networks.
4. **Modality Hierarchy**:
   - Global Cartesian position is critical; finger sensors resolve ambiguity between geometrically collocated signs.
