# Sign Language Recognition using Temporal Classification
**Course:** UE24CS352A Machine Learning Mini-Project (Problem 118)  
**Authors:** Arun Hariharan (PES2UG24AM126), Pavan Kishore (PES2UG24AM111)  
**Reference Paper:** Cate, Dalvi, Hussain. *Sign Language Recognition using Temporal Classification* (2015).

---

## 1. Problem Statement
Sign language recognition bridges communication between hearing and deaf communities. While vision-based recognition is sensitive to occlusion, lighting shifts, and camera perspective, wearable sensor-based capture directly records kinematic state (hand position, rotation, and finger bend) with high temporal resolution.

This project reproduces and extends the work of Cate et al. (2015), which addresses the classification of **95 Australian Sign Language (Auslan) signs** from glove-sensor multivariate time series. The original study observed that standard classifiers (such as linear SVM and logistic regression) attained decent performance on high-quality glove recordings but struggled on noisy, low-quality sensor streams. Furthermore, the paper's deep learning baseline (an LSTM recurrent neural network) completely failed, yielding an F1 score of only 0.066. Our objectives are to:
1. Re-implement and benchmark the baseline classifiers on the low-quality sensor dataset.
2. Conduct systematic feature engineering, spatial/temporal normalization, and modality ablation.
3. Diagnose and resolve the fundamental architectural shortcoming behind the paper's failed LSTM classifier.

---

## 2. Dataset & Cleaning Pipeline
Experiments utilize Mohammed Waleed Kadous’s Auslan benchmark (UCI Machine Learning Repository #114).

### 2.1 Dataset Scope and Discrepancies
The UCI distribution includes two data tiers:
- **High-Quality Data:** Recorded at 200 Hz with dual 5DT gloves and Flock of Birds magnetic trackers (27 samples per sign from one signer). However, the UCI download archives (`tctodd.tar.gz` / `tctodd.tar.bz2`) are broken symlinks and their upstream server links return `403 Forbidden`. Consequently, conforming to realistic noisy conditions, all experimentation is conducted on the low-quality dataset.
- **Low-Quality Data:** Captured at 50 Hz using a single-handed Nintendo PowerGlove across multiple signers, featuring substantial intra-class variation.

### 2.2 Data Cleaning & Channel Pruning
Raw files contain comma-separated sensor rows with trailing hexadecimal status words. The cleaning pipeline implements:
1. **Exclusion of Non-Sign Files:** Dropped 3 calibration recordings (`cal-full-forward`, `cal-full-right`, `cal-full-up`) and 1 corrupted empty file.
2. **Channel Selection:** Analysis of the 11 raw numeric channels revealed that columns 4 and 5 are completely invariant (zero variance across all recordings), while column 10 is an exact duplicate of column 9. Retaining only active, independent physical measurements yields **8 features**:
   - `POS`: $x, y, z$ spatial coordinates (columns 0, 1, 2)
   - `ROT`: roll / wrist rotation (column 3)
   - `F1`–`F4`: bend sensors for thumb, index, middle, and ring fingers (columns 6, 7, 8, 9)
3. **Dataset Verification:** The cleaned corpus contains **6,648 recordings** spanning **95 classes** (69–70 instances per sign, average duration 58.4 frames).

---

## 3. Methodology & Approach

### 3.1 Preprocessing & Representations
Because sign durations vary significantly (range: 10 to >200 frames), signals are temporally standardized:
1. **Temporal Resampling:** Each time series is resampled along the time axis to $T = 57$ frames (the empirical dataset mean) using Fourier-method resampling (`scipy.signal.resample`).
2. **Spatial Feature Scaling:** We contrast **per-example min-max scaling** (normalizing each feature independently within $[0, 1]$ per sign) against **global min-max scaling** across the entire dataset.
3. **Model Representations:**
   - *Flattened Vector:* Each recording becomes a 456-dimensional vector ($57 \text{ frames} \times 8 \text{ features}$) for classical estimators (SVM, Logistic Regression).
   - *Sequential Tensor:* Shape $(N, 57, 8)$ preserved for recurrent sequence modeling.

### 3.2 Resolving the LSTM Formulation
The paper reported an LSTM macro F1 of 0.066, attributing failure to backpropagation at every discrete time step. Crucially, their setup formulated sign recognition as a regression problem with Mean Squared Error (MSE) evaluated at each frame. For whole-gesture classification, this is fundamentally misaligned: intermediate hand configurations are often non-discriminative or identical across gestures.

We rectify this by:
- Enforcing sequence-to-label aggregation via `return_sequences=False`, computing gradients strictly from the final hidden state vector $h_T$.
- Replacing MSE with **Softmax cross-entropy** across the 95 categorical sign classes.
- Incorporating dropout regularization ($p=0.3$) and early stopping based on validation loss.

---

## 4. Empirical Evaluation & Results

All models were evaluated under a rigorous **70% train / 30% test stratified split** (seed 42, 4,653 training samples, 1,995 test samples).

### 4.1 Baseline Classifiers vs. Reference Literature
| Model | Representation / Scaling | Test Accuracy | Macro F1 | Paper Macro F1 |
|---|---|---|---|---|
| Linear SVM ($C=1.0$) | Flattened / Per-example | 0.306 | 0.308 | — |
| Linear SVM ($C=0.1$) | Flattened / Per-example | 0.369 | 0.365 | 0.549 |
| Logistic Regression | Flattened / Per-example | 0.385 | 0.384 | 0.436 |
| RBF SVM ($C=10$) | Flattened / Global | 0.576 | 0.575 | — |
| **RBF SVM ($C=10$)** | **Flattened / Per-example** | **0.602** | **0.601** | **0.549** |

*Key Findings:* While linear classifiers plateau below 39% accuracy, introducing non-linear feature mapping via the RBF kernel dramatically elevates performance to **60.2% accuracy / 0.601 Macro F1**, outperforming Cate et al.’s best reported SVM score (0.549). Per-example scaling proved superior to global scaling by preserving intra-gesture dynamic ranges despite sensor baseline drifts across signers.

### 4.2 Feature Modality Ablation
To understand which physical measurements drive sign separation, we conducted single-group and cumulative ablations on the RBF SVM baseline:

- **Single-Group Removal:**
  - Full feature set: **Test Error = 39.8%** (Accuracy = 60.2%)
  - Removing Position (`POS`, cols 0–2): **Test Error surges to 75.6%** ($\Delta = +35.8\%$)
  - Removing Wrist Roll (`ROT`, col 3): Test Error = 41.8% ($\Delta = +2.0\%$)
  - Removing individual fingers (`F1`–`F4`): Test Error ranges between 38.4% and 44.4%
- **Cumulative Degradation:**  
  Sequentially dropping channels progressively collapses classifier capacity:
  $\text{All} \xrightarrow{-\text{POS}} 75.6\% \xrightarrow{-\text{ROT}} 81.7\% \xrightarrow{-\text{F1}} 82.8\% \xrightarrow{-\text{F2}} 90.0\% \xrightarrow{-\text{F3}} 95.6\% \text{ error}$.

*Interpretation:* Global 3D spatial trajectory is the dominant discriminator. Finger bend provides subtle discriminative boundaries between signs sharing identical gross arm trajectories.

### 4.3 Error Analysis: Most Confused Signs
Confusion matrix inspection identifies the top error pairs:
1. `man` $\rightarrow$ `please` (5 errors)
2. `which` $\rightarrow$ `maybe` (5 errors)
3. `surprise` $\rightarrow$ `more` (5 errors)
4. `exit` $\rightarrow$ `you` (4 errors), `hello` $\rightarrow$ `forget` (4 errors), `hear` $\rightarrow$ `crazy` (4 errors)  
These pairs share nearly identical gross arm motion, differing solely in subtle thumb/index positioning that is easily masked by PowerGlove quantization noise.

### 4.4 LSTM Model Performance & Architectural Study
Contrasting with the paper's near-zero performance (F1 = 0.066), our sequence-level recurrent architectures learn meaningful temporal representations:

| Architecture / Config | Scaling | Units | Dropout | Stacked | Test Accuracy | Macro F1 | Paper F1 (Reference) |
|---|---|---|---|---|---|---|---|
| Baseline LSTM | Per-example | 128 | 0.3 | No | 0.383 | 0.370 | 0.066 |
| Hidden 64 | Per-example | 64 | 0.3 | No | 0.350 | 0.339 | — |
| Hidden 256 | Per-example | 256 | 0.3 | No | 0.409 | 0.396 | — |
| GRU | Per-example | 128 | 0.3 | No | 0.334 | 0.315 | — |
| Bidirectional LSTM | Per-example | 128 | 0.3 | No | 0.437 | 0.431 | — |
| No Dropout | Per-example | 128 | 0.0 | No | 0.347 | 0.332 | — |
| **Stacked LSTM (2x128)** | Per-example | 128 | 0.3 | **Yes** | **0.447** | **0.440** | — |
| Global Scaling LSTM | Global | 128 | 0.3 | No | 0.380 | 0.368 | — |

*Discussion:*
1. **Validation of the Structural Fix:** Moving from per-step MSE to final-state softmax cross-entropy (`return_sequences=False`) elevates macro F1 from **0.066 to 0.440** (nearly 7x improvement), definitively validating the paper's hypothesis.
2. **Architecture Impact:** Stacked LSTM (0.440 F1) and Bidirectional LSTM (0.431 F1) demonstrate that hierarchical temporal feature extraction and bidirectional context (combining stroke onset and terminus) are beneficial for gesture dynamics.
3. **Overfitting & Sample Efficiency:** Without dropout, performance drops by ~3.8% F1 due to overfitting. Even the best recurrent network (0.440 F1) trails the flattened RBF SVM (0.601 F1). With only ~49 training instances per class, non-parametric maximum-margin kernel methods exhibit superior sample efficiency over high-capacity deep networks.

---

## 5. Conclusions
1. **Replication & Advancement:** Successfully reproduced and improved upon Cate et al.'s benchmark, achieving **60.1% Macro F1** on low-quality PowerGlove data.
2. **Resolution of Literature Failure:** Proved that the failure of LSTM in the reference literature was a consequence of per-frame MSE loss formulation rather than an inherent limitation of recurrent networks for sign recognition.
3. **Sensor Modality Hierarchy:** 3D hand position accounts for the vast majority of classification power, while finger bend sensors resolve ambiguities between kinematically collocated gestures.
4. **Model Selection Trade-off:** In data-constrained multi-class settings (~70 samples/class), kernelized SVMs on Fourier-resampled representations match or exceed deep recurrent architectures while training orders of magnitude faster.
