# Project Handoff (Final Submission)

**Team:** Arun Hariharan (PES2UG24AM126), Pavan Kishore (PES2UG24AM111)

## Completed

The project is on `main`; the previous instructions to merge the `lstm` branch are obsolete.

1. **Environment & Data Setup:**
   - Extracted and verified the low-quality Auslan dataset (6,648 `.sign` files across 95 classes, ~70 per class).
2. **Task 1: LSTM & Recurrent Model Pipeline (`src/preprocess.py`, `src/lstm_model.py`):**
   - Implemented `preprocess_sequences` in `src/preprocess.py` returning `(N, 57, 8)` temporal sequences.
   - Implemented `src/lstm_model.py` to fix the paper's failed formulation (`return_sequences=False` with categorical cross-entropy and softmax).
   - Executed systematic experiment suite across 8 configurations:
     - Baseline LSTM (128): Acc `0.383`, Macro F1 `0.370` (**5.6x improvement** over paper's `0.066`).
     - Bidirectional LSTM (128): Acc `0.437`, Macro F1 `0.431`.
     - Stacked LSTM (2x128): Acc `0.447`, Macro F1 `0.440` (**6.7x improvement** over paper).
     - Hidden 64, Hidden 256, GRU, Dropout 0.0, Global Scaling benchmarks.
   - Saved training curves to `results/lstm_training.png` and tabular report to `results/lstm_results.md`.
3. **Task 2: Project Write-up (`docs/writeup.md`, `docs/writeup.pdf`):**
   - Detailed two-page formal report covering problem statement, dataset discrepancies (broken UCI high-quality links), data cleaning, feature engineering, baseline vs. literature comparison, ablation study, and LSTM diagnosis.
   - Added both team members' names and SRNs to the README, report, PDF, and slides.
   - Built automated PDF compilation script `docs/generate_pdf.py` using ReportLab; added ReportLab to `requirements.txt`, regenerated the PDF, and verified that it is two pages.
4. **Task 3: Slides & Interactive Demo (`docs/slides.md`, `src/demo.py`):**
   - Formatted an 8-slide presentation in Marp/markdown covering every stage of the project.
   - Implemented `src/demo.py` with fast model caching (`results/svm_model.joblib`), supporting single `.sign` file prediction and `--random` sign classification with top-3 class confidences.
5. **Task 4: Interactive Web Application & 3D Motion Lab (`app.py`, `src/server.py`, `web/`):**
   - Built a modern web interface and REST API (`http://localhost:5001`).
   - Features:
     - Real-time Auslan sign classification with Top-5 probability breakdown.
     - Interactive 3D hand spatial trajectory visualizer (Three.js) with play/pause, scrub slider, and orbit controls.
     - Synchronized multi-channel time-series charts (Chart.js) for 8 sensor streams (X, Y, Z, Roll, Thumb, Forefinger, Middle, Ring).
     - Sign and signer selectors, random sign sampler, and custom `.sign` file upload / text paste.
     - Comprehensive model benchmarks dashboard (classical models + 8 RNN variants vs paper baseline).
     - Feature modality ablation explorer and error analysis with confusion matrix & 3D PCA viewers.
     - Complete 95 Auslan signs dictionary browser with instant search.
6. **Sequential Pattern Mining (`src/spm_model.py`):**
   - Implemented SPM feature extraction, chi-square pattern selection, and SVM evaluation. Reused the final SPM feature matrices for both classifiers and added progress output.
   - Tuned five configurations on an inner validation split; the selected config was `both`, window 20, max length 3, and 2,000 patterns (validation F1 `0.619`).
   - Final 70/30 held-out test results are in `results/spm_tuned.md`: SPM + linear SVM Macro F1 `0.648`; SPM + flattened features + RBF SVM Macro F1 `0.653`.

## Remaining

### 1. Verify Reproducible Environment
- The current `.venv` reports Python 3.14, while the README setup targets Python 3.11 or 3.12. Recreate a supported environment before rerunning TensorFlow experiments, or update the documented support after validating the newer runtime.

### 2. Demo Rehearsal & Presentation Prep
- Test the demo script during practice:
  ```bash
  python src/demo.py --random
  python src/demo.py --file data/raw/low_quality/extracted/signs/john4/joke1.sign
  ```
- Review the presentation flow using `docs/slides.md`. Both team members should be ready to explain:
  - Why high-quality data was unusable (UCI broken symlinks / 403).
  - Why RBF kernel gave massive gains over linear SVM.
  - The feature ablation finding (removing position `POS` drops accuracy from 60.2% to 24.4%).
  - Exactly why the paper's LSTM failed and how our sequence-to-label formulation fixed it.

### 3. Final Review & Git
- Review the report, slides, results, and assignment-specific submission requirements.
- Review `git status` and the diff before staging. The current worktree contains uncommitted changes and untracked files; inspect each untracked file and stage only intended project deliverables.
- Commit and push the approved changes to `main` when ready. No branch merge is currently pending.

### 4. SPM Test Coverage
- `tests/test_spm.py` is currently empty. Add focused tests for interval extraction, pattern matching/mining, and feature transformation if the SPM code will be maintained or extended.