# Next Steps (Handoff for Third Sprint / Final Push)

## Status: What Was Completed in this Sprint

Branch: `lstm` (Ready for PR into `main`)

1. **Environment & Data Setup:**
   - Established Python 3.11 virtual environment (`.venv`) ensuring full compatibility with TensorFlow 2.21, scikit-learn, scipy, pandas, matplotlib, and reportlab.
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
   - Built automated PDF compilation script `docs/generate_pdf.py` using ReportLab, outputting `docs/writeup.pdf`.
4. **Task 3: Slides & Interactive Demo (`docs/slides.md`, `src/demo.py`):**
   - Formatted an 8-slide presentation in Marp/markdown covering every stage of the project.
   - Implemented `src/demo.py` with fast model caching (`results/svm_model.joblib`), supporting single `.sign` file prediction and `--random` sign classification with top-3 class confidences.

---

## Remaining Tasks for Next Sprint

### 1. Verification & Submission Fill-ins
- **Team Information**: Update placeholders `<Name 1 (SRN)>` and `<Name 2 (SRN)>` in:
  - `README.md`
  - `docs/writeup.md`
  - `docs/generate_pdf.py`
  - `docs/slides.md`
- **Recompile PDF**:
  ```bash
  python docs/generate_pdf.py
  ```
- **Verify Clean PDF**: Check `docs/writeup.pdf` to ensure formatting, margins, and content fit cleanly within 2 pages as required by faculty guidelines.

### 2. Optional Stretch: Sequential Pattern Mining (`src/spm_model.py`)
If seeking bonus technical marks or fulfilling Section 4.3 of the paper:
- **Approach**:
  1. Discretize each of the 8 continuous channels into symbolic tokens (e.g., $k=5$ quantile bins or SAX representation per frame).
  2. Mine frequent sequential patterns across sign sequences (e.g., using `prefixspan` or Apriori-like sequential pattern mining).
  3. Perform Chi-Square ($\chi^2$) ranking to select the top $K$ discriminative sequential patterns.
  4. Encode each sign recording as a binary presence vector of length $K$.
  5. Train and evaluate an SVM classifier on this binary feature space.
- *Note:* The paper's own SPM implementation scored F1 $\approx 0.065$, so treat this as exploratory and benchmark against the RBF SVM ($0.601$) and Stacked LSTM ($0.440$).

### 3. Demo Rehearsal & Presentation Prep
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

---

## Git Workflow: Merging this Push

To merge the `lstm` branch into `main`:

```bash
# 1. Inspect git status on lstm branch
git status

# 2. Checkout main and merge
git checkout main
git merge lstm

# 3. Push main to origin
git push origin main
```

Or open a Pull Request on GitHub:
- **Head branch**: `lstm`
- **Base branch**: `main`
- **PR Title**: `Add LSTM Sequence Classifier, Ablation Study, Write-up, Slides, and Demo`