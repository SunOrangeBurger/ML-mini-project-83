"""
Sequential Pattern Mining (SPM) + SVM for multivariate time-series classification.

Follows Batal et al. (2009) "Multivariate Time Series Classification with Temporal
Abstractions", as used in Section 4.3 of Cate, Dalvi, Hussain (2015):

  1. Discretization      each channel -> symbolic levels, consecutive equal symbols merged
                         into intervals (state = channel + level, e.g. "H:1")
  2. Candidate patterns  Apriori-style level-wise growth. A pattern is a chain of states
                         joined by a relation ("before" / "overlap"). Pruned by minimum
                         per-class support (anti-monotone, so Apriori pruning is valid).
  3. Binary vectors      chi-square ranks patterns, keep top MAX_PATTERNS, each recording
                         becomes a 0/1 vector (pattern present / absent) -> SVM.

Hyperparameters
  mode          "raw"   discretize signal value (levels low/mid/high or 5 levels)
                "delta" discretize rate of change (decreasing/steady/increasing)
                "both"  union of the two state sets
  n_levels      3 or 5 levels for raw mode
  window        max span (frames) between first and last state start of a pattern
  min_support   fraction of a class's training examples a pattern must appear in
                (pattern kept if it reaches this in ANY class)
  max_len       max number of states per pattern
  max_patterns  K patterns kept after chi-square ranking
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import uniform_filter1d
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

BEFORE, OVERLAP = 1, 0
REL_NAME = {BEFORE: "b", OVERLAP: "o"}

RAW_NAMES = {3: ["L", "M", "H"], 5: ["VL", "L", "M", "H", "VH"]}
DELTA_NAMES = ["D", "S", "I"]  # decreasing / steady / increasing


# ----------------------------------------------------------------------------
# 1. Discretization
# ----------------------------------------------------------------------------
def _encode(kind, ch, level):
    """kind 0 = raw, 1 = delta. Packs into one int."""
    return (kind * 100 + ch) * 10 + level


def decode_state(state):
    level = state % 10
    ch = (state // 10) % 100
    kind = state // 1000
    name = DELTA_NAMES[level] if kind == 1 else RAW_NAMES[_decode_levels.get("n", 3)][level]
    return f"{name}{'d' if kind else ''}:{ch}"


_decode_levels = {"n": 3}


def _runs(symbols):
    """Run-length encode a 1-D int array -> list of (start, end_inclusive, symbol)."""
    out = []
    start = 0
    for t in range(1, len(symbols) + 1):
        if t == len(symbols) or symbols[t] != symbols[start]:
            out.append((start, t - 1, int(symbols[start])))
            start = t
    return out


def extract_intervals(seq, mode="raw", n_levels=3, delta_eps=0.01, smooth=3):
    """
    seq: (T, C) array scaled to [0, 1]. Returns list of (start, end, state) sorted by
    (start, state, end). Consecutive equal symbols per channel are merged into one interval.
    """
    T, C = seq.shape
    intervals = []
    if mode in ("raw", "both"):
        thr = np.linspace(0, 1, n_levels + 1)[1:-1]
        lv = np.digitize(seq, thr)  # (T, C) in 0..n_levels-1
        for c in range(C):
            for s, e, l in _runs(lv[:, c]):
                intervals.append((s, e, _encode(0, c, l)))
    if mode in ("delta", "both"):
        sm = uniform_filter1d(seq, size=smooth, axis=0, mode="nearest") if smooth > 1 else seq
        d = np.gradient(sm, axis=0)
        lv = np.where(d < -delta_eps, 0, np.where(d > delta_eps, 2, 1))
        for c in range(C):
            for s, e, l in _runs(lv[:, c]):
                intervals.append((s, e, _encode(1, c, l)))
    intervals.sort(key=lambda iv: (iv[0], iv[2], iv[1]))
    return intervals


# ----------------------------------------------------------------------------
# 2. Pattern matching + mining
# ----------------------------------------------------------------------------
def pattern_occurs(ivs, pattern, window):
    """True if pattern = (states, rels) occurs in the interval list ivs."""
    states, rels = pattern
    # cur: last matched interval index -> latest first-start (later start = more window slack)
    cur = {i: iv[0] for i, iv in enumerate(ivs) if iv[2] == states[0]}
    for st_next, rel in zip(states[1:], rels):
        nxt = {}
        for last, first_start in cur.items():
            last_end = ivs[last][1]
            for j in range(last + 1, len(ivs)):
                s, _, st = ivs[j]
                if s - first_start > window:
                    break
                if st == st_next and (OVERLAP if s <= last_end else BEFORE) == rel:
                    if nxt.get(j, -1) < first_start:
                        nxt[j] = first_start
        cur = nxt
        if not cur:
            return False
    return True


def mine_patterns(inst_intervals, y, min_support=0.2, max_len=3, window=20,
                  max_candidates=5000, verbose=True):
    """
    Level-wise (Apriori) mining over all training instances at once. A pattern is kept
    if it occurs in >= min_support fraction of ANY single class.

    Returns dict: pattern -> np.ndarray of training-instance indices containing it.
    """
    y = np.asarray(y)
    n_classes = int(y.max()) + 1
    class_sizes = np.bincount(y, minlength=n_classes)
    need = np.maximum(1, np.ceil(min_support * class_sizes)).astype(int)

    def best_ratio(inst_ids):
        counts = np.bincount(y[inst_ids], minlength=n_classes)
        return (counts >= need).any(), (counts / class_sizes).max()

    def prune(kept):
        if len(kept) > max_candidates:
            kept = dict(sorted(kept.items(), key=lambda kv: -kv[1][2])[:max_candidates])
        return kept

    # level 1
    level = defaultdict(lambda: defaultdict(dict))
    for i, ivs in enumerate(inst_intervals):
        for idx, (s, _, st) in enumerate(ivs):
            level[((st,), ())][i][idx] = s
    kept = {}
    for pat, matches in level.items():
        ids = np.fromiter(matches.keys(), dtype=int)
        ok, ratio = best_ratio(ids)
        if ok:
            kept[pat] = (matches, ids, ratio)
    del level
    kept = prune(kept)

    result = {}
    k = 1
    while kept:
        for pat, (_, ids, _) in kept.items():
            result[pat] = ids
        if verbose:
            print(f"  length {k}: {len(kept)} frequent patterns", flush=True)
        if k >= max_len:
            break
        # Grow each parent and filter its children immediately (a child's key contains its
        # parent, so children of different parents never merge). Keeps memory bounded.
        nxt = {}
        for pat, (matches, _, _) in kept.items():
            states, rels = pat
            kids = defaultdict(lambda: defaultdict(dict))
            for i, ms in matches.items():
                ivs = inst_intervals[i]
                n = len(ivs)
                for last, fs in ms.items():
                    last_end = ivs[last][1]
                    for j in range(last + 1, n):
                        s, _, st = ivs[j]
                        if s - fs > window:
                            break
                        rel = OVERLAP if s <= last_end else BEFORE
                        d = kids[(st, rel)][i]
                        if d.get(j, -1) < fs:
                            d[j] = fs
            for (st, rel), km in kids.items():
                ids = np.fromiter(km.keys(), dtype=int)
                ok, ratio = best_ratio(ids)
                if ok:
                    nxt[(states + (st,), rels + (rel,))] = (km, ids, ratio)
            if len(nxt) > 2 * max_candidates:
                nxt = prune(nxt)
        kept = prune(nxt)
        k += 1
    return result


def format_pattern(pattern):
    states, rels = pattern
    out = decode_state(states[0])
    for r, s in zip(rels, states[1:]):
        out += f" -{REL_NAME[r]}-> {decode_state(s)}"
    return out


# ----------------------------------------------------------------------------
# 3. Feature transformer
# ----------------------------------------------------------------------------
class SPMFeatures:
    """sklearn-style: fit_transform(train sequences, y) / transform(new sequences)."""

    def __init__(self, mode="raw", n_levels=3, delta_eps=0.01, window=20,
                 min_support=0.2, max_len=3, max_patterns=1000, max_candidates=5000,
                 verbose=True):
        self.mode, self.n_levels, self.delta_eps = mode, n_levels, delta_eps
        self.window, self.min_support, self.max_len = window, min_support, max_len
        self.max_patterns, self.max_candidates = max_patterns, max_candidates
        self.verbose = verbose
        _decode_levels["n"] = n_levels

    def _intervals(self, X):
        return [extract_intervals(s, self.mode, self.n_levels, self.delta_eps) for s in X]

    def fit_transform(self, X, y):
        t0 = time.time()
        ivs = self._intervals(X)
        if self.verbose:
            print(f"mining ({len(X)} instances, avg {np.mean([len(v) for v in ivs]):.1f} intervals each)")
        mined = mine_patterns(ivs, y, self.min_support, self.max_len, self.window,
                              self.max_candidates, self.verbose)
        pats = list(mined.keys())
        if not pats:
            raise RuntimeError("No frequent patterns found. Lower min_support.")
        rows = np.concatenate([mined[p] for p in pats])
        cols = np.concatenate([np.full(len(mined[p]), j) for j, p in enumerate(pats)])
        B = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(len(X), len(pats)))
        scores, _ = chi2(B, y)
        scores = np.nan_to_num(scores)
        top = np.argsort(-scores)[: self.max_patterns]
        self.patterns_ = [pats[j] for j in top]
        self.scores_ = scores[top]
        if self.verbose:
            print(f"selected {len(self.patterns_)} / {len(pats)} patterns by chi2 "
                  f"({time.time() - t0:.1f}s)")
        return B[:, top].toarray()

    def transform(self, X):
        ivs = self._intervals(X)
        out = np.zeros((len(X), len(self.patterns_)), dtype=np.float32)
        # index patterns by first state so each instance only tests plausible patterns
        by_first = defaultdict(list)
        for j, p in enumerate(self.patterns_):
            by_first[p[0][0]].append(j)
        for i, iv in enumerate(ivs):
            present = {st for _, _, st in iv}
            for st in present:
                for j in by_first.get(st, ()):
                    if pattern_occurs(iv, self.patterns_[j], self.window):
                        out[i, j] = 1
        return out


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def make_svm(kind="linear", C=1.0):
    if kind == "linear":
        return LinearSVC(C=C, max_iter=20000)
    return SVC(kernel="rbf", C=C, gamma="scale")


def evaluate(clf, F_tr, y_tr, F_te, y_te):
    clf.fit(F_tr, y_tr)
    pred = clf.predict(F_te)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)
    return {
        "acc": accuracy_score(y_te, pred), "precision": p, "recall": r, "f1": f1,
        "train_err": 1 - accuracy_score(y_tr, clf.predict(F_tr)),
        "test_err": 1 - accuracy_score(y_te, pred),
    }


def run_config(Xs_tr, y_tr, Xs_te, y_te, cfg, svm="linear", C=1.0, flat_tr=None, flat_te=None,
               verbose=False):
    spm = SPMFeatures(verbose=verbose, **cfg)
    F_tr = spm.fit_transform(Xs_tr, y_tr)
    F_te = spm.transform(Xs_te)
    if flat_tr is not None:  # concatenate SPM features with raw flattened features
        F_tr = np.hstack([F_tr, flat_tr])
        F_te = np.hstack([F_te, flat_te])
    return evaluate(make_svm(svm, C), F_tr, y_tr, F_te, y_te), spm


def tune(Xs, y, grid, svm="linear", seed=42):
    """Pick the best config on an inner validation split of the TRAINING data only."""
    Xa, Xv, ya, yv = train_test_split(Xs, y, test_size=0.2, stratify=y, random_state=seed)
    rows = []
    for i, cfg in enumerate(grid, start=1):
        t0 = time.time()
        print(f"  tuning config {i}/{len(grid)}: {cfg}", flush=True)
        m, _ = run_config(Xa, ya, Xv, yv, cfg, svm=svm, verbose=True)
        rows.append((cfg, m))
        print(f"  {cfg} -> val F1={m['f1']:.3f} acc={m['acc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return max(rows, key=lambda r: r[1]["f1"])[0], rows


DEFAULT_GRID = [
    dict(mode="raw", n_levels=3, window=10, min_support=0.2, max_len=3, max_patterns=1000),
    dict(mode="raw", n_levels=5, window=10, min_support=0.2, max_len=3, max_patterns=1000),
    dict(mode="delta", window=10, min_support=0.2, max_len=3, max_patterns=1000),
    dict(mode="both", n_levels=3, window=10, min_support=0.2, max_len=3, max_patterns=1000),
    dict(mode="both", n_levels=3, window=20, min_support=0.2, max_len=3, max_patterns=2000),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true", help="grid search on inner validation split")
    ap.add_argument("--mode", default="both", choices=["raw", "delta", "both"])
    ap.add_argument("--levels", type=int, default=3, choices=[3, 5])
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--min-support", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=3)
    ap.add_argument("--max-patterns", type=int, default=1000)
    ap.add_argument("--svm", default="linear", choices=["linear", "rbf"])
    ap.add_argument("--out", default="results/spm_results.md")
    args = ap.parse_args()

    from load_data import load_low_quality
    from preprocess import preprocess_sequences

    COLS = [0, 1, 2, 3, 6, 7, 8, 9]
    signals, labels, _ = load_low_quality()
    Xs, y, classes = preprocess_sequences(signals, labels, COLS)  # (n, 57, 8), per-example 0-1
    print("sequences:", Xs.shape, "classes:", len(classes))
    Xs_tr, Xs_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, stratify=y, random_state=42)

    if args.tune:
        print("tuning on inner validation split...")
        cfg, tune_rows = tune(Xs_tr, y_tr, DEFAULT_GRID, svm=args.svm)
    else:
        cfg = dict(mode=args.mode, n_levels=args.levels, window=args.window,
                   min_support=args.min_support, max_len=args.max_len,
                   max_patterns=args.max_patterns)
        tune_rows = []
    print("final config:", cfg)

    flat_tr, flat_te = Xs_tr.reshape(len(Xs_tr), -1), Xs_te.reshape(len(Xs_te), -1)
    results = {}
    spm = SPMFeatures(verbose=True, **cfg)
    F_tr = spm.fit_transform(Xs_tr, y_tr)
    print(f"transforming {len(Xs_te)} test sequences...", flush=True)
    F_te = spm.transform(Xs_te)
    print(f"training {args.svm} SVM on SPM features...", flush=True)
    m = evaluate(make_svm(args.svm), F_tr, y_tr, F_te, y_te)
    results["SPM + SVM"] = m
    # paper's final analysis: SPM features concatenated with raw flattened features (RBF SVM)
    print("training RBF SVM on SPM + flattened features...", flush=True)
    m2 = evaluate(make_svm("rbf", C=10), np.hstack([F_tr, flat_tr]), y_tr,
                  np.hstack([F_te, flat_te]), y_te)
    results["SPM + flattened, RBF SVM"] = m2
    for k, v in results.items():
        print(f"{k}: acc={v['acc']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} train_err={v['train_err']:.3f} test_err={v['test_err']:.3f}")

    top = [(format_pattern(p), s) for p, s in zip(spm.patterns_[:10], spm.scores_[:10])]
    lines = ["# Sequential Pattern Mining Results", "",
             "Low-quality Auslan, 70/30 stratified split (seed 42). Paper's SPM: F1 = 0.065.", "",
             f"Config: `{cfg}`, SVM: `{args.svm}`", "",
             "| Model | Accuracy | Precision | Recall | Macro F1 | Train err | Test err |",
             "|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        lines.append(f"| {k} | {v['acc']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} | "
                     f"{v['f1']:.3f} | {v['train_err']:.3f} | {v['test_err']:.3f} |")
    if tune_rows:
        lines += ["", "## Tuning (inner validation split)", "", "| Config | Val acc | Val F1 |", "|---|---|---|"]
        for c, m_ in tune_rows:
            lines.append(f"| `{c}` | {m_['acc']:.3f} | {m_['f1']:.3f} |")
    lines += ["", "## Top 10 patterns by chi-square", "",
              "State `H:3` = channel 3 high; suffix `d` = rate-of-change state (D/S/I); "
              "`-b->` before, `-o->` overlap.", "", "| Pattern | chi2 |", "|---|---|"]
    lines += [f"| `{p}` | {s:.1f} |" for p, s in top]
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()