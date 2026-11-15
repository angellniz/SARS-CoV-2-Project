import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import os
import json
import time

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
K_VALUES    = [3, 4, 5, 6]
FEAT_DIR    = "features"
RESULTS_DIR = "results_exp2"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
def load(k, split):
    X = sparse.load_npz(os.path.join(FEAT_DIR, f"kmer_k{k}_{split}.npz")).tocsr()
    y = np.load(os.path.join(FEAT_DIR, f"labels_{split}.npy"), allow_pickle=True)
    return X, y

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
def make_rf():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

def make_svm():
    return make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=20000,
            random_state=42,
        ),
    )

# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------
def evaluate(y_true, y_pred):
    labels_sorted = sorted(set(y_true))
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    report = classification_report(y_true, y_pred, zero_division=0)
    return {
        "accuracy":         float(acc),
        "macro_f1":         float(f_macro),
        "weighted_f1":      float(f_weight),
        "macro_precision":  float(p_macro),
        "macro_recall":     float(r_macro),
        "confusion_matrix": cm.tolist(),
        "labels":           [str(x) for x in labels_sorted],
        "report":           report,
    }

# ------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------
all_results = {}

print("=" * 78)
print("EXPERIMENT 2 — K-MER SIZE SWEEP")
print("=" * 78)

for k in K_VALUES:
    print(f"\n\n########## k = {k} ##########")
    all_results[k] = {}

    for split_name in ["random", "time"]:
        t0 = time.time()

        if split_name == "random":
            X_train, y_train = load(k, "train")
            X_test,  y_test  = load(k, "test")
        else:
            X_train, y_train = load(k, "train_time")
            X_test,  y_test  = load(k, "test_time")

        # RF
        rf = make_rf()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_eval = evaluate(y_test, rf_pred)

        # SVM
        svm = make_svm()
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_eval = evaluate(y_test, svm_pred)

        t1 = time.time()

        all_results[k][split_name] = {
            "dim":          int(X_train.shape[1]),
            "train_size":   int(X_train.shape[0]),
            "test_size":    int(X_test.shape[0]),
            "seconds":      round(t1 - t0, 2),
            "random_forest": rf_eval,
            "linear_svm":    svm_eval,
        }

        print(f"  [{split_name:6s}] dim={X_train.shape[1]:5d}  "
              f"RF acc={rf_eval['accuracy']:.4f} F1={rf_eval['macro_f1']:.4f}  |  "
              f"SVM acc={svm_eval['accuracy']:.4f} F1={svm_eval['macro_f1']:.4f}  "
              f"({t1-t0:.1f}s)")

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "experiment2_kmer_sweep.json"),
          "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

# ------------------------------------------------------------------
# Compact summary table
# ------------------------------------------------------------------
lines = []
lines.append("EXPERIMENT 2 — K-MER SWEEP SUMMARY")
lines.append("=" * 100)
lines.append("")
header = (f"{'k':>2} | {'dim':>5} | {'split':>6} | "
          f"{'RF acc':>8} | {'RF F1':>8} | {'SVM acc':>8} | {'SVM F1':>8} | {'time (s)':>8}")
lines.append(header)
lines.append("-" * len(header))

for k in K_VALUES:
    for split_name in ["random", "time"]:
        r = all_results[k][split_name]
        lines.append(
            f"{k:>2} | {r['dim']:>5} | {split_name:>6} | "
            f"{r['random_forest']['accuracy']:>8.4f} | "
            f"{r['random_forest']['macro_f1']:>8.4f} | "
            f"{r['linear_svm']['accuracy']:>8.4f} | "
            f"{r['linear_svm']['macro_f1']:>8.4f} | "
            f"{r['seconds']:>8.2f}"
        )

lines.append("")
lines.append("Full per-class results and confusion matrices: results_exp2/experiment2_kmer_sweep.json")

summary = "\n".join(lines)
print()
print(summary)

with open(os.path.join(RESULTS_DIR, "experiment2_summary.txt"),
          "w", encoding="utf-8") as f:
    f.write(summary)

print(f"\nAll results saved to {RESULTS_DIR}/")