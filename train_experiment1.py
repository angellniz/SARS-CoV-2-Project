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

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
K             = 4
FEAT_DIR      = "features"
RESULTS_DIR   = "results_exp1"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
def load(split):
    X = sparse.load_npz(os.path.join(FEAT_DIR, f"kmer_k{K}_{split}.npz")).tocsr()
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
    # Linear SVM on standardized features.
    # We standardize (with_mean=False because data is sparse but we
    # densify in the pipeline to keep StandardScaler simple).
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
def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    labels_sorted = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    report = classification_report(y_true, y_pred, zero_division=0)

    print(f"\n--- {name} ---")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Macro F1         : {f_macro:.4f}")
    print(f"Weighted F1      : {f_weight:.4f}")
    print(f"Macro Precision  : {p_macro:.4f}")
    print(f"Macro Recall     : {r_macro:.4f}")
    print("\nClassification report:")
    print(report)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(cm)

    return {
        "name": name,
        "accuracy": float(acc),
        "macro_f1": float(f_macro),
        "weighted_f1": float(f_weight),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "confusion_matrix": cm.tolist(),
        "labels": [str(x) for x in labels_sorted],
        "report": report,
    }

# ------------------------------------------------------------------
# Run on both splits
# ------------------------------------------------------------------
all_results = {"k": K, "splits": {}}

for split_name in ["random", "time"]:
    print(f"\n\n########## {split_name.upper()} SPLIT ##########")

    if split_name == "random":
        X_train, y_train = load("train")
        X_test,  y_test  = load("test")
    else:
        X_train, y_train = load("train_time")
        X_test,  y_test  = load("test_time")

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # RF
    print("\n=== Training Random Forest ===")
    rf = make_rf()
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_result = evaluate("Random Forest", y_test, rf_pred)

    # Linear SVM
    print("\n=== Training Linear SVM ===")
    svm = make_svm()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_result = evaluate("Linear SVM", y_test, svm_pred)

    all_results["splits"][split_name] = {
        "train_size": int(X_train.shape[0]),
        "test_size":  int(X_test.shape[0]),
        "random_forest": rf_result,
        "linear_svm": svm_result,
    }

    np.save(os.path.join(RESULTS_DIR, f"y_test_{split_name}_k{K}.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, f"rf_pred_{split_name}_k{K}.npy"), rf_pred)
    np.save(os.path.join(RESULTS_DIR, f"svm_pred_{split_name}_k{K}.npy"), svm_pred)

# ------------------------------------------------------------------
# Save JSON summary
# ------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, f"experiment1_k{K}.json"), "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll results saved to {RESULTS_DIR}/")