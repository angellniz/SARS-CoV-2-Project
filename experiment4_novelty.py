import numpy as np
from scipy import sparse
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os
import json
import csv
from collections import defaultdict

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
K              = 4
FEAT_DIR       = "features"
RESULTS_DIR    = "results_exp4"
RANDOM_SEED    = 42
HOLDOUT_FRAC   = 0.20   # fraction of the 4 known lineages held out for evaluation
THRESHOLD_PCT  = 5      # threshold = 5th percentile of training decision scores

os.makedirs(RESULTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------------
# Load training features + labels
# ------------------------------------------------------------------
X_all = sparse.load_npz(os.path.join(FEAT_DIR, f"kmer_k{K}_train.npz")).tocsr()
y_all = np.load(os.path.join(FEAT_DIR, "labels_train.npy"), allow_pickle=True)

lineages = sorted(set(y_all))
print(f"Loaded X_train: {X_all.shape}")
print(f"Lineages: {lineages}")

# ------------------------------------------------------------------
# Standardize once (used by all One-Class SVMs)
# ------------------------------------------------------------------
scaler = StandardScaler(with_mean=False)
X_all_std = scaler.fit_transform(X_all)

# ------------------------------------------------------------------
# Run one-vs-rest novelty detection for each lineage
# ------------------------------------------------------------------
all_results = {}

for held_out in lineages:
    print(f"\n=== Hold-out: {held_out} ===")

    known_mask   = (y_all != held_out)
    unknown_mask = (y_all == held_out)

    X_known   = X_all_std[known_mask]
    y_known   = y_all[known_mask]
    X_unknown = X_all_std[unknown_mask]

    # Split the known set into "train OC-SVM" and "held-known for testing"
    known_lineages = [l for l in lineages if l != held_out]
    n_train_per_class = {}
    train_idx, test_known_idx = [], []

    for lin in known_lineages:
        idx = np.where(y_known == lin)[0]
        rng.shuffle(idx)
        n_hold = max(1, int(round(len(idx) * HOLDOUT_FRAC)))
        test_known_idx.extend(idx[:n_hold].tolist())
        train_idx.extend(idx[n_hold:].tolist())

    X_train_oc   = X_known[train_idx]
    X_test_known = X_known[test_known_idx]

    print(f"  Train OC-SVM: {X_train_oc.shape[0]} genomes "
          f"(from {len(known_lineages)} lineages)")
    print(f"  Test known  : {X_test_known.shape[0]} genomes")
    print(f"  Test novel  : {X_unknown.shape[0]} genomes ({held_out})")

    # Fit One-Class SVM
    oc = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05,
    )
    oc.fit(X_train_oc)

    # Decision scores: positive = "in-distribution", negative = "out"
    scores_train   = oc.decision_function(X_train_oc)
    scores_known   = oc.decision_function(X_test_known)
    scores_unknown = oc.decision_function(X_unknown)

    # Threshold at the THRESHOLD_PCT percentile of training scores
    threshold = np.percentile(scores_train, THRESHOLD_PCT)

    # Predicted labels: True = novelty (score below threshold)
    pred_known_novel   = (scores_known   < threshold)
    pred_unknown_novel = (scores_unknown < threshold)

    # AUC: score held-out lineage as positive (novel) vs known as negative.
    # We use -scores so that "novel" has higher value = lower decision score.
    y_true  = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
    y_score = np.concatenate([-scores_known, -scores_unknown])
    auc = roc_auc_score(y_true, y_score)

    # Flag rates
    flag_rate_known   = float(pred_known_novel.mean())
    flag_rate_unknown = float(pred_unknown_novel.mean())

    print(f"  Threshold (5th pct of train scores): {threshold:.6f}")
    print(f"  Mean decision score, known  : {scores_known.mean():.4f}")
    print(f"  Mean decision score, novel  : {scores_unknown.mean():.4f}")
    print(f"  AUC (novel vs known)        : {auc:.4f}")
    print(f"  Flag rate, known  (FPR)     : {flag_rate_known:.4f}")
    print(f"  Flag rate, novel  (TPR)     : {flag_rate_unknown:.4f}")

    all_results[held_out] = {
        "held_out":             held_out,
        "known_lineages":       known_lineages,
        "n_train_oc":           int(X_train_oc.shape[0]),
        "n_test_known":         int(X_test_known.shape[0]),
        "n_test_unknown":       int(X_unknown.shape[0]),
        "threshold":            float(threshold),
        "mean_score_known":     float(scores_known.mean()),
        "mean_score_novel":     float(scores_unknown.mean()),
        "auc":                  float(auc),
        "flag_rate_known":      flag_rate_known,
        "flag_rate_unknown":    flag_rate_unknown,
    }

# ------------------------------------------------------------------
# Compact summary
# ------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("NOVELTY DETECTION SUMMARY (leave-one-lineage-out)")
print("=" * 80)
print(f"{'held_out':>14s} | {'AUC':>7s} | {'flag_known':>11s} | {'flag_novel':>11s} | {'mean_nov - mean_known':>22s}")
print("-" * 80)

for held_out in lineages:
    r = all_results[held_out]
    diff = r["mean_score_novel"] - r["mean_score_known"]
    print(f"{held_out:>14s} | {r['auc']:>7.4f} | {r['flag_rate_known']:>11.4f} | "
          f"{r['flag_rate_unknown']:>11.4f} | {diff:>22.4f}")

# ------------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "experiment4_novelty.json"),
          "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

# ------------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "experiment4_novelty.csv"),
          "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["held_out","auc","flag_rate_known","flag_rate_novel",
                     "mean_score_known","mean_score_novel",
                     "n_train","n_test_known","n_test_novel","threshold"])
    for held_out in lineages:
        r = all_results[held_out]
        writer.writerow([
            held_out, r["auc"], r["flag_rate_known"], r["flag_rate_unknown"],
            r["mean_score_known"], r["mean_score_novel"],
            r["n_train_oc"], r["n_test_known"], r["n_test_unknown"], r["threshold"],
        ])

print(f"\nSaved: {RESULTS_DIR}/experiment4_novelty.json")
print(f"Saved: {RESULTS_DIR}/experiment4_novelty.csv")