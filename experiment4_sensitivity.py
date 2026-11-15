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
HOLDOUT_FRAC   = 0.20
THRESHOLD_PCT  = 5
NU_VALUES      = [0.01, 0.05, 0.10]

os.makedirs(RESULTS_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
X_all = sparse.load_npz(os.path.join(FEAT_DIR, f"kmer_k{K}_train.npz")).tocsr()
y_all = np.load(os.path.join(FEAT_DIR, "labels_train.npy"), allow_pickle=True)

lineages = sorted(set(y_all))
print(f"Loaded X_train: {X_all.shape}")
print(f"Lineages: {lineages}")
print(f"nu values: {NU_VALUES}")

scaler = StandardScaler(with_mean=False)
X_all_std = scaler.fit_transform(X_all)

# ------------------------------------------------------------------
# Pre-compute the train/test index splits ONCE (fixed seed)
# so that all nu values see the exact same data partitions.
# ------------------------------------------------------------------
splits = {}
for held_out in lineages:
    known_mask = (y_all != held_out)
    y_known = y_all[known_mask]
    X_known = X_all_std[known_mask]
    X_unknown = X_all_std[~known_mask]

    known_lineages = [l for l in lineages if l != held_out]
    train_idx, test_known_idx = [], []
    for lin in known_lineages:
        idx = np.where(y_known == lin)[0]
        rng_local = np.random.default_rng(RANDOM_SEED + hash(lin) % 1000)
        rng_local.shuffle(idx)
        n_hold = max(1, int(round(len(idx) * HOLDOUT_FRAC)))
        test_known_idx.extend(idx[:n_hold].tolist())
        train_idx.extend(idx[n_hold:].tolist())

    splits[held_out] = {
        "X_train_oc":   X_known[train_idx],
        "X_test_known": X_known[test_known_idx],
        "X_unknown":    X_unknown,
        "known_lineages": known_lineages,
    }

# ------------------------------------------------------------------
# Run for each nu
# ------------------------------------------------------------------
all_results = {}

for nu in NU_VALUES:
    print(f"\n\n########## nu = {nu} ##########")
    all_results[nu] = {}

    for held_out in lineages:
        s = splits[held_out]

        oc = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        oc.fit(s["X_train_oc"])

        scores_train   = oc.decision_function(s["X_train_oc"])
        scores_known   = oc.decision_function(s["X_test_known"])
        scores_unknown = oc.decision_function(s["X_unknown"])

        threshold = np.percentile(scores_train, THRESHOLD_PCT)

        flag_rate_known   = float((scores_known   < threshold).mean())
        flag_rate_unknown = float((scores_unknown < threshold).mean())

        y_true  = np.concatenate([np.zeros(len(scores_known)),
                                  np.ones (len(scores_unknown))])
        y_score = np.concatenate([-scores_known, -scores_unknown])
        auc = roc_auc_score(y_true, y_score)

        all_results[nu][held_out] = {
            "held_out":          held_out,
            "nu":                nu,
            "auc":               float(auc),
            "flag_rate_known":   flag_rate_known,
            "flag_rate_unknown": flag_rate_unknown,
            "mean_score_known":  float(scores_known.mean()),
            "mean_score_novel":  float(scores_unknown.mean()),
            "threshold":         float(threshold),
            "n_train_oc":        int(s["X_train_oc"].shape[0]),
            "n_test_known":      int(s["X_test_known"].shape[0]),
            "n_test_unknown":    int(s["X_unknown"].shape[0]),
        }

        print(f"  {held_out:>14s} | AUC={auc:.4f} | "
              f"FPR={flag_rate_known:.4f} | TPR={flag_rate_unknown:.4f}")

# ------------------------------------------------------------------
# Summary table: rows = held-out lineage, cols = nu values (AUC only)
# ------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("SENSITIVITY ANALYSIS — AUC by held-out lineage and nu")
print("=" * 80)
header = f"{'held_out':>14s} | " + " | ".join(f"{'nu='+str(nu):>10s}" for nu in NU_VALUES)
print(header)
print("-" * len(header))
for held_out in lineages:
    row = f"{held_out:>14s} | " + " | ".join(
        f"{all_results[nu][held_out]['auc']:>10.4f}" for nu in NU_VALUES
    )
    print(row)

# ------------------------------------------------------------------
# Save JSON
# ------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "experiment4_sensitivity.json"),
          "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

# ------------------------------------------------------------------
# Save CSV (long form: one row per (nu, held_out))
# ------------------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, "experiment4_sensitivity.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["nu","held_out","auc","flag_rate_known","flag_rate_unknown",
                     "mean_score_known","mean_score_novel","threshold",
                     "n_train_oc","n_test_known","n_test_unknown"])
    for nu in NU_VALUES:
        for held_out in lineages:
            r = all_results[nu][held_out]
            writer.writerow([
                r["nu"], r["held_out"], r["auc"],
                r["flag_rate_known"], r["flag_rate_unknown"],
                r["mean_score_known"], r["mean_score_novel"],
                r["threshold"], r["n_train_oc"],
                r["n_test_known"], r["n_test_unknown"],
            ])

# ------------------------------------------------------------------
# Save summary TXT
# ------------------------------------------------------------------
lines = []
lines.append("SENSITIVITY ANALYSIS — NOVELTY DETECTION")
lines.append("=" * 80)
lines.append(f"k = {K}, threshold percentile = {THRESHOLD_PCT}")
lines.append("")
lines.append(header)
lines.append("-" * len(header))
for held_out in lineages:
    row = f"{held_out:>14s} | " + " | ".join(
        f"{all_results[nu][held_out]['auc']:>10.4f}" for nu in NU_VALUES
    )
    lines.append(row)
lines.append("")

# Per-lineage mini-tables for TPR
lines.append("Flag rates on held-out (novel) lineages (TPR):")
lines.append(header)
lines.append("-" * len(header))
for held_out in lineages:
    row = f"{held_out:>14s} | " + " | ".join(
        f"{all_results[nu][held_out]['flag_rate_unknown']:>10.4f}" for nu in NU_VALUES
    )
    lines.append(row)

with open(os.path.join(RESULTS_DIR, "experiment4_sensitivity_summary.txt"),
          "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\nSaved: {RESULTS_DIR}/experiment4_sensitivity.json")
print(f"Saved: {RESULTS_DIR}/experiment4_sensitivity.csv")
print(f"Saved: {RESULTS_DIR}/experiment4_sensitivity_summary.txt")