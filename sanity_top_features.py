import numpy as np
from scipy import sparse
from sklearn.feature_selection import f_classif
from collections import defaultdict
import os
import csv

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
K              = 4
FEAT_DIR       = "features"
RESULTS_DIR    = "results_exp3"
TOP_N          = 20

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load training data (k=4)
# ------------------------------------------------------------------
X_train = sparse.load_npz(os.path.join(FEAT_DIR, f"kmer_k{K}_train.npz")).tocsr()
y_train = np.load(os.path.join(FEAT_DIR, "labels_train.npy"), allow_pickle=True)

X = X_train.toarray()
print(f"X_train dense shape: {X.shape}")

# ------------------------------------------------------------------
# Per-feature F-statistic
# ------------------------------------------------------------------
F, p = f_classif(X, y_train)
print(f"F-statistic: min={F.min():.2f}  max={F.max():.2f}  median={np.median(F):.2f}")

# ------------------------------------------------------------------
# Top-N features
# ------------------------------------------------------------------
top_idx = np.argsort(F)[::-1][:TOP_N]
print(f"\nTop {TOP_N} discriminative 4-mer indices:")
print("  " + ", ".join(str(int(i)) for i in top_idx))

# ------------------------------------------------------------------
# Mean frequency per lineage for top features
# ------------------------------------------------------------------
lineages = sorted(set(y_train))
mean_per_lineage = {}
std_per_lineage  = {}

for lin in lineages:
    mask = (y_train == lin)
    mean_per_lineage[lin] = X[mask][:, top_idx].mean(axis=0)
    std_per_lineage[lin]  = X[mask][:, top_idx].std(axis=0)

# ------------------------------------------------------------------
# Print mean table
# ------------------------------------------------------------------
print("\nMean frequency of top features, per lineage:")
header = "feature idx | " + " | ".join(f"{lin:>11s}" for lin in lineages)
print(header)
print("-" * len(header))

for i, fidx in enumerate(top_idx):
    row = f"{fidx:>11d} | " + " | ".join(f"{mean_per_lineage[lin][i]:>11.6f}" for lin in lineages)
    print(row)

# ------------------------------------------------------------------
# Compute "spread" between lineage means, and "spread" within lineages.
# If the between-lineage spread is much larger than within-lineage std,
# the feature is genuinely discriminative at the mean level.
# ------------------------------------------------------------------
print("\nDiscriminative strength at the mean level:")
print(f"{'feature':>8s} | {'between_std':>12s} | {'avg_within_std':>15s} | {'ratio':>8s}")
print("-" * 55)

for i, fidx in enumerate(top_idx):
    means = np.array([mean_per_lineage[lin][i] for lin in lineages])
    between_std = means.std()
    within_std  = np.mean([std_per_lineage[lin][i] for lin in lineages])
    ratio = between_std / (within_std + 1e-12)
    print(f"{fidx:>8d} | {between_std:>12.6f} | {within_std:>15.6f} | {ratio:>8.4f}")

# ------------------------------------------------------------------
# Save table to CSV
# ------------------------------------------------------------------
out_path = os.path.join(RESULTS_DIR, "top_features_summary.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["feature_index", "F_statistic"] + [f"mean_{lin}" for lin in lineages]
                    + [f"std_{lin}" for lin in lineages])
    for i, fidx in enumerate(top_idx):
        writer.writerow(
            [int(fidx), float(F[fidx])]
            + [float(mean_per_lineage[lin][i]) for lin in lineages]
            + [float(std_per_lineage[lin][i])  for lin in lineages]
        )

print(f"\nSaved: {out_path}")

# ------------------------------------------------------------------
# How many features are needed to separate all 5 lineages?
# ----------------------------------------------------------------
print("\nCumulative discriminative power (sorted by F-statistic):")
sorted_F = np.sort(F)[::-1]
for n in [1, 2, 5, 10, 20, 50, 100, 256]:
    cumulative = sorted_F[:n].sum() / sorted_F.sum()
    print(f"  Top {n:3d} features  -> {cumulative*100:6.2f}% of total F-statistic mass")