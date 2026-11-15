import numpy as np
from scipy import sparse
from scipy.spatial.distance import cosine, euclidean
from collections import defaultdict
import os
import csv
import json

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
K            = 4
FEAT_DIR     = "features"
RESULTS_DIR  = "results_exp3"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load training features + labels
# ------------------------------------------------------------------
X_train = sparse.load_npz(os.path.join(FEAT_DIR, f"kmer_k{K}_train.npz")).tocsr()
y_train = np.load(os.path.join(FEAT_DIR, "labels_train.npy"), allow_pickle=True)

print(f"Loaded X_train: {X_train.shape}")
print(f"Loaded y_train: {y_train.shape}")

# ------------------------------------------------------------------
# Compute centroids per lineage
# ------------------------------------------------------------------
lineages = sorted(set(y_train))
centroids = {}

for lin in lineages:
    mask = (y_train == lin)
    subset = X_train[mask]
    centroid = np.asarray(subset.mean(axis=0)).ravel()
    centroids[lin] = centroid
    print(f"  Centroid for {lin:15s} (n={subset.shape[0]})  "
          f"sum={centroid.sum():.4f}  "
          f"max={centroid.max():.6f}  "
          f"nonzero={np.count_nonzero(centroid)}")

# ------------------------------------------------------------------
# Pairwise cosine distance matrix (5x5)
# ------------------------------------------------------------------
print("\nComputing pairwise cosine distances...")
n = len(lineages)
cos_dist = np.zeros((n, n))
euc_dist = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        cos_dist[i, j] = cosine(centroids[lineages[i]], centroids[lineages[j]])
        euc_dist[i, j] = euclidean(centroids[lineages[i]], centroids[lineages[j]])

# ------------------------------------------------------------------
# Print readable matrix
# ------------------------------------------------------------------
def print_matrix(title, mat):
    print()
    print(title)
    header = " " * 16 + "".join(f"{l:>14s}" for l in lineages)
    print(header)
    for i, lin in enumerate(lineages):
        row = f"{lin:>15s} " + "".join(f"{mat[i,j]:>14.4f}" for j in range(n))
        print(row)

print_matrix("Cosine distance between lineage centroids:", cos_dist)
print_matrix("Euclidean distance between lineage centroids:", euc_dist)

# ------------------------------------------------------------------
# Per-lineage distinctiveness score
# ------------------------------------------------------------------
print("\nPer-lineage distinctiveness (mean cosine distance to other lineages):")
distinctiveness = {}
for i, lin in enumerate(lineages):
    others = [cos_dist[i, j] for j in range(n) if j != i]
    score = float(np.mean(others))
    distinctiveness[lin] = score
    print(f"  {lin:15s} : {score:.4f}")

# ------------------------------------------------------------------
# Save matrices as CSV
# ------------------------------------------------------------------
def save_matrix_csv(path, mat, row_labels, col_labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + col_labels)
        for i, rl in enumerate(row_labels):
            writer.writerow([rl] + [f"{mat[i,j]:.6f}" for j in range(len(col_labels))])

save_matrix_csv(os.path.join(RESULTS_DIR, "cosine_distance_matrix.csv"),
                cos_dist, lineages, lineages)
save_matrix_csv(os.path.join(RESULTS_DIR, "euclidean_distance_matrix.csv"),
                euc_dist, lineages, lineages)

# ------------------------------------------------------------------
# Save centroids themselves for Experiment 4
# ------------------------------------------------------------------
np.savez(
    os.path.join(RESULTS_DIR, f"lineage_centroids_k{K}.npz"),
    **{lin: centroids[lin] for lin in lineages}
)
print(f"\nCentroids saved to {RESULTS_DIR}/lineage_centroids_k{K}.npz")

# ------------------------------------------------------------------
# Save JSON summary
# ------------------------------------------------------------------
summary = {
    "k": K,
    "lineages": lineages,
    "cosine_distance_matrix": cos_dist.tolist(),
    "euclidean_distance_matrix": euc_dist.tolist(),
    "distinctiveness_score": distinctiveness,
    "train_size": int(X_train.shape[0]),
    "dim": int(X_train.shape[1]),
}
with open(os.path.join(RESULTS_DIR, "experiment3_similarity.json"),
          "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# ------------------------------------------------------------------
# Write human-readable summary
# ------------------------------------------------------------------
lines = []
lines.append("EXPERIMENT 3 — SIMILARITY ANALYSIS SUMMARY")
lines.append("=" * 70)
lines.append(f"Feature space: k = {K}  ({X_train.shape[1]} features)")
lines.append(f"Training genomes: {X_train.shape[0]}")
lines.append("")

lines.append("Cosine distance between lineage centroids:")
header = " " * 16 + "".join(f"{l:>14s}" for l in lineages)
lines.append(header)
for i, lin in enumerate(lineages):
    lines.append(f"{lin:>15s} " + "".join(f"{cos_dist[i,j]:>14.4f}" for j in range(n)))
lines.append("")

lines.append("Distinctiveness score (mean cosine distance to other lineages):")
for lin in lineages:
    lines.append(f"  {lin:15s} : {distinctiveness[lin]:.4f}")
lines.append("")

# Find closest pair (excluding self)
min_d, min_pair = float("inf"), None
for i in range(n):
    for j in range(i+1, n):
        if cos_dist[i, j] < min_d:
            min_d = cos_dist[i, j]
            min_pair = (lineages[i], lineages[j])

max_d, max_pair = -float("inf"), None
for i in range(n):
    for j in range(i+1, n):
        if cos_dist[i, j] > max_d:
            max_d = cos_dist[i, j]
            max_pair = (lineages[i], lineages[j])

lines.append(f"Closest pair : {min_pair[0]} vs {min_pair[1]}  (cosine = {min_d:.4f})")
lines.append(f"Furthest pair: {max_pair[0]} vs {max_pair[1]}  (cosine = {max_d:.4f})")

summary_text = "\n".join(lines)
print()
print(summary_text)

with open(os.path.join(RESULTS_DIR, "experiment3_summary.txt"),
          "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nAll results saved to {RESULTS_DIR}/")