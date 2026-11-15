import csv
import numpy as np
from scipy import sparse
from collections import Counter

# ------------------------------------------------------------------
# 1. Check for train/test overlap at the accession level
# ------------------------------------------------------------------
def load_accessions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [r["accession"] for r in csv.DictReader(f)]

train_acc = set(load_accessions("train.csv"))
test_acc  = set(load_accessions("test.csv"))

overlap = train_acc & test_acc
print(f"Train accessions: {len(train_acc)}")
print(f"Test accessions : {len(test_acc)}")
print(f"Overlap         : {len(overlap)}")
if overlap:
    print(f"  Examples: {list(overlap)[:5]}")
print()

# ------------------------------------------------------------------
# 2. Check for sequence-level overlap (identical sequences)
# ------------------------------------------------------------------
def load_sequences(path):
    with open(path, "r", encoding="utf-8") as f:
        return [r["sequence"] for r in csv.DictReader(f)]

train_seq = load_sequences("train.csv")
test_seq  = load_sequences("test.csv")

train_set = set(train_seq)
test_set  = set(test_seq)
seq_overlap = train_set & test_set
print(f"Train unique sequences: {len(train_set)}")
print(f"Test unique sequences : {len(test_set)}")
print(f"Sequence overlap      : {len(seq_overlap)}")
print()

# ------------------------------------------------------------------
# 3. Check that label order matches feature order
# ------------------------------------------------------------------
def load_csv_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [r["lineage"] for r in csv.DictReader(f)]

train_csv_labels = load_csv_labels("train.csv")
test_csv_labels  = load_csv_labels("test.csv")

train_npy_labels = np.load("features/labels_train.npy", allow_pickle=True).tolist()
test_npy_labels  = np.load("features/labels_test.npy",  allow_pickle=True).tolist()

print("Train: CSV labels == NPY labels ?", train_csv_labels == train_npy_labels)
print("Test : CSV labels == NPY labels ?", test_csv_labels  == test_npy_labels)
print()

if train_csv_labels != train_npy_labels:
    print("MISMATCH DETECTED. First 10 CSV labels:")
    print(train_csv_labels[:10])
    print("First 10 NPY labels:")
    print(train_npy_labels[:10])
print()

# ------------------------------------------------------------------
# 4. Check feature matrix shape consistency
# ------------------------------------------------------------------
X_train = sparse.load_npz("features/kmer_k4_train.npz")
X_test  = sparse.load_npz("features/kmer_k4_test.npz")
print(f"X_train shape: {X_train.shape}, expected ({len(train_csv_labels)}, 256)")
print(f"X_test shape : {X_test.shape}, expected ({len(test_csv_labels)}, 256)")
print()

# ------------------------------------------------------------------
# 5. Look for a single feature that trivially separates classes
# ------------------------------------------------------------------
labels = np.array(train_csv_labels)
features = X_train.toarray()

from sklearn.feature_selection import f_classif
F, p = f_classif(features, labels)
top_idx = np.argsort(F)[::-1][:10]

print("Top 10 most discriminative single features (F-statistic):")
for i in top_idx:
    print(f"  feature {i:4d}  F = {F[i]:12.2f}  p = {p[i]:.2e}")

# ------------------------------------------------------------------
# 6. Check row-sum distribution per split
# ------------------------------------------------------------------
train_sums = np.asarray(X_train.sum(axis=1)).ravel()
test_sums  = np.asarray(X_test.sum(axis=1)).ravel()
print()
print(f"Train row-sum: min {train_sums.min():.4f}  max {train_sums.max():.4f}  mean {train_sums.mean():.4f}")
print(f"Test  row-sum: min {test_sums.min():.4f}  max {test_sums.max():.4f}  mean {test_sums.mean():.4f}")

# ------------------------------------------------------------------
# 7. Print label distribution in each set
# ------------------------------------------------------------------
print()
print("Train label distribution:", Counter(train_csv_labels))
print("Test  label distribution:", Counter(test_csv_labels))