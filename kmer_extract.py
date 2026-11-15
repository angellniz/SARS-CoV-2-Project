import csv
import os
import numpy as np
from scipy import sparse
from collections import Counter

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
INPUTS = {
    "train":      "train.csv",
    "test":       "test.csv",
    "train_time": "train_time.csv",
    "test_time":  "test_time.csv",
}

OUTPUT_DIR = "features"
K_VALUES   = [3, 4, 5, 6]
BASES      = ["A", "C", "G", "T"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Build the full list of k-mers for a given k, in a fixed order.
# We exclude any k-mer containing N because N is not a real base.
# ------------------------------------------------------------------
def generate_kmers(k):
    from itertools import product
    return ["".join(p) for p in product(BASES, repeat=k)]

# ------------------------------------------------------------------
# Count k-mers in one sequence and return a normalized frequency vector.
# ------------------------------------------------------------------
def kmer_vector(seq, kmers, kmer_index, k):
    seq = seq.upper()
    counts = np.zeros(len(kmers), dtype=np.float32)
    n_windows = 0
    for i in range(len(seq) - k + 1):
        sub = seq[i:i+k]
        if "N" in sub:
            continue
        idx = kmer_index.get(sub)
        if idx is not None:
            counts[idx] += 1.0
            n_windows += 1
    if n_windows > 0:
        counts /= n_windows
    return counts

# ------------------------------------------------------------------
# Load a CSV into lists of (sequence, label)
# ------------------------------------------------------------------
def load_csv(path):
    seqs, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            seqs.append(r["sequence"])
            labels.append(r["lineage"])
    return seqs, labels

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
summary_lines = []
summary_lines.append("KMER FEATURE EXTRACTION SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"k values: {K_VALUES}")
summary_lines.append("")

for k in K_VALUES:
    print(f"\n=== k = {k} ===")
    kmers = generate_kmers(k)
    kmer_index = {km: i for i, km in enumerate(kmers)}
    dim = len(kmers)
    print(f"  Number of possible k-mers: {dim}")

    summary_lines.append(f"k = {k}: {dim} features")

    for split_name, csv_path in INPUTS.items():
        if not os.path.exists(csv_path):
            print(f"  SKIP {split_name}: {csv_path} not found")
            continue

        print(f"  Processing {split_name} from {csv_path}...")
        seqs, labels = load_csv(csv_path)
        n = len(seqs)

        # Build sparse matrix row by row
        rows, cols, data = [], [], []
        for i, seq in enumerate(seqs):
            vec = kmer_vector(seq, kmers, kmer_index, k)
            nz = np.nonzero(vec)[0]
            for j in nz:
                rows.append(i)
                cols.append(int(j))
                data.append(float(vec[j]))

        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n, dim),
            dtype=np.float32,
        )

        out_path = os.path.join(OUTPUT_DIR, f"kmer_k{k}_{split_name}.npz")
        sparse.save_npz(out_path, mat)
        print(f"    Saved {out_path}  (shape {mat.shape}, nnz {mat.nnz})")

        # Save labels once per split (independent of k)
        if k == K_VALUES[0]:
            label_path = os.path.join(OUTPUT_DIR, f"labels_{split_name}.npy")
            np.save(label_path, np.array(labels))
            print(f"    Saved {label_path}")

with open(os.path.join(OUTPUT_DIR, "kmer_feature_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\nDONE.")
print("Summary written to features/kmer_feature_summary.txt")