import numpy as np
from scipy import sparse
from sklearn.feature_selection import f_classif
import matplotlib
matplotlib.use("Agg")   # no GUI window, just save file
import matplotlib.pyplot as plt
import os

K         = 4
FEAT_DIR  = "features"
FIG_DIR   = os.path.join("results_exp3", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Load
X = sparse.load_npz(os.path.join(FEAT_DIR, f"kmer_k{K}_train.npz")).tocsr().toarray()
y = np.load(os.path.join(FEAT_DIR, "labels_train.npy"), allow_pickle=True)

print(f"X shape: {X.shape}")

# F-statistics
F, _ = f_classif(X, y)
F_sorted = np.sort(F)[::-1]
cumulative = np.cumsum(F_sorted) / F_sorted.sum() * 100.0
n_features = np.arange(1, len(F_sorted) + 1)

# Figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_features, cumulative, color="#1f4e79", linewidth=2)
ax.set_xlabel("Number of top-ranked features (by F-statistic)", fontsize=11)
ax.set_ylabel("Cumulative fraction of total F-statistic mass (%)", fontsize=11)
ax.set_title(f"Distribution of discriminative signal across k={K} features", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, len(F_sorted))
ax.set_ylim(0, 105)

# Annotate the top-20 line
cum_20 = cumulative[19]
ax.axvline(20, color="gray", linestyle="--", alpha=0.5)
ax.axhline(cum_20, color="gray", linestyle="--", alpha=0.5)
ax.plot(20, cum_20, "o", color="#d9534f", markersize=8)
ax.annotate(f"top 20 features\n= {cum_20:.1f}% of signal",
            xy=(20, cum_20), xytext=(40, cum_20 - 15),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"))

# Annotate the top-50 line
cum_50 = cumulative[49]
ax.plot(50, cum_50, "o", color="#5cb85c", markersize=8)
ax.annotate(f"top 50 features\n= {cum_50:.1f}% of signal",
            xy=(50, cum_50), xytext=(80, cum_50 - 25),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"))

plt.tight_layout()
out_path = os.path.join(FIG_DIR, f"fig1_fstat_cumulative_k{K}.png")
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved: {out_path}")