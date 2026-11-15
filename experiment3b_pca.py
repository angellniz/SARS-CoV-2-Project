import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
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

# Standardize (dense, so full centering)
X_std = StandardScaler().fit_transform(X)

# PCA to 2 components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)

print(f"Explained variance (PC1, PC2): "
      f"{pca.explained_variance_ratio_[0]*100:.2f}%, "
      f"{pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total (2 comps): {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Colors
lineages = ["Alpha", "Beta", "Gamma", "Delta", "Omicron_BA1"]
colors = {
    "Alpha":       "#1f77b4",
    "Beta":        "#ff7f0e",
    "Gamma":       "#2ca02c",
    "Delta":       "#d62728",
    "Omicron_BA1": "#9467bd",
}
markers = {
    "Alpha":       "o",
    "Beta":        "s",
    "Gamma":       "^",
    "Delta":       "D",
    "Omicron_BA1": "P",
}

# Plot
fig, ax = plt.subplots(figsize=(9, 7))
for lin in lineages:
    mask = (y == lin)
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=colors[lin], marker=markers[lin],
        s=22, alpha=0.6, edgecolors="none",
        label=f"{lin} (n={mask.sum()})",
    )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
ax.set_title(f"PCA of k={K} (256-feature) SARS-CoV-2 k-mer frequency space", fontsize=12)
ax.legend(loc="best", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, f"fig2_pca_k{K}.png")
plt.savefig(out_path, dpi=200)
plt.close()
print(f"Saved: {out_path}")