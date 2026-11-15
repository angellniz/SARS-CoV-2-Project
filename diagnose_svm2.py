import numpy as np
from scipy import sparse
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
from collections import Counter

K = 4
FEAT_DIR = "features"

def load(split):
    X = sparse.load_npz(f"{FEAT_DIR}/kmer_k{K}_{split}.npz").tocsr()
    y = np.load(f"{FEAT_DIR}/labels_{split}.npy", allow_pickle=True)
    return X, y

X_train_s, y_train = load("train")
X_test_s,  y_test  = load("test")

# Dense, standardized
X_train_d = X_train_s.toarray()
X_test_d  = X_test_s.toarray()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_d)
X_test  = scaler.transform(X_test_d)

print(f"Dense train shape: {X_train.shape}")
print(f"Feature std after scaling (first 5): {X_train.std(axis=0)[:5]}")
print()

def run(name, model, Xtr, Xte):
    t0 = time.time()
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    acc = accuracy_score(y_test, pred)
    t1 = time.time()
    dist = Counter(pred.tolist())
    print(f"--- {name} ---")
    print(f"  accuracy: {acc:.4f}   time: {t1-t0:.2f}s")
    print(f"  predictions: {dist}")
    print()

# Linear SVM with increasing C
for C in [1.0, 1e3, 1e6, 1e9]:
    run(f"LinearSVC(C={C})",
        LinearSVC(C=C, max_iter=20000, class_weight="balanced", random_state=42),
        X_train, X_test)

# SVC linear with increasing C
for C in [1.0, 1e3, 1e6]:
    run(f"SVC(linear, C={C})",
        SVC(kernel="linear", C=C, class_weight="balanced", random_state=42),
        X_train, X_test)

# SVC RBF with increasing C
for C in [1.0, 1e3, 1e6]:
    run(f"SVC(rbf, C={C}, gamma=1.0)",
        SVC(kernel="rbf", C=C, gamma=1.0, class_weight="balanced", random_state=42),
        X_train, X_test)