import numpy as np
from scipy import sparse
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import time

K = 4
FEAT_DIR = "features"

def load(split):
    X = sparse.load_npz(f"{FEAT_DIR}/kmer_k{K}_{split}.npz").tocsr()
    y = np.load(f"{FEAT_DIR}/labels_{split}.npy", allow_pickle=True)
    return X, y

X_train, y_train = load("train")
X_test,  y_test  = load("test")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"X_train dtype: {X_train.dtype}, min: {X_train.min()}, max: {X_train.max()}")
print(f"X_train row sum (first 3): {np.asarray(X_train[:3].sum(axis=1)).ravel()}")
print(f"X_train column variance (first 5): {np.asarray(X_train.power(2).mean(axis=0)).ravel()[:5]}")
print()

def run(name, model):
    t0 = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    t1 = time.time()
    print(f"--- {name} ---")
    print(f"  accuracy : {acc:.4f}")
    print(f"  time     : {t1-t0:.2f}s")
    from collections import Counter
    print(f"  prediction distribution: {Counter(pred.tolist())}")
    print()

# 1. Linear SVM, no scaling, balanced
run("LinearSVC, no scaling, balanced",
    LinearSVC(class_weight="balanced", max_iter=10000, random_state=42))

# 2. Linear SVM, no scaling, no weighting
run("LinearSVC, no scaling, no weights",
    LinearSVC(max_iter=10000, random_state=42))

# 3. RBF SVM on dense centered data
X_train_dense = X_train.toarray()
X_test_dense  = X_test.toarray()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_dense)
X_test_c  = scaler.transform(X_test_dense)

run("SVC RBF, dense centered, gamma='scale'",
    SVC(kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", random_state=42))

run("SVC RBF, dense centered, gamma=1.0",
    SVC(kernel="rbf", C=1.0, gamma=1.0,
        class_weight="balanced", random_state=42))

run("SVC RBF, dense centered, gamma=10.0",
    SVC(kernel="rbf", C=1.0, gamma=10.0,
        class_weight="balanced", random_state=42))

# 4. SVC with TruncatedSVD (dimensionality reduction then RBF)
svd = TruncatedSVD(n_components=50, random_state=42)
X_train_svd = svd.fit_transform(X_train)
X_test_svd  = svd.transform(X_test)
scaler2 = StandardScaler()
X_train_svd = scaler2.fit_transform(X_train_svd)
X_test_svd  = scaler2.transform(X_test_svd)

run("SVC RBF after SVD(50) + scaling",
    SVC(kernel="rbf", C=1.0, gamma="scale",
        class_weight="balanced", random_state=42))