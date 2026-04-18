"""
PCA using sklearn (drop-in replacement for pca.py)

Same interface as pca.py — just swap the import in main.py:
  from pca import apply_pca, optimal_dim          ← handcrafted
  from pca_sklearn import apply_pca, optimal_dim   ← sklearn

All function signatures and return types are identical.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_pca(X_train_tfidf, X_test_tfidf, n_components=100):
    """
    Apply sklearn PCA to TF-IDF features.
    Same interface as pca.py apply_pca().

    :param X_train_tfidf: sparse TF-IDF matrix (train)
    :param X_test_tfidf: sparse TF-IDF matrix (test)
    :param n_components: target dimension
    :return: X_train_pca, X_test_pca, scaler, pca
    """
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"\n[sklearn PCA] {X_train_dense.shape[1]}D → {n_components}D")
    print(f"  cumulative variance: {explained:.2f}%")
    return X_train_pca, X_test_pca, scaler, pca


def optimal_dim(X_train_tfidf, threshold=85):
    """
    Find optimal number of principal components using sklearn PCA.
    Same interface as pca.py optimal_dim().

    :param X_train_tfidf: sparse TF-IDF matrix
    :param threshold: variance explanation threshold (%)
    :return: n_optimal, cumulative_var
    """
    X_dense = X_train_tfidf.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    max_comp = min(X_scaled.shape[0], X_scaled.shape[1])
    pca_full = PCA(n_components=max_comp, random_state=123)
    pca_full.fit(X_scaled)

    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_) * 100

    n_90 = int(np.argmax(cumulative_var >= 90) + 1)
    n_95 = int(np.argmax(cumulative_var >= 95) + 1)
    n_optimal = int(np.argmax(cumulative_var >= threshold) + 1)

    print(f"\n[sklearn PCA dimensional analysis]")
    print(f"  90% variance: {n_90} components")
    print(f"  95% variance: {n_95} components")
    print(f"  {threshold}% variance: {n_optimal} components")

    return n_optimal, cumulative_var