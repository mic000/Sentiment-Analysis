"""
Stage 3b

Math steps:
  Step 1: Center       X_c = X - mean(X)
  Step 2: Covariance   S = (1/(N-1))*X_c'*X_c
  Step 3: Eigendecomp  S = W*Λ*W'
  Step 4: Select top-k W_k = W[:, :k]
  Step 5: Project      X_reduced = X_c @ W_k

Standardization is also handcrafted (no sklearn.preprocessing.StandardScaler).

Pipeline position:
  data_cleaner → feature_engineering (TF-IDF) → [HERE: PCA] → svm_classifier
"""

import numpy as np


class MyPCA:
    """
    PCA — replaces sklearn.decomposition.PCA

    optimization:
      max_W  trace(W'*S*W)
      s.t.   W'*W = I

    equivalent：
      min_W  Σ abs(x_i - W*W'*x_i)^2
      s.t.   W'*W = I

    Solution: columns of W are eigenvectors of S corresponding to the k largest eigenvalues.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None                 # W_k, shape (D, k)
        self.eigenvalues_ = None                # top k eigenvalues
        self.all_eigenvalues_ = None            # all eigenvalues (for variance curve)
        self.explained_variance_ratio_ = None   # variance of principal components

    def fit(self, X):
        """
        fit PCA in datasets

        :param X: datasets, shape (N, D), N=number of samples, D=dimension
        """
        N, D = X.shape

        # ================================================================
        # Step 1: Centering
        #
        # mathematic equation: X_centered = X - μ, where μ = (1/N) Σ x_i
        # ================================================================
        self.mean_ = X.mean(axis=0)  # shape (D,)
        X_centered = X - self.mean_  # shape (N, D)

        # ================================================================
        # Step 2: Covariance Matrix
        #
        # mathematic equation: S = (1/(N-1)) X_centered'*X_centered
        # S is D×D symmetric matrix:
        #   S[i, j] = covariance of i and j
        #   S[i, i] = variance of i
        # ================================================================
        S = (X_centered.T @ X_centered) / (N - 1)

        print(f"    shape of covariance: {S.shape}")
        print(f"    validation: max|S - S.T| = {np.max(np.abs(S - S.T)):.2e}")
        print(f"    mean of diagonal matrix: {np.diag(S).mean():.4f}")

        # ================================================================
        # Step 3: Eigendecomposition
        #
        # mathematic equation: S = W*Λ*W'
        #   eigh: for symmetric matrices, numerically stable, returns real values
        # ================================================================
        eigenvalues, eigenvectors = np.linalg.eigh(S)

        # eigh returns reverse to descending order
        eigenvalues = eigenvalues[::-1]  # shape (D,)
        eigenvectors = eigenvectors[:, ::-1]  # shape (D, D)

        eigenvalues = np.maximum(eigenvalues, 0)        # nonzero clip numerical noise (eigenvalues should be ≥ 0)

        print(f"    first 5 eigenvalues: {eigenvalues[:5]}")
        print(f"    smallest eigenvalues: {eigenvalues[-1]:.6f}")
        print(f"    size of eigenvectors: {eigenvectors.shape}")
        self.all_eigenvalues_ = eigenvalues.copy()      # store all eigenvalues (for variance curve)

        # ================================================================
        # Step 4: take top k of principal compound
        # ================================================================
        k = self.n_components
        self.components_ = eigenvectors[:, :k]  # shape (D, k)
        self.eigenvalues_ = eigenvalues[:k]  # shape (k,)

        # variance ratio by principal component = λ_i / Σλ_j
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:k] / total_variance

        print(f"  Step 4: completed ")
        print(f"    shape of W_k: {self.components_.shape}")
        print(f"    W_k'*W_k: "
              f"max|I - W_k'*W_k| = "
              f"{np.max(np.abs(np.eye(k) - self.components_.T @ self.components_)):.2e}")
        print(f"    accumulated explained variance ratio: "
              f"{sum(self.explained_variance_ratio_) * 100:.2f}%")
        return self

    def transform(self, X):
        """
        Project data to lower dimension.

        mathematic equation: X_reduced = (X - μ) @ W_k

        :param X: input datasets, shape (N, D)
        :return: datasets reduced dimension, shape (N, k)
        """
        X_centered = X - self.mean_
        X_reduced = X_centered @ self.components_
        return X_reduced

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_reduced):
        """
        mathematic equation: X_reconstructed = X_reduced @ W_k' + μ
        """
        return X_reduced @ self.components_.T + self.mean_

    def get_cumulative_variance(self):
        """
        get cumulative variance curve for all components.
        """
        total_var = self.all_eigenvalues_.sum()
        return np.cumsum(self.all_eigenvalues_) / total_var * 100


class MyStandardScaler:
    """
    mathematic equation: x_scaled = (x - mean) / std
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Learn mean and std from training data.
        """
        self.mean_ = X.mean(axis=0)                                 # shape (D,)
        self.std_ = X.std(axis=0, ddof=0)                           # population std
        self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)     # prevent division by zero
        return self

    def transform(self, X):
        """
        Apply standardization using learned mean/std.
        """
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def apply_pca(X_train_tfidf, X_test_tfidf, n_components=100):
    """
    Apply handcrafted PCA to TF-IDF features.
    Same interface as the old sklearn-based apply_pca.

    :param X_train_tfidf: sparse TF-IDF matrix (train)
    :param X_test_tfidf: sparse TF-IDF matrix (test)
    :param n_components: target dimension
    :return: X_train_pca, X_test_pca, scaler, pca
    """
    # sparse → dense
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()
    n_components = min(n_components, X_train_dense.shape[1] - 1)

    # handcrafted standardization
    scaler = MyStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)

    # handcrafted PCA
    pca = MyPCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"\n[MyPCA] {X_train_dense.shape[1]}D → {n_components}D")
    print(f"  cumulative variance: {explained:.2f}%")
    return X_train_pca, X_test_pca, scaler, pca


def optimal_dim(X_train_tfidf, threshold=85):
    """
    Find optimal number of principal components using handcrafted PCA.

    :param X_train_tfidf: sparse TF-IDF matrix
    :param threshold: variance explanation threshold (%)
    :return: n_optimal, cumulative_var
    """
    X_dense = X_train_tfidf.toarray()
    scaler = MyStandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    # fit full PCA to get all eigenvalues
    max_comp = min(X_scaled.shape[0], X_scaled.shape[1])
    pca_full = MyPCA(n_components=max_comp)
    pca_full.fit(X_scaled)

    cumulative_var = pca_full.get_cumulative_variance()

    n_90 = int(np.argmax(cumulative_var >= 90) + 1)
    n_95 = int(np.argmax(cumulative_var >= 95) + 1)
    n_optimal = int(np.argmax(cumulative_var >= threshold) + 1)
    print(f"\n[MyPCA dimensional analysis]")
    print(f"  90% variance: {n_90} components")
    print(f"  95% variance: {n_95} components")
    print(f"  {threshold}% variance: {n_optimal} components")
    return n_optimal, cumulative_var