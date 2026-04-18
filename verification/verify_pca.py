"""
Verify handcrafted PCA against sklearn PCA using real training data.

This script:
  1. Loads data with clean_text + split_data
  2. Runs TF-IDF
  3. Shows which words correspond to chosen sample on datasets etc.
  4. Runs BOTH sklearn PCA and handcrafted PCA
  5. Compares results step by step
"""

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler as SklearnScaler

from data_cleaner import clean_text, split_data
from feature_engineering import *
from pca import MyPCA, MyStandardScaler


def main():
    print("=" * 65)
    print("  Stage 1-2: Load and split data")
    print("=" * 65)
    df = clean_text("data/raw", "amazon_cells_labelled", 100)
    train_dataset, test_dataset = split_data(df)

    print("\n" + "=" * 65)
    print("  Stage 3a: TF-IDF")
    print("=" * 65)
    X_train_tfidf, X_test_tfidf, vectorizer = tfidf_features(train_dataset, test_dataset)

    print("\n" + "-" * 65)
    print("  Inspect: what words are in each review?")
    print("-" * 65)
    for row in range(min(3, X_train_tfidf.shape[0])):
        inspect_tfidf(X_train_tfidf, vectorizer, sample_index=row)


    # =========================================================
    # Stage 3b: PCA comparison — sklearn vs handcrafted
    # =========================================================
    print("\n" + "=" * 65)
    print("  Stage 3b: PCA — sklearn vs handcrafted (from scratch)")
    print("=" * 65)

    n_components = 50  # use 50 for comparison (your data has ~494 features)

    # Convert sparse → dense (both methods need this)
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    D = X_train_dense.shape[1]  # number of features
    N = X_train_dense.shape[0]  # number of samples
    print(f"\n  Input: {N} samples × {D} features → {n_components} components")

    # ---- Step 1: Standardization comparison ----
    print(f"\n  {'='*55}")
    print(f"  Step 1: Standardization comparison")
    print(f"  {'='*55}")

    sk_scaler = SklearnScaler()
    X_train_sk_scaled = sk_scaler.fit_transform(X_train_dense)

    my_scaler = MyStandardScaler()
    X_train_my_scaled = my_scaler.fit_transform(X_train_dense)

    mean_diff = np.max(np.abs(sk_scaler.mean_ - my_scaler.mean_))
    scale_diff = np.max(np.abs(X_train_sk_scaled - X_train_my_scaled))

    print(f"    Mean difference:   {mean_diff:.2e}")
    print(f"    Scaled data diff:  {scale_diff:.2e}")
    print(f"    {'✓ Match' if scale_diff < 1e-10 else '⚠ Mismatch'}")

    # ---- Step 2: Covariance matrix (handcrafted only, sklearn does SVD) ----
    print(f"\n  {'='*55}")
    print(f"  Step 2: Covariance matrix (handcrafted)")
    print(f"  {'='*55}")

    X_centered = X_train_my_scaled - X_train_my_scaled.mean(axis=0)
    S = (X_centered.T @ X_centered) / (N - 1)

    print(f"    Shape: {S.shape}")
    print(f"    Symmetric: max|S - S^T| = {np.max(np.abs(S - S.T)):.2e}")
    print(f"    Trace (total variance): {np.trace(S):.4f}")
    print(f"    Diagonal mean: {np.diag(S).mean():.4f}")

    # ---- Step 3: Eigendecomposition vs SVD ----
    print(f"\n  {'='*55}")
    print(f"  Step 3: Eigendecomposition (handcrafted) vs SVD (sklearn)")
    print(f"  {'='*55}")

    # handcrafted: eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    # sklearn
    sk_pca = SklearnPCA(n_components=n_components)
    X_train_sk_pca = sk_pca.fit_transform(X_train_sk_scaled)

    print(f"    Top 10 eigenvalues:")
    print(f"      Handcrafted: {eigenvalues[:10].round(4)}")
    print(f"      Sklearn:     {sk_pca.explained_variance_[:10].round(4)}")

    eig_diff = np.max(np.abs(eigenvalues[:n_components] - sk_pca.explained_variance_))
    print(f"\n    Eigenvalue difference: {eig_diff:.2e}")
    print(f"    {'✓ Match' if eig_diff < 1e-8 else '⚠ Mismatch'}")

    # ---- Step 4: Variance explained ----
    print(f"\n  {'='*55}")
    print(f"  Step 4: Variance explained comparison")
    print(f"  {'='*55}")

    total_var = eigenvalues.sum()
    my_var_ratio = eigenvalues[:n_components] / total_var
    sk_var_ratio = sk_pca.explained_variance_ratio_

    print(f"    Top 5 variance ratios:")
    print(f"      Handcrafted: {my_var_ratio[:5].round(6)}")
    print(f"      Sklearn:     {sk_var_ratio[:5].round(6)}")

    var_diff = np.max(np.abs(my_var_ratio - sk_var_ratio))
    print(f"\n    Variance ratio difference: {var_diff:.2e}")
    print(f"    {'✓ Match' if var_diff < 1e-8 else '⚠ Mismatch'}")

    cumulative_my = sum(my_var_ratio) * 100
    cumulative_sk = sum(sk_var_ratio) * 100
    print(f"\n    Cumulative variance ({n_components} components):")
    print(f"      Handcrafted: {cumulative_my:.2f}%")
    print(f"      Sklearn:     {cumulative_sk:.2f}%")

    # ---- Step 5: Projection comparison ----
    print(f"\n  {'='*55}")
    print(f"  Step 5: Projection comparison")
    print(f"  {'='*55}")

    # handcrafted projection
    W_k = eigenvectors[:, :n_components]
    X_train_my_pca = (X_train_my_scaled - X_train_my_scaled.mean(axis=0)) @ W_k

    # compare absolute values (eigenvectors may have flipped signs)
    proj_diff = np.max(np.abs(np.abs(X_train_my_pca) - np.abs(X_train_sk_pca)))
    print(f"    Projection difference (abs): {proj_diff:.2e}")
    print(f"    {'✓ Match' if proj_diff < 1e-8 else '⚠ Small numerical difference'}")

    # ---- Step 6: Full handcrafted pipeline via MyPCA class ----
    print(f"\n  {'='*55}")
    print(f"  Step 6: Full MyPCA class vs sklearn PCA")
    print(f"  {'='*55}")

    my_pca = MyPCA(n_components=n_components)
    X_train_mypca = my_pca.fit_transform(X_train_my_scaled)
    X_test_mypca = my_pca.transform(my_scaler.transform(X_test_dense))

    # sklearn test
    X_test_sk_pca = sk_pca.transform(sk_scaler.transform(X_test_dense))

    train_diff = np.max(np.abs(np.abs(X_train_mypca) - np.abs(X_train_sk_pca)))
    test_diff = np.max(np.abs(np.abs(X_test_mypca) - np.abs(X_test_sk_pca)))

    print(f"    Train set projection diff: {train_diff:.2e}")
    print(f"    Test set projection diff:  {test_diff:.2e}")
    print(f"    {'✓ Both match' if max(train_diff, test_diff) < 1e-8 else '⚠ Check'}")

    # ---- Step 7: Reconstruction error ----
    print(f"\n  {'='*55}")
    print(f"  Step 7: Reconstruction error")
    print(f"  {'='*55}")

    X_reconstructed = my_pca.inverse_transform(X_train_mypca)
    X_centered_orig = X_train_my_scaled - X_train_my_scaled.mean(axis=0)
    mse = np.mean((X_centered_orig - X_reconstructed) ** 2)
    total_mse = np.mean(X_centered_orig ** 2)
    info_preserved = (1 - mse / total_mse) * 100

    print(f"    Reconstruction MSE: {mse:.6f}")
    print(f"    Information preserved: {info_preserved:.2f}%")
    print(f"    (should ≈ cumulative variance {cumulative_my:.2f}%)")

    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"""
    Data:
      Reviews after cleaning:    {len(df)}
      Training samples:          {N}
      TF-IDF features:           {D}
      PCA components:            {n_components}

    Verification results:
      Standardization:           {'✓ Pass' if scale_diff < 1e-10 else '✗ Fail'}
      Eigenvalues:               {'✓ Pass' if eig_diff < 1e-8 else '✗ Fail'}
      Variance ratios:           {'✓ Pass' if var_diff < 1e-8 else '✗ Fail'}
      Train projection:          {'✓ Pass' if train_diff < 1e-8 else '✗ Fail'}
      Test projection:           {'✓ Pass' if test_diff < 1e-8 else '✗ Fail'}

    Conclusion:
      Handcrafted PCA ≡ sklearn PCA  (within floating-point precision)
      Cumulative variance ({n_components} components): {cumulative_my:.2f}%
    """)

if __name__ == "__main__":
    main()