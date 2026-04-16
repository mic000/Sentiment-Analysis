import numpy as np
from pandas.core.common import random_state
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def tfidf_features(train_dataset, test_dataset, max = 1000, ngram = (1, 2)):
    """
    stage 3a: cleaned text changed to numerical or mathematical features by TF-IDF theory

    :param train_dataset:
    :param test_dataset:
    :param max_features: build a vocabulary that only consider the top default 3000
    term which frequency across the corpus
    :param ngram_range: consider default bigram vocabulary
    :return:
    """
    X_train = train_dataset[0]
    X_test = test_dataset[0]

    vectorizer = TfidfVectorizer(
        max_features=max,
        stop_words='english',
        ngram_range=ngram,
        min_df=2,  # term been keep at least number of times shown in dataset
        max_df=0.90,  # remove terms that appear over number% of the doc.
        sublinear_tf=True) # replace tf with 1 + log(TF), impact on reduce high frequency term

    X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocabulary and IDF, return document-term matrix
    X_test_tfidf = vectorizer.transform(X_test)  # transform doc. to doc.-term matrix and df learned by fit_transform

    feature_names = vectorizer.get_feature_names_out()  # to retrieve the names of features produced by a transformer
    print(f"\n[TF-IDF] feature extraction is completed")
    print(f"  size of wordlist: {len(feature_names)}")
    print(f"  top feature words: {list(feature_names[:10])}")
    return X_train_tfidf, X_test_tfidf, vectorizer


def apply_pca(X_train_tfidf, X_test_tfidf, n_components = 100, seed = 123):
    """
    Stage 3b: PCA dimensionality reduction

    principle：
      1. Standardize: x' = (x - mean) / std
      2. Covariance:  S = (1/N) X'^T X'
      3. Eigendecomp: S = W Λ W^T
      4. Project:     X_reduced = X' @ W_k

    :param X_train_tfidf:
    :param X_test_tfidf:
    :param n_components: new dimension after reduced
    :param seed:
    :return:
    """
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    # standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)

    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"\n[PCA] completed: {X_train_dense.shape[1]}D → {n_components}D")
    print(f"  cumulative variance interpretation rate: {explained_var:.2f}%")
    return X_train_pca, X_test_pca, scaler, pca


def optimal_dim(X_train_tfidf, threshold = 85, max_components = 500, seed = 123):
    """
    find the optimal number of principal components required to reach variance for drawing the variance curve graph

    :param X_train_tfidf:
    :param threshold: var. interpretation rate threshold
    :param max_components: max. number of principal components to calculate
    :return:
    """
    X_dense = X_train_tfidf.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    max_comp = min(max_components, X_scaled.shape[0], X_scaled.shape[1])
    pca_full = PCA(n_components=max_comp, random_state=seed)
    pca_full.fit(X_scaled)

    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
    n_90 = int(np.argmax(cumulative_var >= 90) + 1)
    n_95 = int(np.argmax(cumulative_var >= 95) + 1)
    n_optimal = int(np.argmax(cumulative_var >= threshold) + 1)

    print(f"\n[PCA dimensional analysis]")
    print(f"  achieve to 90%: need number of {n_90} principal components")
    print(f"  achieve to 95%: need number of {n_95} principal components")
    print(f"  achieve to {threshold}%: need number of {n_optimal} principal components")
    return n_optimal, cumulative_var
