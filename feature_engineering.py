import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


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
    print(f"\n[TF-IDF] feature extraction completed")
    print(f"  size of vocabulary: {len(feature_names)}")
    print(f"  top feature words: {list(feature_names[:10])}")
    print(f"  stored non-zero elements: {X_train_tfidf.nnz}")
    print(f"  sparsity: {1 - X_train_tfidf.nnz / np.prod(X_train_tfidf.shape):.4f}")
    return X_train_tfidf, X_test_tfidf, vectorizer


def inspect_tfidf(X_tfidf, vectorizer, sample_index=0, top_n=20):
    """
    Inspect a specific row in the TF-IDF matrix: see which words and their weights.

    :param X_tfidf: TF-IDF sparse matrix
    :param vectorizer: fitted TfidfVectorizer (to get word names)
    :param sample_index: which document (row) to inspect
    :param top_n: show top N words by weight
    """
    feature_names = vectorizer.get_feature_names_out()
    row = X_tfidf[sample_index]
    nonzero_indices = row.nonzero()[1]

    print(f"\n[Inspect] Row {sample_index}: {len(nonzero_indices)} non-zero words")
    print(f"  {'Index':<8} {'Word':<25} {'TF-IDF Weight':<15}")
    print(f"  {'-'*48}")

    words_weights = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        weight = row[0, idx]
        words_weights.append((idx, word, weight))

    # sort by weight descending
    words_weights.sort(key=lambda x: x[2], reverse=True)
    for idx, word, weight in words_weights[:top_n]:
        print(f"  [{idx:<5}] {word:<25} {weight:.6f}")

    # verify L2 normalization
    l2_norm = np.sqrt(sum(w**2 for _, _, w in words_weights))
    print(f"\n  L2 norm = {l2_norm:.6f}  (should be 1.0)")


def full_vocabulary(vectorizer):
    """Print the complete vocabulary with index numbers."""
    feature_names = vectorizer.get_feature_names_out()
    print(f"\n[Vocabulary] Total: {len(feature_names)} words")
    for i, word in enumerate(feature_names):
        print(f"  [{i:>4}] {word}")