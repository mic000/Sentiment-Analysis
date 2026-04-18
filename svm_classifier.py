"""
Stage 4: SVM classification + Cross Validation

Optimization:
  min_{w,b}  (1/2)||w||^2 + C * (1/N) Σ max(0, 1 - y_i(w'x_i + b))
             └──────┬──────┘  └────────────────┬──────────────────┘
         Regularization                 Hinge Loss
Parameter adjusted: 5-Fold Cross Validation
"""

import numpy as np
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def run_svm(X_train, X_test, y_train, y_test,
            name="SVM", kernel='linear', C=1.0, gamma='scale', seed=123):
    """
    Run a single SVM experiment: train + test + evaluate.

    :param name: experiment label (for plots)
    :param kernel: 'linear' or 'rbf'
    :param C: regularization parameter
    :param seed: random seed
    :return: dict with all results
    """
    if kernel == 'linear':
        model = LinearSVC(C=C, max_iter=10000, random_state=seed, loss='hinge')
    else:
        model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=seed)

    # train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # test
    t0 = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - t0

    # evaluate
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"{'='*52}")
    print(f"  Kernel={kernel}, C={C}")
    print(f"  Train time:  {train_time:.4f} s")
    print(f"  Test time:   {test_time:.6f} s")
    print(f"  Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Negative','Positive'])}")

    return {
        'name': name, 'kernel': kernel, 'C': C,
        'accuracy': acc, 'f1': f1, 'cm': cm,
        'train_time': train_time, 'test_time': test_time,
        'y_pred': y_pred, 'model': model
    }


def tune_C_with_cv(X_train, y_train, kernel='linear', cv_folds=5,
                    C_values=None):
    """
    Tune regularization parameter C using K-Fold Cross Validation.
    Only uses training data — test set is untouched.

    :param cv_folds: number of folds (default 5)
    :param C_values: list of candidate C values
    :return: C_values, cv_mean, cv_std, best_C
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    cv_mean = []
    cv_std = []

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print(f"\n{'='*60}")
    print(f"  {cv_folds}-Fold Cross Validation (kernel={kernel})")
    print(f"{'='*60}")
    print(f"  {'C':<10} {'CV Acc (mean)':<18} {'CV Std':<12}")
    print(f"  {'-'*40}")

    for C in C_values:
        if kernel == 'linear':
            model = LinearSVC(C=C, max_iter=10000, random_state=42, loss='hinge')
        else:
            model = SVC(kernel=kernel, C=C, gamma='scale', random_state=42)

        scores = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring='accuracy', n_jobs=-1)

        cv_mean.append(scores.mean())
        cv_std.append(scores.std())
        print(f"  {C:<10} {scores.mean():<18.4f} {scores.std():<12.4f}")

    best_idx = int(np.argmax(cv_mean))
    best_C = C_values[best_idx]
    print(f"\n  Best C = {best_C}")
    print(f"  Best CV Accuracy = {cv_mean[best_idx]:.4f} ± {cv_std[best_idx]:.4f}")

    return C_values, cv_mean, cv_std, best_C


def run_all_experiments(X_train_tfidf, X_test_tfidf, pca_results,
                         y_train, y_test, best_C=1.0):
    """
    Run 4 comparison experiments.

    Exp 1: Linear SVM + TF-IDF (494D)     ← baseline
    Exp 2: Linear SVM + PCA-50D           ← aggressive reduction
    Exp 3: Linear SVM + PCA-200D          ← moderate reduction
    Exp 4: RBF SVM + TF-IDF (494D)        ← nonlinear kernel

    :return: list of result dicts
    """
    results = []

    # Exp 1: baseline — no PCA
    results.append(run_svm(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        name="Linear SVM + TF-IDF", kernel='linear', C=best_C
    ))

    for dim in sorted(pca_results.keys()):
        results.append(run_svm(
            pca_results[dim]['X_train'], pca_results[dim]['X_test'],
            y_train, y_test,
            name=f"Linear SVM + PCA-{dim}D", kernel='linear', C=best_C
        ))

    # Exp 4: RBF kernel — no PCA
    results.append(run_svm(
        X_train_tfidf, X_test_tfidf, y_train, y_test,
        name="RBF SVM + TF-IDF", kernel='rbf', C=best_C
    ))

    return results