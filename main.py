import re

from data_cleaner import text_combined, split_data
from feature_engineering import tfidf_features, inspect_tfidf
from pca import apply_pca, optimal_dim
from svm_classifier import run_svm, tune_C_with_cv, run_all_experiments
from visualization import (
    plot_pca_variance, plot_confusion_matrices,
    plot_accuracy, plot_time, plot_cv_curve, print_summary
)
from predictor import save_pipeline, SentimentPredictor, interactive_predict


CONFIG = {
    # data
    'data_folder':     "data/raw",
    'data_files': ["amazon_cells_labelled", "imdb_labelled", "yelp_labelled"],
    'test_size':       0.2,
    'random_seed':     123,

    # TF-IDF
    'max_features':    3000,
    'ngram_range':     (1, 2),

    # PCA
    'pca_dims':        [500, 1600],   # dimensions to test

    # Cross Validation
    'cv_folds':        5,
    'C_values':        [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],

    # prediction
    'interactive_mode': False,      # True = enter interactive mode after training
}


def main():
    print(f"\n{'='*60}")
    print(f"  Data Cleaning & Splitting")
    print(f"{'='*60}")

    df = text_combined(CONFIG['data_folder'], CONFIG['data_files'])

    train_dataset, test_dataset = split_data(
        df, test_size=CONFIG['test_size'], seed=CONFIG['random_seed']
    )
    y_train = train_dataset[1]
    y_test = test_dataset[1]
    print(f"number of train datasets are {len(train_dataset[0])}"
          f"(positive ={sum(y_train == 1)}, negative={sum(y_train == 0)})")
    print(f"number of test datasets are {len(test_dataset[0])}"
          f"(positive={sum(y_test == 1)}, negative={sum(y_test == 0)})")


    print(f"\n{'='*60}")
    print(f"  TF-IDF Feature Extraction")
    print(f"{'='*60}")

    X_train_tfidf, X_test_tfidf, vectorizer = tfidf_features(
        train_dataset, test_dataset,
        max=CONFIG['max_features'],
        ngram=CONFIG['ngram_range']
    )

    # show first 3 reviews' words
    for i in range(min(3, X_train_tfidf.shape[0])):
        inspect_tfidf(X_train_tfidf, vectorizer, sample_index=i)

    print(f"\n{'='*60}")
    print(f"  PCA Dimensionality Reduction")
    print(f"{'='*60}")

    # find optimal dimension + plot variance curve
    n_optimal, cumvar = optimal_dim(X_train_tfidf, threshold=85)
    plot_pca_variance(cumvar)

    # run PCA at each target dimension
    pca_results = {}
    for n_comp in CONFIG['pca_dims']:
        print(f"\n  --- PCA {n_comp}D ---")
        X_tr_pca, X_te_pca, scaler, pca_model = apply_pca(
            X_train_tfidf, X_test_tfidf, n_components=n_comp
        )
        pca_results[n_comp] = {
            'X_train': X_tr_pca,
            'X_test': X_te_pca,
            'scaler': scaler,
            'pca': pca_model
        }

    print(f"\n{'='*60}")
    print(f"  Cross Validation — Tune C")
    print(f"{'='*60}")

    C_values, cv_mean, cv_std, best_C = tune_C_with_cv(
        X_train_tfidf, y_train,
        kernel='linear',
        cv_folds=CONFIG['cv_folds'],
        C_values=CONFIG['C_values']
    )
    plot_cv_curve(C_values, cv_mean, cv_std)

    print(f"\n{'='*60}")
    print(f"  4 SVM Experiments (best C={best_C})")
    print(f"{'='*60}")

    all_results = run_all_experiments(
        X_train_tfidf, X_test_tfidf,
        pca_results, y_train, y_test,
        best_C=best_C
    )

    print(f"\n{'='*60}")
    print(f"  Visualization")
    print(f"{'='*60}")

    print_summary(all_results)
    plot_confusion_matrices(all_results)
    plot_accuracy(all_results)
    plot_time(all_results)

    print(f"\n{'='*60}")
    print(f"  Save Model & Prediction Demo")
    print(f"{'='*60}")

    # pick the best experiment by accuracy
    best_exp = max(all_results, key=lambda x: x['accuracy'])
    print(f"  Best model: {best_exp['name']} ({best_exp['accuracy']*100:.2f}%)")

    # check if best model used PCA
    use_pca = 'PCA' in best_exp['name']
    n_comp = None

    if use_pca:
        match = re.search(r'PCA-(\d+)D', best_exp['name'])
        n_comp = int(match.group(1))
        save_pipeline(
            vectorizer=vectorizer,
            svm_model=best_exp['model'],
            scaler=pca_results[n_comp]['scaler'],
            pca=pca_results[n_comp]['pca'],
            use_pca=True
        )
    else:
        save_pipeline(
            vectorizer=vectorizer,
            svm_model=best_exp['model'],
            use_pca=False
        )

    # create predictor
    predictor = SentimentPredictor(
        vectorizer=vectorizer,
        svm_model=best_exp['model'],
        scaler=pca_results[n_comp]['scaler'] if use_pca else None,
        pca=pca_results[n_comp]['pca'] if use_pca else None,
        use_pca=use_pca)

    # demo predictions
    demo_texts = [
        "This product is absolutely amazing! I love it.",
        "Terrible quality, broke after one day. Waste of money.",
        "Best purchase I've ever made, highly recommend!",
        "Do not buy this, poor customer service.",
        "It works as expected, good value for the price.",
    ]

    print(f"\n  --- Prediction Demo ---")
    for t in demo_texts:
        predictor.predict(t)

    # optional: interactive mode
    if CONFIG['interactive_mode']:
        interactive_predict(predictor)

    print(f"\n{'='*60}")
    print(f"  COMPLETE — all figures saved, model at saved_models/best_model.pkl")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()