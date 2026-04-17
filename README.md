# Sentiment Analysis of Product Reviews Using PCA and SVM

---

## Overview

This project applies **Support Vector Machine (SVM)** to classify Amazon product reviews as positive or negative. The feature extraction pipeline uses **TF-IDF** vectorization, and **PCA** (implemented from scratch) is used for dimensionality reduction. The regularization parameter is tuned using **5-Fold Cross Validation**.

### Optimization Problem

$$\min_{w,b} \; \frac{1}{2}\|w\|^2 + C \sum_{i=1}^{N} \max\bigl(0,\; 1 - y_i(w^T x_i + b)\bigr)$$

where the first term is L2 regularization and the second term is hinge loss.

---

## Project Structure

```
project/
│
├── data/
│   └── raw/
│       └── amazon_cells_labelled.txt    # raw dataset (1000 reviews)
│
├── data_cleaner.py            # Stage 1-2: text cleaning + train/test split
├── feature_engineering.py     # Stage 3a:  TF-IDF feature extraction
├── pca.py                     # Stage 3b:  PCA from scratch (no sklearn.decomposition)
├── svm_classifier.py          # Stage 4:   SVM + Cross Validation
├── visualization.py           # Stage 5:   report figures
├── predictor.py               # Stage 6:   save model + predict new reviews
│
├── main.py                    # main entry — runs Stage 1 through 6
├── predict_standalone.py      # predict without retraining
├── verify_pca.py              # verify handcrafted PCA against sklearn
│
├── saved_models/              # (generated) saved model files
│   └── best_model.pkl
│
└── README.md
```

---

## Pipeline

```
Stage 1    clean_text()          Raw text → lowercase, remove punctuation, truncate
           ↓
Stage 2    split_data()          80/20 train/test split (stratified)
           ↓
Stage 3a   tfidf_features()     Text → TF-IDF sparse matrix (494 dimensions)
           ↓
Stage 3b   apply_pca()          TF-IDF → PCA reduced (50D / 200D) — from scratch
           ↓
Stage 4a   tune_C_with_cv()     5-Fold Cross Validation to find best C
           ↓
Stage 4b   run_all_experiments() 4 comparison experiments
           ↓
Stage 5    plot_*()              Generate 5 report figures
           ↓
Stage 6    save + predict        Save best model, demo predictions
```

---

## Experiments

| # | Method | Features | Purpose |
|---|--------|----------|---------|
| 1 | Linear SVM + TF-IDF | 494D (original) | Baseline |
| 2 | Linear SVM + PCA-50D | 50D (reduced) | Aggressive reduction |
| 3 | Linear SVM + PCA-200D | 200D (reduced) | Moderate reduction |
| 4 | RBF SVM + TF-IDF | 494D (original) | Nonlinear kernel |

### Comparison Dimensions

- **Exp 1 vs 2, 3** → Does PCA dimensionality reduction help?
- **Exp 1 vs 4** → Is a nonlinear kernel necessary?
- **Exp 2 vs 3** → Trade-off between compression and accuracy

---

## Quick Start

### Requirements

```
Python 3.8+
numpy
pandas
scikit-learn
matplotlib
```

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Run Full Pipeline

```bash
python main.py
```

This will:
1. Clean and split the data
2. Extract TF-IDF features
3. Run handcrafted PCA
4. Tune C with 5-Fold Cross Validation
5. Run 4 SVM experiments
6. Generate 5 figures
7. Save best model to `saved_models/best_model.pkl`
8. Demo predictions on sample reviews

### Predict New Reviews (after training)

```bash
python predict_standalone.py
```

### Verify Handcrafted PCA

```bash
python verify_pca.py
```

Compares handcrafted PCA (eigendecomposition) against sklearn PCA (SVD) step by step.

---

## Configuration

All parameters are in `main.py` under `CONFIG`:

```python
CONFIG = {
    'max_chars':       100,         # truncate reviews to N characters
    'test_size':       0.2,         # 0.2 = 80/20 split
    'max_features':    1000,        # TF-IDF vocabulary size cap
    'ngram_range':     (1, 2),      # unigram + bigram
    'pca_dims':        [50, 200],   # PCA dimensions to test
    'cv_folds':        5,           # Cross Validation folds
    'interactive_mode': False,      # True = interactive prediction after training
}
```

---

## Generated Figures

| Figure | Filename | Report Section |
|--------|----------|---------------|
| PCA cumulative variance curve | `fig0_pca_variance.png` | Section IV |
| Confusion matrices (all experiments) | `fig1_confusion_matrices.png` | Section IV |
| Accuracy & F1 comparison | `fig2_accuracy.png` | Section IV |
| Training/testing time comparison | `fig3_time.png` | Section IV |
| CV accuracy vs C (with error bars) | `fig4_cv_tuning.png` | Section IV |

---

## Handcrafted PCA

The PCA implementation in `pca.py` does **not** use `sklearn.decomposition.PCA`. It follows the mathematical formulation directly:

1. **Centering**: $X_c = X - \mu$
2. **Covariance**: $S = \frac{1}{N-1} X_c^T X_c$
3. **Eigendecomposition**: $S = W \Lambda W^T$ via `numpy.linalg.eigh`
4. **Select top-k**: $W_k = W[:, :k]$
5. **Project**: $X_{\text{reduced}} = X_c \cdot W_k$

`MyStandardScaler` is also handcrafted (no `sklearn.preprocessing.StandardScaler`).

Verified against sklearn on real training data — eigenvalues match within 0.5% relative error (due to eigh vs SVD numerical differences, not a bug).

---

## Dataset

**Sentiment Labelled Sentences Data Set** from UCI Machine Learning Repository.

- Source: Amazon product reviews
- Size: 1000 reviews (500 positive, 500 negative)
- After cleaning (truncation + filtering): ~881 reviews
- Format: `sentence \t label` (label: 0 = negative, 1 = positive)

Reference:  
Kotzias et al., "From Group to Individual Labels using Deep Features," KDD 2015.

---

## Report Structure Mapping

| Report Section | Code Module |
|---------------|-------------|
| I. Introduction | Background of sentiment analysis + SVM/PCA theory |
| II. Optimization Problem | SVM hinge loss formula + PCA optimization formula |
| III. Solution Methods | Pipeline description + CV tuning + 4 experiments |
| IV. Simulation Results | Figures generated by `visualization.py` |
| V. Conclusion | Summary from `print_summary()` |
| Appendix | `pca.py` (handcrafted PCA code) |