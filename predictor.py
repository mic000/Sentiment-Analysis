"""
Stage 6: Save trained model + predict new reviews
"""

import os
import re
import pickle


def clean_single_text(text, char=None):
    """
    Clean a single input text — same logic as data_cleaner.clean_text()
    so that prediction uses identical preprocessing.
    """
    if char is not None:
        text = text[:int(char)]
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_pipeline(vectorizer, svm_model, save_dir="saved_models",
                   scaler=None, pca=None, use_pca=False, name="best_model"):
    """
    Save the full pipeline to disk.

    :param vectorizer: trained TfidfVectorizer
    :param svm_model: trained SVM model
    :param scaler: MyStandardScaler (only if PCA used)
    :param pca: MyPCA (only if PCA used)
    :param use_pca: whether PCA was used
    """
    os.makedirs(save_dir, exist_ok=True)

    pipeline = {
        'vectorizer': vectorizer,
        'svm_model': svm_model,
        'scaler': scaler,
        'pca': pca,
        'use_pca': use_pca,
    }

    path = os.path.join(save_dir, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)

    print(f"\n[Save] Pipeline saved to: {path}")
    return path


class SentimentPredictor:
    """
    Predict sentiment for new reviews.

    Usage:
        # from trained components
        predictor = SentimentPredictor(vectorizer, svm_model)
        result = predictor.predict("This product is amazing!")

        # from saved file
        predictor = SentimentPredictor.from_file("saved_models/best_model.pkl")
        result = predictor.predict("Terrible waste of money")
    """

    def __init__(self, vectorizer, svm_model,
                  scaler=None, pca=None, use_pca=False, max_chars=100):
        self.vectorizer = vectorizer
        self.svm_model = svm_model
        self.scaler = scaler
        self.pca = pca
        self.use_pca = use_pca
        self.max_chars = max_chars
        self.label_map = {0: "Negative", 1: "Positive"}

    @classmethod
    def from_file(cls, path, max_chars=100):
        """Load from saved pkl file."""
        with open(path, 'rb') as f:
            p = pickle.load(f)
        print(f"[Load] Pipeline loaded from {path}")
        return cls(
            vectorizer=p['vectorizer'], svm_model=p['svm_model'],
            scaler=p.get('scaler'), pca=p.get('pca'),
            use_pca=p.get('use_pca', False), max_chars=max_chars
        )

    def predict(self, text, show=True):
        """
        Predict sentiment of a single text.

        Pipeline: clean → TF-IDF (transform) → [PCA (transform)] → SVM (predict)
        """
        cleaned = clean_single_text(text, char=self.max_chars)
        if len(cleaned) == 0:
            return {'label': None, 'sentiment': "Unknown"}

        # TF-IDF — must use transform (not fit_transform!)
        X = self.vectorizer.transform([cleaned])

        # PCA if used
        if self.use_pca and self.scaler and self.pca:
            X_dense = X.toarray()
            X_scaled = self.scaler.transform(X_dense)
            X_final = self.pca.transform(X_scaled)
        else:
            X_final = X

        # SVM predict
        label = int(self.svm_model.predict(X_final)[0])
        sentiment = self.label_map[label]

        # confidence (distance to hyperplane)
        confidence = None
        try:
            confidence = abs(float(self.svm_model.decision_function(X_final)[0]))
        except AttributeError:
            pass

        result = {
            'text': text, 'cleaned': cleaned,
            'label': label, 'sentiment': sentiment,
            'confidence': confidence
        }

        if show:
            icon = "+" if label == 1 else "-"
            print(f"\n  [{icon}] {sentiment}")
            print(f"      Input:   {text}")
            print(f"      Cleaned: {cleaned}")
            if confidence is not None:
                level = "high" if confidence > 1.0 else ("mid" if confidence > 0.3 else "low")
                print(f"      Confidence: {confidence:.4f} ({level})")

        return result

    def predict_batch(self, texts):
        """Predict multiple texts."""
        return [self.predict(t, show=False) for t in texts]


def interactive_predict(predictor):
    """Interactive mode — user types reviews, model predicts."""
    print(f"\n{'='*50}")
    print(f"  Sentiment Predictor — Interactive Mode")
    print(f"  Type a review and press Enter. Type 'quit' to exit.")
    print(f"{'='*50}")

    while True:
        text = input("\n> Review: ").strip()
        if text.lower() in ('quit', 'exit', 'q'):
            print("Bye!")
            break
        if not text:
            print("(empty input, try again)")
            continue
        predictor.predict(text)