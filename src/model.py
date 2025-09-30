from sklearn.linear_model import LogisticRegression
import joblib
import os


def build_model():
    """Create and return a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000)
    return model


def get_result_dir():
    """Return the path to the result folder, create it if missing."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "..", "result")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_model(model, vectorizer):
    """Save trained model and vectorizer into /result directory."""
    result_dir = get_result_dir()
    model_path = os.path.join(result_dir, "sentiment_model.pkl")
    vectorizer_path = os.path.join(result_dir, "tfidf_vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Vectorizer saved to: {vectorizer_path}")


def load_model():
    """Load trained model and vectorizer from /result directory."""
    result_dir = get_result_dir()
    model_path = os.path.join(result_dir, "sentiment_model.pkl")
    vectorizer_path = os.path.join(result_dir, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("❌ Model or vectorizer not found in /result. Please train first!")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
