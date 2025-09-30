from data_preprocessing import load_and_preprocess_data
from model import build_model, save_model, load_model
from sklearn.metrics import accuracy_score, classification_report


def main():
    print("🔹 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data()

    print("🔹 Building model...")
    model = build_model()

    print("🔹 Training model...")
    model.fit(X_train, y_train)

    print("🔹 Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("🔹 Saving model and vectorizer...")
    save_model(model, vectorizer)

    # === Sentiment Prediction from User Input ===
    print("\n💬 Sentiment Prediction Mode (type 'exit' to quit):")
    model, vectorizer = load_model()

    while True:
        text = input("\nEnter a movie review: ")
        if text.lower() == "exit":
            print("👋 Exiting Sentiment Prediction.")
            break

        text_tfidf = vectorizer.transform([text])
        pred = model.predict(text_tfidf)[0]
        sentiment = "Positive 😄" if pred == 1 else "Negative 😞"
        print(f"👉 Sentiment: {sentiment}")


if __name__ == "__main__":
    main()
