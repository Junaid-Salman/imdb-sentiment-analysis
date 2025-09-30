import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import string
import os


def clean_text(text):
    """Clean review text."""
    text = BeautifulSoup(text, "html.parser").get_text()  # remove HTML tags
    text = re.sub(r"[^a-zA-Z]", " ", text)                # keep only letters
    text = text.lower()                                   # lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text


def load_and_preprocess_data():
    """Load CSV, clean data, and return TF-IDF vectors + labels."""
    # Construct dataset path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "IMDB Dataset.csv")

    # Load dataset
    df = pd.read_csv(data_path)
    
    # Clean text
    df['review'] = df['review'].apply(clean_text)
    
    # Map sentiment to binary (positive -> 1, negative -> 0)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer
