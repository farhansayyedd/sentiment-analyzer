"""
Model Training Script — Sentiment Analyzer
Trains a Logistic Regression classifier on the NLTK movie_reviews dataset.
Run: python model/train.py
Author: Farhan Sayyed
"""
import os, pickle
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
from nltk.corpus import stopwords, movie_reviews
from nltk.stem import PorterStemmer

nltk.download('movie_reviews', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

print("Loading dataset...")
documents = [
    (' '.join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
texts, labels = zip(*documents)
labels = ['positive' if l == 'pos' else 'negative' for l in labels]
texts = [preprocess(t) for t in texts]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2))),
    ('clf',   LogisticRegression(max_iter=500, C=1.0))
])

print("Training model...")
pipeline.fit(X_train, y_train)

print("\nEvaluation:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(__file__), exist_ok=True)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"\nModel saved → {MODEL_PATH}")
