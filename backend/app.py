"""
Sentiment Analyzer — Flask REST API
Author: Farhan Sayyed
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')

# ── Preprocessing ────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)              # keep letters only
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ── Load or train model ──────────────────────────────────────────────
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

pipeline = load_model()

# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return jsonify({'status': 'Sentiment Analyzer API running', 'endpoints': ['/api/analyze', '/api/health']})

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': pipeline is not None})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({'error': 'Provide {"text": "your text here"}'}), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    if pipeline is None:
        return jsonify({'error': 'Model not loaded. Run: python model/train.py'}), 503

    clean = preprocess(text)
    prediction = pipeline.predict([clean])[0]
    probabilities = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    confidence = float(max(probabilities))

    labels = {'positive': 'Positive 😊', 'negative': 'Negative 😔', 'neutral': 'Neutral 😐'}
    scores = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}

    return jsonify({
        'sentiment': prediction,
        'label': labels.get(prediction, prediction.capitalize()),
        'confidence': round(confidence, 4),
        'scores': scores,
        'word_count': len(text.split())
    })

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    data = request.get_json(silent=True)
    if not data or 'texts' not in data:
        return jsonify({'error': 'Provide {"texts": ["text1", "text2", ...]}'}), 400
    if pipeline is None:
        return jsonify({'error': 'Model not loaded'}), 503
    results = []
    for text in data['texts'][:50]:  # limit to 50
        clean = preprocess(str(text))
        pred = pipeline.predict([clean])[0]
        conf = float(max(pipeline.predict_proba([clean])[0]))
        results.append({'text': text[:100], 'sentiment': pred, 'confidence': round(conf, 4)})
    return jsonify({'results': results, 'count': len(results)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
