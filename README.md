# Sentiment Analyzer

An NLP-powered web application that classifies text as **Positive**, **Negative**, or **Neutral** using machine learning. Built with a Python/Flask backend and a React frontend.

## 🚀 Features

- Real-time text sentiment analysis
- Confidence score with visual probability bar
- Batch analysis (upload a `.txt` file)
- REST API endpoint — integrate with any app
- Clean dark-mode UI

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Tailwind CSS |
| Backend | Python, Flask, Flask-CORS |
| ML Model | scikit-learn (Naive Bayes / Logistic Regression) |
| NLP | NLTK, TF-IDF Vectorizer |
| Deployment | Gunicorn + Render |

## 📂 Project Structure

```
sentiment-analyzer/
├── backend/
│   ├── app.py            # Flask API
│   ├── model/
│   │   ├── train.py      # Model training script
│   │   └── model.pkl     # Serialised trained model
│   ├── utils/
│   │   └── preprocess.py # Text cleaning utilities
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   └── index.html
└── README.md
```

## ⚙️ Getting Started

```bash
# Clone
git clone https://github.com/farhansayyedd/sentiment-analyzer.git
cd sentiment-analyzer

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python model/train.py     # Train & save the model
python app.py             # Start Flask server at :5000

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev               # React dev server at :5173
```

## 🔌 API Usage

```bash
POST /api/analyze
Content-Type: application/json

{ "text": "I love this course!" }

# Response
{ "sentiment": "positive", "confidence": 0.94, "label": "Positive 😊" }
```

## 📄 License

MIT © Farhan Sayyed
