# 📊 Customer Feedback Sentiment Analysis Platform

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to classify customer reviews and log evaluations into a relational storage system. This project was developed as part of the **WEX 328 — End-to-End AI System Deployment** course requirements.

---

## 🛠️ System Architecture & Workflow

1. **Data Preprocessing Pipeline (`src/pipeline.py`):** Cleans text inputs by lowering case, removing HTML tags/URLs/digits/punctuation, tokenizing text via NLTK, stripping out standard English stopwords, and vectorizing strings into features using a `TfidfVectorizer` (configured for 2,500 max features).
2. **Model Training & Evaluation (`src/train.py`):** Parses data in fastText format, trains three separate classifiers (Logistic Regression, Naive Bayes, and SVM), prints comparative evaluation metrics, and saves the highest-performing serialized weights (`.pkl`) into the `models/` directory.
3. **Database Logging Layer (`app.py`):** Automatically initializes a local relational `SQLite3` database (`sentiment_logs.db`) to record all live inputs, target predictions, confidence probabilities, and absolute execution timestamps.
4. **Interactive UI (`app.py` & `run.py`):** A modern, responsive Streamlit dashboard providing immediate real-time inferences, live database querying, and a historical data distribution bar chart built with Seaborn.

---

## 📊 Evaluation & Model Performance

The comparative metrics obtained during the workspace cross-validation phase on a 10,000-sample slice of the Amazon/Kaggle text dataset:

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **84.55%** | **0.84** | **0.85** | **0.84** |
| **Naive Bayes** | **83.15%** | **0.83** | **0.81** | **0.82** |
| **Support Vector Machine (SVM)** | *Omitted* | *Omitted* | *Omitted* | *Omitted* |

> 📌 **Deployment Decision:** **Logistic Regression** was selected as the active web application backend classifier because it demonstrated the highest classification performance metric (84.55% total accuracy).

---

## 📂 Repository Directory Tree

```text
sentiment-analysis-dashboard/
├── data/
│   └── train.ft.txt           # Raw Amazon/Kaggle text dataset source
├── models/
│   ├── best_model.pkl         # Serialized high-performing model weights
│   └── pipeline.pkl           # Fitted TF-IDF preprocessing vectorizer
├── src/
│   ├── pipeline.py            # Custom NLP preprocessing pipeline class
│   └── train.py               # Model execution and metrics exporter script
├── app.py                     # Streamlit dashboard and SQLite database engine
├── run.py                     # Automated PyCharm local server launcher
├── sentiment_logs.db          # Live generated SQLite database file
├── requirements.txt           # Manifest of required Python packages
└── README.md                  # Comprehensive technical documentation