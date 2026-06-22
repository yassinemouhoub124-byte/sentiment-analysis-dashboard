import os
import sys
import sqlite3
import pickle
import pandas as pd
import datetime
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Fix the module path issue so Python can see pipeline.py inside 'src'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Set page style
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Paths setup
DB_PATH = os.path.join(PROJECT_ROOT, "sentiment_logs.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "pipeline.pkl")


# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS sentiment_logs
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       review_text
                       TEXT,
                       predicted_sentiment
                       TEXT,
                       confidence_score
                       REAL,
                       timestamp
                       TEXT
                   )
                   ''')
    conn.commit()
    conn.close()


def log_prediction(text, sentiment, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
                   INSERT INTO sentiment_logs (review_text, predicted_sentiment, confidence_score, timestamp)
                   VALUES (?, ?, ?, ?)
                   ''', (text, sentiment, confidence, now))
    conn.commit()
    conn.close()


def fetch_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sentiment_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df


# Initialize database table
init_db()


# --- LOAD MODELS ---
@st.cache_resource(show_spinner=False)
def load_ml_components():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return model, pipeline


model, pipeline = load_ml_components()

# --- UI HEADER ---
st.title("📊 Customer Feedback Sentiment Analysis Platform")
st.markdown("WEX 328 — End-to-End AI System Deployment")
st.write("---")

if model is None:
    st.error(
        "⏳ Best performing model files not found. Please run your `train.py` script first to generate the model files!")
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Evaluate New Custom Review")
        user_input = st.text_area("Paste customer review feedback here:", height=150,
                                  placeholder="Type something like: 'This device is wonderful, I highly recommend it!'")

        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip() == "":
                st.warning("Please type a valid review before analyzing.")
            else:
                # Run text through pre-trained feature extraction pipeline
                transformed_text = pipeline.transform(pd.Series([user_input]))

                # Make prediction
                pred_class = model.predict(transformed_text)[0]
                probabilities = model.predict_proba(transformed_text)[0]

                sentiment_label = "Positive" if pred_class == 1 else "Negative"
                confidence = probabilities[pred_class]

                # Log straight into SQL database
                log_prediction(user_input, sentiment_label, confidence)

                # Visual Alert Result
                if sentiment_label == "Positive":
                    st.success(
                        f"🎯 **Predicted Sentiment:** Positive  \n👍 **Confidence Score:** {confidence * 100:.2f}%")
                else:
                    st.error(f"🎯 **Predicted Sentiment:** Negative  \n👎 **Confidence Score:** {confidence * 100:.2f}%")

    with col2:
        st.subheader("📈 Relational Database Logs & Trends")
        df_logs = fetch_logs()

        if df_logs.empty:
            st.info(
                "The SQL database is currently empty. Analyze a few custom reviews on the left to generate graphs instantly!")
        else:
            st.metric("Total Logged Requests", len(df_logs))

            # Matplotlib Chart Generation
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sentiment_counts = df_logs['predicted_sentiment'].value_counts()

            # Explicit category rendering
            all_categories = ['Positive', 'Negative']
            counts = [sentiment_counts.get(cat, 0) for cat in all_categories]
            colors = ['#2ecc71', '#e74c3c']

            sns.barplot(x=all_categories, y=counts, palette=colors, ax=ax)
            ax.set_title("Historical Sentiment Distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    if not df_logs.empty:
        st.write("---")
        st.subheader("📋 Raw Database Log History (`SQLite3`)")
        st.dataframe(df_logs, use_container_width=True)

