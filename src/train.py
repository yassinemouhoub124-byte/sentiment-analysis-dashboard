import os
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# AUTOMATIC NLTK DOWNLOADER - DOWNLOADS ALL DATA AUTOMATICALLY
import nltk

print("📦 Checking and downloading required text processing data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Import the pipeline we created earlier
from pipeline import TextPreprocessingPipeline


def load_fasttext_data(file_path, num_samples=10000):
    reviews = []
    labels = []

    print(f"📖 Reading a sample of {num_samples} reviews from the dataset...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            match = re.match(r'__label__(\d)\s+(.*)', line.strip())
            if match:
                # __label__2 (Positive) -> 1, __label__1 (Negative) -> 0
                label = 1 if match.group(1) == '2' else 0
                review = match.group(2)
                labels.append(label)
                reviews.append(review)

    return pd.Series(reviews), pd.Series(labels)


def train_models():
    # Bulletproof absolute paths tracking
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")

    if not os.path.exists(data_dir):
        print(f"❌ Error: Cannot find data directory at {data_dir}")
        return

    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]

    if not txt_files:
        print("❌ Error: Could not find any .txt dataset file inside your 'data' folder.")
        return

    data_path = os.path.join(data_dir, txt_files[0])
    print(f"Found dataset file at: {data_path}")

    # 1. Load data sample format
    X_raw, y = load_fasttext_data(data_path, num_samples=10000)

    print("🧹 Running NLP Preprocessing Pipeline (Cleaning, Tokenizing, TF-IDF)...")
    pipeline = TextPreprocessingPipeline()
    X_tfidf, _ = pipeline.fit_transform(X_raw)

    # 2. Split data into 80% Train and 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # 3. Initialize the 3 models required by your project
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Machine (SVM)": SVC(kernel='linear', probability=True)
    }

    best_accuracy = 0
    best_model = None
    best_model_name = ""

    print("\n🏋️ Training and evaluating models...")
    print("-" * 50)

    # 4. Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        print(f"⭐ Model: {name}")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        print("-" * 50)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    # 5. Save everything for the dashboard execution inside the root directory
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(models_dir, "pipeline.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\n✅ Done! The best model was '{best_model_name}' with {best_accuracy * 100:.2f}% accuracy.")
    print("Saved files into the 'models' folder.")


if __name__ == "__main__":
    train_models()
