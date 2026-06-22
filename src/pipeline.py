import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

class TextPreprocessingPipeline:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=2500)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        cleaned_tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(cleaned_tokens)

    def fit_transform(self, text_series: pd.Series):
        cleaned_series = text_series.apply(self.clean_text)
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_series)
        return tfidf_matrix, cleaned_series

    def transform(self, text_series: pd.Series):
        cleaned_series = text_series.apply(self.clean_text)
        return self.vectorizer.transform(cleaned_series)
