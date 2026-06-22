
CREATE TABLE IF NOT EXISTS sentiment_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text TEXT NOT NULL,
    predicted_sentiment TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_timestamp 
ON sentiment_logs (timestamp DESC);
