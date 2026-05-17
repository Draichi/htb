"""
Sentiment classification for HTB skills assessment.
Pipeline: HTML cleanup → lowercase → CountVectorizer → MultinomialNB
Follows spam_classification.ipynb structure.
"""

import json
import re
import joblib
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("skills_assessment_data")
TRAIN_FILE      = DATA_DIR / "train.json"
TEST_FILE       = DATA_DIR / "test.json"
MODEL_FILE_PATH = "skills_assessment.joblib"

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Lowercase
    text = text.lower()
    # Remove non-essential punctuation, keep $ and !
    text = re.sub(r"[^a-z\s$!]", "", text)
    # Tokenize and rejoin (no stopword removal, no stemming — negation matters for sentiment)
    tokens = word_tokenize(text)
    return " ".join(tokens)

# ── Data loading ───────────────────────────────────────────────────────────────
def load(path: Path) -> tuple[list[str], np.ndarray]:
    records = json.loads(path.read_text())
    texts  = [preprocess(r["text"]) for r in records]
    labels = np.array([r["label"] for r in records])
    return texts, labels

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=== Loading & preprocessing data ===")
    train_texts, y_train = load(TRAIN_FILE)
    test_texts,  y_test  = load(TEST_FILE)
    print(f"  Train: {len(train_texts)}  |  Test: {len(test_texts)}")

    print("\n=== Building pipeline ===")
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2), max_features=50_000)),
        ("classifier", MultinomialNB()),
    ])

    param_grid = {
        "classifier__alpha": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
    }

    print("=== Grid search (5-fold CV, scoring=f1) ===")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    grid_search.fit(train_texts, y_train)

    best_model = grid_search.best_estimator_
    print(f"\n  Best params: {grid_search.best_params_}")

    print("\n=== Evaluation on test set ===")
    preds = best_model.predict(test_texts)
    acc   = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["negative", "positive"]))

    joblib.dump(best_model, MODEL_FILE_PATH)
    print(f"Model saved → {MODEL_FILE_PATH}")

if __name__ == "__main__":
    main()
