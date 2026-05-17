import json
import re
import joblib

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s$!]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    tokens = [STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)


def load_json(path: str) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


VECTORIZER = CountVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2))

CANDIDATES = [
    (
        "LogisticRegression",
        Pipeline([("vectorizer", VECTORIZER), ("classifier", LogisticRegression(max_iter=1000))]),
        {"classifier__C": [0.1, 0.5, 1.0, 5.0, 10.0]},
    ),
    (
        "LinearSVC",
        Pipeline([("vectorizer", VECTORIZER), ("classifier", LinearSVC(max_iter=2000))]),
        {"classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0]},
    ),
]


def main():
    print("Loading data...")
    train_df = load_json("skills_assessment_data/train.json")
    test_df = load_json("skills_assessment_data/test.json")

    print("Preprocessing...")
    train_df["processed"] = train_df["text"].apply(preprocess)
    test_df["processed"] = test_df["text"].apply(preprocess)

    X_train, y_train = train_df["processed"], train_df["label"]
    X_test, y_test = test_df["processed"], test_df["label"]

    best_score, best_model, best_name = 0.0, None, ""

    for name, pipeline, param_grid in CANDIDATES:
        print(f"\n--- {name} ---")
        gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        print(f"Best params: {gs.best_params_}  |  CV F1: {gs.best_score_:.4f}")

        y_pred = gs.best_estimator_.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_
            best_name = name

    print(f"\nBest classifier: {best_name} (CV F1={best_score:.4f})")
    model_path = "skills_assessment.joblib"
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
