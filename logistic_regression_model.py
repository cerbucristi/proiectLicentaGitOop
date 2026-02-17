
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from base_model import BaseModel


class LogisticRegressionModel(BaseModel):


    def __init__(self):
        self.vectorizer = TfidfVectorizer (
            analyzer="word",
            token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
            ngram_range=(1,3),
            max_features=50000
        )

        self.model = LogisticRegression(class_weight="balanced")

    def train(self, samples):
        X = [s["code"] for s in samples]
        y = [s["label"] for s in samples]

        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        return super().train(samples)
    
    def predict(self, code):
        X = self.vectorizer.transform([code])
        return self.model.predict_proba(X)[0][1]
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump((self.vectorizer, self.model), path)

    def load(self, path):
        self.vectorizer, self.model = joblib.load(path)