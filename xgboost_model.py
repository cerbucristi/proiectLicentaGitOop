import os
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from base_model import BaseModel

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed. Install dependencies from requirements.txt (pip install -r requirements.txt)."
    ) from exc


class XGBoostModel(BaseModel):
    def __init__(
        self,
        feature_mode="word_char_tfidf",
        max_features=20000,
        random_state=42,
        output_dir="artifacts/classical/xgboost",
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.7,
    ):
        self.feature_mode = feature_mode
        self.max_features = max_features
        self.random_state = random_state
        self.output_dir = output_dir

        self.vectorizer = self._create_vectorizer()
        self.word_vectorizer = None
        self.char_vectorizer = None

        if self.feature_mode == "word_char_tfidf":
            split_features = max(1, self.max_features // 2)
            self.word_vectorizer = self._create_word_vectorizer(split_features)
            self.char_vectorizer = self._create_char_vectorizer(split_features)

        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_bin=64,
            random_state=self.random_state,
            tree_method="hist",
            n_jobs=-1,
        )

    def _create_word_vectorizer(self, max_features):
        return TfidfVectorizer(
            analyzer="word",
            token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
            ngram_range=(1, 3),
            max_features=max_features,
            dtype=np.float32,
        )

    def _create_char_vectorizer(self, max_features):
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
            dtype=np.float32,
        )

    def _create_vectorizer(self):
        if self.feature_mode == "word_tfidf":
            return self._create_word_vectorizer(self.max_features)

        if self.feature_mode == "char_tfidf":
            return self._create_char_vectorizer(self.max_features)

        if self.feature_mode == "word_char_tfidf":
            return None

        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def _fit_transform_features(self, X):
        if self.feature_mode == "word_char_tfidf":
            X_word = self.word_vectorizer.fit_transform(X)
            X_char = self.char_vectorizer.fit_transform(X)
            return hstack([X_word, X_char], format="csr")
        return self.vectorizer.fit_transform(X)

    def _transform_features(self, X):
        if self.feature_mode == "word_char_tfidf":
            X_word = self.word_vectorizer.transform(X)
            X_char = self.char_vectorizer.transform(X)
            return hstack([X_word, X_char], format="csr")
        return self.vectorizer.transform(X)

    def train(self, samples):
        os.makedirs(self.output_dir, exist_ok=True)

        X = [s["code"] for s in samples]
        y = np.array([s["label"] for s in samples], dtype=np.int32)

        X_vec = self._fit_transform_features(X)
        if X_vec.dtype != np.float32:
            X_vec = X_vec.astype(np.float32)

        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos > 0:
            self.model.set_params(scale_pos_weight=float(neg / pos))

        self.model.fit(X_vec, y)

    def predict(self, code: str) -> float:
        X = self._transform_features([code])
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return float(self.model.predict_proba(X)[0][1])

    def predict_batch(self, codes):
        X = self._transform_features(codes)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "word_vectorizer": self.word_vectorizer,
                "char_vectorizer": self.char_vectorizer,
                "model": self.model,
                "feature_mode": self.feature_mode,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "output_dir": self.output_dir,
            },
            path,
        )

    def load(self, path: str):
        artifact = joblib.load(path)
        self.vectorizer = artifact["vectorizer"]
        self.word_vectorizer = artifact.get("word_vectorizer")
        self.char_vectorizer = artifact.get("char_vectorizer")
        self.model = artifact["model"]
        self.feature_mode = artifact.get("feature_mode", "word_char_tfidf")
        self.max_features = artifact.get("max_features", 50000)
        self.random_state = artifact.get("random_state", 42)
        self.output_dir = artifact.get("output_dir", "artifacts/classical/xgboost")
