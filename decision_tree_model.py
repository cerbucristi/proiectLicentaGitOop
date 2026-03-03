import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, feature_mode="word_tfidf", max_features=50000, random_state=42):
        self.feature_mode = feature_mode
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = self._create_vectorizer()
        self.model = DecisionTreeClassifier(
            criterion="entropy",
            random_state=self.random_state
        )

    def _create_vectorizer(self):
        if self.feature_mode == "word_tfidf":
            return TfidfVectorizer(
                analyzer="word",
                token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
                ngram_range=(1, 3),
                max_features=self.max_features
            )

        if self.feature_mode == "char_tfidf":
            return TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=self.max_features
            )

        raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

    def _row_signature(self, sparse_row):
        indices = sparse_row.indices
        values = sparse_row.data
        if len(indices) == 0:
            return "__EMPTY__"

        pairs = [f"{idx}:{val:.8f}" for idx, val in zip(indices, values)]
        return "|".join(pairs)

    def analyze_feature_label_conflicts(self, X_vec, y, output_dir="artifacts/classical/id3"):
        os.makedirs(output_dir, exist_ok=True)

        signatures = [self._row_signature(X_vec.getrow(i)) for i in range(X_vec.shape[0])]
        df = pd.DataFrame({"feature_signature": signatures, "label": y})

        grouped = df.groupby("feature_signature")["label"]
        duplicate_groups = grouped.size()
        conflicting = grouped.nunique()

        duplicate_count = int((duplicate_groups > 1).sum())
        conflict_mask = conflicting > 1
        conflict_count = int(conflict_mask.sum())

        conflicts_df = pd.DataFrame({
            "group_size": duplicate_groups[conflict_mask],
            "distinct_labels": conflicting[conflict_mask]
        }).sort_values(by="group_size", ascending=False)

        report_path = os.path.join(output_dir, f"id3_{self.feature_mode}_feature_conflicts.csv")
        if not conflicts_df.empty:
            conflicts_df.to_csv(report_path)
            print(f"[{self.feature_mode}] Found {conflict_count} conflicting duplicate feature groups.")
            print(f"[{self.feature_mode}] Conflict report saved to: {report_path}")
        else:
            print(f"[{self.feature_mode}] No conflicting duplicate feature groups found.")

        print(f"[{self.feature_mode}] Duplicate feature groups (any label): {duplicate_count}")

    def _save_loss_plot(self, X_vec, y, output_dir="artifacts/classical/id3", max_points=20):
        os.makedirs(output_dir, exist_ok=True)

        fitted_depth = self.model.get_depth()
        if fitted_depth <= 0:
            print(f"[{self.feature_mode}] Tree depth is 0, skipping loss plot generation.")
            return

        depth_values = np.linspace(1, fitted_depth, num=min(max_points, fitted_depth), dtype=int)
        depth_values = sorted(set(int(d) for d in depth_values))

        losses = []
        for depth in depth_values:
            temp_model = DecisionTreeClassifier(
                criterion="entropy",
                random_state=self.random_state,
                max_depth=depth
            )
            temp_model.fit(X_vec, y)
            y_prob = temp_model.predict_proba(X_vec)
            losses.append(float(log_loss(y, y_prob, labels=[0, 1])))

        plot_path = os.path.join(output_dir, f"id3_{self.feature_mode}_loss_vs_depth.png")
        plt.figure(figsize=(10, 6))
        plt.plot(depth_values, losses, marker="o", linewidth=1.5)
        plt.title(f"ID3 ({self.feature_mode}) Training Loss vs Tree Depth")
        plt.xlabel("Tree Depth (max_depth)")
        plt.ylabel("Log Loss (train)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[{self.feature_mode}] Saved ID3 loss plot to: {plot_path}")

    def train(self, samples):
        X = [s["code"] for s in samples]
        y = [s["label"] for s in samples]

        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self._save_loss_plot(X_vec, y)
        self.analyze_feature_label_conflicts(X_vec, y)

    def predict(self, code: str) -> float:
        X = self.vectorizer.transform([code])
        return self.model.predict_proba(X)[0][1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "model": self.model,
                "feature_mode": self.feature_mode,
                "max_features": self.max_features,
                "random_state": self.random_state,
            },
            path,
        )

    def load(self, path: str):
        artifact = joblib.load(path)
        self.vectorizer = artifact["vectorizer"]
        self.model = artifact["model"]
        self.feature_mode = artifact.get("feature_mode", "word_tfidf")
        self.max_features = artifact.get("max_features", 50000)
        self.random_state = artifact.get("random_state", 42)
