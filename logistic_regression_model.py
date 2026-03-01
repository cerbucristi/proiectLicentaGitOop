
import joblib
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from base_model import BaseModel


class LogisticRegressionModel(BaseModel):


    def __init__(self):
        self.vectorizer = TfidfVectorizer (
            analyzer="word",
            token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
            ngram_range=(1,3),
            max_features=50000
        )

        self.model = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            warm_start=True,
            max_iter=1,
            random_state=42
        )
        self.training_epochs = 30

    def _save_loss_plot(self, epochs, losses, output_dir="artifacts/classical"):
        if not losses:
            print("No Logistic Regression loss values found, skipping loss plot generation.")
            return

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "loss_vs_epoch.png")

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker="o", linewidth=1.5)
        plt.title("Logistic Regression Training Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Saved Logistic Regression loss plot to: {plot_path}")

    def train(self, samples):
        X = [s["code"] for s in samples]
        y = [s["label"] for s in samples]

        X_vec = self.vectorizer.fit_transform(X)
        epochs = []
        losses = []

        for epoch in range(1, self.training_epochs + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                self.model.fit(X_vec, y)

            y_prob = self.model.predict_proba(X_vec)[:, 1]
            current_loss = log_loss(y, y_prob, labels=[0, 1])

            epochs.append(epoch)
            losses.append(float(current_loss))

        self._save_loss_plot(epochs, losses)
    
    def predict(self, code):
        X = self.vectorizer.transform([code])
        return self.model.predict_proba(X)[0][1]
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump((self.vectorizer, self.model), path)

    def load(self, path):
        self.vectorizer, self.model = joblib.load(path)