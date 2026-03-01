
from base_model import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
from code_dataset import CodeDataset
import os
import matplotlib.pyplot as plt



class CodeBertModel(BaseModel):

    def __init__(self, model_name="microsoft/codebert-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CodeBertModel using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(self.device)


    def _build_dataset(self, samples):
        return CodeDataset(samples, self.tokenizer)

        # texts = [s["code"] for s in samples]
        # labels = [s["label"] for s in samples]

        # encodings = self.tokenizer (
        #     texts,
        #     truncation=True,
        #     padding=True,
        #     max_length=512
        # )

        # return CodeDataset(encodings, labels)

    def _save_loss_plot(self, log_history, output_dir):
        epochs = []
        losses = []

        for entry in log_history:
            if "loss" in entry and "epoch" in entry:
                epochs.append(float(entry["epoch"]))
                losses.append(float(entry["loss"]))

        if not losses:
            print("No training loss logs found, skipping loss plot generation.")
            return

        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "loss_vs_epoch.png")

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker="o", linewidth=1.5)
        plt.title("CodeBERT Training Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Saved CodeBERT loss plot to: {plot_path}")

    def train(self, samples):
        dataset = self._build_dataset(samples)

        args = TrainingArguments(
            output_dir="artifacts/codebert",
            num_train_epochs=3,
            per_device_train_batch_size=8, #with 2 will take more time
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none"
            # logging_steps=50,
            # save_steps=500,
            # save_total_limit=1,
            # report_to="none",
            # fp16=torch.cuda.is_available()
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            data_collator=data_collator
        )

        # trainer = Trainer(
        #     model=self.model,
        #     args=args,
        #     train_dataset=dataset
        # )

        trainer.train()
        self._save_loss_plot(trainer.state.log_history, args.output_dir)

    def predict(self, code: str) -> float:
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        return probs[0][1].item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
