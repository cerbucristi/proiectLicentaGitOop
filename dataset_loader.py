from datasets import load_dataset
from code_processor import CodePreprocessor
import pandas as pd

class MegaVulDataset:
    def __init__(self, streaming=True):
        self.streaming = streaming

    def load(self, limit=None):
        dataset = load_dataset(
            "hitoshura25/megavul",
            split="train",
            streaming=self.streaming
        )
    

        samples = []
        codeProcessor = CodePreprocessor()
        for row in dataset:
            # if row["language"] != self.language:
            #     continue

            if row["vulnerable_code"]:
                samples.append({"code": codeProcessor.clean(row["vulnerable_code"]), "label": 1})
            if row["fixed_code"]:
                samples.append({"code": codeProcessor.clean(row["fixed_code"]), "label": 0})

            if limit and len(samples) >= limit:
                break

        return samples

    def load_from_csv(self, csv_path, limit=None):
        df = pd.read_csv(csv_path)

        if "code" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'code' and 'label' columns.")

        if limit is not None:
            df = df.head(limit)

        samples = [
            {"code": row["code"], "label": int(row["label"])}
            for _, row in df.iterrows()
        ]
        return samples
    