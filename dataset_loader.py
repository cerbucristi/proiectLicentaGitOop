from datasets import load_dataset
from code_processor import CodePreprocessor

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
    