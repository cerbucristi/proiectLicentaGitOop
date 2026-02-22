# import torch

# class CodeDataset(torch.utils.data.Dataset):

#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {k: v[idx] for k, v in self.encodings.items()}
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item
    
#     def __len__(self):
#         return len (self.labels)

from torch.utils.data import Dataset
import torch

class CodeDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoding = self.tokenizer(
            sample["code"],
            truncation=True,
            max_length=512,   
            padding=True,       # dynamic padding
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(sample["label"], dtype=torch.long)

        return item
