"""
HuggingFace Dataset wrapper for SSD-2026 fine-tuning.
"""

import torch
from torch.utils.data import Dataset


class SSDDataset(Dataset):
    """Generalized dataset for any HuggingFace tokenizer."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
