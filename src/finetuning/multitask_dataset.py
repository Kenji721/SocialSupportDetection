"""
Multi-task dataset for SSD-2026 that returns labels for all 3 tasks.

Uses -100 to mask invalid labels:
  - Task 2 = -100 when Task 1 is Non-Supportive
  - Task 3 = -100 when Task 2 is not Group
"""

import torch
from torch.utils.data import Dataset

from src.data.label_maps import TASK1_LABEL2ID, TASK2_LABEL2ID, TASK3_LABEL2ID


class MultiTaskSSDDataset(Dataset):
    """Dataset returning tokenized inputs + labels for all 3 tasks."""

    IGNORE_INDEX = -100

    def __init__(self, texts: list[str], task1_labels: list[str],
                 task2_labels: list[str], task3_labels: list[str],
                 tokenizer, max_len: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )

        self.labels1 = torch.tensor(
            [TASK1_LABEL2ID[l] for l in task1_labels], dtype=torch.long
        )

        # Task 2: valid only for Supportive samples
        self.labels2 = torch.tensor([
            TASK2_LABEL2ID.get(l, -1) if t1 == "Supportive" else self.IGNORE_INDEX
            for l, t1 in zip(task2_labels, task1_labels)
        ], dtype=torch.long)
        # Map unknown task2 values to IGNORE_INDEX
        self.labels2[self.labels2 == -1] = self.IGNORE_INDEX

        # Task 3: valid only for Group samples
        self.labels3 = torch.tensor([
            TASK3_LABEL2ID.get(l, -1) if t2 == "Group" else self.IGNORE_INDEX
            for l, t2 in zip(task3_labels, task2_labels)
        ], dtype=torch.long)
        self.labels3[self.labels3 == -1] = self.IGNORE_INDEX

    def __len__(self):
        return len(self.labels1)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels1"] = self.labels1[idx]
        item["labels2"] = self.labels2[idx]
        item["labels3"] = self.labels3[idx]
        return item
