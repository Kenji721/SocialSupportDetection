"""
Shared utilities for SSD-2026 Social Support Detection.
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Text Preprocessing ────────────────────────────────────────────────────────
EMOJI_WORDS = [
    'smiling', 'crying', 'face', 'heart', 'hearts', 'eyes', 'red', 'two',
    'pensive', 'loudly', 'signround', 'pushpin', 'greating', 'hearteyes',
    'facehearteyessmiling', 'faceloudly', 'hearteyesloudly', 'hearteyessmiling',
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    for word in sorted(EMOJI_WORDS, key=len, reverse=True):
        text = text.replace(word, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Dataset ───────────────────────────────────────────────────────────────────
class SupportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── EDA ───────────────────────────────────────────────────────────────────────
def run_eda(df, text_col, label_col):
    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print("=" * 60)
    print(f"Total samples : {len(df)}")
    print(f"\nClass distribution:")
    vc = df[label_col].value_counts()
    for cls, cnt in vc.items():
        print(f"  {cls:<20} {cnt:>6}  ({cnt / len(df) * 100:.1f}%)")
    df["_len"] = df[text_col].str.split().str.len()
    print(f"\nText length (words):")
    print(f"  mean={df['_len'].mean():.1f}  median={df['_len'].median():.0f}"
          f"  max={df['_len'].max()}  min={df['_len'].min()}")
    df.drop(columns=["_len"], inplace=True)
    print("=" * 60 + "\n")


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, class_weights):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)
