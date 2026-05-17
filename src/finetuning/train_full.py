"""
Train on the FULL dataset (no val/test split) for final submission models.

Trains for a fixed number of epochs (the best_epoch from prior experiments)
using all available training data. No early stopping since there's no
validation set — we know the optimal epoch count from previous experiments.

Usage:
    python -m src.finetuning.train_full
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.label_maps import get_label_config
from src.data.loading import load_ssd_data
from src.data.splits import get_task_subset
from src.finetuning.datasets import SSDDataset
from src.preprocessing import preprocess_transformer
from src.utils import get_device

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EN_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-english.csv")
_ES_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-spanish.csv")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _train_epoch(model, loader, optimizer, scheduler, device, class_weights):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_full(
    task: int,
    language: str,
    model_name: str,
    seed: int,
    epochs: int,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_len: int = 128,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    use_class_weights: bool = True,
) -> str:
    """Train on full dataset for a fixed number of epochs. Returns save path."""

    _set_seed(seed)
    device = get_device()
    task_col = f"task{task}"
    labels_list, label2id, id2label, num_labels = get_label_config(task)
    model_short = model_name.split("/")[-1]

    print(f"\n{'=' * 60}")
    print(f"FULL TRAIN — {model_short} — Task {task} — {language} — seed {seed}")
    print(f"{'=' * 60}")
    print(f"Device: {device} | Epochs: {epochs} (fixed, no early stopping)")

    # Load & subset
    csv_path = _EN_CSV if language == "en" else _ES_CSV
    df = load_ssd_data(csv_path, language)
    df = get_task_subset(df, task)
    df = df[df[task_col].isin(label2id)].copy()
    df["label"] = df[task_col].map(label2id)

    # Preprocess
    df["text_clean"] = df["text"].apply(preprocess_transformer)
    texts = df["text_clean"].tolist()
    labels = df["label"].tolist()

    print(f"Training on ALL {len(labels)} samples")
    for name, lid in label2id.items():
        cnt = labels.count(lid)
        print(f"  {name}: {cnt} ({cnt / len(labels) * 100:.1f}%)")

    # Tokenize
    use_fast = "bertweet" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    train_ds = SSDDataset(texts, labels, tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    ).to(device)

    # Class weights
    if use_class_weights:
        counts = np.bincount(labels, minlength=num_labels)
        cw = torch.tensor(
            [len(labels) / (num_labels * c) if c > 0 else 1.0 for c in counts],
            dtype=torch.float,
        )
        print(f"Class weights: {cw.tolist()}")
    else:
        cw = torch.ones(num_labels)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train for fixed epochs
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device, cw)
        print(f"  Epoch {epoch}/{epochs}  loss={train_loss:.4f}")

    # Save
    experiment_name = f"finetuning_{model_short}_task{task}_{language}_s{seed}_full"
    artifact_dir = os.path.join(_ARTIFACTS_DIR, experiment_name)
    save_dir = os.path.join(artifact_dir, "model")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.cpu().state_dict(), os.path.join(save_dir, "pytorch_model.pt"))
    model.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "base_model.json"), "w") as f:
        json.dump({"model_name": model_name, "num_labels": num_labels}, f)

    with open(os.path.join(artifact_dir, "training_config.json"), "w") as f:
        json.dump({
            "task": task, "language": language, "model_name": model_name,
            "seed": seed, "epochs": epochs, "lr": lr, "batch_size": batch_size,
            "max_len": max_len, "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio, "num_samples": len(labels),
            "full_train": True,
        }, f, indent=2)

    print(f"Model saved to {save_dir}")
    return save_dir


# ── Best configs from prior experiments ──────────────────────────────────────
# BERTweet English: best seeds and epochs per task
# RoBERTuito Spanish: best seeds and epochs per task

FULL_TRAIN_CONFIGS = [
    # BERTweet English
    {"task": 1, "language": "en", "model_name": "vinai/bertweet-base",
     "seed": 13, "epochs": 3, "batch_size": 16},
    {"task": 2, "language": "en", "model_name": "vinai/bertweet-base",
     "seed": 13, "epochs": 5, "batch_size": 16},
    {"task": 3, "language": "en", "model_name": "vinai/bertweet-base",
     "seed": 77, "epochs": 5, "batch_size": 16},
    # RoBERTuito Spanish
    {"task": 1, "language": "es", "model_name": "pysentimiento/robertuito-base-uncased",
     "seed": 42, "epochs": 2, "batch_size": 16},
    {"task": 2, "language": "es", "model_name": "pysentimiento/robertuito-base-uncased",
     "seed": 13, "epochs": 3, "batch_size": 16},
    {"task": 3, "language": "es", "model_name": "pysentimiento/robertuito-base-uncased",
     "seed": 42, "epochs": 4, "batch_size": 16},
]


def main():
    print("Training final models on FULL datasets (no val/test split)")
    print("Using best hyperparameters from prior experiments\n")

    for cfg in FULL_TRAIN_CONFIGS:
        train_full(**cfg)

    if get_device().type == "mps":
        torch.mps.empty_cache()

    print("\n" + "=" * 60)
    print("All full-dataset models trained!")
    print("=" * 60)


if __name__ == "__main__":
    main()
