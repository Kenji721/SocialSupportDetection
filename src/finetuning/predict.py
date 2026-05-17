"""
Transformer inference for SSD-2026.
"""

import json
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.finetuning.datasets import SSDDataset
from src.utils import get_device


def _load_model(model_dir: str, device):
    """Load model from directory, supporting both torch.save and save_pretrained formats."""
    pt_path = os.path.join(model_dir, "pytorch_model.pt")
    base_model_path = os.path.join(model_dir, "base_model.json")

    if os.path.exists(pt_path) and os.path.exists(base_model_path):
        # Load using torch state dict + base model info
        with open(base_model_path) as f:
            info = json.load(f)
        model = AutoModelForSequenceClassification.from_pretrained(
            info["model_name"], num_labels=info["num_labels"]
        )
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # Fallback: standard from_pretrained
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    return model.to(device)


def predict_transformer(
    model_dir: str,
    texts: list[str],
    batch_size: int = 32,
    max_len: int = 128,
) -> list[int]:
    """Load model + tokenizer from model_dir and run inference."""
    device = get_device()

    # Load tokenizer: prefer base model name (BERTweet tokenizer doesn't round-trip)
    base_model_path = os.path.join(model_dir, "base_model.json")
    if os.path.exists(base_model_path):
        with open(base_model_path) as f:
            info = json.load(f)
        use_fast = "bertweet" not in info["model_name"].lower()
        tokenizer = AutoTokenizer.from_pretrained(info["model_name"], use_fast=use_fast)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = _load_model(model_dir, device)
    model.eval()

    # Create dummy labels for dataset
    dummy_labels = [0] * len(texts)
    dataset = SSDDataset(texts, dummy_labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())

    return [int(p) for p in all_preds]
