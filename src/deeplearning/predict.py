"""
Deep learning inference for SSD-2026.
"""

import json
import os

import numpy as np
import torch

from src.deeplearning.embeddings import texts_to_indices
from src.deeplearning.models import BiLSTM, BiLSTMAttention, TextCNN
from src.preprocessing import preprocess_deep
from src.utils import get_device

_MODEL_CLASSES = {
    "textcnn": TextCNN,
    "bilstm": BiLSTM,
    "bilstm_attn": BiLSTMAttention,
}


def predict_deep(
    model_dir: str,
    texts: list[str],
    batch_size: int = 64,
    max_len: int = 128,
) -> list[int]:
    """Load model + vocab from model_dir and run inference."""
    device = get_device()

    # Load config and vocab
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "vocab.json")) as f:
        vocab = json.load(f)

    # Preprocess
    cleaned = [preprocess_deep(t) for t in texts]
    X = texts_to_indices(cleaned, vocab, max_len)

    # Build model
    ModelClass = _MODEL_CLASSES[config["model_type"]]
    model_kwargs = {
        "vocab_size": config["vocab_size"],
        "embedding_dim": config["embedding_dim"],
        "num_classes": config["num_classes"],
        "dropout": config["dropout"],
    }
    if config["model_type"] in ("bilstm", "bilstm_attn") and config.get("hidden_dim"):
        model_kwargs["hidden_dim"] = config["hidden_dim"]

    model = ModelClass(**model_kwargs)
    state_dict = torch.load(
        os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Inference
    all_preds = []
    X_tensor = torch.tensor(X, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size].to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())

    return [int(p) for p in all_preds]
