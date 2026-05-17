"""
Deep learning training pipeline for SSD-2026.

Supports TextCNN, BiLSTM, and BiLSTM+Attention with pretrained embeddings.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.label_maps import get_label_config
from src.data.loading import load_ssd_data
from src.data.splits import get_task_subset, stratified_split
from src.deeplearning.embeddings import (
    build_embedding_matrix,
    build_vocab,
    load_fasttext,
    load_glove,
    texts_to_indices,
)
from src.deeplearning.models import BiLSTM, BiLSTMAttention, TextCNN
from src.metrics import compute_metrics
from src.preprocessing import preprocess_deep
from src.results import log_result
from src.utils import get_device

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EN_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-english.csv")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")

_MODEL_CLASSES = {
    "textcnn": TextCNN,
    "bilstm": BiLSTM,
    "bilstm_attn": BiLSTMAttention,
}

_EMBEDDING_LOADERS = {
    "fasttext": load_fasttext,
    "glove": load_glove,
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _make_loader(X, y, batch_size, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_deep(
    task: int,
    language: str = "en",
    model_type: str = "textcnn",
    embedding_type: str = "fasttext",
    seed: int = 42,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 30,
    max_len: int = 128,
    patience: int = 5,
    embedding_dim: int = 300,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    freeze_embeddings: bool = False,
    min_freq: int = 2,
) -> dict:
    """Full deep learning training pipeline with class weights and early stopping."""

    _set_seed(seed)
    device = get_device()
    task_col = f"task{task}"
    labels_list, label2id, id2label, num_labels = get_label_config(task)

    print(f"\n{'=' * 60}")
    print(f"Deep Learning — {model_type} + {embedding_type} — Task {task} — {language} — seed {seed}")
    print(f"{'=' * 60}")
    print(f"Device: {device}")

    # 1. Load & subset
    df = load_ssd_data(_EN_CSV, language)
    df = get_task_subset(df, task)
    df = df[df[task_col].isin(label2id)].copy()
    df["label"] = df[task_col].map(label2id)

    # 2. Preprocess
    df["text_clean"] = df["text"].apply(preprocess_deep)

    texts = df["text_clean"].tolist()
    raw_texts = df["text"].tolist()
    labels = df["label"].tolist()

    print(f"Total samples: {len(labels)}")
    for name, lid in label2id.items():
        cnt = labels.count(lid)
        print(f"  {name}: {cnt} ({cnt / len(labels) * 100:.1f}%)")

    # 3. Split
    X_train_t, X_val_t, X_test_t, y_train, y_val, y_test = stratified_split(
        texts, labels, seed=seed
    )
    raw_train, raw_val, raw_test, _, _, _ = stratified_split(
        raw_texts, labels, seed=seed
    )
    print(f"Split: train={len(X_train_t)}, val={len(X_val_t)}, test={len(X_test_t)}")

    # 4. Build vocab from training data only
    vocab = build_vocab(X_train_t, min_freq=min_freq)
    print(f"Vocab size: {len(vocab)}")

    # 5. Convert to indices
    X_train = texts_to_indices(X_train_t, vocab, max_len)
    X_val = texts_to_indices(X_val_t, vocab, max_len)
    X_test = texts_to_indices(X_test_t, vocab, max_len)

    # 6. Load pretrained embeddings
    loader_fn = _EMBEDDING_LOADERS[embedding_type]
    raw_embeddings = loader_fn(dim=embedding_dim)
    embedding_matrix = build_embedding_matrix(vocab, raw_embeddings, dim=embedding_dim)
    pretrained = torch.tensor(embedding_matrix, dtype=torch.float)

    # 7. Build model
    ModelClass = _MODEL_CLASSES[model_type]
    model_kwargs = {
        "vocab_size": len(vocab),
        "embedding_dim": embedding_dim,
        "num_classes": num_labels,
        "dropout": dropout,
        "pretrained_embeddings": pretrained,
        "freeze_embeddings": freeze_embeddings,
    }
    if model_type in ("bilstm", "bilstm_attn"):
        model_kwargs["hidden_dim"] = hidden_dim

    model = ModelClass(**model_kwargs).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 8. Class weights
    counts = np.bincount(y_train, minlength=num_labels)
    cw = torch.tensor(
        [len(y_train) / (num_labels * c) if c > 0 else 1.0 for c in counts],
        dtype=torch.float,
    ).to(device)
    print(f"Class weights: {cw.tolist()}")

    loss_fn = torch.nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 9. Data loaders
    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size)
    test_loader = _make_loader(X_test, y_test, batch_size)

    # 10. Training loop
    best_f1, best_state, patience_counter, best_epoch = 0.0, None, 0, 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(y_batch.numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 11. Restore best & evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    test_preds_list, test_labels_list = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            test_preds_list.extend(logits.argmax(dim=-1).cpu().numpy())
            test_labels_list.extend(y_batch.numpy())

    test_labels_arr = np.array(test_labels_list)
    test_preds_arr = np.array(test_preds_list)
    test_metrics = compute_metrics(test_labels_arr, test_preds_arr, labels_list)

    print(f"\nTest — Macro F1: {test_metrics['macro_f1']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print(f"Test per-class F1: {test_metrics['per_class_f1']}")

    # 12. Log results
    experiment_name = f"deeplearning_{model_type}_{embedding_type}_task{task}_{language}_s{seed}"

    log_result(
        approach="deeplearning",
        model=model_type,
        task=task,
        macro_f1=test_metrics["macro_f1"],
        split="test",
        embedding=embedding_type,
        dataset=f"train-{language}.csv",
        accuracy=test_metrics["accuracy"],
        precision_macro=test_metrics["precision_macro"],
        recall_macro=test_metrics["recall_macro"],
        per_class_f1=test_metrics["per_class_f1"],
        epochs=best_epoch,
        early_stop_epoch=best_epoch if patience_counter >= patience else None,
        hyperparams={
            "lr": lr,
            "batch_size": batch_size,
            "max_len": max_len,
            "hidden_dim": hidden_dim if model_type != "textcnn" else None,
            "dropout": dropout,
            "freeze_embeddings": freeze_embeddings,
            "min_freq": min_freq,
            "seed": seed,
            "patience": patience,
        },
        notes=experiment_name,
    )

    # 13. Save artifacts
    artifact_dir = os.path.join(_ARTIFACTS_DIR, experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({"val_best_f1": best_f1, "test": test_metrics}, f, indent=2)

    pred_df = pd.DataFrame({
        "text": raw_test,
        "y_true": [id2label[y] for y in test_labels_arr.tolist()],
        "y_pred": [id2label[int(y)] for y in test_preds_arr.tolist()],
    })
    pred_df.to_csv(os.path.join(artifact_dir, "predictions.csv"), index=False)

    # Save model weights + vocab
    model_dir = os.path.join(artifact_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, "model.pt"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "model_type": model_type,
            "vocab_size": len(vocab),
            "embedding_dim": embedding_dim,
            "num_classes": num_labels,
            "hidden_dim": hidden_dim if model_type != "textcnn" else None,
            "dropout": dropout,
        }, f, indent=2)
    with open(os.path.join(model_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    print(f"Model saved to {model_dir}")
    return test_metrics


# ── Experiment grid ──────────────────────────────────────────────────────────

_SEEDS = [13, 42, 77]
_TASKS = [1, 2, 3]


def run_experiment_grid(
    model_types=None,
    embedding_types=None,
    seeds=None,
    tasks=None,
):
    """Run the Phase 5 DL experiment grid.

    Default: TextCNN + BiLSTM × FastText × 3 tasks × 3 seeds = 18 experiments.
    """
    model_types = model_types or ["textcnn", "bilstm"]
    embedding_types = embedding_types or ["fasttext"]
    seeds = seeds or _SEEDS
    tasks = tasks or _TASKS

    all_results = []
    total = len(model_types) * len(embedding_types) * len(tasks) * len(seeds)
    i = 0

    for mt in model_types:
        for emb in embedding_types:
            for task in tasks:
                pat = 7 if task == 3 else 5
                for seed in seeds:
                    i += 1
                    print(f"\n{'#' * 60}")
                    print(f"# [{i}/{total}] {mt} + {emb} / task{task} / seed={seed}")
                    print(f"{'#' * 60}")

                    try:
                        metrics = train_deep(
                            task=task,
                            language="en",
                            model_type=mt,
                            embedding_type=emb,
                            seed=seed,
                            patience=pat,
                        )
                        all_results.append({
                            "model": mt,
                            "embedding": emb,
                            "task": task,
                            "seed": seed,
                            "macro_f1": metrics["macro_f1"],
                        })
                    except Exception as e:
                        print(f"ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append({
                            "model": mt,
                            "embedding": emb,
                            "task": task,
                            "seed": seed,
                            "macro_f1": None,
                            "error": str(e),
                        })

    # Summary
    print(f"\n{'=' * 80}")
    print(f"{'PHASE 5 DL SUMMARY':^80}")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'Embedding':<12} {'Task':<6} {'Mean F1':<10} {'Std':<10} {'Seeds'}")
    print(f"{'-' * 80}")

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["embedding"], r["task"])
        if r["macro_f1"] is not None:
            grouped[key].append(r["macro_f1"])

    for (model, emb, task), f1s in sorted(grouped.items()):
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        seeds_str = ", ".join(f"{f:.4f}" for f in f1s)
        print(f"{model:<15} {emb:<12} {task:<6} {mean_f1:<10.4f} {std_f1:<10.4f} [{seeds_str}]")

    print(f"{'=' * 80}")
    return all_results


if __name__ == "__main__":
    run_experiment_grid()
