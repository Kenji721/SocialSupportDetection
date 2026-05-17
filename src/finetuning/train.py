"""
Unified transformer fine-tuning for SSD-2026.

Supports: BERTweet, RoBERTuito, XLM-R (and any AutoModel-compatible checkpoint).
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data.label_maps import get_label_config
from src.data.loading import load_multilingual, load_ssd_data
from src.data.splits import get_task_subset, stratified_split
from src.finetuning.datasets import SSDDataset
from src.metrics import compute_metrics
from src.preprocessing import preprocess_transformer
from src.results import log_result
from src.utils import get_device

# Paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EN_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-english.csv")
_ES_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-spanish.csv")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")


def _fix_layernorm_naming(model):
    """Rename legacy LayerNorm.gamma/beta to weight/bias in-place.

    BERTweet uses old naming convention that breaks save_pretrained/from_pretrained
    round-trip because from_pretrained expects weight/bias keys.
    """
    for name, module in model.named_modules():
        if hasattr(module, "gamma") and isinstance(module.gamma, torch.nn.Parameter):
            module.weight = module.gamma
            del module.gamma
        if hasattr(module, "beta") and isinstance(module.beta, torch.nn.Parameter):
            module.bias = module.beta
            del module.beta


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _load_data(language: str):
    if language == "en":
        return load_ssd_data(_EN_CSV, "en")
    elif language == "es":
        return load_ssd_data(_ES_CSV, "es")
    elif language == "multilingual":
        return load_multilingual(_EN_CSV, _ES_CSV)
    else:
        raise ValueError(f"Unknown language: {language}")


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


def _evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)


def train_transformer(
    task: int,
    language: str = "en",
    model_name: str = "vinai/bertweet-base",
    seed: int = 42,
    lr: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 5,
    max_len: int = 128,
    patience: int = 2,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    use_class_weights: bool = True,
    save_dir: str = None,
) -> dict:
    """Full transformer fine-tuning pipeline. Returns test metrics dict."""

    _set_seed(seed)
    device = get_device()
    task_col = f"task{task}"
    labels_list, label2id, id2label, num_labels = get_label_config(task)

    # Short model name for logging
    model_short = model_name.split("/")[-1]

    print(f"\n{'=' * 60}")
    print(f"Transformer — {model_short} — Task {task} — {language} — seed {seed}")
    print(f"{'=' * 60}")
    print(f"Device: {device}")

    # 1. Load & subset
    df = _load_data(language)
    df = get_task_subset(df, task)
    df = df[df[task_col].isin(label2id)].copy()
    df["label"] = df[task_col].map(label2id)

    # 2. Preprocess (minimal for transformers)
    df["text_clean"] = df["text"].apply(preprocess_transformer)

    texts = df["text_clean"].tolist()
    raw_texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Print distributions
    print(f"Total samples: {len(labels)}")
    for name, lid in label2id.items():
        cnt = labels.count(lid)
        print(f"  {name}: {cnt} ({cnt / len(labels) * 100:.1f}%)")

    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        texts, labels, seed=seed
    )
    raw_train, raw_val, raw_test, _, _, _ = stratified_split(
        raw_texts, labels, seed=seed
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 4. Tokenize
    # BERTweet requires use_fast=False
    use_fast = "bertweet" not in model_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    train_ds = SSDDataset(X_train, y_train, tokenizer, max_len)
    val_ds = SSDDataset(X_val, y_val, tokenizer, max_len)
    test_ds = SSDDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 5. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    ).to(device)

    # 6. Class weights
    if use_class_weights:
        counts = np.bincount(y_train, minlength=num_labels)
        cw = torch.tensor(
            [len(y_train) / (num_labels * c) if c > 0 else 1.0 for c in counts],
            dtype=torch.float,
        )
        print(f"Class weights: {cw.tolist()}")
    else:
        cw = torch.ones(num_labels)

    # 7. Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 8. Training loop
    best_f1, best_state, patience_counter, best_epoch = 0.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device, cw)
        val_labels, val_preds = _evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        print(f"  Epoch {epoch}/{epochs}  loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}")

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

    # 9. Restore best & evaluate on test
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_labels, test_preds = _evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_preds, labels_list)

    print(f"\nTest — Macro F1: {test_metrics['macro_f1']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print(f"Test per-class F1: {test_metrics['per_class_f1']}")

    # 10. Log results
    experiment_name = f"finetuning_{model_short}_task{task}_{language}_s{seed}"

    log_result(
        approach="finetuning",
        model=model_short,
        task=task,
        macro_f1=test_metrics["macro_f1"],
        split="test",
        embedding="—",
        dataset=f"train-{language}.csv" if language != "multilingual" else "multilingual",
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
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "seed": seed,
            "patience": patience,
        },
        notes=experiment_name,
    )

    # 11. Save artifacts
    artifact_dir = os.path.join(_ARTIFACTS_DIR, experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({"val_best_f1": best_f1, "test": test_metrics}, f, indent=2)

    pred_df = pd.DataFrame({
        "text": raw_test,
        "y_true": [id2label[y] for y in test_labels.tolist()],
        "y_pred": [id2label[int(y)] for y in test_preds.tolist()],
    })
    pred_df.to_csv(os.path.join(artifact_dir, "predictions.csv"), index=False)

    # 12. Save model + tokenizer
    if save_dir is None:
        save_dir = os.path.join(artifact_dir, "model")
    os.makedirs(save_dir, exist_ok=True)

    # Save state dict with torch.save (avoids BERTweet LayerNorm gamma/beta
    # key mapping issues with save_pretrained/from_pretrained round-trip)
    torch.save(model.cpu().state_dict(), os.path.join(save_dir, "pytorch_model.pt"))
    model.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "base_model.json"), "w") as f:
        json.dump({"model_name": model_name, "num_labels": num_labels}, f)
    model.to(device)
    print(f"Model saved to {save_dir}")

    return test_metrics


# ── Experiment grid ──────────────────────────────────────────────────────────

# Model configs: (hf_name, language, batch_size)
_EXPERIMENT_CONFIGS = [
    ("vinai/bertweet-base", "en", 16),
    ("pysentimiento/robertuito-base-uncased", "es", 16),
    ("xlm-roberta-base", "en", 32),
    ("xlm-roberta-base", "es", 32),
    ("xlm-roberta-base", "multilingual", 32),
]

_SEEDS = [13, 42, 77]
_TASKS = [1, 2, 3]


def run_experiment_grid(
    configs=None,
    seeds=None,
    tasks=None,
):
    """Run the full Phase 2 experiment grid.

    Default: 5 configs × 3 tasks × 3 seeds = 45 experiments.
    """
    configs = configs or _EXPERIMENT_CONFIGS
    seeds = seeds or _SEEDS
    tasks = tasks or _TASKS

    all_results = []
    total = len(configs) * len(tasks) * len(seeds)
    i = 0

    for model_name, language, bs in configs:
        for task in tasks:
            # Task 3 gets more patience
            pat = 3 if task == 3 else 2

            for seed in seeds:
                i += 1
                model_short = model_name.split("/")[-1]
                print(f"\n{'#' * 60}")
                print(f"# [{i}/{total}] {model_short} / task{task} / {language} / seed={seed}")
                print(f"{'#' * 60}")

                try:
                    metrics = train_transformer(
                        task=task,
                        language=language,
                        model_name=model_name,
                        seed=seed,
                        batch_size=bs,
                        patience=pat,
                    )
                    all_results.append({
                        "model": model_short,
                        "task": task,
                        "language": language,
                        "seed": seed,
                        "macro_f1": metrics["macro_f1"],
                    })
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "model": model_short,
                        "task": task,
                        "language": language,
                        "seed": seed,
                        "macro_f1": None,
                        "error": str(e),
                    })

    # Summary with mean ± std across seeds
    print(f"\n{'=' * 80}")
    print(f"{'PHASE 2 SUMMARY':^80}")
    print(f"{'=' * 80}")
    print(f"{'Model':<30} {'Task':<6} {'Lang':<14} {'Mean F1':<10} {'Std':<10} {'Seeds'}")
    print(f"{'-' * 80}")

    # Group by (model, task, language)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["task"], r["language"])
        if r["macro_f1"] is not None:
            grouped[key].append(r["macro_f1"])

    for (model, task, lang), f1s in sorted(grouped.items()):
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        seeds_str = ", ".join(f"{f:.4f}" for f in f1s)
        print(f"{model:<30} {task:<6} {lang:<14} {mean_f1:<10.4f} {std_f1:<10.4f} [{seeds_str}]")

    print(f"{'=' * 80}")
    return all_results


if __name__ == "__main__":
    run_experiment_grid()
