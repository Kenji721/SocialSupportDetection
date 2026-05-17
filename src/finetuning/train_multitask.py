"""
Multi-task XLM-T fine-tuning for SSD-2026.

Two-phase training:
  Phase 1 — Freeze bottom 8/12 encoder layers, warm up heads at LR 1e-4
  Phase 2 — Unfreeze all, discriminative LR (1e-5 → 2e-5 → 5e-5)

Usage:
    python -m src.finetuning.train_multitask
    python -m src.finetuning.train_multitask --encoder microsoft/mdeberta-v3-base
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.label_maps import (
    TASK1_LABELS,
    TASK2_LABELS,
    TASK3_LABELS,
    get_label_config,
)
from src.data.loading import load_multilingual
from src.data.splits import stratified_split
from src.finetuning.multitask_dataset import MultiTaskSSDDataset
from src.finetuning.multitask_model import MultiTaskSSDModel, multitask_loss
from src.metrics import compute_metrics
from src.preprocessing import preprocess_transformer
from src.results import log_result
from src.utils import get_device

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EN_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-english.csv")
_ES_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-spanish.csv")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def _evaluate_multitask(model, loader, device):
    """Run evaluation, return per-task predictions and labels."""
    model.eval()
    all_l1, all_p1 = [], []
    all_l2, all_p2 = [], []
    all_l3, all_p3 = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels1 = batch["labels1"]
            labels2 = batch["labels2"]
            labels3 = batch["labels3"]

            logits1, logits2, logits3 = model(input_ids, attention_mask)

            # Task 1: all samples
            all_l1.extend(labels1.numpy())
            all_p1.extend(logits1.argmax(dim=-1).cpu().numpy())

            # Task 2: only valid (non-masked) samples
            mask2 = labels2 != -100
            if mask2.any():
                all_l2.extend(labels2[mask2].numpy())
                all_p2.extend(logits2[mask2].argmax(dim=-1).cpu().numpy())

            # Task 3: only valid samples
            mask3 = labels3 != -100
            if mask3.any():
                all_l3.extend(labels3[mask3].numpy())
                all_p3.extend(logits3[mask3].argmax(dim=-1).cpu().numpy())

    results = {}
    results["task1"] = (np.array(all_l1), np.array(all_p1))

    if all_l2:
        results["task2"] = (np.array(all_l2), np.array(all_p2))
    if all_l3:
        results["task3"] = (np.array(all_l3), np.array(all_p3))

    return results


def _macro_f1_summary(eval_results: dict) -> dict[str, float]:
    """Compute macro F1 for each task from eval results."""
    f1s = {}
    for task_key, (true, pred) in eval_results.items():
        f1s[task_key] = float(f1_score(true, pred, average="macro", zero_division=0))
    return f1s


def train_multitask(
    encoder_name: str = "cardiffnlp/twitter-xlm-roberta-base",
    seed: int = 42,
    max_len: int = 128,
    batch_size: int = 16,
    # Phase 1
    phase1_epochs: int = 3,
    phase1_lr: float = 1e-4,
    phase1_freeze_layers: int = 8,
    # Phase 2
    phase2_epochs: int = 10,
    phase2_lr: float = 2e-5,
    patience: int = 3,
    # Loss
    lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    grad_accum: int = 2,
):
    """Full multi-task training pipeline."""
    _set_seed(seed)
    device = get_device()
    encoder_short = encoder_name.split("/")[-1]

    print(f"\n{'=' * 60}")
    print(f"Multi-Task Training — {encoder_short}")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_multilingual(_EN_CSV, _ES_CSV)
    df["text_clean"] = df["text"].apply(preprocess_transformer)

    # Normalize task labels
    df = df[df["task1"].isin(["Supportive", "Non-Supportive"])].copy()

    print(f"Total samples: {len(df)}")
    print(f"  Task 1: {df['task1'].value_counts().to_dict()}")
    print(f"  Task 2 (Supportive only): {df[df['task1']=='Supportive']['task2'].value_counts().to_dict()}")
    print(f"  Task 3 (Group only): {df[df['task2']=='Group']['task3'].value_counts().to_dict()}")

    # ── 2. Split (stratified on task1) ────────────────────────────────────────
    texts = df["text_clean"].tolist()
    t1_labels = df["task1"].tolist()
    t2_labels = df["task2"].tolist()
    t3_labels = df["task3"].tolist()

    # Use task1 for stratification (primary task)
    from src.data.label_maps import TASK1_LABEL2ID
    strat_labels = [TASK1_LABEL2ID[l] for l in t1_labels]

    X_train, X_val, X_test, y_train_s, y_val_s, y_test_s = stratified_split(
        texts, strat_labels, seed=seed
    )

    # We need to carry all 3 label columns through the split
    # Re-split using indices
    indices = list(range(len(texts)))
    idx_train, idx_val, idx_test, _, _, _ = stratified_split(
        indices, strat_labels, seed=seed
    )

    def gather(lst, idxs):
        return [lst[i] for i in idxs]

    t1_train, t1_val, t1_test = gather(t1_labels, idx_train), gather(t1_labels, idx_val), gather(t1_labels, idx_test)
    t2_train, t2_val, t2_test = gather(t2_labels, idx_train), gather(t2_labels, idx_val), gather(t2_labels, idx_test)
    t3_train, t3_val, t3_test = gather(t3_labels, idx_train), gather(t3_labels, idx_val), gather(t3_labels, idx_test)
    txt_train, txt_val, txt_test = gather(texts, idx_train), gather(texts, idx_val), gather(texts, idx_test)
    raw_test = gather(df["text"].tolist(), idx_test)

    print(f"Split: train={len(txt_train)}, val={len(txt_val)}, test={len(txt_test)}")

    # ── 3. Tokenize + datasets ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    train_ds = MultiTaskSSDDataset(txt_train, t1_train, t2_train, t3_train, tokenizer, max_len)
    val_ds = MultiTaskSSDDataset(txt_val, t1_val, t2_val, t3_val, tokenizer, max_len)
    test_ds = MultiTaskSSDDataset(txt_test, t1_test, t2_test, t3_test, tokenizer, max_len)

    # MPS config: num_workers=0, pin_memory=False
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             num_workers=0, pin_memory=False)

    # ── 4. Class weights ──────────────────────────────────────────────────────
    cw1 = _compute_class_weights(train_ds.labels1.tolist(), len(TASK1_LABELS))
    # Task 2 weights from valid (non-masked) labels only
    valid_t2 = [l.item() for l in train_ds.labels2 if l.item() != -100]
    cw2 = _compute_class_weights(valid_t2, len(TASK2_LABELS)) if valid_t2 else None
    # Task 3 weights
    valid_t3 = [l.item() for l in train_ds.labels3 if l.item() != -100]
    cw3 = _compute_class_weights(valid_t3, len(TASK3_LABELS)) if valid_t3 else None

    print(f"Class weights — Task1: {cw1.tolist()}")
    if cw2 is not None:
        print(f"Class weights — Task2: {cw2.tolist()}")
    if cw3 is not None:
        print(f"Class weights — Task3: {cw3.tolist()}")

    # ── 5. Model ──────────────────────────────────────────────────────────────
    model = MultiTaskSSDModel(
        encoder_name=encoder_name,
        num_task1=len(TASK1_LABELS),
        num_task2=len(TASK2_LABELS),
        num_task3=len(TASK3_LABELS),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ── 6. Phase 1: Freeze encoder, warm up heads ────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"Phase 1: Warm-up heads ({phase1_epochs} epochs, LR={phase1_lr})")
    print(f"{'─' * 60}")

    model.freeze_encoder_layers(phase1_freeze_layers)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer1 = AdamW(trainable, lr=phase1_lr, weight_decay=weight_decay)
    total_steps1 = (len(train_loader) // grad_accum) * phase1_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=int(warmup_ratio * total_steps1),
        num_training_steps=total_steps1,
    )

    best_f1, best_state = 0.0, None

    for epoch in range(1, phase1_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer1.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels1 = batch["labels1"].to(device)
            labels2 = batch["labels2"].to(device)
            labels3 = batch["labels3"].to(device)

            logits1, logits2, logits3 = model(input_ids, attention_mask)
            loss, l1, l2, l3 = multitask_loss(
                logits1, logits2, logits3,
                labels1, labels2, labels3,
                cw1, cw2, cw3, lambdas,
            )
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer1.step()
                scheduler1.step()
                optimizer1.zero_grad()

            total_loss += loss.item() * grad_accum

        avg_loss = total_loss / len(train_loader)
        eval_results = _evaluate_multitask(model, val_loader, device)
        f1s = _macro_f1_summary(eval_results)
        avg_f1 = np.mean(list(f1s.values()))

        f1_str = "  ".join(f"{k}={v:.4f}" for k, v in f1s.items())
        print(f"  P1 Epoch {epoch}/{phase1_epochs}  loss={avg_loss:.4f}  {f1_str}  avg={avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── 7. Phase 2: Full fine-tuning with discriminative LR ──────────────────
    print(f"\n{'─' * 60}")
    print(f"Phase 2: Full fine-tuning ({phase2_epochs} epochs, base LR={phase2_lr})")
    print(f"{'─' * 60}")

    # Restore best Phase 1 state
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.unfreeze_all()

    # Discriminative learning rates
    layer_groups = model.get_layer_groups()
    lr_multipliers = {
        "embeddings": 0.5,
        "layers_0-3": 0.5,
        "layers_4-7": 1.0,
        "layers_8-11": 1.5,
        "heads": 2.5,
    }
    param_groups = []
    for group in layer_groups:
        mult = lr_multipliers.get(group["name"], 1.0)
        param_groups.append({
            "params": group["params"],
            "lr": phase2_lr * mult,
        })
        trainable_count = sum(p.numel() for p in group["params"] if p.requires_grad)
        print(f"  {group['name']}: lr={phase2_lr * mult:.2e}, params={trainable_count:,}")

    optimizer2 = AdamW(param_groups, weight_decay=weight_decay)
    total_steps2 = (len(train_loader) // grad_accum) * phase2_epochs
    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=int(warmup_ratio * total_steps2),
        num_training_steps=total_steps2,
    )

    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, phase2_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer2.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels1 = batch["labels1"].to(device)
            labels2 = batch["labels2"].to(device)
            labels3 = batch["labels3"].to(device)

            logits1, logits2, logits3 = model(input_ids, attention_mask)
            loss, l1, l2, l3 = multitask_loss(
                logits1, logits2, logits3,
                labels1, labels2, labels3,
                cw1, cw2, cw3, lambdas,
            )
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer2.step()
                scheduler2.step()
                optimizer2.zero_grad()

            total_loss += loss.item() * grad_accum

        avg_loss = total_loss / len(train_loader)
        eval_results = _evaluate_multitask(model, val_loader, device)
        f1s = _macro_f1_summary(eval_results)
        avg_f1 = np.mean(list(f1s.values()))

        f1_str = "  ".join(f"{k}={v:.4f}" for k, v in f1s.items())
        print(f"  P2 Epoch {epoch}/{phase2_epochs}  loss={avg_loss:.4f}  {f1_str}  avg={avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── 8. Evaluate on test set ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Test Evaluation (best model from P2 epoch {best_epoch})")
    print(f"{'=' * 60}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_results = _evaluate_multitask(model, test_loader, device)
    test_f1s = _macro_f1_summary(test_results)

    _, _, _, num_t1 = get_label_config(1)
    _, _, _, num_t2 = get_label_config(2)
    _, _, _, num_t3 = get_label_config(3)

    all_metrics = {}
    for task_num, task_key, labels_list in [
        (1, "task1", TASK1_LABELS),
        (2, "task2", TASK2_LABELS),
        (3, "task3", TASK3_LABELS),
    ]:
        if task_key in test_results:
            true, pred = test_results[task_key]
            metrics = compute_metrics(true, pred, labels_list)
            all_metrics[task_key] = metrics
            print(f"\nTask {task_num} — Macro F1: {metrics['macro_f1']:.4f}  Acc: {metrics['accuracy']:.4f}")
            print(f"  Per-class F1: {metrics['per_class_f1']}")

            # Log each task result
            log_result(
                approach="multitask",
                model=encoder_short,
                task=task_num,
                macro_f1=metrics["macro_f1"],
                split="test",
                embedding="—",
                dataset="multilingual",
                accuracy=metrics["accuracy"],
                precision_macro=metrics["precision_macro"],
                recall_macro=metrics["recall_macro"],
                per_class_f1=metrics["per_class_f1"],
                epochs=phase1_epochs + best_epoch,
                early_stop_epoch=best_epoch if patience_counter >= patience else None,
                hyperparams={
                    "encoder": encoder_name,
                    "phase1_epochs": phase1_epochs,
                    "phase1_lr": phase1_lr,
                    "phase2_lr": phase2_lr,
                    "batch_size": batch_size,
                    "grad_accum": grad_accum,
                    "max_len": max_len,
                    "lambdas": list(lambdas),
                    "seed": seed,
                    "patience": patience,
                },
                notes=f"multitask_{encoder_short}_s{seed}",
            )

    # ── 9. Save model ─────────────────────────────────────────────────────────
    experiment_name = f"multitask_{encoder_short}_s{seed}"
    artifact_dir = os.path.join(_ARTIFACTS_DIR, experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({
            "test_f1s": test_f1s,
            "test_metrics": {k: v for k, v in all_metrics.items()},
        }, f, indent=2)

    # Save model state dict + config
    model_dir = os.path.join(artifact_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, "multitask_model.pt"))
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "encoder_name": encoder_name,
            "num_task1": len(TASK1_LABELS),
            "num_task2": len(TASK2_LABELS),
            "num_task3": len(TASK3_LABELS),
        }, f)
    print(f"\nModel saved to {model_dir}")

    # Save test predictions
    if "task1" in test_results:
        t1_true, t1_pred = test_results["task1"]
        _, _, id2label_t1, _ = get_label_config(1)
        pred_df = pd.DataFrame({
            "text": raw_test,
            "task1_true": [id2label_t1[y] for y in t1_true.tolist()],
            "task1_pred": [id2label_t1[int(y)] for y in t1_pred.tolist()],
        })
        pred_df.to_csv(os.path.join(artifact_dir, "predictions.csv"), index=False)

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="cardiffnlp/twitter-xlm-roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--phase1_epochs", type=int, default=3)
    parser.add_argument("--phase1_lr", type=float, default=1e-4)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--phase2_lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad_accum", type=int, default=2)
    args = parser.parse_args()

    train_multitask(
        encoder_name=args.encoder,
        seed=args.seed,
        batch_size=args.batch_size,
        max_len=args.max_len,
        phase1_epochs=args.phase1_epochs,
        phase1_lr=args.phase1_lr,
        phase2_epochs=args.phase2_epochs,
        phase2_lr=args.phase2_lr,
        patience=args.patience,
        grad_accum=args.grad_accum,
    )
