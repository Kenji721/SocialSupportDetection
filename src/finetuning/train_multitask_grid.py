"""
Experiment grid for multi-task training across encoders and languages.

Usage:
    python -m src.finetuning.train_multitask_grid
    python -m src.finetuning.train_multitask_grid --seeds 42 77
"""

import argparse
import json
import os
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd

from src.data.label_maps import TASK1_LABEL2ID, TASK1_LABELS, TASK2_LABELS, TASK3_LABELS
from src.data.loading import load_multilingual, load_ssd_data
from src.data.splits import stratified_split
from src.finetuning.multitask_dataset import MultiTaskSSDDataset
from src.finetuning.multitask_model import MultiTaskSSDModel, multitask_loss
from src.metrics import compute_metrics
from src.preprocessing import preprocess_transformer
from src.results import log_result
from src.utils import get_device

import random
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EN_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-english.csv")
_ES_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-spanish.csv")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")

# ── Grid configs ──────────────────────────────────────────────────────────────

ENCODERS = [
    "cardiffnlp/twitter-xlm-roberta-base",
    "microsoft/mdeberta-v3-base",
    "xlm-roberta-base",
]

LANGUAGES = ["en", "es", "multilingual"]

SEEDS = [42, 77, 13]


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


def _evaluate_multitask(model, loader, device):
    model.eval()
    all_l1, all_p1 = [], []
    all_l2, all_p2 = [], []
    all_l3, all_p3 = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels1, labels2, labels3 = batch["labels1"], batch["labels2"], batch["labels3"]

            logits1, logits2, logits3 = model(input_ids, attention_mask)

            all_l1.extend(labels1.numpy())
            all_p1.extend(logits1.argmax(dim=-1).cpu().numpy())

            mask2 = labels2 != -100
            if mask2.any():
                all_l2.extend(labels2[mask2].numpy())
                all_p2.extend(logits2[mask2].argmax(dim=-1).cpu().numpy())

            mask3 = labels3 != -100
            if mask3.any():
                all_l3.extend(labels3[mask3].numpy())
                all_p3.extend(logits3[mask3].argmax(dim=-1).cpu().numpy())

    results = {"task1": (np.array(all_l1), np.array(all_p1))}
    if all_l2:
        results["task2"] = (np.array(all_l2), np.array(all_p2))
    if all_l3:
        results["task3"] = (np.array(all_l3), np.array(all_p3))
    return results


def _macro_f1_summary(eval_results: dict) -> dict[str, float]:
    return {k: float(f1_score(t, p, average="macro", zero_division=0))
            for k, (t, p) in eval_results.items()}


def _load_data(language: str) -> pd.DataFrame:
    if language == "en":
        return load_ssd_data(_EN_CSV, "en")
    elif language == "es":
        return load_ssd_data(_ES_CSV, "es")
    elif language == "multilingual":
        return load_multilingual(_EN_CSV, _ES_CSV)
    else:
        raise ValueError(f"Unknown language: {language}")


def run_single(
    encoder_name: str,
    language: str,
    seed: int = 42,
    max_len: int = 128,
    batch_size: int = 16,
    phase1_epochs: int = 3,
    phase1_lr: float = 1e-4,
    phase1_freeze_layers: int = 8,
    phase2_epochs: int = 10,
    phase2_lr: float = 2e-5,
    patience: int = 3,
    lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0),
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    grad_accum: int = 2,
) -> dict:
    """Train one encoder/language/seed combination. Returns test metrics."""
    _set_seed(seed)
    device = get_device()
    encoder_short = encoder_name.split("/")[-1]

    print(f"\n{'=' * 60}")
    print(f"Multi-Task — {encoder_short} — {language} — seed {seed}")
    print(f"{'=' * 60}")

    # Load & prep data
    df = _load_data(language)
    df["text_clean"] = df["text"].apply(preprocess_transformer)
    df = df[df["task1"].isin(["Supportive", "Non-Supportive"])].copy()
    print(f"Samples: {len(df)}")

    texts = df["text_clean"].tolist()
    t1_labels = df["task1"].tolist()
    t2_labels = df["task2"].tolist()
    t3_labels = df["task3"].tolist()

    strat_labels = [TASK1_LABEL2ID[l] for l in t1_labels]
    indices = list(range(len(texts)))
    idx_train, idx_val, idx_test, _, _, _ = stratified_split(indices, strat_labels, seed=seed)

    def gather(lst, idxs):
        return [lst[i] for i in idxs]

    t1_train = gather(t1_labels, idx_train)
    t2_train = gather(t2_labels, idx_train)
    t3_train = gather(t3_labels, idx_train)
    txt_train = gather(texts, idx_train)
    txt_val = gather(texts, idx_val)
    txt_test = gather(texts, idx_test)
    t1_val, t2_val, t3_val = gather(t1_labels, idx_val), gather(t2_labels, idx_val), gather(t3_labels, idx_val)
    t1_test, t2_test, t3_test = gather(t1_labels, idx_test), gather(t2_labels, idx_test), gather(t3_labels, idx_test)
    raw_test = gather(df["text"].tolist(), idx_test)

    print(f"Split: train={len(txt_train)}, val={len(txt_val)}, test={len(txt_test)}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    train_ds = MultiTaskSSDDataset(txt_train, t1_train, t2_train, t3_train, tokenizer, max_len)
    val_ds = MultiTaskSSDDataset(txt_val, t1_val, t2_val, t3_val, tokenizer, max_len)
    test_ds = MultiTaskSSDDataset(txt_test, t1_test, t2_test, t3_test, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, pin_memory=False)

    # Class weights
    cw1 = _compute_class_weights(train_ds.labels1.tolist(), len(TASK1_LABELS))
    valid_t2 = [l.item() for l in train_ds.labels2 if l.item() != -100]
    cw2 = _compute_class_weights(valid_t2, len(TASK2_LABELS)) if valid_t2 else None
    valid_t3 = [l.item() for l in train_ds.labels3 if l.item() != -100]
    cw3 = _compute_class_weights(valid_t3, len(TASK3_LABELS)) if valid_t3 else None

    # Model — force float32 for DeBERTa (MPS mixed-dtype crash)
    model = MultiTaskSSDModel(
        encoder_name=encoder_name,
        num_task1=len(TASK1_LABELS),
        num_task2=len(TASK2_LABELS),
        num_task3=len(TASK3_LABELS),
    )
    if "deberta" in encoder_name.lower():
        model = model.float()
    model = model.to(device)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print(f"Phase 1: heads warm-up ({phase1_epochs} ep, LR={phase1_lr})")
    model.freeze_encoder_layers(phase1_freeze_layers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer1 = AdamW(trainable, lr=phase1_lr, weight_decay=weight_decay)
    total_steps1 = (len(train_loader) // grad_accum) * phase1_epochs
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, int(warmup_ratio * total_steps1), total_steps1)

    best_f1, best_state = 0.0, None

    for epoch in range(1, phase1_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer1.zero_grad()
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            l1, l2, l3 = batch["labels1"].to(device), batch["labels2"].to(device), batch["labels3"].to(device)
            lo1, lo2, lo3 = model(ids, mask)
            loss, _, _, _ = multitask_loss(lo1, lo2, lo3, l1, l2, l3, cw1, cw2, cw3, lambdas)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer1.step()
                scheduler1.step()
                optimizer1.zero_grad()
            total_loss += loss.item()

        eval_results = _evaluate_multitask(model, val_loader, device)
        f1s = _macro_f1_summary(eval_results)
        avg_f1 = np.mean(list(f1s.values()))
        f1_str = "  ".join(f"{k}={v:.4f}" for k, v in f1s.items())
        print(f"  P1 E{epoch}  loss={total_loss/len(train_loader):.4f}  {f1_str}  avg={avg_f1:.4f}")
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print(f"Phase 2: full fine-tune ({phase2_epochs} ep, LR={phase2_lr})")
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.unfreeze_all()

    layer_groups = model.get_layer_groups()
    lr_mult = {"embeddings": 0.5, "layers_0-3": 0.5, "layers_4-7": 1.0, "layers_8-11": 1.5, "heads": 2.5}
    param_groups = [{"params": g["params"], "lr": phase2_lr * lr_mult.get(g["name"], 1.0)} for g in layer_groups]
    optimizer2 = AdamW(param_groups, weight_decay=weight_decay)
    total_steps2 = (len(train_loader) // grad_accum) * phase2_epochs
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, int(warmup_ratio * total_steps2), total_steps2)

    patience_counter, best_epoch = 0, 0

    for epoch in range(1, phase2_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer2.zero_grad()
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            l1, l2, l3 = batch["labels1"].to(device), batch["labels2"].to(device), batch["labels3"].to(device)
            lo1, lo2, lo3 = model(ids, mask)
            loss, _, _, _ = multitask_loss(lo1, lo2, lo3, l1, l2, l3, cw1, cw2, cw3, lambdas)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer2.step()
                scheduler2.step()
                optimizer2.zero_grad()
            total_loss += loss.item()

        eval_results = _evaluate_multitask(model, val_loader, device)
        f1s = _macro_f1_summary(eval_results)
        avg_f1 = np.mean(list(f1s.values()))
        f1_str = "  ".join(f"{k}={v:.4f}" for k, v in f1s.items())
        print(f"  P2 E{epoch}  loss={total_loss/len(train_loader):.4f}  {f1_str}  avg={avg_f1:.4f}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_results = _evaluate_multitask(model, test_loader, device)
    test_f1s = _macro_f1_summary(test_results)

    all_metrics = {}
    for task_num, task_key, labels_list in [(1, "task1", TASK1_LABELS), (2, "task2", TASK2_LABELS), (3, "task3", TASK3_LABELS)]:
        if task_key in test_results:
            true, pred = test_results[task_key]
            metrics = compute_metrics(true, pred, labels_list)
            all_metrics[task_key] = metrics
            print(f"  Test Task {task_num}: F1={metrics['macro_f1']:.4f}  {metrics['per_class_f1']}")

            log_result(
                approach="multitask",
                model=encoder_short,
                task=task_num,
                macro_f1=metrics["macro_f1"],
                split="test",
                embedding="—",
                dataset=f"train-{language}" if language != "multilingual" else "multilingual",
                accuracy=metrics["accuracy"],
                precision_macro=metrics["precision_macro"],
                recall_macro=metrics["recall_macro"],
                per_class_f1=metrics["per_class_f1"],
                epochs=phase1_epochs + best_epoch,
                hyperparams={"encoder": encoder_name, "language": language, "seed": seed,
                             "phase1_lr": phase1_lr, "phase2_lr": phase2_lr,
                             "batch_size": batch_size, "grad_accum": grad_accum},
                notes=f"multitask_{encoder_short}_{language}_s{seed}",
            )

    # Save model
    experiment_name = f"multitask_{encoder_short}_{language}_s{seed}"
    artifact_dir = os.path.join(_ARTIFACTS_DIR, experiment_name)
    model_dir = os.path.join(artifact_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(model_dir, "multitask_model.pt"))
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"encoder_name": encoder_name, "num_task1": len(TASK1_LABELS),
                    "num_task2": len(TASK2_LABELS), "num_task3": len(TASK3_LABELS)}, f)
    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({"test_f1s": test_f1s, "test_metrics": all_metrics}, f, indent=2)
    print(f"  Saved → {artifact_dir}")

    # Clean up GPU memory
    del model, optimizer1, optimizer2
    if device.type == "mps":
        torch.mps.empty_cache()

    return {"encoder": encoder_short, "language": language, "seed": seed, **test_f1s}


def run_grid(encoders=None, languages=None, seeds=None, **kwargs):
    """Run full experiment grid and print summary."""
    encoders = encoders or ENCODERS
    languages = languages or LANGUAGES
    seeds = seeds or SEEDS

    total = len(encoders) * len(languages) * len(seeds)
    results = []
    i = 0

    for encoder in encoders:
        for lang in languages:
            for seed in seeds:
                i += 1
                enc_short = encoder.split("/")[-1]
                print(f"\n{'#' * 60}")
                print(f"# [{i}/{total}] {enc_short} / {lang} / seed={seed}")
                print(f"{'#' * 60}")
                try:
                    r = run_single(encoder, lang, seed=seed, **kwargs)
                    results.append(r)
                except Exception as e:
                    print(f"ERROR: {e}")
                    traceback.print_exc()
                    results.append({"encoder": enc_short, "language": lang, "seed": seed, "error": str(e)})

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"{'MULTI-TASK GRID SUMMARY':^90}")
    print(f"{'=' * 90}")
    print(f"{'Encoder':<35} {'Lang':<14} {'T1 F1':<10} {'T2 F1':<10} {'T3 F1':<10} {'Avg F1':<10} {'Seeds'}")
    print(f"{'-' * 90}")

    grouped = defaultdict(list)
    for r in results:
        if "error" not in r:
            grouped[(r["encoder"], r["language"])].append(r)

    for (enc, lang), runs in sorted(grouped.items()):
        t1s = [r.get("task1", 0) for r in runs]
        t2s = [r.get("task2", 0) for r in runs]
        t3s = [r.get("task3", 0) for r in runs]
        avgs = [np.mean([r.get("task1", 0), r.get("task2", 0), r.get("task3", 0)]) for r in runs]
        n = len(runs)
        print(f"{enc:<35} {lang:<14} "
              f"{np.mean(t1s):.4f}±{np.std(t1s):.3f} "
              f"{np.mean(t2s):.4f}±{np.std(t2s):.3f} "
              f"{np.mean(t3s):.4f}±{np.std(t3s):.3f} "
              f"{np.mean(avgs):.4f}  "
              f"n={n}")

    print(f"{'=' * 90}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=None,
                        help="Override encoder list")
    parser.add_argument("--languages", nargs="+", default=None,
                        choices=["en", "es", "multilingual"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--phase1_epochs", type=int, default=3)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    run_grid(
        encoders=args.encoders,
        languages=args.languages,
        seeds=args.seeds,
        batch_size=args.batch_size,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        patience=args.patience,
    )
