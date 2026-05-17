"""
Generate competition submission CSVs from best models.

Loads the best model for each language × task combination, runs inference
on the test data, maps internal label IDs to submission label IDs, and
produces two CSV files (English + Spanish) packaged in a zip.

Submission label encoding (different from our internal encoding!):
  Task 1: Non-Supportive=0, Supportive=1
  Task 2: Group=0, Individual=1
  Task 3: Black Community=0, LGBTQ=1, Nation=2, Other=3, Religion=4, Women=5

Usage:
    python -m src.predict_submission
"""

import json
import os
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.finetuning.datasets import SSDDataset
from src.finetuning.multitask_model import MultiTaskSSDModel
from src.preprocessing import preprocess_transformer
from src.utils import get_device

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_ARTIFACTS = os.path.join(_PROJECT_ROOT, "artifacts")
_TEST_DIR = os.path.join(_PROJECT_ROOT, "data", "Test_phase_data")

# ── Submission label mappings ─────────────────────────────────────────────────
# Our internal → submission ID conversion

# Task 1: internal [Non-Supportive=0, Supportive=1] → submission same
TASK1_TO_SUB = {0: 0, 1: 1}

# Task 2: internal [Individual=0, Group=1] → submission [Group=0, Individual=1]
TASK2_TO_SUB = {0: 1, 1: 0}  # Individual→1, Group→0

# Task 3: internal [Nation=0, Other=1, LGBTQ=2, BlackComm=3, Religion=4, Women=5]
#       → submission [BlackComm=0, LGBTQ=1, Nation=2, Other=3, Religion=4, Women=5]
TASK3_TO_SUB = {0: 2, 1: 3, 2: 1, 3: 0, 4: 4, 5: 5}

# ── Model configs ─────────────────────────────────────────────────────────────
# Best models per language and task based on validation results

ENGLISH_MODELS = {
    "task1": {
        "type": "single",
        "path": os.path.join(_ARTIFACTS, "finetuning_bertweet-base_task1_en_s13", "model"),
        "note": "BERTweet s13, F1=0.831",
    },
    "task2": {
        "type": "multitask",
        "path": os.path.join(_ARTIFACTS, "multitask_twitter-xlm-roberta-base_multilingual_s42", "model"),
        "note": "XLM-T multitask multilingual, T2=0.903",
    },
    "task3": {
        "type": "single",
        "path": os.path.join(_ARTIFACTS, "finetuning_xlm-roberta-base_task3_multilingual_s13", "model"),
        "note": "XLM-R multi s13, F1=0.858",
    },
}

SPANISH_MODELS = {
    "task1": {
        "type": "single",
        "path": os.path.join(_ARTIFACTS, "finetuning_xlm-roberta-base_task1_es_s42", "model"),
        "note": "XLM-R es s42, F1=0.877",
    },
    "task2": {
        "type": "single",
        "path": os.path.join(_ARTIFACTS, "finetuning_robertuito-base-uncased_task2_es_s13", "model"),
        "note": "RoBERTuito s13, F1=0.953",
    },
    "task3": {
        "type": "single",
        "path": os.path.join(_ARTIFACTS, "finetuning_robertuito-base-uncased_task3_es_s42", "model"),
        "note": "RoBERTuito s42, F1=0.974",
    },
}


def _load_single_task_model(model_dir: str, device):
    """Load a single-task AutoModelForSequenceClassification."""
    base_path = os.path.join(model_dir, "base_model.json")
    with open(base_path) as f:
        info = json.load(f)

    use_fast = "bertweet" not in info["model_name"].lower()
    tokenizer = AutoTokenizer.from_pretrained(info["model_name"], use_fast=use_fast)

    model = AutoModelForSequenceClassification.from_pretrained(
        info["model_name"], num_labels=info["num_labels"]
    )
    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.pt"),
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, tokenizer


def _load_multitask_model(model_dir: str, device):
    """Load a MultiTaskSSDModel."""
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])

    model = MultiTaskSSDModel(
        encoder_name=config["encoder_name"],
        num_task1=config["num_task1"],
        num_task2=config["num_task2"],
        num_task3=config["num_task3"],
    )
    state_dict = torch.load(
        os.path.join(model_dir, "multitask_model.pt"),
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, tokenizer


def _predict_single_task(model, tokenizer, texts, device, batch_size=32, max_len=128):
    """Run single-task inference, return list of predicted class IDs."""
    dummy_labels = [0] * len(texts)
    dataset = SSDDataset(texts, dummy_labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
    return [int(p) for p in all_preds]


def _predict_multitask(model, tokenizer, texts, device, batch_size=32, max_len=128):
    """Run multitask inference, return (task1_preds, task2_preds, task3_preds)."""
    dummy_labels = [0] * len(texts)
    dataset = SSDDataset(texts, dummy_labels, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    p1, p2, p3 = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits1, logits2, logits3 = model(input_ids, attention_mask)
            p1.extend(logits1.argmax(dim=-1).cpu().numpy())
            p2.extend(logits2.argmax(dim=-1).cpu().numpy())
            p3.extend(logits3.argmax(dim=-1).cpu().numpy())
    return [int(x) for x in p1], [int(x) for x in p2], [int(x) for x in p3]


def predict_english(device):
    """Generate English predictions."""
    print("\n" + "=" * 60)
    print("English Predictions")
    print("=" * 60)

    # Load test data
    test_path = os.path.join(_TEST_DIR, "test_phase_english.csv")
    df = pd.read_csv(test_path)
    texts = df["text"].apply(preprocess_transformer).tolist()
    ids = df["id"].tolist()
    print(f"Loaded {len(texts)} English test samples")

    # Task 1: BERTweet
    cfg = ENGLISH_MODELS["task1"]
    print(f"\nTask 1: {cfg['note']}")
    model_t1, tok_t1 = _load_single_task_model(cfg["path"], device)
    preds_t1 = _predict_single_task(model_t1, tok_t1, texts, device)
    del model_t1

    # Task 2: XLM-T multitask (use task2 output)
    cfg = ENGLISH_MODELS["task2"]
    print(f"Task 2: {cfg['note']}")
    model_mt, tok_mt = _load_multitask_model(cfg["path"], device)
    _, preds_t2, _ = _predict_multitask(model_mt, tok_mt, texts, device)
    del model_mt

    # Task 3: XLM-R multilingual s13 (single-task, best for Task 3)
    cfg = ENGLISH_MODELS["task3"]
    print(f"Task 3: {cfg['note']}")
    model_t3, tok_t3 = _load_single_task_model(cfg["path"], device)
    preds_t3 = _predict_single_task(model_t3, tok_t3, texts, device)
    del model_t3

    if device.type == "mps":
        torch.mps.empty_cache()

    # Convert to submission IDs
    sub_t1 = [TASK1_TO_SUB[p] for p in preds_t1]
    sub_t2 = [TASK2_TO_SUB[p] for p in preds_t2]
    sub_t3 = [TASK3_TO_SUB[p] for p in preds_t3]

    # Build submission DataFrame
    result = pd.DataFrame({
        "id": ids,
        "en_support_pred": sub_t1,
        "en_individual_pred": sub_t2,
        "en_multiclass_pred": sub_t3,
    })

    # Print distribution summary
    print(f"\nPrediction distribution:")
    print(f"  Task 1: Non-Supportive={sub_t1.count(0)}, Supportive={sub_t1.count(1)}")
    print(f"  Task 2: Group={sub_t2.count(0)}, Individual={sub_t2.count(1)}")
    t3_names = {0: "BlackComm", 1: "LGBTQ", 2: "Nation", 3: "Other", 4: "Religion", 5: "Women"}
    for k, v in sorted(t3_names.items()):
        print(f"  Task 3 {v}: {sub_t3.count(k)}")

    return result


def predict_spanish(device):
    """Generate Spanish predictions."""
    print("\n" + "=" * 60)
    print("Spanish Predictions")
    print("=" * 60)

    # Load test data
    test_path = os.path.join(_TEST_DIR, "test_phase_spanish.csv")
    df = pd.read_csv(test_path)
    texts = df["comment"].apply(preprocess_transformer).tolist()
    ids = df["id"].tolist()
    print(f"Loaded {len(texts)} Spanish test samples")

    # Task 1: RoBERTuito
    cfg = SPANISH_MODELS["task1"]
    print(f"\nTask 1: {cfg['note']}")
    model_t1, tok_t1 = _load_single_task_model(cfg["path"], device)
    preds_t1 = _predict_single_task(model_t1, tok_t1, texts, device)
    del model_t1

    # Task 2: RoBERTuito
    cfg = SPANISH_MODELS["task2"]
    print(f"Task 2: {cfg['note']}")
    model_t2, tok_t2 = _load_single_task_model(cfg["path"], device)
    preds_t2 = _predict_single_task(model_t2, tok_t2, texts, device)
    del model_t2

    # Task 3: RoBERTuito
    cfg = SPANISH_MODELS["task3"]
    print(f"Task 3: {cfg['note']}")
    model_t3, tok_t3 = _load_single_task_model(cfg["path"], device)
    preds_t3 = _predict_single_task(model_t3, tok_t3, texts, device)
    del model_t3

    if device.type == "mps":
        torch.mps.empty_cache()

    # Convert to submission IDs
    sub_t1 = [TASK1_TO_SUB[p] for p in preds_t1]
    sub_t2 = [TASK2_TO_SUB[p] for p in preds_t2]
    sub_t3 = [TASK3_TO_SUB[p] for p in preds_t3]

    # Build submission DataFrame
    result = pd.DataFrame({
        "id": ids,
        "es_support_pred": sub_t1,
        "es_individual_pred": sub_t2,
        "es_multiclass_pred": sub_t3,
    })

    # Print distribution summary
    print(f"\nPrediction distribution:")
    print(f"  Task 1: Non-Supportive={sub_t1.count(0)}, Supportive={sub_t1.count(1)}")
    print(f"  Task 2: Group={sub_t2.count(0)}, Individual={sub_t2.count(1)}")
    t3_names = {0: "BlackComm", 1: "LGBTQ", 2: "Nation", 3: "Other", 4: "Religion", 5: "Women"}
    for k, v in sorted(t3_names.items()):
        print(f"  Task 3 {v}: {sub_t3.count(k)}")

    return result


def main():
    device = get_device()
    print(f"Device: {device}")

    out_dir = os.path.join(_PROJECT_ROOT, "submissions")
    os.makedirs(out_dir, exist_ok=True)

    # Generate predictions
    en_df = predict_english(device)
    es_df = predict_spanish(device)

    # Save CSVs
    en_path = os.path.join(out_dir, "english_submission.csv")
    es_path = os.path.join(out_dir, "spanish_submission.csv")
    en_df.to_csv(en_path, index=False)
    es_df.to_csv(es_path, index=False)
    print(f"\nSaved: {en_path}")
    print(f"Saved: {es_path}")

    # Create zip (no folders within, just the CSVs)
    zip_en = os.path.join(out_dir, "submission_english.zip")
    zip_es = os.path.join(out_dir, "submission_spanish.zip")

    with zipfile.ZipFile(zip_en, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(en_path, "english_submission.csv")
    with zipfile.ZipFile(zip_es, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(es_path, "spanish_submission.csv")

    print(f"\nZips: {zip_en}")
    print(f"      {zip_es}")

    # Sanity checks
    print(f"\n{'=' * 60}")
    print("Sanity Checks")
    print(f"{'=' * 60}")
    print(f"EN rows: {len(en_df)} (expected 2000)")
    print(f"ES rows: {len(es_df)} (expected 651)")
    print(f"EN IDs unique: {en_df['id'].nunique()}")
    print(f"ES IDs unique: {es_df['id'].nunique()}")
    print(f"EN Task 1 values: {sorted(en_df['en_support_pred'].unique())}")
    print(f"EN Task 2 values: {sorted(en_df['en_individual_pred'].unique())}")
    print(f"EN Task 3 values: {sorted(en_df['en_multiclass_pred'].unique())}")
    print(f"ES Task 1 values: {sorted(es_df['es_support_pred'].unique())}")
    print(f"ES Task 2 values: {sorted(es_df['es_individual_pred'].unique())}")
    print(f"ES Task 3 values: {sorted(es_df['es_multiclass_pred'].unique())}")


if __name__ == "__main__":
    main()
