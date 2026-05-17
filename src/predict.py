"""
Pipeline Inference: runs all three task models sequentially on new data.
Produces a submission CSV with predictions for all three subtasks.

Usage:
    python src/predict.py \
      --csv Test_phase_data/test_phase_english.csv \
      --text_col text \
      --task1_model ./task1_model \
      --task2_model ./task2_model \
      --task3_model ./task3_model \
      --output predictions.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

from src.utils import get_device, clean_text


# Label maps (must match training scripts)
TASK1_ID2LABEL = {0: "Non-Supportive", 1: "Supportive"}
TASK2_ID2LABEL = {0: "Individual", 1: "Group"}
TASK3_ID2LABEL = {0: "Nation", 1: "Other", 2: "LGBTQ", 3: "Black Community", 4: "Religion", 5: "Women"}

# Numeric encoding for submission format
TASK1_SUBMIT = {"Non-Supportive": 0, "Supportive": 1}
TASK2_SUBMIT = {"No": 0, "Individual": 1, "Group": 2}
TASK3_SUBMIT = {"No": 0, "Nation": 1, "Other": 2, "LGBTQ": 3, "Black Community": 4, "Religion": 5, "Women": 6}


def load_model(model_dir, device):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def predict_batch(model, tokenizer, texts, device, max_len=128):
    encodings = tokenizer(
        texts, truncation=True, padding=True,
        max_length=max_len, return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits.argmax(dim=-1).cpu().numpy()


def main(args):
    device = get_device()
    print(f"Device: {device}")

    # Load models
    print("Loading Task 1 model...")
    task1_model, task1_tok = load_model(args.task1_model, device)
    print("Loading Task 2 model...")
    task2_model, task2_tok = load_model(args.task2_model, device)
    print("Loading Task 3 model...")
    task3_model, task3_tok = load_model(args.task3_model, device)

    # Load data
    df = pd.read_csv(args.csv)
    assert args.text_col in df.columns, f"Column '{args.text_col}' not found"
    ids = df["id"].tolist()
    texts = df[args.text_col].apply(clean_text).tolist()

    # Stage 1: Task 1 predictions (all texts)
    print("\nRunning Task 1 predictions...")
    task1_preds = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[i:i + args.batch_size]
        preds = predict_batch(task1_model, task1_tok, batch, device, args.max_len)
        task1_preds.extend(preds)
    task1_labels = [TASK1_ID2LABEL[p] for p in task1_preds]

    # Stage 2: Task 2 predictions (Supportive only)
    supportive_indices = [i for i, l in enumerate(task1_labels) if l == "Supportive"]
    task2_labels = ["No"] * len(texts)

    if supportive_indices:
        print(f"\nRunning Task 2 predictions on {len(supportive_indices)} Supportive texts...")
        supportive_texts = [texts[i] for i in supportive_indices]
        task2_preds = []
        for i in tqdm(range(0, len(supportive_texts), args.batch_size)):
            batch = supportive_texts[i:i + args.batch_size]
            preds = predict_batch(task2_model, task2_tok, batch, device, args.max_len)
            task2_preds.extend(preds)
        for idx, pred in zip(supportive_indices, task2_preds):
            task2_labels[idx] = TASK2_ID2LABEL[pred]

    # Stage 3: Task 3 predictions (Group only)
    group_indices = [i for i, l in enumerate(task2_labels) if l == "Group"]
    task3_labels = ["No"] * len(texts)

    if group_indices:
        print(f"\nRunning Task 3 predictions on {len(group_indices)} Group texts...")
        group_texts = [texts[i] for i in group_indices]
        task3_preds = []
        for i in tqdm(range(0, len(group_texts), args.batch_size)):
            batch = group_texts[i:i + args.batch_size]
            preds = predict_batch(task3_model, task3_tok, batch, device, args.max_len)
            task3_preds.extend(preds)
        for idx, pred in zip(group_indices, task3_preds):
            task3_labels[idx] = TASK3_ID2LABEL[pred]

    # Build output DataFrame
    result = pd.DataFrame({
        "id": ids,
        "task1": task1_labels,
        "task2": task2_labels,
        "task3": task3_labels,
    })

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(result)}")
    print(f"\nTask 1: {result['task1'].value_counts().to_dict()}")
    print(f"Task 2: {result['task2'].value_counts().to_dict()}")
    print(f"Task 3: {result['task3'].value_counts().to_dict()}")
    print("=" * 60)

    # Save predictions
    result.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")

    # Also save submission format
    submission_path = args.output.replace(".csv", "_submission.csv")
    submission = pd.DataFrame({
        "id": ids,
        "en_support_pred": [TASK1_SUBMIT[l] for l in task1_labels],
        "en_individual_pred": [TASK2_SUBMIT[l] for l in task2_labels],
        "en_multiclass_pred": [TASK3_SUBMIT[l] for l in task3_labels],
    })
    submission.to_csv(submission_path, index=False)
    print(f"Submission format saved to: {submission_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Inference - Social Support Detection")
    parser.add_argument("--csv",         required=True,              help="Path to test CSV file")
    parser.add_argument("--text_col",    default="text",             help="Column name for text")
    parser.add_argument("--task1_model", default="./task1_model",    help="Path to Task 1 model")
    parser.add_argument("--task2_model", default="./task2_model",    help="Path to Task 2 model")
    parser.add_argument("--task3_model", default="./task3_model",    help="Path to Task 3 model")
    parser.add_argument("--output",      default="predictions.csv",  help="Output CSV path")
    parser.add_argument("--max_len",     type=int, default=128)
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()
    main(args)
