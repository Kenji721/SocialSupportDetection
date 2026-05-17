"""
Task 2: Individual vs Group Classification
Model: distilbert-base-uncased fine-tuned
Trains on Supportive comments only.
Classifies: Individual (0) vs Group (1)
Usage: python src/train_task2.py --csv Train_Data_SSD26/train-english.csv --text_col text --label_col task2
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.utils import (
    get_device, clean_text, SupportDataset, run_eda, train_epoch, evaluate,
)


def main(args):
    device = get_device()
    print(f"Device: {device}")

    df = pd.read_csv(args.csv)
    assert args.text_col in df.columns,  f"Column '{args.text_col}' not found"
    assert args.label_col in df.columns, f"Column '{args.label_col}' not found"

    # Filter to Supportive comments only
    df = df[df["task1"] == "Supportive"][[args.text_col, args.label_col]].dropna()
    print(f"Filtered to Supportive comments: {len(df)} samples")

    run_eda(df, args.text_col, args.label_col)

    label_map = {"Individual": 0, "Group": 1}
    id2label  = {v: k for k, v in label_map.items()}
    df["label"] = df[args.label_col].map(label_map)
    assert df["label"].notna().all(), "Unexpected label values found"

    df[args.text_col] = df[args.text_col].apply(clean_text)
    texts  = df[args.text_col].tolist()
    labels = df["label"].tolist()

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    print(f"Split → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    counts = np.bincount(y_train)
    class_weights = torch.tensor(
        [len(y_train) / (len(counts) * c) for c in counts], dtype=torch.float
    )
    print(f"Class weights: {class_weights.tolist()}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_ds = SupportDataset(X_train, y_train, tokenizer, args.max_len)
    val_ds   = SupportDataset(X_val,   y_val,   tokenizer, args.max_len)
    test_ds  = SupportDataset(X_test,  y_test,  tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label_map,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1, best_state, patience_counter = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights
        )
        val_labels, val_preds = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_labels, test_preds = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(classification_report(
        test_labels, test_preds,
        target_names=["Individual", "Group"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"\nMacro F1: {macro_f1:.4f}")
    print("=" * 60)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        print(f"\nModel saved to: {args.save_dir}")

    return macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2 - Individual vs Group Classification")
    parser.add_argument("--csv",       required=True,  help="Path to CSV file")
    parser.add_argument("--text_col",  default="text", help="Column name for text")
    parser.add_argument("--label_col", default="task2",help="Column name for labels")
    parser.add_argument("--max_len",   type=int, default=128)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--epochs",    type=int, default=5)
    parser.add_argument("--lr",        type=float, default=2e-5)
    parser.add_argument("--patience",  type=int, default=2,  help="Early stopping patience")
    parser.add_argument("--save_dir",  default="./task2_model", help="Where to save best model")
    args = parser.parse_args()
    main(args)
