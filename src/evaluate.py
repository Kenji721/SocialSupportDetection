"""
Unified evaluation script for all three subtasks.
Computes macro F1, per-class metrics, and confusion matrices.

Usage:
    python src/evaluate.py \
      --gold Train_Data_SSD26/train-english.csv \
      --pred predictions.csv
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def evaluate_task(gold_labels, pred_labels, task_name, target_names):
    print(f"\n{'=' * 60}")
    print(f"{task_name} RESULTS")
    print(f"{'=' * 60}")
    print(f"Samples evaluated: {len(gold_labels)}")
    print(classification_report(gold_labels, pred_labels, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(gold_labels, pred_labels, labels=target_names))
    macro_f1 = f1_score(gold_labels, pred_labels, average="macro")
    print(f"\nMacro F1: {macro_f1:.4f}")
    return macro_f1


def main(args):
    gold = pd.read_csv(args.gold)
    pred = pd.read_csv(args.pred)

    # Merge on id
    merged = gold.merge(pred, on="id", suffixes=("_gold", "_pred"))
    print(f"Matched {len(merged)} / {len(gold)} gold samples")

    # Task 1: All samples
    f1_task1 = evaluate_task(
        merged["task1_gold"].tolist(),
        merged["task1_pred"].tolist(),
        "TASK 1 (Supportive vs Non-Supportive)",
        ["Non-Supportive", "Supportive"],
    )

    # Task 2: Only where gold task1 == Supportive
    task2_mask = merged["task1_gold"] == "Supportive"
    task2_df = merged[task2_mask]
    f1_task2 = evaluate_task(
        task2_df["task2_gold"].tolist(),
        task2_df["task2_pred"].tolist(),
        "TASK 2 (Individual vs Group)",
        ["Individual", "Group"],
    )

    # Task 3: Only where gold task2 == Group
    task3_mask = merged["task2_gold"] == "Group"
    task3_df = merged[task3_mask]
    f1_task3 = evaluate_task(
        task3_df["task3_gold"].tolist(),
        task3_df["task3_pred"].tolist(),
        "TASK 3 (Community Classification)",
        ["Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"],
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Task 1 Macro F1: {f1_task1:.4f}")
    print(f"  Task 2 Macro F1: {f1_task2:.4f}")
    print(f"  Task 3 Macro F1: {f1_task3:.4f}")
    print(f"{'=' * 60}")

    return f1_task1, f1_task2, f1_task3


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SSD predictions")
    parser.add_argument("--gold", required=True, help="Path to gold-standard CSV")
    parser.add_argument("--pred", required=True, help="Path to predictions CSV")
    args = parser.parse_args()
    main(args)
