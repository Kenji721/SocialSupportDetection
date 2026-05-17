"""
Evaluation metrics for SSD-2026.
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, target_names: list[str]) -> dict:
    """Compute standard classification metrics.

    Returns dict with: macro_f1, accuracy, precision_macro, recall_macro,
    per_class_f1, confusion_matrix.
    """
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {name: round(float(score), 4) for name, score in zip(target_names, per_class)},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
