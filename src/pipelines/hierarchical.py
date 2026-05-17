"""
Hierarchical pipeline: T1 → filter Supportive → T2 → filter Group → T3.
"""

import pandas as pd

from src.data.label_maps import TASK1_ID2LABEL, TASK2_ID2LABEL, TASK3_ID2LABEL


def _predict(model_dir: str, texts: list[str], model_type: str) -> list[int]:
    if model_type == "transformer":
        from src.finetuning.predict import predict_transformer
        return predict_transformer(model_dir, texts)
    elif model_type == "traditional":
        from src.traditional.predict import predict_traditional
        return predict_traditional(model_dir, texts)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_hierarchical_pipeline(
    task1_model_dir: str,
    task2_model_dir: str,
    task3_model_dir: str,
    texts: list[str],
    model_type: str = "transformer",
) -> pd.DataFrame:
    """Cascade: T1 → filter Supportive → T2 → filter Group → T3.

    Non-Supportive texts get task2="No", task3="No".
    Individual texts get task3="No".
    Returns DataFrame with columns: task1_pred, task2_pred, task3_pred.
    """
    n = len(texts)

    # Stage 1: Task 1 on all texts
    print("Hierarchical pipeline — predicting Task 1...")
    t1_ids = _predict(task1_model_dir, texts, model_type)
    t1_labels = [TASK1_ID2LABEL[p] for p in t1_ids]

    # Stage 2: Task 2 on Supportive texts only
    t2_labels = ["No"] * n
    supportive_indices = [i for i, l in enumerate(t1_labels) if l == "Supportive"]

    if supportive_indices:
        print(f"Hierarchical pipeline — predicting Task 2 on {len(supportive_indices)} Supportive texts...")
        supportive_texts = [texts[i] for i in supportive_indices]
        t2_ids = _predict(task2_model_dir, supportive_texts, model_type)
        for idx, pred in zip(supportive_indices, t2_ids):
            t2_labels[idx] = TASK2_ID2LABEL[pred]

    # Stage 3: Task 3 on Group texts only
    t3_labels = ["No"] * n
    group_indices = [i for i, l in enumerate(t2_labels) if l == "Group"]

    if group_indices:
        print(f"Hierarchical pipeline — predicting Task 3 on {len(group_indices)} Group texts...")
        group_texts = [texts[i] for i in group_indices]
        t3_ids = _predict(task3_model_dir, group_texts, model_type)
        for idx, pred in zip(group_indices, t3_ids):
            t3_labels[idx] = TASK3_ID2LABEL[pred]

    return pd.DataFrame({
        "task1_pred": t1_labels,
        "task2_pred": t2_labels,
        "task3_pred": t3_labels,
    })
