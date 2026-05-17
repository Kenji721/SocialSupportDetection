"""
Independent pipeline: runs T1, T2, T3 as separate models on ALL texts.
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


def run_independent_pipeline(
    task1_model_dir: str,
    task2_model_dir: str,
    task3_model_dir: str,
    texts: list[str],
    model_type: str = "transformer",
) -> pd.DataFrame:
    """Run all three models independently on ALL texts.

    Task 2 and Task 3 predict on all texts (no filtering).
    Returns DataFrame with columns: task1_pred, task2_pred, task3_pred.
    """
    print("Independent pipeline — predicting Task 1...")
    t1_ids = _predict(task1_model_dir, texts, model_type)
    t1_labels = [TASK1_ID2LABEL[p] for p in t1_ids]

    print("Independent pipeline — predicting Task 2...")
    t2_ids = _predict(task2_model_dir, texts, model_type)
    t2_labels = [TASK2_ID2LABEL[p] for p in t2_ids]

    print("Independent pipeline — predicting Task 3...")
    t3_ids = _predict(task3_model_dir, texts, model_type)
    t3_labels = [TASK3_ID2LABEL[p] for p in t3_ids]

    return pd.DataFrame({
        "task1_pred": t1_labels,
        "task2_pred": t2_labels,
        "task3_pred": t3_labels,
    })
