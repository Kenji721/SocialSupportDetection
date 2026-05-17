"""
Traditional ML training pipeline: TF-IDF + Logistic Regression / Linear SVM.
"""

import json
import os

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from src.data.label_maps import get_label_config
from src.data.loading import load_multilingual, load_ssd_data
from src.data.splits import get_task_subset, stratified_split
from src.traditional.features import extract_features
from src.metrics import compute_metrics
from src.preprocessing import preprocess_traditional
from src.results import log_result
from src.traditional.vectorizers import build_tfidf_pipeline

# Paths
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EN_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-english.csv")
_ES_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-spanish.csv")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")


def _build_model(model_type: str):
    if model_type == "lr":
        return LogisticRegression(
            class_weight="balanced", max_iter=3000, solver="lbfgs", C=1.0
        )
    elif model_type == "svm":
        return LinearSVC(
            class_weight="balanced", max_iter=2000, C=1.0
        )
    elif model_type == "sgd":
        return SGDClassifier(
            class_weight="balanced", loss="hinge", max_iter=1000, random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'lr', 'svm', or 'sgd'.")


def _load_data(language: str):
    if language == "en":
        return load_ssd_data(_EN_CSV, "en")
    elif language == "es":
        return load_ssd_data(_ES_CSV, "es")
    elif language == "multilingual":
        return load_multilingual(_EN_CSV, _ES_CSV)
    else:
        raise ValueError(f"Unknown language: {language}")


def _extract_handcrafted(texts, language):
    """Extract handcrafted features as a numpy array."""
    from src.traditional.features import ALL_FEATURES
    records = [extract_features(t, language) for t in texts]
    return np.array([[r[f] for f in ALL_FEATURES] for r in records])


def train_traditional(
    task: int,
    language: str = "en",
    model_type: str = "lr",
    tfidf_mode: str = "combined",
    use_handcrafted: bool = True,
    seed: int = 42,
    max_features: int = 50000,
) -> dict:
    """Full traditional ML training pipeline.

    Returns metrics dict from test evaluation.
    """
    task_col = f"task{task}"
    labels_list, label2id, id2label, num_labels = get_label_config(task)

    # 1. Load data
    df = _load_data(language)

    # 2. Subset for task
    df = get_task_subset(df, task)

    # Encode labels
    df = df[df[task_col].isin(label2id)].copy()
    df["label"] = df[task_col].map(label2id)

    # 3. Preprocess
    lang_for_preprocess = "es" if language == "es" else "en"
    df["text_clean"] = df["text"].apply(preprocess_traditional)

    texts = df["text_clean"].tolist()
    raw_texts = df["text"].tolist()
    labels = df["label"].tolist()

    # Print label distributions
    print(f"\n{'=' * 60}")
    print(f"Traditional ML — Task {task} — {language} — {model_type} — {tfidf_mode}")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(labels)}")
    for name, lid in label2id.items():
        cnt = labels.count(lid)
        print(f"  {name}: {cnt} ({cnt / len(labels) * 100:.1f}%)")

    # 4. Split 80/10/10
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        texts, labels, seed=seed
    )
    raw_train, raw_val, raw_test, _, _, _ = stratified_split(
        raw_texts, labels, seed=seed
    )

    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 5. Vectorize (TF-IDF)
    tfidf = build_tfidf_pipeline(mode=tfidf_mode, max_features=max_features)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    # 6. Optionally append handcrafted features
    if use_handcrafted:
        lang_feat = lang_for_preprocess
        hc_train = csr_matrix(_extract_handcrafted(raw_train, lang_feat))
        hc_val = csr_matrix(_extract_handcrafted(raw_val, lang_feat))
        hc_test = csr_matrix(_extract_handcrafted(raw_test, lang_feat))

        X_train_feat = hstack([X_train_tfidf, hc_train])
        X_val_feat = hstack([X_val_tfidf, hc_val])
        X_test_feat = hstack([X_test_tfidf, hc_test])
    else:
        X_train_feat = X_train_tfidf
        X_val_feat = X_val_tfidf
        X_test_feat = X_test_tfidf

    # 7. Train
    model = _build_model(model_type)
    y_train_arr = np.array(y_train)
    model.fit(X_train_feat, y_train_arr)

    # 8. Evaluate on val + test
    y_val_pred = model.predict(X_val_feat)
    y_test_pred = model.predict(X_test_feat)

    val_metrics = compute_metrics(y_val, y_val_pred, labels_list)
    test_metrics = compute_metrics(y_test, y_test_pred, labels_list)

    print(f"\nVal  — Macro F1: {val_metrics['macro_f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
    print(f"Test — Macro F1: {test_metrics['macro_f1']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    print(f"Test per-class F1: {test_metrics['per_class_f1']}")

    # 9. Log results
    hc_str = "+hc" if use_handcrafted else ""
    experiment_name = f"traditional_{model_type}_{tfidf_mode}{hc_str}_task{task}_{language}_s{seed}"

    log_result(
        approach="traditional",
        model=model_type,
        task=task,
        macro_f1=test_metrics["macro_f1"],
        split="test",
        embedding=f"tfidf_{tfidf_mode}{hc_str}",
        dataset=f"train-{language}.csv" if language != "multilingual" else "multilingual",
        accuracy=test_metrics["accuracy"],
        precision_macro=test_metrics["precision_macro"],
        recall_macro=test_metrics["recall_macro"],
        per_class_f1=test_metrics["per_class_f1"],
        hyperparams={
            "max_features": max_features,
            "seed": seed,
            "use_handcrafted": use_handcrafted,
            "tfidf_mode": tfidf_mode,
        },
        notes=experiment_name,
    )

    # 10. Save artifacts
    artifact_dir = os.path.join(_ARTIFACTS_DIR, experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    # Save metrics
    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    # Save predictions CSV
    import pandas as pd
    pred_df = pd.DataFrame({
        "text": raw_test,
        "y_true": [id2label[y] for y in y_test],
        "y_pred": [id2label[int(y)] for y in y_test_pred],
    })
    pred_df.to_csv(os.path.join(artifact_dir, "predictions.csv"), index=False)

    # Save model pipeline
    joblib.dump({"tfidf": tfidf, "model": model, "use_handcrafted": use_handcrafted},
                os.path.join(artifact_dir, "pipeline.joblib"))

    print(f"Artifacts saved to {artifact_dir}")
    return test_metrics


def run_experiment_grid():
    """Run the full Phase 1 experiment grid (18 experiments)."""
    configs = [
        # (model_type, task, language)
        ("lr", 1, "en"), ("lr", 2, "en"), ("lr", 3, "en"),
        ("lr", 1, "es"), ("lr", 2, "es"), ("lr", 3, "es"),
        ("svm", 1, "en"), ("svm", 2, "en"), ("svm", 3, "en"),
        ("svm", 1, "es"), ("svm", 2, "es"), ("svm", 3, "es"),
        ("lr", 1, "multilingual"), ("lr", 2, "multilingual"), ("lr", 3, "multilingual"),
        ("svm", 1, "multilingual"), ("svm", 2, "multilingual"), ("svm", 3, "multilingual"),
    ]

    all_results = []
    for model_type, task, language in configs:
        print(f"\n{'#' * 60}")
        print(f"# Experiment: {model_type} / task{task} / {language}")
        print(f"{'#' * 60}")
        try:
            metrics = train_traditional(
                task=task,
                language=language,
                model_type=model_type,
                tfidf_mode="combined",
                use_handcrafted=True,
                seed=42,
            )
            all_results.append({
                "model": model_type, "task": task, "language": language,
                "macro_f1": metrics["macro_f1"],
            })
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "model": model_type, "task": task, "language": language,
                "macro_f1": None, "error": str(e),
            })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"{'PHASE 1 SUMMARY':^70}")
    print(f"{'=' * 70}")
    print(f"{'Model':<8} {'Task':<6} {'Language':<14} {'Macro F1':<10}")
    print(f"{'-' * 70}")
    for r in all_results:
        f1 = f"{r['macro_f1']:.4f}" if r['macro_f1'] is not None else "FAILED"
        print(f"{r['model']:<8} {r['task']:<6} {r['language']:<14} {f1:<10}")
    print(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    run_experiment_grid()
