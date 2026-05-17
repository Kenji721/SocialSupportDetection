"""
LLM judge evaluation for SSD-2026.

Calls a local Ollama model (or OpenAI-compatible API) for zero-shot / few-shot
classification on all three tasks. Supports parallel API calls for speed.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.data.label_maps import get_label_config
from src.data.loading import load_ssd_data
from src.data.splits import get_task_subset, stratified_split
from src.llm_judge.parse import parse_label
from src.llm_judge.prompts import get_prompt
from src.metrics import compute_metrics
from src.results import log_result

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_EN_CSV = os.path.join(_PROJECT_ROOT, "data", "Train_Data_SSD26", "train-english.csv")
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")

# Default Ollama endpoint
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")


def _call_ollama(prompt: str, model: str, max_retries: int = 3) -> str | None:
    """Call Ollama generate API and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 20,
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(_OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return None


def _call_openai_compatible(
    prompt: str, model: str, api_base: str, api_key: str, max_retries: int = 5
) -> str | None:
    """Call OpenAI-compatible chat API."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 100,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return None


def _call_single(
    idx: int,
    text: str,
    prompt_template: str,
    task: int,
    model: str,
    api_backend: str,
    api_base: str,
    api_key: str,
    max_retries: int,
) -> tuple[int, str | None]:
    """Process a single text and return (index, parsed_label)."""
    prompt = prompt_template.format(text=text)

    if api_backend == "ollama":
        raw = _call_ollama(prompt, model, max_retries)
    else:
        raw = _call_openai_compatible(
            prompt, model, api_base or "https://api.openai.com/v1",
            api_key or os.environ.get("OPENAI_API_KEY", ""),
            max_retries,
        )

    if raw is None:
        return idx, None
    return idx, parse_label(raw, task)


def run_llm_judge(
    task: int,
    texts: list[str],
    model: str = "qwen2.5:7b",
    mode: str = "zero",
    max_retries: int = 3,
    api_backend: str = "ollama",
    api_base: str = None,
    api_key: str = None,
    max_workers: int = 20,
) -> list[str | None]:
    """Run LLM judge on a list of texts with parallel API calls.

    Args:
        task: Task number (1, 2, or 3).
        texts: List of input texts.
        model: Model name (Ollama model or OpenAI model ID).
        mode: "zero" for zero-shot, "few" for few-shot.
        max_retries: Number of retries per API call.
        api_backend: "ollama" or "openai".
        api_base: Base URL for OpenAI-compatible API.
        api_key: API key for OpenAI-compatible API.
        max_workers: Number of parallel threads (default: 20).

    Returns:
        List of predicted labels (None for failed parses).
    """
    prompt_template = get_prompt(task, mode)
    predictions = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _call_single, i, text, prompt_template, task,
                model, api_backend, api_base, api_key, max_retries,
            ): i
            for i, text in enumerate(texts)
        }

        with tqdm(total=len(texts), desc=f"LLM Judge T{task} ({mode}-shot)") as pbar:
            for future in as_completed(futures):
                idx, label = future.result()
                predictions[idx] = label
                pbar.update(1)

    return predictions


def evaluate_llm_judge(
    task: int,
    language: str = "en",
    model: str = "qwen2.5:7b",
    mode: str = "zero",
    seed: int = 42,
    max_samples: int = None,
    api_backend: str = "ollama",
    api_base: str = None,
    api_key: str = None,
    max_workers: int = 20,
) -> dict:
    """Evaluate LLM judge on the test split of a task."""
    task_col = f"task{task}"
    labels_list, label2id, id2label, num_labels = get_label_config(task)

    print(f"\n{'=' * 60}")
    print(f"LLM Judge — {model} — {mode}-shot — Task {task} — {language}")
    print(f"{'=' * 60}")

    # Load & subset
    df = load_ssd_data(_EN_CSV, language)
    df = get_task_subset(df, task)
    df = df[df[task_col].isin(label2id)].copy()

    texts = df["text"].tolist()
    labels = df[task_col].tolist()
    labels_int = [label2id[l] for l in labels]

    # Use only test split
    _, _, X_test, _, _, y_test = stratified_split(texts, labels_int, seed=seed)
    _, _, raw_test, _, _, _ = stratified_split(texts, labels_int, seed=seed)

    if max_samples and max_samples < len(X_test):
        X_test = X_test[:max_samples]
        y_test = y_test[:max_samples]
        raw_test = raw_test[:max_samples]

    print(f"Test samples: {len(X_test)}")
    print(f"Parallel workers: {max_workers}")

    # Run judge
    preds_raw = run_llm_judge(
        task=task, texts=X_test, model=model, mode=mode,
        api_backend=api_backend, api_base=api_base, api_key=api_key,
        max_workers=max_workers,
    )

    # Count parse failures
    valid_mask = [p is not None for p in preds_raw]
    n_valid = sum(valid_mask)
    n_failed = len(preds_raw) - n_valid
    parse_rate = n_valid / len(preds_raw) * 100 if preds_raw else 0
    print(f"Parse success: {n_valid}/{len(preds_raw)} ({parse_rate:.1f}%)")
    print(f"Parse failures: {n_failed}")

    # Filter to valid predictions for metrics
    y_true_valid = [y for y, v in zip(y_test, valid_mask) if v]
    y_pred_valid = [label2id[p] for p, v in zip(preds_raw, valid_mask) if v]

    if not y_pred_valid:
        print("No valid predictions — cannot compute metrics.")
        return {"macro_f1": 0.0, "parse_rate": parse_rate}

    test_metrics = compute_metrics(
        np.array(y_true_valid), np.array(y_pred_valid), labels_list
    )

    print(f"Macro F1: {test_metrics['macro_f1']:.4f} (on {n_valid} valid samples)")
    print(f"Per-class F1: {test_metrics['per_class_f1']}")

    # Log results
    model_short = model.replace("/", "_").replace(":", "_")
    experiment_name = f"llm_judge_{model_short}_{mode}_task{task}_{language}"

    log_result(
        approach="llm_judge",
        model=model_short,
        task=task,
        macro_f1=test_metrics["macro_f1"],
        split="test",
        embedding="—",
        dataset=f"train-{language}.csv",
        accuracy=test_metrics.get("accuracy"),
        precision_macro=test_metrics.get("precision_macro"),
        recall_macro=test_metrics.get("recall_macro"),
        per_class_f1=test_metrics.get("per_class_f1"),
        hyperparams={
            "mode": mode,
            "model": model,
            "parse_rate": round(parse_rate, 1),
            "n_valid": n_valid,
            "n_total": len(preds_raw),
        },
        notes=experiment_name,
    )

    # Save artifacts
    artifact_dir = os.path.join(_ARTIFACTS_DIR, experiment_name)
    os.makedirs(artifact_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        "text": raw_test if len(raw_test) == len(preds_raw) else X_test,
        "y_true": [id2label[y] for y in y_test],
        "y_pred": [p if p is not None else "PARSE_FAIL" for p in preds_raw],
    })
    pred_df.to_csv(os.path.join(artifact_dir, "predictions.csv"), index=False)

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({
            "test": test_metrics,
            "parse_rate": parse_rate,
            "n_valid": n_valid,
            "n_failed": n_failed,
        }, f, indent=2)

    print(f"Artifacts saved to {artifact_dir}")
    return test_metrics


def run_judge_grid(
    model: str = "qwen2.5:7b",
    modes=None,
    tasks=None,
    language: str = "en",
    max_samples: int = None,
    api_backend: str = "ollama",
    api_base: str = None,
    api_key: str = None,
    max_workers: int = 20,
):
    """Run LLM judge on all tasks × modes."""
    modes = modes or ["zero", "few"]
    tasks = tasks or [1, 2, 3]

    all_results = []
    for mode in modes:
        for task in tasks:
            try:
                metrics = evaluate_llm_judge(
                    task=task, language=language, model=model,
                    mode=mode, max_samples=max_samples,
                    api_backend=api_backend, api_base=api_base, api_key=api_key,
                    max_workers=max_workers,
                )
                all_results.append({
                    "mode": mode, "task": task,
                    "macro_f1": metrics.get("macro_f1"),
                })
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "mode": mode, "task": task,
                    "macro_f1": None, "error": str(e),
                })

    # Summary
    print(f"\n{'=' * 60}")
    print(f"{'LLM JUDGE SUMMARY':^60}")
    print(f"{'=' * 60}")
    for r in all_results:
        f1 = f"{r['macro_f1']:.4f}" if r.get("macro_f1") is not None else "ERROR"
        print(f"  {r['mode']}-shot  Task {r['task']}  →  Macro F1: {f1}")
    print(f"{'=' * 60}")

    return all_results


def _call_combined_single(
    idx: int,
    text: str,
    prompt_template: str,
    model: str,
    api_backend: str,
    api_base: str,
    api_key: str,
    max_retries: int,
) -> tuple[int, dict | None]:
    """Process a single text with combined prompt, return (index, {task1, task2, task3})."""
    prompt = prompt_template.format(text=text)

    if api_backend == "ollama":
        raw = _call_ollama(prompt, model, max_retries)
    else:
        raw = _call_openai_compatible(
            prompt, model, api_base or "https://api.openai.com/v1",
            api_key or os.environ.get("OPENAI_API_KEY", ""),
            max_retries,
        )

    if raw is None:
        return idx, None

    # Parse JSON response
    try:
        raw_clean = raw.strip()
        # Handle markdown code blocks
        if raw_clean.startswith("```"):
            raw_clean = raw_clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw_clean)
        return idx, {
            "task1": result.get("task1"),
            "task2": result.get("task2"),
            "task3": result.get("task3"),
        }
    except (json.JSONDecodeError, AttributeError):
        return idx, None


def run_combined_judge(
    texts: list[str],
    model: str = "gpt-5.4-mini",
    max_retries: int = 3,
    api_backend: str = "openai",
    api_base: str = None,
    api_key: str = None,
    max_workers: int = 10,
) -> list[dict | None]:
    """Run combined 3-task LLM judge on a list of texts with parallel API calls.

    Returns:
        List of dicts with keys task1, task2, task3 (or None for failures).
    """
    prompt_template = get_prompt("combined", "few")
    predictions = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _call_combined_single, i, text, prompt_template,
                model, api_backend, api_base, api_key, max_retries,
            ): i
            for i, text in enumerate(texts)
        }

        with tqdm(total=len(texts), desc="LLM Judge (combined)") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                predictions[idx] = result
                pbar.update(1)

    return predictions


def evaluate_combined_judge(
    language: str = "en",
    model: str = "gpt-5.4-mini",
    seed: int = 42,
    max_samples: int = None,
    api_backend: str = "openai",
    api_base: str = None,
    api_key: str = None,
    max_workers: int = 10,
) -> dict:
    """Evaluate combined LLM judge on all 3 tasks at once."""
    from src.data.label_maps import get_label_config

    print(f"\n{'=' * 60}")
    print(f"Combined LLM Judge — {model} — {language}")
    print(f"{'=' * 60}")

    # Load full dataset (Task 1 has all samples)
    df = load_ssd_data(_EN_CSV, language)
    task1_col = "task1"
    labels1, label2id1, id2label1, _ = get_label_config(1)
    labels2, label2id2, id2label2, _ = get_label_config(2)
    labels3, label2id3, id2label3, _ = get_label_config(3)

    df = df[df[task1_col].isin(label2id1)].copy()
    texts = df["text"].tolist()
    labels1_int = [label2id1[l] for l in df["task1"]]

    # Split using Task 1 stratification
    _, _, X_test, _, _, y1_test = stratified_split(texts, labels1_int, seed=seed)

    # Get Task 2 and Task 3 ground truth aligned with test split
    all_task2 = df["task2"].tolist()
    all_task3 = df["task3"].tolist()
    _, _, task2_test, _, _, _ = stratified_split(all_task2, labels1_int, seed=seed)
    _, _, task3_test, _, _, _ = stratified_split(all_task3, labels1_int, seed=seed)

    if max_samples and max_samples < len(X_test):
        X_test = X_test[:max_samples]
        y1_test = y1_test[:max_samples]
        task2_test = task2_test[:max_samples]
        task3_test = task3_test[:max_samples]

    print(f"Test samples: {len(X_test)}")
    print(f"Parallel workers: {max_workers}")

    # Run combined judge
    preds = run_combined_judge(
        texts=X_test, model=model, api_backend=api_backend,
        api_base=api_base, api_key=api_key, max_workers=max_workers,
    )

    # Evaluate each task
    all_metrics = {}
    for task_num, labels_list, label2id, gt_labels in [
        (1, labels1, label2id1, [id2label1[y] for y in y1_test]),
        (2, labels2, label2id2, task2_test),
        (3, labels3, label2id3, task3_test),
    ]:
        task_key = f"task{task_num}"
        y_true, y_pred = [], []

        for pred, gt in zip(preds, gt_labels):
            if pred is None:
                continue
            pred_label = pred.get(task_key)
            if pred_label is None or pred_label == "No" or gt == "No" or gt not in label2id:
                continue
            if pred_label not in label2id:
                continue
            y_true.append(label2id[gt])
            y_pred.append(label2id[pred_label])

        if y_pred:
            metrics = compute_metrics(np.array(y_true), np.array(y_pred), labels_list)
            print(f"\nTask {task_num}: Macro F1 = {metrics['macro_f1']:.4f} ({len(y_pred)} samples)")
            print(f"  Per-class: {metrics['per_class_f1']}")
            all_metrics[task_num] = metrics

            model_short = model.replace("/", "_").replace(":", "_")
            log_result(
                approach="llm_judge",
                model=model_short,
                task=task_num,
                macro_f1=metrics["macro_f1"],
                split="test",
                embedding="—",
                dataset=f"train-{language}.csv",
                accuracy=metrics.get("accuracy"),
                precision_macro=metrics.get("precision_macro"),
                recall_macro=metrics.get("recall_macro"),
                per_class_f1=metrics.get("per_class_f1"),
                hyperparams={"mode": "combined_few", "model": model},
                notes=f"llm_judge_combined_{model_short}_task{task_num}_{language}",
            )
        else:
            print(f"\nTask {task_num}: No valid predictions")
            all_metrics[task_num] = {"macro_f1": 0.0}

    # Save artifacts
    artifact_dir = os.path.join(_ARTIFACTS_DIR, f"llm_judge_combined_{model.replace('/', '_')}_{language}")
    os.makedirs(artifact_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        "text": X_test,
        "pred_task1": [p["task1"] if p else None for p in preds],
        "pred_task2": [p["task2"] if p else None for p in preds],
        "pred_task3": [p["task3"] if p else None for p in preds],
        "true_task1": [id2label1[y] for y in y1_test],
        "true_task2": task2_test,
        "true_task3": task3_test,
    })
    pred_df.to_csv(os.path.join(artifact_dir, "predictions.csv"), index=False)

    with open(os.path.join(artifact_dir, "metrics.json"), "w") as f:
        json.dump({f"task{k}": v for k, v in all_metrics.items()}, f, indent=2, default=str)

    print(f"\nArtifacts saved to {artifact_dir}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"{'COMBINED JUDGE SUMMARY':^60}")
    print(f"{'=' * 60}")
    parse_ok = sum(1 for p in preds if p is not None)
    print(f"  Parse success: {parse_ok}/{len(preds)} ({parse_ok/len(preds)*100:.1f}%)")
    for t in [1, 2, 3]:
        f1 = all_metrics.get(t, {}).get("macro_f1", 0)
        print(f"  Task {t}: Macro F1 = {f1:.4f}")
    print(f"{'=' * 60}")

    return all_metrics


if __name__ == "__main__":
    run_judge_grid()
