"""
Experiment results logger.

Logs experiment metadata as JSON Lines to results/experiments.jsonl.
"""

import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
JSONL_FILE = os.path.join(RESULTS_DIR, "experiments.jsonl")


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def log_result(
    approach: str,
    model: str,
    task: int,
    macro_f1: float,
    split: str = "test",
    embedding: str = "—",
    dataset: str = "train-english.csv",
    accuracy: float = None,
    precision_macro: float = None,
    recall_macro: float = None,
    per_class_f1: dict = None,
    epochs: int = None,
    early_stop_epoch: int = None,
    hyperparams: dict = None,
    notes: str = "",
):
    """Log a single experiment result to JSONL."""
    _ensure_dir()

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "approach": approach,
        "model": model,
        "embedding": embedding,
        "task": task,
        "dataset": dataset,
        "split": split,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "precision_macro": round(precision_macro, 4) if precision_macro is not None else None,
        "recall_macro": round(recall_macro, 4) if recall_macro is not None else None,
        "per_class_f1": per_class_f1,
        "epochs": epochs,
        "early_stop_epoch": early_stop_epoch,
        "hyperparams": hyperparams,
        "notes": notes,
    }

    # Write to unified log
    with open(JSONL_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Also write to per-approach log file
    approach_file = os.path.join(RESULTS_DIR, f"experiments_{approach}.jsonl")
    with open(approach_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"[Results] Logged to {JSONL_FILE} + {approach_file}")


def load_results() -> list[dict]:
    """Load all results from JSONL."""
    results = []

    if os.path.exists(JSONL_FILE):
        with open(JSONL_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

    return results


def print_summary():
    """Print a formatted summary of all experiment results."""
    results = load_results()
    if not results:
        print("No results logged yet.")
        return

    print(f"\n{'=' * 90}")
    print(f"{'EXPERIMENT RESULTS':^90}")
    print(f"{'=' * 90}")
    print(f"{'Approach':<14} {'Model':<22} {'Embedding':<10} {'Task':<6} {'Split':<6} {'Macro F1':<10} {'Date'}")
    print(f"{'-' * 90}")
    for r in results:
        ts = r.get("timestamp", "")
        date_str = ts[:10] if ts else ""
        print(
            f"{r.get('approach', ''):<14} {r.get('model', ''):<22} {r.get('embedding', '—'):<10} "
            f"{str(r.get('task', '')):<6} {r.get('split', ''):<6} {str(r.get('macro_f1', '')):<10} {date_str}"
        )
    print(f"{'=' * 90}")
