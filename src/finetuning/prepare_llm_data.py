"""
Prepare SSD training data in chat-format JSONL for mlx-lm LoRA fine-tuning.

Converts the CSV training data into the chat template format expected by mlx-lm:
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Key design decisions:
  - Uses the same COMBINED_FEW_SHOT prompt from LLM Judge so training and inference
    see the same instructions/examples.
  - Oversamples the minority (Supportive) class to reduce majority-class bias.
  - No system message — the full prompt goes in the user message (matching
    inference-time usage).

Usage:
    python -m src.finetuning.prepare_llm_data --langs en es --out_dir data/llm_finetune/combined
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

from src.llm_judge.prompts import COMBINED_FEW_SHOT

# Split COMBINED_FEW_SHOT into instructions+examples vs the {text} placeholder
_PROMPT_PREFIX = COMBINED_FEW_SHOT.rsplit("Comment: {text}", 1)[0].strip()

# Label mappings from CSV values to JSON output
TASK1_MAP = {
    "Supportive": "Supportive",
    "Non-Supportive": "Non-Supportive",
    "Support": "Supportive",
    "Non Support": "Non-Supportive",
}

TASK2_MAP = {
    "Individual": "Individual",
    "Group": "Group",
    "No": "No",
}

TASK3_MAP = {
    "Nation": "Nation",
    "Other": "Other",
    "LGBTQ": "LGBTQ",
    "Black Community": "Black Community",
    "Religion": "Religion",
    "Women": "Women",
    "No": "No",
}


def row_to_chat(row: pd.Series, text_col: str) -> dict:
    """Convert a single row to chat-format dict."""
    text = str(row[text_col]).strip()
    t1 = TASK1_MAP.get(str(row["task1"]).strip(), "Non-Supportive")
    t2 = TASK2_MAP.get(str(row["task2"]).strip(), "No")
    t3 = TASK3_MAP.get(str(row["task3"]).strip(), "No")

    answer = json.dumps({"task1": t1, "task2": t2, "task3": t3})

    # Full prompt with instructions + examples + this comment
    user_content = _PROMPT_PREFIX + f"\nComment: {text}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "_task1": t1,  # metadata for balancing (stripped before writing)
    }


def load_lang(lang: str) -> list[dict]:
    """Load and convert a single language CSV to chat records."""
    base = Path(__file__).resolve().parents[2] / "data" / "Train_Data_SSD26"

    if lang == "en":
        csv_path = base / "train-english.csv"
        text_col = "text"
    elif lang == "es":
        csv_path = base / "train-spanish.csv"
        text_col = "comment"
    else:
        raise ValueError(f"Unknown lang: {lang}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")
    return [row_to_chat(row, text_col) for _, row in df.iterrows()]


def balance_oversample(records: list[dict], seed: int = 42) -> list[dict]:
    """Oversample minority class (Supportive) to match majority class count."""
    rng = random.Random(seed)

    supportive = [r for r in records if r["_task1"] == "Supportive"]
    non_supportive = [r for r in records if r["_task1"] == "Non-Supportive"]

    print(f"Before balancing: Supportive={len(supportive)}, "
          f"Non-Supportive={len(non_supportive)}")

    # Oversample supportive to match non-supportive count
    if len(supportive) < len(non_supportive):
        extra_needed = len(non_supportive) - len(supportive)
        oversampled = rng.choices(supportive, k=extra_needed)
        supportive = supportive + oversampled

    balanced = supportive + non_supportive
    rng.shuffle(balanced)

    print(f"After balancing: Supportive={len(supportive)}, "
          f"Non-Supportive={len(non_supportive)}, Total={len(balanced)}")

    return balanced


def prepare_data(
    langs: list[str],
    out_dir: str,
    test_ratio: float = 0.1,
    seed: int = 42,
    balance: bool = True,
):
    """Load CSV(s), convert to chat JSONL, split into train/valid/test."""
    all_records = []
    for lang in langs:
        all_records.extend(load_lang(lang))

    print(f"\nTotal combined records: {len(all_records)}")

    # Shuffle and split BEFORE balancing (test/val should reflect real distribution)
    random.seed(seed)
    random.shuffle(all_records)

    n = len(all_records)
    n_test = int(n * test_ratio)
    n_val = int(n * test_ratio)

    test_set = all_records[:n_test]
    val_set = all_records[n_test : n_test + n_val]
    train_set = all_records[n_test + n_val :]

    # Balance training set only
    if balance:
        train_set = balance_oversample(train_set, seed=seed)

    print(f"Split: train={len(train_set)}, valid={len(val_set)}, test={len(test_set)}")

    # Write JSONL files (strip _task1 metadata)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train_set), ("valid", val_set), ("test", test_set)]:
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for record in data:
                clean = {"messages": record["messages"]}
                f.write(json.dumps(clean) + "\n")
        print(f"Wrote {path} ({len(data)} samples)")

    # Print class distribution
    t1_dist = Counter(r["_task1"] for r in train_set)
    print(f"\nTrain Task 1 distribution: {dict(t1_dist)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langs", nargs="+", default=["en", "es"], choices=["en", "es"],
    )
    parser.add_argument("--out_dir", default="data/llm_finetune/combined")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_balance", action="store_true")
    args = parser.parse_args()
    prepare_data(args.langs, args.out_dir, args.test_ratio, args.seed,
                 balance=not args.no_balance)
