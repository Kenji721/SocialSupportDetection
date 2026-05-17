"""
Evaluate fine-tuned Qwen3.5 model on SSD test set.

Usage:
    python -m src.finetuning.evaluate_llm \
        --adapter_path adapters/qwen3.5-9b-ssd \
        --test_file data/llm_finetune/combined/test.jsonl \
        --max_samples 200
"""

import argparse
import json
import re
from pathlib import Path

from sklearn.metrics import classification_report, f1_score


SYSTEM_PROMPT = """\
Classify the comment on 3 tasks. Respond with JSON only.
Task1: "Supportive" or "Non-Supportive" (support/encouragement/admiration toward a person, group, or cause with a clear target).
Task2: "Individual", "Group", or "No" (if Non-Supportive).
Task3: "Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women", or "No" (if not Group).\
"""


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response, handling potential noise."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON in the text
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def evaluate(
    adapter_path: str,
    test_file: str,
    max_samples: int = 0,
    enable_thinking: bool = False,
):
    from mlx_lm import load, generate

    print(f"Loading model with adapter: {adapter_path}")
    model, tokenizer = load(
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        adapter_path=adapter_path,
    )

    # Load test data
    records = []
    with open(test_file) as f:
        for line in f:
            records.append(json.loads(line))

    if max_samples > 0:
        records = records[:max_samples]

    print(f"Evaluating on {len(records)} samples...")

    true_t1, pred_t1 = [], []
    true_t2, pred_t2 = [], []
    true_t3, pred_t3 = [], []
    parse_failures = 0

    for i, record in enumerate(records):
        # Get true labels
        true_labels = json.loads(record["messages"][2]["content"])

        # Build prompt
        user_msg = record["messages"][1]["content"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100)

        pred = parse_json_response(response)
        if pred is None:
            parse_failures += 1
            if i < 5:
                print(f"  [PARSE FAIL] {response[:100]}")
            continue

        # Collect predictions
        true_t1.append(true_labels["task1"])
        pred_t1.append(pred.get("task1", "Non-Supportive"))

        if true_labels["task1"] == "Supportive":
            true_t2.append(true_labels["task2"])
            pred_t2.append(pred.get("task2", "No"))

            if true_labels["task2"] == "Group":
                true_t3.append(true_labels["task3"])
                pred_t3.append(pred.get("task3", "No"))

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    # Results
    print(f"\n{'='*60}")
    print(f"Results ({len(true_t1)} valid / {len(records)} total, "
          f"{parse_failures} parse failures)")
    print(f"{'='*60}")

    print(f"\nTask 1 (Supportive vs Non-Supportive):")
    print(f"  Macro F1: {f1_score(true_t1, pred_t1, average='macro'):.3f}")
    print(classification_report(true_t1, pred_t1))

    if true_t2:
        print(f"\nTask 2 (Individual vs Group) [Supportive only]:")
        print(f"  Macro F1: {f1_score(true_t2, pred_t2, average='macro'):.3f}")
        print(classification_report(true_t2, pred_t2))

    if true_t3:
        print(f"\nTask 3 (Community) [Group only]:")
        print(f"  Macro F1: {f1_score(true_t3, pred_t3, average='macro'):.3f}")
        print(classification_report(true_t3, pred_t3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="adapters/qwen3.5-9b-ssd")
    parser.add_argument(
        "--test_file", default="data/llm_finetune/combined/test.jsonl"
    )
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()
    evaluate(args.adapter_path, args.test_file, args.max_samples, args.enable_thinking)
