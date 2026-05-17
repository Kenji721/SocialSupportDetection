"""
Evaluate fine-tuned Qwen3.5 with the full LLM Judge COMBINED_FEW_SHOT prompt.

Usage:
    python -m src.finetuning.evaluate_llm_judge_prompt \
        --adapter_path adapters/qwen3.5-9b-ssd \
        --max_samples 200
"""

import argparse
import json
import re

from sklearn.metrics import classification_report, f1_score


def parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def evaluate(adapter_path: str, test_file: str, max_samples: int = 200):
    from mlx_lm import load, generate
    from src.llm_judge.prompts import COMBINED_FEW_SHOT

    # Split the combined prompt into system part + user template
    system_and_examples = COMBINED_FEW_SHOT.rsplit("Comment: {text}", 1)[0].strip()

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
        true_labels = json.loads(record["messages"][2]["content"])

        # Extract comment text from user message
        user_content = record["messages"][1]["content"]  # "Comment: ..."
        comment_text = user_content.replace("Comment: ", "", 1)

        # Build prompt with the full LLM Judge few-shot template
        full_prompt = system_and_examples + f"\nComment: {comment_text}"

        messages = [
            {"role": "user", "content": full_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100)

        pred = parse_json_response(response)
        if pred is None:
            parse_failures += 1
            if parse_failures <= 5:
                print(f"  [PARSE FAIL #{parse_failures}] {response[:120]}")
            continue

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

    print(f"\n{'='*60}")
    print(f"Results ({len(true_t1)} valid / {len(records)} total, "
          f"{parse_failures} parse failures)")
    print(f"{'='*60}")

    print(f"\nTask 1 (Supportive vs Non-Supportive):")
    f1_t1 = f1_score(true_t1, pred_t1, average="macro")
    print(f"  Macro F1: {f1_t1:.3f}")
    print(classification_report(true_t1, pred_t1, zero_division=0))

    if true_t2:
        # Filter to only valid task2 labels
        valid_t2 = [(t, p) for t, p in zip(true_t2, pred_t2)
                     if t in ("Individual", "Group")]
        if valid_t2:
            t2_true, t2_pred = zip(*valid_t2)
            f1_t2 = f1_score(t2_true, t2_pred, average="macro",
                             labels=["Individual", "Group"], zero_division=0)
            print(f"\nTask 2 (Individual vs Group) [Supportive only]:")
            print(f"  Macro F1: {f1_t2:.3f}")
            print(classification_report(t2_true, t2_pred, zero_division=0))

    if true_t3:
        labels_t3 = ["Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"]
        f1_t3 = f1_score(true_t3, pred_t3, average="macro",
                         labels=labels_t3, zero_division=0)
        print(f"\nTask 3 (Community) [Group only]:")
        print(f"  Macro F1: {f1_t3:.3f}")
        print(classification_report(true_t3, pred_t3, zero_division=0,
                                    labels=labels_t3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", default="adapters/qwen3.5-9b-ssd")
    parser.add_argument("--test_file",
                        default="data/llm_finetune/combined/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.adapter_path, args.test_file, args.max_samples)
