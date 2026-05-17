"""
Hybrid submission: BERTweet pre-labels + LLM Judge correction.

Loads BERTweet predictions as pre-labels, then asks gpt-5.4-mini to
confirm or correct each classification using the COMBINED_FEW_SHOT prompt
augmented with the pre-label.

Usage:
    python -m src.predict_submission_hybrid --lang en --max_workers 5
"""

import argparse
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from src.llm_judge.prompts import COMBINED_FEW_SHOT
from src.preprocessing import preprocess_transformer
from src.predict_submission_llm import TASK1_MAP, TASK2_MAP, TASK3_MAP, DEFAULT_T1, DEFAULT_T2, DEFAULT_T3

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_TEST_DIR = os.path.join(_PROJECT_ROOT, "data", "Test_phase_data")

# Reverse maps: submission ID → label string
SUB_TO_TASK1 = {0: "Non-Supportive", 1: "Supportive"}
SUB_TO_TASK2 = {0: "Group", 1: "Individual"}
SUB_TO_TASK3 = {0: "Black Community", 1: "LGBTQ", 2: "Nation", 3: "Other", 4: "Religion", 5: "Women"}


def _build_hybrid_prompt(text: str, pre_t1: int, pre_t2: int, pre_t3: int) -> str:
    """Build prompt with BERTweet pre-labels appended."""
    base = COMBINED_FEW_SHOT.format(text=text)

    # Convert submission IDs to label strings
    t1_label = SUB_TO_TASK1[pre_t1]
    t2_label = SUB_TO_TASK2.get(pre_t2, "No") if pre_t1 == 1 else "No"
    t3_label = SUB_TO_TASK3.get(pre_t3, "No") if pre_t2 == 0 and pre_t1 == 1 else "No"

    prelabel_hint = (
        f'\n\nA BERTweet classifier predicted: {{"task1": "{t1_label}", "task2": "{t2_label}", "task3": "{t3_label}"}}'
        f"\nIf you agree with BERTweet's prediction, use the same labels. If you disagree, label it yourself based on the guidelines above."
        f"\nRespond ONLY with a JSON object."
    )
    return base + prelabel_hint


def _call_openai(prompt: str, model: str, api_base: str, api_key: str, max_retries: int = 5) -> str | None:
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
                headers=headers, json=payload, timeout=90,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                print(f"  API error after {max_retries} retries: {e}")
                return None


def _call_single(idx, text, pre_t1, pre_t2, pre_t3, model, api_base, api_key, max_retries):
    """Process one sample: build hybrid prompt, call API, parse JSON."""
    prompt = _build_hybrid_prompt(text, pre_t1, pre_t2, pre_t3)
    raw = _call_openai(prompt, model, api_base, api_key, max_retries)

    if raw is None:
        return idx, None

    try:
        raw_clean = raw.strip()
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


def run_hybrid(
    texts: list[str],
    pre_t1s: list[int],
    pre_t2s: list[int],
    pre_t3s: list[int],
    model: str,
    api_base: str | None,
    api_key: str | None,
    max_workers: int,
    max_retries: int,
) -> list[dict | None]:
    """Run hybrid judge on all texts."""
    api_base = api_base or "https://api.openai.com/v1"
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    predictions = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _call_single, i, text, pre_t1s[i], pre_t2s[i], pre_t3s[i],
                model, api_base, api_key, max_retries,
            ): i
            for i, text in enumerate(texts)
        }

        with tqdm(total=len(texts), desc="Hybrid LLM Judge") as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                predictions[idx] = result
                pbar.update(1)

    return predictions


def _convert_one(pred: dict | None, fallback_t1: int, fallback_t2: int, fallback_t3: int) -> tuple[int, int, int]:
    """Convert LLM prediction to submission IDs, falling back to BERTweet pre-labels."""
    if pred is None:
        return fallback_t1, fallback_t2, fallback_t3

    t1 = TASK1_MAP.get(pred.get("task1"), fallback_t1)
    t2 = TASK2_MAP.get(pred.get("task2", "No"), fallback_t2)
    t3 = TASK3_MAP.get(pred.get("task3", "No"), fallback_t3)
    return t1, t2, t3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prelabel_csv", required=True, help="Path to BERTweet submission CSV")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--lang_prefix", default="en")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--output_suffix", default="hybrid")
    args = parser.parse_args()

    lp = args.lang_prefix

    # Load test data
    test_df = pd.read_csv(args.test_csv)
    texts = test_df[args.text_col].apply(preprocess_transformer).tolist()
    ids = test_df["id"].tolist()
    print(f"Loaded {len(texts)} {lp.upper()} test samples")

    # Load BERTweet pre-labels
    pre_df = pd.read_csv(args.prelabel_csv)
    pre_t1s = pre_df[f"{lp}_support_pred"].tolist()
    pre_t2s = pre_df[f"{lp}_individual_pred"].tolist()
    pre_t3s = pre_df[f"{lp}_multiclass_pred"].tolist()
    print(f"Loaded {len(pre_t1s)} pre-labels from {args.prelabel_csv}")

    assert len(texts) == len(pre_t1s), "Test data and pre-labels must have same length"

    # Run hybrid LLM judge
    preds = run_hybrid(
        texts=texts,
        pre_t1s=pre_t1s, pre_t2s=pre_t2s, pre_t3s=pre_t3s,
        model=args.model, api_base=args.api_base, api_key=args.api_key,
        max_workers=args.max_workers, max_retries=args.max_retries,
    )

    # Retry failures
    for attempt in range(1, 4):
        failed = [i for i, p in enumerate(preds) if p is None]
        if not failed:
            break
        print(f"\n  Retry round {attempt}: {len(failed)} failures")
        retry_preds = run_hybrid(
            texts=[texts[i] for i in failed],
            pre_t1s=[pre_t1s[i] for i in failed],
            pre_t2s=[pre_t2s[i] for i in failed],
            pre_t3s=[pre_t3s[i] for i in failed],
            model=args.model, api_base=args.api_base, api_key=args.api_key,
            max_workers=1, max_retries=args.max_retries + 2,
        )
        for idx, retry_pred in zip(failed, retry_preds):
            if retry_pred is not None:
                preds[idx] = retry_pred

    # Convert to submission IDs (fall back to BERTweet if LLM failed)
    t1_ids, t2_ids, t3_ids = [], [], []
    for i, pred in enumerate(preds):
        t1, t2, t3 = _convert_one(pred, pre_t1s[i], pre_t2s[i], pre_t3s[i])
        t1_ids.append(t1)
        t2_ids.append(t2)
        t3_ids.append(t3)

    # Count how many the LLM changed vs BERTweet
    changed_t1 = sum(1 for i in range(len(preds)) if t1_ids[i] != pre_t1s[i])
    changed_t2 = sum(1 for i in range(len(preds)) if t2_ids[i] != pre_t2s[i])
    changed_t3 = sum(1 for i in range(len(preds)) if t3_ids[i] != pre_t3s[i])
    n_failed = sum(1 for p in preds if p is None)
    print(f"\n  Parse success: {len(preds) - n_failed}/{len(preds)}")
    print(f"  LLM changed vs BERTweet: T1={changed_t1}, T2={changed_t2}, T3={changed_t3}")

    result = pd.DataFrame({
        "id": ids,
        f"{lp}_support_pred": t1_ids,
        f"{lp}_individual_pred": t2_ids,
        f"{lp}_multiclass_pred": t3_ids,
    })

    # Distribution
    print(f"\n  Prediction distribution:")
    print(f"    Task 1: Non-Supportive={t1_ids.count(0)}, Supportive={t1_ids.count(1)}")
    print(f"    Task 2: Group={t2_ids.count(0)}, Individual={t2_ids.count(1)}")
    t3_names = {0: "BlackComm", 1: "LGBTQ", 2: "Nation", 3: "Other", 4: "Religion", 5: "Women"}
    for k, v in sorted(t3_names.items()):
        print(f"    Task 3 {v}: {t3_ids.count(k)}")

    # Save
    out_dir = os.path.join(_PROJECT_ROOT, "submissions")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{lp}_submission_{args.output_suffix}.csv")
    result.to_csv(csv_path, index=False)
    zip_path = os.path.join(out_dir, f"submission_{lp}_{args.output_suffix}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, os.path.basename(csv_path))
    print(f"\n  Saved: {zip_path}")


if __name__ == "__main__":
    main()
