"""
Generate competition submission CSVs using LLM Judge (combined 3-task prompt).

Uses the COMBINED_FEW_SHOT prompt with gpt-5.4-mini for all 3 tasks at once.
For Spanish, texts are sent as-is (gpt-5.4-mini handles Spanish natively).

Built-in retry: any parse failures are retried up to 3 times individually.

Usage:
    python -m src.predict_submission_llm --lang es
    python -m src.predict_submission_llm --lang en --max_workers 5
"""

import argparse
import os
import zipfile

import pandas as pd

from src.llm_judge.judge import run_combined_judge
from src.preprocessing import preprocess_transformer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_TEST_DIR = os.path.join(_PROJECT_ROOT, "data", "Test_phase_data")

# LLM string labels → submission numeric IDs
TASK1_MAP = {"Non-Supportive": 0, "Supportive": 1}
TASK2_MAP = {"Group": 0, "Individual": 1}
TASK3_MAP = {
    "Black Community": 0, "LGBTQ": 1, "Nation": 2,
    "Other": 3, "Religion": 4, "Women": 5,
}

DEFAULT_T1 = 0  # Non-Supportive
DEFAULT_T2 = 0  # Group
DEFAULT_T3 = 3  # Other


def _convert_one(pred: dict | None) -> tuple[int, int, int]:
    """Convert a single LLM prediction dict to (t1, t2, t3) submission IDs."""
    if pred is None:
        return DEFAULT_T1, DEFAULT_T2, DEFAULT_T3

    t1 = TASK1_MAP.get(pred.get("task1", "Non-Supportive"), DEFAULT_T1)

    t2_label = pred.get("task2", "No")
    t2 = TASK2_MAP.get(t2_label, DEFAULT_T2)

    t3_label = pred.get("task3", "No")
    t3 = TASK3_MAP.get(t3_label, DEFAULT_T3)

    return t1, t2, t3


def predict_with_retry(
    texts: list[str],
    model: str,
    api_backend: str,
    api_base: str | None,
    api_key: str | None,
    max_workers: int,
    max_retries: int,
    retry_rounds: int = 3,
) -> list[dict | None]:
    """Run combined judge, then retry failed indices individually."""

    # Main pass
    preds = run_combined_judge(
        texts=texts, model=model, api_backend=api_backend,
        api_base=api_base, api_key=api_key,
        max_workers=max_workers, max_retries=max_retries,
    )

    # Retry loop for failures
    for attempt in range(1, retry_rounds + 1):
        failed_idxs = [i for i, p in enumerate(preds) if p is None]
        if not failed_idxs:
            break
        print(f"\n  Retry round {attempt}: {len(failed_idxs)} failures")
        failed_texts = [texts[i] for i in failed_idxs]
        retry_preds = run_combined_judge(
            texts=failed_texts, model=model, api_backend=api_backend,
            api_base=api_base, api_key=api_key,
            max_workers=1, max_retries=max_retries + 2,
        )
        for idx, retry_pred in zip(failed_idxs, retry_preds):
            if retry_pred is not None:
                preds[idx] = retry_pred

    final_fails = sum(1 for p in preds if p is None)
    print(f"  Final: {len(preds) - final_fails}/{len(preds)} parsed ({final_fails} failures)")
    return preds


def predict_language(
    csv_path: str,
    text_col: str,
    lang_prefix: str,
    model: str,
    api_backend: str,
    api_base: str | None,
    api_key: str | None,
    max_workers: int,
    max_retries: int,
) -> pd.DataFrame:
    """Run LLM judge on a test CSV and return submission DataFrame."""
    df = pd.read_csv(csv_path)
    texts = df[text_col].apply(preprocess_transformer).tolist()
    ids = df["id"].tolist()
    print(f"Loaded {len(texts)} {lang_prefix.upper()} test samples")

    preds = predict_with_retry(
        texts=texts, model=model, api_backend=api_backend,
        api_base=api_base, api_key=api_key,
        max_workers=max_workers, max_retries=max_retries,
    )

    t1_ids, t2_ids, t3_ids = [], [], []
    for pred in preds:
        t1, t2, t3 = _convert_one(pred)
        t1_ids.append(t1)
        t2_ids.append(t2)
        t3_ids.append(t3)

    result = pd.DataFrame({
        "id": ids,
        f"{lang_prefix}_support_pred": t1_ids,
        f"{lang_prefix}_individual_pred": t2_ids,
        f"{lang_prefix}_multiclass_pred": t3_ids,
    })

    print(f"\n  Prediction distribution:")
    print(f"    Task 1: Non-Supportive={t1_ids.count(0)}, Supportive={t1_ids.count(1)}")
    print(f"    Task 2: Group={t2_ids.count(0)}, Individual={t2_ids.count(1)}")
    t3_names = {0: "BlackComm", 1: "LGBTQ", 2: "Nation", 3: "Other", 4: "Religion", 5: "Women"}
    for k, v in sorted(t3_names.items()):
        print(f"    Task 3 {v}: {t3_ids.count(k)}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api_backend", default="openai")
    parser.add_argument("--api_base", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--lang", choices=["en", "es", "both"], default="both")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_ROOT, "submissions")
    os.makedirs(out_dir, exist_ok=True)

    if args.lang in ("en", "both"):
        print("\n" + "=" * 60)
        print("English — LLM Judge")
        print("=" * 60)
        en_df = predict_language(
            csv_path=os.path.join(_TEST_DIR, "test_phase_english.csv"),
            text_col="text", lang_prefix="en",
            model=args.model, api_backend=args.api_backend,
            api_base=args.api_base, api_key=args.api_key,
            max_workers=args.max_workers, max_retries=args.max_retries,
        )
        en_path = os.path.join(out_dir, "english_submission_llm.csv")
        en_df.to_csv(en_path, index=False)
        zip_en = os.path.join(out_dir, "submission_english_llm.zip")
        with zipfile.ZipFile(zip_en, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(en_path, "english_submission_llm.csv")
        print(f"\nSaved: {zip_en}")

    if args.lang in ("es", "both"):
        print("\n" + "=" * 60)
        print("Spanish — LLM Judge")
        print("=" * 60)
        es_df = predict_language(
            csv_path=os.path.join(_TEST_DIR, "test_phase_spanish.csv"),
            text_col="comment", lang_prefix="es",
            model=args.model, api_backend=args.api_backend,
            api_base=args.api_base, api_key=args.api_key,
            max_workers=args.max_workers, max_retries=args.max_retries,
        )
        es_path = os.path.join(out_dir, "spanish_submission_llm.csv")
        es_df.to_csv(es_path, index=False)
        zip_es = os.path.join(out_dir, "submission_spanish_llm.zip")
        with zipfile.ZipFile(zip_es, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(es_path, "spanish_submission_llm.csv")
        print(f"\nSaved: {zip_es}")


if __name__ == "__main__":
    main()
