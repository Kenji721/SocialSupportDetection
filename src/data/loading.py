"""
Data loading and normalization for SSD-2026.
"""

import pandas as pd


# Spanish → English label normalization for task1
_ES_TASK1_MAP = {
    "Support": "Supportive",
    "Non Support": "Non-Supportive",
}


def load_ssd_data(csv_path: str, language: str = "en") -> pd.DataFrame:
    """Load and normalize an SSD-2026 CSV into a standard DataFrame.

    Returns DataFrame with columns: [id, text, task1, task2, task3, language]
    """
    df = pd.read_csv(csv_path)

    if language == "es":
        df = df.rename(columns={"comment": "text"})
        df["task1"] = df["task1"].map(_ES_TASK1_MAP)

    df["language"] = language

    expected_cols = ["id", "text", "task1", "task2", "task3", "language"]
    return df[expected_cols].copy()


def load_multilingual(en_path: str, es_path: str) -> pd.DataFrame:
    """Load and concatenate English + Spanish data."""
    en = load_ssd_data(en_path, language="en")
    es = load_ssd_data(es_path, language="es")
    return pd.concat([en, es], ignore_index=True)
