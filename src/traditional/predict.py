"""
Traditional ML inference.
"""

import joblib


def predict_traditional(model_path: str, texts: list[str]) -> list[int]:
    """Load a saved sklearn pipeline and predict."""
    pipeline = joblib.load(model_path)
    return pipeline.predict(texts).tolist()
