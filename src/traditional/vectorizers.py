"""
TF-IDF vectorization pipelines for traditional ML.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf_pipeline(mode: str = "combined", max_features: int = 50000):
    """Build a TF-IDF vectorizer pipeline.

    Args:
        mode: "word", "char", or "combined" (word + char FeatureUnion).
        max_features: Max vocabulary size per vectorizer.

    Returns:
        sklearn transformer (TfidfVectorizer or FeatureUnion).
    """
    if mode == "word":
        return TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
    elif mode == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
            sublinear_tf=True,
        )
    elif mode == "combined":
        return FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=max_features,
                sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=max_features,
                sublinear_tf=True,
            )),
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'word', 'char', or 'combined'.")
