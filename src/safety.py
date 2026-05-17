"""
Safety module: harm detection and override logic for SSD-2026.
"""

import re

HARM_LEXICON_EN = [
    "kill", "attack", "destroy", "deport", "hate", "die", "shoot", "burn",
    "murder", "assault", "execute", "eliminate", "exterminate", "slaughter",
    "lynch", "bomb", "stab", "genocide", "terrorize", "threaten",
]

HARM_LEXICON_ES = [
    "matar", "odio", "destruir", "atacar", "deportar", "muerte",
    "asesinar", "quemar", "bombardear", "eliminar", "exterminar",
    "amenazar", "linchar", "apuñalar",
]

_WORD_BOUNDARY = re.compile(r"\b\w+\b")


def harm_score(text: str, language: str = "en") -> float:
    """Compute a harm score in [0, 1] based on lexicon hits.

    Score = (number of harm word hits) / (total words), capped at 1.0.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    lexicon = set(HARM_LEXICON_ES if language == "es" else HARM_LEXICON_EN)
    words = [w.lower() for w in _WORD_BOUNDARY.findall(text)]

    if not words:
        return 0.0

    hits = sum(1 for w in words if w in lexicon)
    return min(hits / len(words), 1.0)


def has_harm_signal(text: str, language: str = "en") -> bool:
    """Return True if any harm lexicon word is found in the text."""
    return harm_score(text, language) > 0.0


def apply_harm_override(predictions_df, texts, language: str = "en"):
    """Override Supportive predictions to Non-Supportive when harm is detected.

    If task1 is flipped to Non-Supportive, task2 and task3 are also set to "No"
    (consistent with the hierarchical pipeline logic).

    Args:
        predictions_df: DataFrame with at least a 'task1' column.
            May also have 'task2' and 'task3' columns.
        texts: Iterable of text strings aligned with predictions_df rows.
        language: "en" or "es".

    Returns:
        Modified copy of predictions_df with count of overrides applied.
    """
    df = predictions_df.copy()
    overrides = 0
    for i, text in enumerate(texts):
        if df.iloc[i]["task1"] == "Supportive" and has_harm_signal(text, language):
            df.iloc[i, df.columns.get_loc("task1")] = "Non-Supportive"
            if "task2" in df.columns:
                df.iloc[i, df.columns.get_loc("task2")] = "No"
            if "task3" in df.columns:
                df.iloc[i, df.columns.get_loc("task3")] = "No"
            overrides += 1
    return df, overrides
