"""
Handcrafted feature extraction for SSD-2026.

Six feature groups (~21 features total):
  - Pronoun counts
  - Group lexicon hits
  - Support lexicon hits
  - Harm/violence lexicon hits
  - Punctuation/emotion cues
  - Surface features
"""

import re

import pandas as pd

_WORD_RE = re.compile(r"\b\w+\b")

# ── Lexicons ─────────────────────────────────────────────────────────────────

_PRONOUN_EN = {"you": "pronoun_you", "we": "pronoun_we", "they": "pronoun_they"}
_PRONOUN_ES = {"ustedes": "pronoun_you", "nosotros": "pronoun_we", "nosotras": "pronoun_we",
               "ellos": "pronoun_they", "ellas": "pronoun_they"}

_GROUP_LEXICON = {
    "women": "lex_women", "woman": "lex_women", "mujer": "lex_women", "mujeres": "lex_women",
    "lgbtq": "lex_lgbtq", "lgbt": "lex_lgbtq", "gay": "lex_lgbtq", "queer": "lex_lgbtq",
    "trans": "lex_lgbtq", "pride": "lex_lgbtq", "orgullo": "lex_lgbtq",
    "religion": "lex_religion", "god": "lex_religion", "church": "lex_religion",
    "muslim": "lex_religion", "christian": "lex_religion", "bible": "lex_religion",
    "dios": "lex_religion", "iglesia": "lex_religion",
    "black": "lex_black", "blm": "lex_black", "negro": "lex_black", "negra": "lex_black",
    "nation": "lex_nation", "country": "lex_nation", "patriot": "lex_nation",
    "america": "lex_nation", "patria": "lex_nation", "nacion": "lex_nation",
}

_SUPPORT_LEXICON = {
    "strength", "love", "bless", "prayers", "pray", "support",
    "proud", "solidarity", "hope", "courage", "hero",
    "animo", "fuerza", "orgullo", "apoyo", "bendicion", "esperanza",
}

_HARM_LEXICON = {
    "kill", "attack", "destroy", "deport", "hate", "die", "shoot", "burn",
    "matar", "odio", "destruir", "atacar", "deportar", "muerte",
}

# Heart, prayer, flag emojis (common Unicode codepoints)
_EMOTION_EMOJIS = {"❤", "♥", "💕", "💖", "💗", "💓", "💛", "💙", "💚", "🧡", "🤍", "🖤",
                   "🙏", "😢", "😭", "😍", "🥰", "✊", "💪"}
_FLAG_PATTERN = re.compile(r"[\U0001F1E0-\U0001F1FF]{2}")

# ── Feature name constants ───────────────────────────────────────────────────

PRONOUN_FEATURES = ["pronoun_you", "pronoun_we", "pronoun_they"]
GROUP_LEXICON_FEATURES = ["lex_women", "lex_lgbtq", "lex_religion", "lex_black", "lex_nation"]
SUPPORT_FEATURES = ["lex_support"]
HARM_FEATURES = ["lex_harm"]
PUNCTUATION_FEATURES = ["count_exclamation", "count_question", "count_heart_emoji",
                        "count_prayer_emoji", "count_flag_emoji", "count_allcaps_words"]
SURFACE_FEATURES = ["comment_length", "word_count", "uppercase_ratio", "avg_word_length"]

ALL_FEATURES = (PRONOUN_FEATURES + GROUP_LEXICON_FEATURES + SUPPORT_FEATURES +
                HARM_FEATURES + PUNCTUATION_FEATURES + SURFACE_FEATURES)


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_features(text: str, language: str = "en") -> dict:
    """Extract handcrafted features from a single text string."""
    if not isinstance(text, str):
        text = ""

    lower = text.lower()
    words = _WORD_RE.findall(lower)

    feats = {}

    # Pronoun counts
    pronoun_map = _PRONOUN_ES if language == "es" else _PRONOUN_EN
    for feat_name in PRONOUN_FEATURES:
        feats[feat_name] = 0
    for w in words:
        if w in pronoun_map:
            feats[pronoun_map[w]] += 1

    # Group lexicon
    for feat_name in GROUP_LEXICON_FEATURES:
        feats[feat_name] = 0
    for w in words:
        if w in _GROUP_LEXICON:
            feats[_GROUP_LEXICON[w]] += 1

    # Support lexicon
    feats["lex_support"] = sum(1 for w in words if w in _SUPPORT_LEXICON)

    # Harm lexicon
    feats["lex_harm"] = sum(1 for w in words if w in _HARM_LEXICON)

    # Punctuation / emotion cues
    feats["count_exclamation"] = text.count("!")
    feats["count_question"] = text.count("?")
    feats["count_heart_emoji"] = sum(1 for ch in text if ch in {"❤", "♥", "💕", "💖", "💗", "💓", "💛", "💙", "💚", "🧡", "🤍", "🖤"})
    feats["count_prayer_emoji"] = text.count("🙏")
    feats["count_flag_emoji"] = len(_FLAG_PATTERN.findall(text))
    feats["count_allcaps_words"] = sum(1 for w in text.split() if w.isupper() and len(w) > 1)

    # Surface features
    raw_words = text.split()
    feats["comment_length"] = len(text)
    feats["word_count"] = len(raw_words)
    if len(text) > 0:
        feats["uppercase_ratio"] = sum(1 for c in text if c.isupper()) / len(text)
    else:
        feats["uppercase_ratio"] = 0.0
    if raw_words:
        feats["avg_word_length"] = sum(len(w) for w in raw_words) / len(raw_words)
    else:
        feats["avg_word_length"] = 0.0

    return feats


def extract_features_df(df, text_col: str = "text", language: str = "en") -> pd.DataFrame:
    """Extract features for all rows in a DataFrame. Returns a DataFrame of features."""
    records = [extract_features(row[text_col], language) for _, row in df.iterrows()]
    return pd.DataFrame(records, index=df.index)
