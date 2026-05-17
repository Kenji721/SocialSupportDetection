"""
Text preprocessing pipelines for different model types.
"""

import re
import unicodedata

import spacy

try:
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    _nlp = None

try:
    import emoji as _emoji_mod
except ImportError:
    _emoji_mod = None

# Common social-media slang → expansion
_SLANG_MAP = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "b4": "before",
    "bc": "because",
    "w/": "with",
    "w/o": "without",
    "ppl": "people",
    "govt": "government",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "smh": "shaking my head",
    "ngl": "not going to lie",
    "idk": "i don't know",
    "irl": "in real life",
    "rn": "right now",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
}


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def _remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def _emoji_to_text(text: str) -> str:
    if _emoji_mod is None:
        return text
    return _emoji_mod.demojize(text, delimiters=(" ", " "))


def _expand_slang(text: str) -> str:
    tokens = text.split()
    return " ".join(_SLANG_MAP.get(t, t) for t in tokens)


def _lemmatize(text: str) -> str:
    if _nlp is None:
        return text
    doc = _nlp(text)
    return " ".join(token.lemma_ for token in doc)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess_traditional(text: str) -> str:
    """Aggressive preprocessing for traditional ML models.

    lowercase, remove URLs/mentions, emoji→text, slang normalize, lemmatize.
    """
    if not isinstance(text, str):
        return ""
    text = _normalize_unicode(text)
    text = text.lower()
    text = _remove_urls(text)
    text = _remove_mentions(text)
    text = _emoji_to_text(text)
    text = _expand_slang(text)
    text = _lemmatize(text)
    text = _collapse_whitespace(text)
    return text


def preprocess_deep(text: str) -> str:
    """Moderate preprocessing for deep learning (LSTM, CNN).

    lowercase, keep punctuation/stopwords, minimal normalization.
    """
    if not isinstance(text, str):
        return ""
    text = _normalize_unicode(text)
    text = text.lower()
    text = _remove_urls(text)
    text = _remove_mentions(text)
    text = _emoji_to_text(text)
    text = _collapse_whitespace(text)
    return text


def preprocess_transformer(text: str) -> str:
    """Minimal preprocessing for transformer models.

    Normalize encoding, convert emojis to text descriptions,
    and collapse whitespace — let the tokenizer handle the rest.
    """
    if not isinstance(text, str):
        return ""
    text = _normalize_unicode(text)
    text = _emoji_to_text(text)
    text = _collapse_whitespace(text)
    return text
