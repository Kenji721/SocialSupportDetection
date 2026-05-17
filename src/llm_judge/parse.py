"""
Output parsing and validation for LLM judge responses.
"""

from src.data.label_maps import get_label_config

# Fuzzy label mappings for common LLM output variations
_FUZZY_MAP = {
    # Task 1
    "supportive": "Supportive",
    "non-supportive": "Non-Supportive",
    "non supportive": "Non-Supportive",
    "nonsupportive": "Non-Supportive",
    "not supportive": "Non-Supportive",
    # Task 2
    "individual": "Individual",
    "group": "Group",
    # Task 3
    "nation": "Nation",
    "other": "Other",
    "lgbtq": "LGBTQ",
    "lgbtq+": "LGBTQ",
    "lgbt": "LGBTQ",
    "black community": "Black Community",
    "black": "Black Community",
    "african american": "Black Community",
    "religion": "Religion",
    "religious": "Religion",
    "women": "Women",
    "woman": "Women",
}


def parse_label(raw_output: str, task: int) -> str | None:
    """Parse LLM output into a valid label for the given task.

    Returns the canonical label string, or None if parsing fails.
    """
    labels, _, _, _ = get_label_config(task)
    valid_set = set(labels)

    # Clean up: take first line, strip whitespace and quotes
    text = raw_output.strip().split("\n")[0].strip().strip("\"'").strip()

    # Exact match
    if text in valid_set:
        return text

    # Fuzzy match (case-insensitive)
    lower = text.lower()
    if lower in _FUZZY_MAP:
        candidate = _FUZZY_MAP[lower]
        if candidate in valid_set:
            return candidate

    # Substring match: check if any valid label appears in the output
    for label in labels:
        if label.lower() in lower:
            return label

    return None
