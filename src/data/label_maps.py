"""
Canonical label definitions for all three SSD-2026 tasks.
"""

# ── Task 1: Binary (Supportive vs Non-Supportive) ───────────────────────────
TASK1_LABELS = ["Non-Supportive", "Supportive"]
TASK1_LABEL2ID = {label: i for i, label in enumerate(TASK1_LABELS)}
TASK1_ID2LABEL = {i: label for i, label in enumerate(TASK1_LABELS)}

# ── Task 2: Binary (Individual vs Group) — trained on Supportive subset only ─
TASK2_LABELS = ["Individual", "Group"]
TASK2_LABEL2ID = {label: i for i, label in enumerate(TASK2_LABELS)}
TASK2_ID2LABEL = {i: label for i, label in enumerate(TASK2_LABELS)}

# ── Task 3: 6-class community — trained on Group subset only ────────────────
TASK3_LABELS = ["Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"]
TASK3_LABEL2ID = {label: i for i, label in enumerate(TASK3_LABELS)}
TASK3_ID2LABEL = {i: label for i, label in enumerate(TASK3_LABELS)}

_CONFIGS = {
    1: (TASK1_LABELS, TASK1_LABEL2ID, TASK1_ID2LABEL),
    2: (TASK2_LABELS, TASK2_LABEL2ID, TASK2_ID2LABEL),
    3: (TASK3_LABELS, TASK3_LABEL2ID, TASK3_ID2LABEL),
}


def get_label_config(task: int):
    """Return (labels, label2id, id2label, num_labels) for a given task number."""
    if task not in _CONFIGS:
        raise ValueError(f"Unknown task: {task}. Must be 1, 2, or 3.")
    labels, label2id, id2label = _CONFIGS[task]
    return labels, label2id, id2label, len(labels)
