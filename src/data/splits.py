"""
Data subsetting and splitting utilities.
"""

from sklearn.model_selection import train_test_split


def get_task_subset(df, task: int):
    """Filter rows according to task-specific subsetting rules.

    - Task 1: all rows (no filtering)
    - Task 2: Supportive rows only (task1 == "Supportive")
    - Task 3: Group rows only (task2 == "Group")
    """
    if task == 1:
        return df.copy()
    elif task == 2:
        return df[df["task1"] == "Supportive"].copy()
    elif task == 3:
        return df[df["task2"] == "Group"].copy()
    else:
        raise ValueError(f"Unknown task: {task}. Must be 1, 2, or 3.")


def stratified_split(texts, labels, seed=42):
    """Split into 80/10/10 train/val/test with stratification.

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
