
# Implementation Plan — SSD-2026 Social Support Detection

## Project Status

| Component | Status |
|---|---|
| `src/train_task1.py` | ✅ Complete — DistilBERT binary classifier (Supportive / Not Supportive) |
| `src/train_task2.py` | ❌ Missing |
| `src/train_task3.py` | ❌ Missing |
| `src/predict.py` | ❌ Missing — pipeline inference |
| `src/evaluate.py` | ❌ Missing — unified evaluation script |
| `main.py` | ❌ Stub only — needs to wire everything together |

---

## Architecture Decision

Use the **pipeline approach**: three independent DistilBERT models trained and
applied sequentially. This directly mirrors the existing `train_task1.py` and
is the simplest path to a working end-to-end system.

```
Input comment
    │
    ▼
[Task 1 Model] ── Not Supportive ──► label = "Non-Supportive", done
    │
  Supportive
    │
    ▼
[Task 2 Model] ── Individual ──► label = "Individual", done
    │
  Group
    │
    ▼
[Task 3 Model] ──► label ∈ {Nation, Other, LGBTQ, Black Community, Religion, Women}
```

---

## Step 1 — `src/train_task2.py`

### What it does
Binary classification on **Supportive comments only**: Individual (0) vs Group (1).

### How to build it
`train_task2.py` should be nearly identical to `train_task1.py`. The only differences are:

1. **Filter the dataframe** to keep only rows where `task1 == "Supportive"` before
   splitting and training.
2. **Label map**: `{"Individual": 0, "Group": 1}`
3. **Target names** in the classification report: `["Individual", "Group"]`
4. **Default `--label_col`**: `task2`
5. **Default `--save_dir`**: `./task2_model`

### Label distribution after filtering
```
Group:      1450  (~80.8%)
Individual:  345  (~19.2%)
```
Class imbalance is significant — the existing inverse-frequency class weighting
from Task 1 handles this automatically, no changes needed.

### Command
```bash
python src/train_task2.py \
  --csv Train_Data_SSD26/train-english.csv \
  --text_col text \
  --label_col task2
```

---

## Step 2 — `src/train_task3.py`

### What it does
6-class classification on **Group-supportive comments only**:
`Nation`, `Other`, `LGBTQ`, `Black Community`, `Religion`, `Women`.

### How to build it
Same pattern as train_task1.py with these differences:

1. **Filter the dataframe** to keep only rows where `task2 == "Group"`.
   Note: `task3` values of `"No"` correspond to non-group rows — after filtering
   on `task2 == "Group"`, all remaining `task3` values should be one of the six
   community labels.
2. **Label map**:
   ```python
   label_map = {
       "Nation": 0,
       "Other": 1,
       "LGBTQ": 2,
       "Black Community": 3,
       "Religion": 4,
       "Women": 5,
   }
   ```
3. **`num_labels=6`** in `DistilBertForSequenceClassification`.
4. **Target names** in classification report:
   `["Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"]`
5. **Default `--label_col`**: `task3`
6. **Default `--save_dir`**: `./task3_model`
7. **Increase patience** to `3` (recommended) because the dataset is small (~1,160
   training examples) and convergence is slower with 6 classes.

### Label distribution after filtering
```
Nation:          786  (~54.2%)
Other:           416  (~28.7%)
LGBTQ:           123  (~8.5%)
Black Community:  91  (~6.3%)
Women:            19  (~1.3%)
Religion:         15  (~1.0%)
```
Severe imbalance — class weights are critical here. No additional changes
needed beyond what Task 1 already does.

### Command
```bash
python src/train_task3.py \
  --csv Train_Data_SSD26/train-english.csv \
  --text_col text \
  --label_col task3
```

---

## Step 3 — `src/predict.py`

### What it does
Loads all three saved models and runs the full pipeline on new data.
Produces a CSV with predictions for all three subtasks.

### Interface
```bash
python src/predict.py \
  --csv path/to/test.csv \
  --text_col text \
  --task1_model ./task1_model \
  --task2_model ./task2_model \
  --task3_model ./task3_model \
  --output predictions.csv
```

### Logic
```python
# Pseudocode
for each comment:
    pred1 = task1_model.predict(comment)          # "Supportive" or "Non-Supportive"

    if pred1 == "Non-Supportive":
        pred2 = "No"
        pred3 = "No"
    else:
        pred2 = task2_model.predict(comment)      # "Individual" or "Group"

        if pred2 == "Individual":
            pred3 = "No"
        else:
            pred3 = task3_model.predict(comment)  # Nation / Other / LGBTQ / ...
```

### Output CSV columns
The output must match the submission format:
```
id, task1, task2, task3
```
Refer to `Submission_examples/submission_example.csv` for the exact format.

### Implementation notes
- Load each model with `DistilBertForSequenceClassification.from_pretrained(model_dir)`
- Load the tokenizer with `DistilBertTokenizerFast.from_pretrained(model_dir)`
- Reuse the `clean_text()` function from `train_task1.py` — move it to a shared
  `src/utils.py` module so all scripts can import it
- Run inference in batches (batch_size=32) for efficiency
- Use `torch.no_grad()` for all inference passes

---

## Step 4 — `src/utils.py` (refactor)

### What it does
Shared utilities used by all training scripts and the prediction script.

### Contents
Extract from `train_task1.py` into this module:

```python
# src/utils.py
EMOJI_WORDS = [...]      # existing list
def clean_text(text)     # existing function
def get_device()         # existing function
class SupportDataset     # existing class
```

Then update `train_task1.py`, `train_task2.py`, `train_task3.py`, and
`predict.py` to import from `src.utils`.

---

## Step 5 — `src/evaluate.py`

### What it does
Given a gold-standard CSV and a predictions CSV, computes official metrics for
all three subtasks.

### Interface
```bash
python src/evaluate.py \
  --gold Train_Data_SSD26/train-english.csv \
  --pred predictions.csv
```

### Metrics to report (per subtask)
- Macro-averaged F1 (primary metric)
- Per-class F1, Precision, Recall
- Confusion matrix

### Implementation notes
- Use `sklearn.metrics.classification_report` and `confusion_matrix`
- Only evaluate Task 2 on rows where gold `task1 == "Supportive"`
- Only evaluate Task 3 on rows where gold `task2 == "Group"`

---

## Step 6 — `main.py` (wire everything together)

Update `main.py` to be a CLI entry point with subcommands:

```bash
python main.py train --task 1   # runs train_task1
python main.py train --task 2   # runs train_task2
python main.py train --task 3   # runs train_task3
python main.py train --task all # runs all three in sequence

python main.py predict --csv test.csv --output predictions.csv
python main.py evaluate --gold gold.csv --pred predictions.csv
```

---

## Step 7 — `pyproject.toml` updates

Add missing dependencies:

```toml
dependencies = [
    "pandas>=2.3.3",
    "scikit-learn>=1.7.2",
    "torch>=2.10.0",
    "transformers>=5.3.0",
    "numpy>=1.26.0",      # add: used throughout but not declared
    "tqdm>=4.66.0",       # add: useful for prediction progress bars
]
```

---

## File Structure After Implementation

```
SocialSupportDetection/
├── src/
│   ├── utils.py           # NEW — shared clean_text, SupportDataset, get_device
│   ├── train_task1.py     # REFACTOR — import from utils.py
│   ├── train_task2.py     # NEW
│   ├── train_task3.py     # NEW
│   ├── predict.py         # NEW
│   └── evaluate.py        # NEW
├── main.py                # UPDATE — CLI entry point
├── task1_model/           # saved after training task 1
├── task2_model/           # saved after training task 2
├── task3_model/           # saved after training task 3
├── Train_Data_SSD26/
│   ├── train-english.csv
│   └── train-spanish.csv
├── Submission_examples/
│   ├── submission_example.csv
│   └── submission_example_2.csv
└── pyproject.toml
```

---

## Training Order

Run training in this exact order (each model depends on the filtered subset
defined by the previous task's labels):

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Train all three models
python src/train_task1.py --csv Train_Data_SSD26/train-english.csv --text_col text --label_col task1
python src/train_task2.py --csv Train_Data_SSD26/train-english.csv --text_col text --label_col task2
python src/train_task3.py --csv Train_Data_SSD26/train-english.csv --text_col text --label_col task3

# 3. Run pipeline on test/submission data
python src/predict.py --csv <test_file.csv> --text_col text --output predictions.csv

# 4. Evaluate against gold labels (internal test set)
python src/evaluate.py --gold Train_Data_SSD26/train-english.csv --pred predictions.csv
```

---

## Notes for the Paper

Once training is complete, record the following numbers for the Results section:

| Metric | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Val Macro F1 | | | |
| Test Macro F1 | | | |
| Test Precision (macro) | | | |
| Test Recall (macro) | | | |

Also note which epoch triggered early stopping for each task — this is useful
for the training analysis paragraph in the paper.
