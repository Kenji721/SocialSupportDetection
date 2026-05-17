# IMPLEMENTATION_PLAN_2.md — Multi-Approach SSD-2026 System

> Detailed, phased implementation plan for Claude Code.
> Based on `docs/training_plan.md`. Supersedes `docs/IMPLEMENTATION_PLAN.md`.

---

## Status of Existing Codebase

The following files already exist and are functional:

| File | Status | Notes |
|---|---|---|
| `src/train_task1.py` | ✅ Complete | DistilBERT binary (Supportive / Non-Supportive) |
| `src/train_task2.py` | ✅ Complete | DistilBERT binary (Individual / Group), Supportive only |
| `src/train_task3.py` | ✅ Complete | DistilBERT 6-class (community), Group only |
| `src/predict.py` | ✅ Complete | Hierarchical pipeline inference (T1→T2→T3) |
| `src/evaluate.py` | ✅ Complete | Unified evaluation with per-task macro F1 |
| `src/results.py` | ✅ Complete | CSV experiment logger |
| `src/utils.py` | ✅ Complete | Device, clean_text, SupportDataset, train_epoch, evaluate |
| `main.py` | ✅ Complete | CLI: train / predict / evaluate / results |

**These files should NOT be deleted.** The existing DistilBERT pipeline serves as the v1 baseline. All new code goes into the new module structure below.

---

## Dataset Schema

### English: `data/Train_Data_SSD26/train-english.csv` (7,998 rows)

| Column | Type | Values |
|---|---|---|
| `id` | int | unique identifier |
| `text` | str | social media comment |
| `task1` | str | `Supportive`, `Non-Supportive` |
| `task2` | str | `Individual`, `Group`, `No` |
| `task3` | str | `Nation`, `Other`, `LGBTQ`, `Black Community`, `Religion`, `Women`, `No` |

### Spanish: `data/Train_Data_SSD26/train-spanish.csv` (3,687 rows)

| Column | Type | Values |
|---|---|---|
| `id` | int | unique identifier |
| `comment` | str | social media comment (**note: column is `comment`, not `text`**) |
| `task1` | str | `Support`, `Non Support` (**note: different label strings**) |
| `task2` | str | `Individual`, `Group`, `No` |
| `task3` | str | `Nation`, `Other`, `LGBTQ`, `Black Community`, `Religion`, `Women`, `No` |

### Subsetting Rules (critical)

- **Task 1**: Use ALL rows.
- **Task 2**: Use only rows where `task1 == "Supportive"` (EN) or `task1 == "Support"` (ES).
- **Task 3**: Use only rows where `task2 == "Group"`.

### Test Data

- `data/Test_phase_data/test_phase_english.csv`
- `data/Test_phase_data/test_phase_spanish.csv`

---

## Target Code Structure

```
src/
├── __init__.py                    # (exists)
├── utils.py                       # (exists — keep as-is)
├── results.py                     # (exists — keep as-is)
├── evaluate.py                    # (exists — keep as-is)
├── predict.py                     # (exists — keep as-is, DistilBERT pipeline)
├── train_task1.py                 # (exists — keep as-is, DistilBERT baseline)
├── train_task2.py                 # (exists — keep as-is)
├── train_task3.py                 # (exists — keep as-is)
│
├── data/
│   ├── __init__.py
│   ├── loading.py                 # Unified CSV loader (EN + ES), column normalization
│   ├── splits.py                  # Stratified train/val/test (80/10/10), fixed seeds
│   └── label_maps.py             # All label→int maps, id2label dicts, EN/ES normalization
│
├── preprocessing.py               # Preprocessing functions per approach type
├── metrics.py                     # Shared metric computation (macro F1, per-class, confusion)
├── safety.py                      # Harm/violence lexicon + detection module
├── error_analysis.py              # Per-class error analysis, confusion matrix viz
│
├── traditional/
│   ├── __init__.py
│   ├── vectorizers.py             # TF-IDF (word + char n-gram) pipelines
│   ├── features.py                # Handcrafted features (pronouns, lexicons, length, etc.)
│   ├── train.py                   # LR + SVM training loop per task
│   └── predict.py                 # Traditional ML inference
│
├── deeplearning/
│   ├── __init__.py
│   ├── embeddings.py              # FastText / GloVe loader
│   ├── models.py                  # TextCNN, BiLSTM, BiLSTM+Attention
│   ├── train.py                   # PyTorch training loop
│   └── predict.py                 # DL inference
│
├── finetuning/
│   ├── __init__.py
│   ├── datasets.py                # HuggingFace Dataset wrappers for all models
│   ├── train.py                   # Unified fine-tuning (BERTweet, RoBERTuito, XLM-R)
│   └── predict.py                 # Transformer inference (auto-detect model type)
│
├── llm_judge/
│   ├── __init__.py
│   ├── prompts.py                 # Zero-shot / few-shot prompt templates per task
│   ├── judge.py                   # LLM API call + retry logic
│   └── parse.py                   # Output parsing + validation
│
└── pipelines/
    ├── __init__.py
    ├── independent.py             # Run T1, T2, T3 as separate models
    ├── hierarchical.py            # T1 → filter → T2 → filter → T3
    └── multitask.py               # Shared encoder + multi-head (optional)
```

---

## Reproducibility Rules

Apply these rules to ALL code written in every phase:

1. **Fixed seeds**: Use `seeds = [13, 42, 77]` for all stochastic operations. Set `random.seed(s)`, `np.random.seed(s)`, `torch.manual_seed(s)`, `torch.cuda.manual_seed_all(s)`.
2. **Data splits**: Always 80/10/10 train/val/test with stratification. Use `sklearn.model_selection.train_test_split` with `random_state=42` (primary seed).
3. **Logging**: Every experiment must call `src.results.log_result(...)` with all fields populated.
4. **Artifacts**: Save per-experiment outputs to `artifacts/<experiment_name>/` containing `metrics.json` and `predictions.csv`.
5. **Package versions**: Pin in `pyproject.toml`. Add new deps as needed per phase.
6. **Print label distributions**: Before every training run, print train/val/test sizes and per-class counts.

---

## PHASE 0: Shared Infrastructure

**Goal**: Build reusable data loading, preprocessing, and metrics modules.

### Step 0.1 — `src/data/loading.py`

Create a unified data loader:

```python
def load_ssd_data(csv_path: str, language: str = "en") -> pd.DataFrame:
    """
    Load and normalize an SSD-2026 CSV file.
    - EN: text_col="text", task1 labels: Supportive / Non-Supportive
    - ES: text_col="comment" → rename to "text", task1 labels: Support / Non Support → normalize
    Returns DataFrame with columns: id, text, task1, task2, task3, language
    """
```

Normalization rules:
- ES `comment` column → rename to `text`
- ES `Support` → `Supportive`, `Non Support` → `Non-Supportive`
- Add `language` column: `"en"` or `"es"`

```python
def load_multilingual(en_path: str, es_path: str) -> pd.DataFrame:
    """Load both EN and ES, concatenate, return unified DataFrame."""
```

### Step 0.2 — `src/data/label_maps.py`

```python
# Task 1
TASK1_LABELS = ["Non-Supportive", "Supportive"]
TASK1_LABEL2ID = {"Non-Supportive": 0, "Supportive": 1}
TASK1_ID2LABEL = {0: "Non-Supportive", 1: "Supportive"}

# Task 2
TASK2_LABELS = ["Individual", "Group"]
TASK2_LABEL2ID = {"Individual": 0, "Group": 1}
TASK2_ID2LABEL = {0: "Individual", 1: "Group"}

# Task 3
TASK3_LABELS = ["Nation", "Other", "LGBTQ", "Black Community", "Religion", "Women"]
TASK3_LABEL2ID = {"Nation": 0, "Other": 1, "LGBTQ": 2, "Black Community": 3, "Religion": 4, "Women": 5}
TASK3_ID2LABEL = {v: k for k, v in TASK3_LABEL2ID.items()}

def get_label_config(task: int):
    """Return (labels, label2id, id2label, num_labels) for a given task number."""
```

### Step 0.3 — `src/data/splits.py`

```python
def get_task_subset(df: pd.DataFrame, task: int) -> pd.DataFrame:
    """
    Apply subsetting rules:
    - Task 1: all rows
    - Task 2: only Supportive rows
    - Task 3: only Group rows
    """

def stratified_split(texts, labels, seed=42, test_size=0.2):
    """80/10/10 stratified split. Returns (X_train, X_val, X_test, y_train, y_val, y_test)."""
```

### Step 0.4 — `src/preprocessing.py`

Three preprocessing modes:

```python
def preprocess_traditional(text: str) -> str:
    """Aggressive: lowercase, remove URLs/mentions, lemmatize (spaCy), emoji→text, slang normalize."""

def preprocess_deep(text: str) -> str:
    """Moderate: lowercase (optional), keep punctuation and stopwords, character repetition normalize."""

def preprocess_transformer(text: str) -> str:
    """Minimal: normalize encoding only, preserve hashtags/emojis/punctuation."""
```

### Step 0.5 — `src/metrics.py`

```python
def compute_metrics(y_true, y_pred, target_names: list) -> dict:
    """
    Return dict with: macro_f1, accuracy, precision_macro, recall_macro,
    per_class_f1 (dict), confusion_matrix (list of lists).
    """
```

### Step 0.6 — `src/safety.py`

```python
HARM_LEXICON = [...]  # violence/harm keywords (EN + ES)

def has_harm_signal(text: str) -> bool:
    """Return True if text contains harm/violence indicators."""

def apply_harm_override(task1_preds, texts) -> list:
    """Post-hoc: if Supportive AND harm_signal → flip to Non-Supportive."""
```

### Step 0.7 — Create `artifacts/` directory

```bash
mkdir -p artifacts
```

### Step 0.8 — Update `pyproject.toml`

Add dependencies needed across all phases:

```toml
dependencies = [
    "pandas>=2.3.3",
    "scikit-learn>=1.7.2",
    "torch>=2.10.0",
    "transformers>=5.3.0",
    "numpy>=1.26.0",
    "tqdm>=4.66.0",
    "spacy>=3.7.0",          # for lemmatization (traditional)
    "xgboost>=2.0.0",        # optional traditional ML
    "emoji>=2.0.0",          # emoji processing
]
```

Also run `python -m spacy download en_core_web_sm` after install.

### Verification

- [ ] `load_ssd_data("data/Train_Data_SSD26/train-english.csv", "en")` returns 7998 rows with columns `[id, text, task1, task2, task3, language]`
- [ ] `load_ssd_data("data/Train_Data_SSD26/train-spanish.csv", "es")` returns 3687 rows with normalized labels
- [ ] `load_multilingual(...)` returns 11685 rows
- [ ] `get_task_subset(df, 2)` filters correctly (EN: ~1795, ES: varies)
- [ ] `stratified_split` produces 80/10/10 with preserved class ratios
- [ ] `compute_metrics` returns correct dict structure

---

## PHASE 1: Traditional ML Baselines

**Goal**: TF-IDF + Logistic Regression / Linear SVM for all 3 tasks × EN/ES.

### Step 1.1 — `src/traditional/vectorizers.py`

```python
def build_tfidf_pipeline(mode="combined", max_features=50000):
    """
    mode: "word", "char", "combined"
    - word: TfidfVectorizer(ngram_range=(1,2), max_features=max_features, sublinear_tf=True)
    - char: TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=max_features, sublinear_tf=True)
    - combined: FeatureUnion of word + char
    Returns sklearn Pipeline or FeatureUnion.
    """
```

### Step 1.2 — `src/traditional/features.py`

Handcrafted features to concatenate with TF-IDF:

```python
def extract_features(text: str, language: str = "en") -> dict:
    """
    Returns dict of:
    - pronoun_you, pronoun_we, pronoun_they (counts)
    - group_lexicon_match (bool for LGBTQ/women/religion/nation keywords)
    - support_lexicon_match (love, strength, solidarity, ánimo, orgullo, etc.)
    - harm_lexicon_match (from safety.py)
    - has_emoji (bool)
    - exclamation_count, question_count
    - text_length_words, text_length_chars
    """
```

### Step 1.3 — `src/traditional/train.py`

```python
def train_traditional(
    task: int,
    language: str,          # "en", "es", "multilingual"
    model_type: str,        # "lr", "svm", "sgd"
    tfidf_mode: str,        # "word", "char", "combined"
    use_handcrafted: bool,  # whether to append handcrafted features
    seed: int = 42,
    max_features: int = 50000,
) -> dict:
    """
    Full training pipeline:
    1. Load data via src.data.loading
    2. Subset for task via src.data.splits
    3. Preprocess via src.preprocessing.preprocess_traditional
    4. Vectorize (TF-IDF ± handcrafted features)
    5. Split 80/10/10
    6. Train model with class_weight="balanced"
    7. Evaluate on val + test
    8. Log results via src.results.log_result
    9. Save artifacts to artifacts/traditional_{model_type}_{tfidf_mode}_task{task}_{language}/
    Returns metrics dict.
    """
```

### Step 1.4 — `src/traditional/predict.py`

```python
def predict_traditional(model_path: str, texts: list[str]) -> list[int]:
    """Load saved sklearn pipeline and predict."""
```

### Step 1.5 — Run experiments

Execute the following grid (18 experiments minimum):

| Model | TF-IDF Mode | Task | Language | Handcrafted |
|---|---|---|---|---|
| LR | combined | 1, 2, 3 | en | yes |
| LR | combined | 1, 2, 3 | es | yes |
| SVM | combined | 1, 2, 3 | en | yes |
| SVM | combined | 1, 2, 3 | es | yes |
| LR | combined | 1, 2, 3 | multilingual | yes |
| SVM | combined | 1, 2, 3 | multilingual | yes |

Run each with `seed=42`. Save results to `artifacts/` and log to `results/experiment_log.csv`.

### Verification

- [ ] All 18+ experiments logged in `results/experiment_log.csv`
- [ ] Each experiment has a folder in `artifacts/` with `metrics.json` and `predictions.csv`
- [ ] Macro F1 for Task 1 EN LR should be roughly 0.60–0.75 (sanity check)
- [ ] Class weights are applied (`class_weight="balanced"`)

---

## PHASE 2: Transformer Fine-tuning

**Goal**: Fine-tune BERTweet (EN), RoBERTuito (ES), XLM-R (multilingual) on all 3 tasks.

### Models (exact HuggingFace identifiers)

| Model | HF Identifier | Language | Domain |
|---|---|---|---|
| BERTweet | `vinai/bertweet-base` | English | Twitter |
| RoBERTuito | `pysentimiento/robertuito-base-uncased` | Spanish | Twitter |
| XLM-R | `xlm-roberta-base` | Multilingual | General |

**Out of scope** (do NOT implement): DistilBERT (already done as v1), BETO, RoBERTa-base.

### Step 2.1 — `src/finetuning/datasets.py`

```python
class SSDDataset(torch.utils.data.Dataset):
    """
    Generalized dataset for any HuggingFace tokenizer.
    Handles: AutoTokenizer (not just DistilBertTokenizerFast).
    Constructor: (texts, labels, tokenizer, max_len=128)
    """
```

### Step 2.2 — `src/finetuning/train.py`

```python
def train_transformer(
    task: int,
    language: str,              # "en", "es", "multilingual"
    model_name: str,            # HF model identifier
    seed: int = 42,
    lr: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 5,
    max_len: int = 128,
    patience: int = 2,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    use_class_weights: bool = True,
    save_dir: str = None,       # auto-generated if None
) -> dict:
    """
    Full fine-tuning pipeline:
    1. Load data (unified loader)
    2. Subset for task
    3. Preprocess via preprocess_transformer (minimal)
    4. Split 80/10/10, stratified, seed=seed
    5. Tokenize with AutoTokenizer.from_pretrained(model_name)
    6. Build DataLoaders
    7. Load AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=N)
    8. Optimizer: AdamW (lr, weight_decay)
    9. Scheduler: linear warmup (warmup_ratio * total_steps)
    10. Loss: CrossEntropyLoss with inverse-frequency class weights
    11. Training loop with gradient clipping (max_norm=1.0)
    12. Early stopping on val macro F1 (patience)
    13. Restore best checkpoint
    14. Evaluate on test set
    15. Log results
    16. Save model + tokenizer to save_dir
    17. Save artifacts (metrics.json, predictions.csv)
    Returns metrics dict.
    """
```

Key implementation notes:
- Use `AutoTokenizer` and `AutoModelForSequenceClassification` (not model-specific classes)
- BERTweet uses `use_fast=False` for its tokenizer
- RoBERTuito is uncased — no lowercasing needed, the tokenizer handles it
- XLM-R supports all languages natively

### Step 2.3 — `src/finetuning/predict.py`

```python
def predict_transformer(model_dir: str, texts: list[str], batch_size: int = 32) -> list[int]:
    """
    Load model + tokenizer from model_dir, run inference.
    Auto-detect model type from config.json.
    """
```

### Step 2.4 — Run experiments

Execute with **3 seeds** each (seeds = [13, 42, 77]):

| Model | Task | Language | Seeds |
|---|---|---|---|
| BERTweet (`vinai/bertweet-base`) | 1, 2, 3 | en | 13, 42, 77 |
| RoBERTuito (`pysentimiento/robertuito-base-uncased`) | 1, 2, 3 | es | 13, 42, 77 |
| XLM-R (`xlm-roberta-base`) | 1, 2, 3 | en | 13, 42, 77 |
| XLM-R (`xlm-roberta-base`) | 1, 2, 3 | es | 13, 42, 77 |
| XLM-R (`xlm-roberta-base`) | 1, 2, 3 | multilingual | 13, 42, 77 |

Total: 5 configs × 3 tasks × 3 seeds = **45 experiments**.

Hyperparameters:
- `lr = 2e-5`
- `batch_size = 16` (BERTweet, RoBERTuito) or `32` (XLM-R)
- `epochs = 5`
- `patience = 2` (Tasks 1, 2) or `3` (Task 3)
- `max_len = 128`
- `warmup_ratio = 0.1`
- `weight_decay = 0.01`
- `use_class_weights = True`

### Verification

- [ ] All 45 experiments logged with mean ± std across seeds
- [ ] BERTweet EN Task 1 macro F1 should be ≥ 0.75 (sanity check)
- [ ] XLM-R multilingual Task 1 should be competitive with monolingual models
- [ ] Model checkpoints saved and loadable
- [ ] No `DistilBert*` imports in `src/finetuning/` (use `Auto*` classes only)

---

## PHASE 3: Hierarchical vs Independent Pipeline Comparison

**Goal**: Compare independent models vs hierarchical cascade (T1→T2→T3) for end-to-end performance.

### Step 3.1 — `src/pipelines/independent.py`

```python
def run_independent_pipeline(
    task1_model_dir: str,
    task2_model_dir: str,
    task3_model_dir: str,
    texts: list[str],
    model_type: str = "transformer",  # or "traditional"
) -> pd.DataFrame:
    """
    Run all three models independently on ALL texts.
    Return DataFrame with columns: task1_pred, task2_pred, task3_pred.
    Note: Task 2 and Task 3 predict on all texts (no filtering).
    """
```

### Step 3.2 — `src/pipelines/hierarchical.py`

```python
def run_hierarchical_pipeline(
    task1_model_dir: str,
    task2_model_dir: str,
    task3_model_dir: str,
    texts: list[str],
    model_type: str = "transformer",
) -> pd.DataFrame:
    """
    Cascade: T1 → filter Supportive → T2 → filter Group → T3.
    Non-Supportive texts get task2="No", task3="No".
    Individual texts get task3="No".
    Return DataFrame with columns: task1_pred, task2_pred, task3_pred.
    """
```

### Step 3.3 — End-to-end evaluation

```python
def evaluate_pipeline(
    pipeline_fn,
    gold_df: pd.DataFrame,
    model_dirs: dict,
    texts: list[str],
) -> dict:
    """
    Run pipeline, compute per-task macro F1 on gold labels.
    Also compute error propagation: what % of Task 2/3 errors are caused by Task 1 mistakes.
    """
```

### Step 3.4 — Run comparison

For the best model from Phase 2 (likely BERTweet EN or XLM-R multilingual):
1. Run `independent` pipeline on test set
2. Run `hierarchical` pipeline on test set
3. Compare per-task macro F1
4. Analyze error propagation in hierarchical mode
5. Log both results

### Verification

- [ ] Both pipelines produce valid output DataFrames
- [ ] Error propagation analysis shows % of T2/T3 errors from upstream misclassification
- [ ] Results compared in a summary table

---

## PHASE 4: Harm Module + Class Imbalance Improvements

**Goal**: Add harm/violence override and address Task 3 class imbalance.

### Step 4.1 — Expand `src/safety.py`

```python
# Build a curated EN + ES harm lexicon
HARM_LEXICON_EN = ["kill", "murder", "violence", "destroy", "attack", "bomb", ...]
HARM_LEXICON_ES = ["matar", "violencia", "destruir", "atacar", "bomba", ...]

def harm_score(text: str, language: str = "en") -> float:
    """Return normalized harm score (0-1) based on lexicon matches."""

def apply_harm_override(predictions: pd.DataFrame, texts: list[str]) -> pd.DataFrame:
    """
    Post-hoc rule: if task1_pred == "Supportive" AND harm_score > threshold → flip to Non-Supportive.
    Also set task2 and task3 to "No".
    """
```

### Step 4.2 — Class imbalance for Task 3

Implement two strategies:

**A. Enhanced class weights** (already using inverse-frequency; ensure it's applied everywhere).

**B. Paraphrase augmentation** (for Women and Religion classes only):

```python
def augment_minority_classes(df: pd.DataFrame, task: int = 3, min_count: int = 50) -> pd.DataFrame:
    """
    For classes with fewer than min_count samples:
    - Back-translate or paraphrase to generate synthetic examples
    - Use simple augmentation: synonym replacement, random word swap
    Target: bring Women (19) and Religion (15) up to ~50 samples each.
    """
```

### Step 4.3 — Re-run best Phase 2 model with improvements

1. Re-train best transformer on Task 1 with harm-aware override → measure improvement
2. Re-train Task 3 with augmented minority classes → measure per-class F1 for Women and Religion
3. Compare: baseline vs harm-override vs augmented vs both

### Verification

- [ ] Harm override flips at least some predictions (sanity check on train set)
- [ ] Women and Religion per-class F1 improves with augmentation
- [ ] Overall macro F1 does not degrade significantly
- [ ] All results logged

---

## PHASE 5: Deep Learning + LLM Judge (If Time Permits)

**Goal**: TextCNN / BiLSTM baselines and LLM zero-shot/few-shot evaluation.

### Step 5.1 — `src/deeplearning/embeddings.py`

```python
def load_fasttext(path: str = None) -> dict:
    """Load FastText embeddings. Download cc.en.300.bin if not present."""

def build_embedding_matrix(vocab: dict, embeddings: dict, dim: int = 300) -> np.ndarray:
    """Build embedding matrix from vocab + pretrained embeddings."""
```

### Step 5.2 — `src/deeplearning/models.py`

```python
class TextCNN(nn.Module):
    """Kim (2014) CNN: multiple filter sizes (3, 4, 5), 100 filters each."""

class BiLSTM(nn.Module):
    """Bidirectional LSTM with hidden_dim=256, dropout=0.3."""

class BiLSTMAttention(nn.Module):
    """BiLSTM + self-attention layer."""
```

Hyperparameters:
- `embedding_dim = 300`
- `hidden_dim = 256`
- `dropout = 0.3`
- `lr = 1e-3`
- `batch_size = 64`
- `epochs = 20`
- `patience = 3`

### Step 5.3 — `src/deeplearning/train.py`

```python
def train_deep(
    task: int,
    language: str,
    model_type: str,     # "textcnn", "bilstm", "bilstm_attn"
    embedding_type: str, # "fasttext", "glove"
    seed: int = 42,
) -> dict:
    """Full DL training loop with class weights and early stopping."""
```

### Step 5.4 — `src/llm_judge/prompts.py`

```python
TASK1_ZERO_SHOT = """Classify the following social media comment as either "Supportive" or "Non-Supportive".
A comment is Supportive if it expresses encouragement, care, solidarity, or help toward someone.
A comment promoting violence is NOT Supportive even if it appears positive.

Comment: {text}
Label:"""

TASK1_FEW_SHOT = """..."""  # Include 3-5 examples per class

# Similar for Task 2 and Task 3
```

### Step 5.5 — `src/llm_judge/judge.py`

```python
def run_llm_judge(
    task: int,
    texts: list[str],
    model: str,           # "qwen2.5:7b" or "gpt-4o-mini"
    mode: str = "zero",   # "zero" or "few"
    max_retries: int = 3,
) -> list[str]:
    """
    Call LLM API, parse output, retry on failure.
    Log invalid outputs separately.
    """
```

### Step 5.6 — Run experiments

Deep Learning:
- TextCNN + FastText: Tasks 1, 2, 3 × EN (seeds 13, 42, 77)
- BiLSTM + FastText: Tasks 1, 2, 3 × EN (seeds 13, 42, 77)

LLM Judge:
- Zero-shot: Tasks 1, 2, 3 (one model, one run)
- Few-shot: Tasks 1, 2, 3 (one model, one run)

### Verification

- [ ] DL models train and converge (loss decreasing)
- [ ] LLM judge produces valid labels for >95% of inputs
- [ ] All results logged and comparable with Phase 1/2

---

## Final Summary Report

After all phases, generate a summary comparison:

```python
# Run: python main.py results
# Should print a formatted table comparing all approaches across all tasks
```

The `results/experiment_log.csv` should contain all experiments. Create a final `artifacts/summary_report.md` with:
1. Best model per task per language
2. Traditional ML vs Transformer vs DL comparison
3. Hierarchical vs Independent pipeline comparison
4. Effect of harm override
5. Effect of class imbalance handling
6. Recommendations for the final submission

---

## Execution Order Summary

| Phase | What | Priority | Est. Experiments |
|---|---|---|---|
| 0 | Shared infrastructure | MUST | 0 (setup only) |
| 1 | Traditional ML baselines | HIGH | ~18 |
| 2 | Transformer fine-tuning | HIGH | ~45 |
| 3 | Pipeline comparison | HIGH | ~4 |
| 4 | Harm + imbalance | MEDIUM | ~8 |
| 5 | Deep learning + LLM | LOW | ~24 |

**Total estimated experiments**: ~99

Start with Phase 0, then proceed sequentially. Phases 1 and 2 are the highest priority and should be completed before moving on.
