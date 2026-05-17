# SSD-2026: Training Plan — Multi-Approach, Hierarchical, Multilingual

## Overview

Goal: Build and compare multiple approaches for Social Support Detection (SSD-2026)
across three subtasks:

1. Task 1: Support Detection (binary)
2. Task 2: Target Type (Individual vs Group)
3. Task 3: Group Classification (multiclass)

We explicitly model:
- Hierarchical dependencies between tasks
- Multilingual setting (English + Spanish)
- Harm/violence constraint (support ≠ prosocial if violent)

---

## Core Experimental Axes

### 1. Modeling Approach
- Traditional ML (TF-IDF + features)
- Deep Learning (CNN / BiLSTM)
- Transformer Fine-tuning
- LLM-as-a-Judge (baseline only)

### 2. Task Strategy
- Independent (separate model per task)
- Hierarchical (T1 → T2 → T3 cascade)
- Multi-task (shared encoder, multi-head)

### 3. Language Strategy
- English-only
- Spanish-only
- Multilingual (shared model)

### 4. Class Imbalance Strategy
- None
- Class weights
- Oversampling / augmentation (Task 3 focus)

### 5. Harm Constraint Handling
- None (baseline)
- Lexicon feature
- Auxiliary classifier
- Post-hoc override

---

## Code Structure
src/
├── data/
│ ├── loading.py
│ ├── splits.py
│ └── label_maps.py
│
├── preprocessing.py
├── metrics.py
├── safety.py # Harm/violence detection
├── results.py # JSON logging
├── error_analysis.py
│
├── traditional/
│ ├── vectorizers.py # TF-IDF (word + char)
│ ├── features.py # handcrafted features
│ ├── train.py
│ └── predict.py
│
├── deeplearning/
│ ├── embeddings.py
│ ├── models.py
│ ├── train.py
│ └── predict.py
│
├── finetuning/
│ ├── datasets.py
│ ├── train.py
│ └── predict.py
│
├── llm_judge/
│ ├── prompts.py
│ ├── judge.py
│ └── parse.py
│
├── pipelines/
│ ├── independent.py
│ ├── hierarchical.py
│ └── multitask.py
│
main.py


---

## 1. Preprocessing

### A. Traditional ML (aggressive)
- Lowercase
- Remove URLs / mentions
- Lemmatization (spaCy)
- Optional stopword removal (experiment)
- Emoji → text tokens
- Slang normalization

### B. Deep Learning (moderate)
- Lowercase (optional)
- Keep punctuation
- Keep stopwords
- Tokenization
- Handle OOV

### C. Transformers 
/ LLM
- Minimal cleaning
- Preserve raw text
- Only normalize encoding if needed

---

## 2. Traditional ML (PRIMARY BASELINE)

### Representation (CRITICAL)

1. TF-IDF word n-grams (1–2 / 1–3)
2. TF-IDF char n-grams (3–5)
3. Combined word + char features

### Handcrafted Features (IMPORTANT)

- Pronoun features (you / we / they / nosotros / ustedes)
- Group lexicon matches (women, LGBTQ, religion, etc.)
- Support lexicon (love, strength, ánimo, orgullo, etc.)
- Harm/violence lexicon
- Emoji indicators
- Punctuation features
- Text length

### Models

- Logistic Regression (primary)
- Linear SVM (primary)
- SGDClassifier (fast baseline)
- XGBoost (optional, dense features)

### Secondary (optional)
- FastText document embeddings
- GloVe (low priority)

---

## 3. Deep Learning

### Architectures

- TextCNN
- BiLSTM
- BiLSTM + Attention

### Embeddings

- FastText (preferred)
- GloVe
- word2vec

### Training

- AdamW
- Early stopping (macro F1)
- Class weights

---

## 4. Transformer Fine-tuning (MAIN MODEL)

### Model families

RoBERTuito for Spanish tweets
BERTweet for English tweets

#### Multilingual
- XLM-RoBERTa

### Rationale
Model selection is guided by both language coverage and domain fit.
Because SSD-2026 is defined over tweets, tweet-native encoders are expected to
better capture hashtags, mentions, emojis, abbreviations, and informal phrasing.
Thus, BERTweet and RoBERTuito are treated as in-domain monolingual models,
while BETO / Spanish RoBERTa / RoBERTa-base / DistilBERT are general-language baselines.
XLM-R serves as the main multilingual encoder for cross-lingual transfer.

### Fine-tuning comparisons

1. In-domain monolingual vs general-domain monolingual
2. Monolingual vs multilingual
3. Independent vs hierarchical vs multi-task
4. Original vs imbalance-handled training
5. With vs without harm-aware constraint handling

### Cross-lingual experiments
- EN-only train → EN test
- ES-only train → ES test
- EN+ES combined train → EN and ES test
- Optional: zero-shot / low-resource transfer with XLM-R

### Input handling
- Minimal preprocessing only
- Preserve hashtags, emojis, punctuation, and tweet-specific markers
- Normalize encoding only when necessary
- Respect model-native tokenization behavior

---

## 5. Harm / Violence Module (CRITICAL)

### Methods

1. Lexicon-based features
2. Auxiliary classifier (binary: harm vs not)
3. Post-hoc override:
   IF Support AND Harm → Not Support

Used in:
- Traditional ML (features)
- Transformers (auxiliary head or filtering)

---

## 6. Task Modeling Strategies

### A. Independent
- Train separate models per task

### B. Hierarchical (IMPORTANT)
- T1 → T2 → T3 cascade

### C. Multi-task
- Shared encoder
- Separate heads

---

## 7. Class Imbalance Handling

Focus: Task 3

Methods:
- Class weights
- Oversampling
- Paraphrase augmentation (controlled)
- Target-aware augmentation

---

## 8. Evaluation

### Metrics

- Macro F1 (primary)
- Accuracy (secondary)
- Per-class F1
- Confusion matrix

### Additional

- Per-language performance
- End-to-end pipeline evaluation
- Error propagation analysis
- Calibration (Task 1)

---

## 9. LLM-as-a-Judge (BASELINE ONLY)

### Modes

- Zero-shot
- Few-shot (3–5 examples)

### Constraints

- Output: label only
- Strict parsing
- Retry on failure
- Log invalid outputs

### Models

- One open model (e.g. Qwen or Llama)
- One closed model (e.g. GPT-5.4-mini)

---

## 10. Experiment Tracking (JSON)

All experiments logged as JSON objects.

Example:

```json
{
  "timestamp": "2026-03-17T22:00:00",
  "approach": "finetuning",
  "model": "xlm-roberta-base",
  "task": 1,
  "language": "multilingual",
  "strategy": "hierarchical",
  "macro_f1": 0.81,
  "accuracy": 0.87,
  "per_class_f1": {
    "Supportive": 0.72,
    "Non-Supportive": 0.90
  },
  "hyperparams": {
    "lr": 2e-5,
    "batch_size": 32
  },
  "notes": "with harm filter + class weights"
}
´´´

---

## 11. Implementation Order

1. Traditional ML (TF-IDF + features)
2. Transformer fine-tuning
3. Hierarchical pipeline + harm module
4. Deep learning
5. LLM judge

---

## Expected Contributions

1. Strong TF-IDF baseline with features
2. Multilingual transformer comparison
3. Hierarchical vs independent modeling
4. Harm-aware support detection
5. Target-aware imbalance handling
