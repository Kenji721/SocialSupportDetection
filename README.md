# Social Support Detection

This repository contains the code and experiment records for our SSD-2026 shared-task system for detecting social support in English and Spanish social media text.

The accompanying paper is titled **"Comparing Transformer Fine-Tuning, Multi-Task Learning, and LLM-as-Judge for Social Support Detection"**. The project studies whether models can identify not only whether a comment is supportive, but also who the support is directed toward and which community is being supported.

## Paper Summary

Social support detection is different from standard sentiment analysis. A comment can be emotional without being supportive, supportive without being strongly positive, or positive in a way that is not directed at anyone. In this project, we treat social support as a hierarchical NLP task:

- **Subtask 1:** detect whether a comment is Supportive or Non-Supportive.
- **Subtask 2:** if supportive, classify whether the target is an Individual or a Group.
- **Subtask 3:** if group-targeted, classify the supported community: Nation, Other, LGBTQ, Black Community, Women, or Religion.

The SSD-2026 dataset contains **7,998 English** and **2,600 Spanish** training comments. The task is highly imbalanced: in English, only 1,795 comments are supportive, and some Subtask 3 classes have fewer than 20 training examples.

## Research Questions

The paper focuses on three main questions:

1. How do domain-adapted transformers compare with general multilingual models and LLM-based judging for social support detection?
2. Does multilingual training help when some classes are extremely underrepresented?
3. Can an LLM complement fine-tuned classifiers, especially for minority classes where supervised data is scarce?

## Modeling Approaches

We compared five modeling paradigms:

- **Traditional ML:** TF-IDF with Logistic Regression and Linear SVM.
- **Deep learning:** TextCNN and BiLSTM with pretrained FastText embeddings.
- **Single-task transformers:** BERTweet for English, RoBERTuito for Spanish, and XLM-R for English, Spanish, and multilingual training.
- **Multi-task learning:** a shared XLM-Twitter encoder with separate heads for the three subtasks.
- **LLM-as-judge:** GPT-5.4-mini with emotion-aware few-shot prompting.

For final submissions, we used hierarchical pipelines: Subtask 1 predictions decide which examples continue to Subtask 2, and Subtask 2 predictions decide which examples continue to Subtask 3.

## Key Results

Official CodaBench scores:

| Language | Best score | Best strategy |
| --- | ---: | --- |
| English | **0.8231** | Three-way tie across English pipeline variants |
| Spanish | **0.8785** | Full Spanish pipeline |

Important findings from the paper:

- **Domain-adapted transformers were strongest overall.** BERTweet performed best for English support detection, while RoBERTuito was especially strong for Spanish target and community classification.
- **Multilingual training helped minority classes.** XLM-R trained on English + Spanish improved English Subtask 3 by 12.5 F1 points over English-only training.
- **The LLM judge was useful for analysis.** GPT-5.4-mini performed well on minority-class examples in internal evaluation, especially for Religion and Women, where fine-tuned models had very little training data.
- **Hybrid LLM verification was weaker than expected.** When BERTweet labels were shown to the LLM as pre-labels, the LLM over-agreed with them, creating anchoring bias instead of reliably correcting mistakes.
- **Prompt wording mattered.** Explicitly treating admiration directed at a person as support recovered many false negatives.

## Repository Structure

```text
src/
  traditional/      TF-IDF baselines
  deeplearning/     FastText + TextCNN/BiLSTM models
  finetuning/       transformer and multi-task training
  llm_judge/        LLM-as-judge prompts and parsing
  pipelines/        hierarchical prediction logic
  data/             data loading and label utilities
results/            lightweight experiment logs and summaries
docs/               project notes and planning documents
scripts/            analysis scripts
notebooks/          exploratory notebooks
```

Large local files are intentionally excluded from GitHub: raw data, trained checkpoints, generated artifacts, CodaBench submissions, logs, and CEUR paper build files.

## Setup

This project uses Python 3.10+.

```bash
uv sync
```

If using plain `pip`, install the dependencies from `pyproject.toml`.

## Expected Data Layout

The training scripts expect the SSD-2026 data under:

```text
data/
  Train_Data_SSD26/
    train-english.csv
    train-spanish.csv
  Test_phase_data/
    test_phase_english.csv
    test_phase_spanish.csv
```

Deep learning experiments also expect downloaded embeddings such as FastText under `data/`.

## Training Entry Points

Traditional TF-IDF baselines:

```bash
python -m src.traditional.train
```

FastText-based neural models:

```bash
python -m src.deeplearning.train
```

Transformer fine-tuning grid:

```bash
python -m src.finetuning.train
```

Multi-task transformer training:

```bash
python -m src.finetuning.train_multitask
```

The scripts write large artifacts to `artifacts/` and compact experiment logs to `results/`.

## Experiment Records

The main lightweight records are:

- `results/experiments_traditional.jsonl`
- `results/experiments_deeplearning.jsonl`
- `results/experiments_finetuning.jsonl`
- `results/experiments_multitask.jsonl`
- `results/experiments_llm_judge.jsonl`
- `results/results-finetuning.md`

## Excluded Local Files

The following are intentionally ignored:

- `data/`: shared-task data and downloaded embeddings.
- `artifacts/`: trained model outputs, metrics dumps, and prediction artifacts.
- `task*_model/` and `adapters/`: saved checkpoints and fine-tuned adapter weights.
- `submissions/`: generated CodaBench submission files.
- `CEURART/`, `CEURART_corregido/`, `CEURART.zip`: submitted paper source/template/build outputs.
- `.env` and `*.log`: local secrets and run logs.
