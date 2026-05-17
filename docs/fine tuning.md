# Building a local multilingual social support classifier on Apple Silicon

**A fine-tuned encoder model—not a large decoder LLM—is your best path to high-accuracy, low-latency classification of bilingual tweets on an M4 Mac.** Specifically, `cardiffnlp/twitter-xlm-roberta-base` (XLM-T), a 278M-parameter XLM-RoBERTa variant pretrained on 198 million multilingual tweets, offers the strongest combination of tweet-domain knowledge, English/Spanish parity, and practical efficiency for this task. Paired with a multi-task classification architecture, LoRA or full fine-tuning via Hugging Face Transformers on MPS, and targeted data augmentation, you can build a production-quality system entirely on-device with 1K–5K labeled examples.

This report covers model selection, architecture design, fine-tuning workflows, and evaluation strategy—all grounded in recent shared-task results and 2024–2025 research.

---

## Encoder models dominate decoder LLMs for classification with limited labeled data

The single most important architectural decision is choosing an **encoder model** over a decoder/generative LLM. Multiple 2024–2025 studies converge on this finding:

- Fine-tuned encoders outperform zero-shot and few-shot LLMs by **10–25% accuracy** across classification benchmarks. Even GPT-4 in few-shot mode generally trails fine-tuned BERT-family models on text classification.
- A September 2025 study on mmBERT concluded that "encoder-only models are significantly better for classification than decoder-only models for a given size, even beating decoders an order of magnitude larger."
- Inference speed is **20–50× faster** with encoder models: BERT-class models process ~277 samples/sec versus ~12 samples/sec for a 2B decoder.
- An arXiv paper titled "Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification" (2406.08660) confirms this pattern holds even against the latest generative models.

The one scenario where decoder LLMs win is **zero labeled data**. With your 1K–5K examples, an encoder is the clear choice. A decoder model like Qwen 2.5 7B is better reserved for data augmentation (generating synthetic training examples) rather than serving as the production classifier.

### The best encoder models for bilingual tweet classification

| Model | Params | Key strength | EN/ES support |
|-------|--------|-------------|---------------|
| **`cardiffnlp/twitter-xlm-roberta-base`** (XLM-T) | 278M | Pretrained on 198M multilingual tweets across 30+ languages; tweet-domain adapted | ✅ Native |
| **`microsoft/mdeberta-v3-base`** | 276M | 3.6% better than XLM-R on XNLI; disentangled attention + ELECTRA-style training | ✅ 100 languages |
| **`jhu-clsp/mmbert-base`** | 307M | Newest (Sep 2025); outperforms XLM-R on XTREME/GLUE/MTEB; Flash Attention, 8192 context | ✅ 1833 languages |
| **`pysentimiento/robertuito-base-uncased`** | ~125M | Pretrained on 500M Spanish tweets; best Spanish social media model | Spanish-focused (some EN transfer) |
| **`Twitter/twhin-bert-large`** | ~355M | Trained on 7B tweets with social graph objective | ✅ 100+ languages |

**Top recommendation:** Start with `cardiffnlp/twitter-xlm-roberta-base`. Domain-specific pretraining on tweets matters more than topical pretraining—BERTweet outperforms HateBERT (pretrained on banned Reddit content) for tweet-based hate speech, and RoBERTuito outperforms all other Spanish PLMs. XLM-T gives you this domain advantage across both languages simultaneously. If you want to experiment, `jhu-clsp/mmbert-base` (mmBERT) is the newest and strongest general multilingual encoder, though it lacks tweet-domain pretraining.

### When decoder models make sense

Use a **7–9B decoder model as a supporting tool**, not as your primary classifier:

- **Data augmentation**: Qwen 2.5 7B Instruct (best structured JSON output, 30+ languages including Spanish) or Aya Expanse 8B (purpose-built for 23 languages) can generate synthetic training examples for underrepresented categories.
- **Zero-shot baseline**: Before fine-tuning, test a decoder model's zero-shot classification to establish a baseline and identify hard examples.
- **Annotation assistance**: Use a decoder model to pre-label unlabeled data for human review.

For decoder inference on M4, use **MLX** or **llama.cpp** with 4-bit quantized GGUF models (~5 GB for an 8B model).

---

## Multi-task learning with three classification heads is the right architecture

Your three tasks—supportive/non-supportive, individual/group, and target category—share deep semantic structure. A person writing hateful content about a religious group activates overlapping linguistic features for all three labels. Multi-task learning exploits this overlap.

**Research strongly supports multi-task over single-task for hate speech:**

- MT-DNN (Liu et al., 2019) showed multi-task BERT outperformed single-task BERT on 8/9 GLUE tasks, with **larger improvements on smaller datasets**—directly relevant to your 1K–5K setting.
- A multi-task approach for Spanish hate speech using BETO achieved **86.58% macro-F1**, outperforming all prior single-task systems on HatEval.
- The EXIST 2025 shared task winner (GrootWatch) used a multi-task headed BERT model, ranking **1st in all soft-soft evaluations**.
- MTL acts as implicit regularization, forcing the shared encoder to learn general features and reducing overfitting—critical with small datasets.

### Recommended architecture with soft label dependencies

```
                    ┌──────────────────────────────────────────┐
                    │     XLM-T / twitter-xlm-roberta-base     │
                    │          (shared encoder, 278M)           │
                    └──────────────┬───────────────────────────┘
                                   │ [CLS] embedding (768-dim)
                    ┌──────────────┴───────────────────────────┐
                    │                                           │
              ┌─────▼─────┐                          ┌─────────▼──────────┐
              │  Head 1:   │                          │                    │
              │ Support    │──── P(support) ────────▶ │  Concat [CLS] +   │
              │ (binary)   │                          │  Head 1 probs      │
              └────────────┘                          └────────┬───────────┘
                                                    ┌──────────┴──────────┐
                                              ┌─────▼─────┐        ┌─────▼─────┐
                                              │  Head 2:   │        │  Head 3:   │
                                              │ Ind/Group  │        │ Category   │
                                              │ (binary)   │        │ (multiclass)│
                                              └────────────┘        └────────────┘
```

Head 1 predicts supportive/non-supportive from the [CLS] token. Heads 2 and 3 receive the **concatenation of the [CLS] embedding and Head 1's softmax probabilities**, capturing the logical dependency that target type and category are conditioned on the support/hate classification. All predictions happen in a **single forward pass**—no cascading errors, no sequential inference latency.

Each head is a small MLP: `Linear(770, 256) → LayerNorm → GELU → Dropout(0.1) → Linear(256, num_classes)`. The combined loss is `L = λ₁·L_support + λ₂·L_target + λ₃·L_category`, where loss weights can be tuned or dynamically adjusted via uncertainty weighting.

**Why not three separate models?** Three XLM-RoBERTa models mean 3× memory at inference (~3.3 GB versus ~1.1 GB), 3× latency, and—critically—each model trains on fewer effective gradient signals. With only 1K–5K samples, you cannot afford to waste the shared representation learning that MTL provides.

**Why not a single decoder with JSON output?** A decoder model would be 20–50× slower at inference, harder to fine-tune on small data (8B parameters vs. 278M), and research shows that imposing JSON format constraints can degrade LLM reasoning quality (Tam et al., 2024).

---

## Fine-tuning workflow optimized for Apple M4 with MPS

### Step 1: Prepare environment and data

**Environment setup:**
```bash
# Use native ARM64 Python 3.12, NOT Rosetta
pip install torch transformers datasets peft accelerate setfit
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Essential for MPS
```

**Data preparation** for the multi-task encoder: Create a single dataset with three label columns. Split into train/validation (80/20) with **iterative stratification** (using `skmultilearn.model_selection.iterative_train_test_split`) to preserve the joint label distribution across splits. For robust evaluation with small data, use **stratified 5-fold cross-validation** and report mean ± standard deviation.

### Step 2: Evaluate pretrained starting points

Before fine-tuning, test these already-trained models as baselines to understand your task's difficulty:

- **`cardiffnlp/twitter-xlm-roberta-base-hate-spanish`** — fine-tuned on HaterNet + HatEval Spanish for hate speech detection
- **`pysentimiento/robertuito-hate-speech`** — fine-tuned on HatEval Subtask B (hateful, targeted, aggressive labels—closely aligned with your tasks)
- **`cardiffnlp/twitter-roberta-base-hate-multiclass-latest`** — English multiclass model with labels including sexism, racism, disability, sexual orientation, religion (overlaps your target categories)
- **`unitary/multilingual-toxic-xlm-roberta`** — multilingual toxic comment classifier covering 7 languages including Spanish

These baselines will reveal which aspects of your task are already well-captured by existing models and where fine-tuning is most needed.

### Step 3: Fine-tune the multi-task model

For an encoder model of ~278M parameters, **full fine-tuning is practical on any M4 Mac** (requires only ~3–5 GB of unified memory). LoRA is unnecessary at this model size—it adds complexity without meaningful memory savings.

**Two-phase training strategy (research-backed for small datasets):**

**Phase 1 — Classifier warm-up (2–4 epochs):** Freeze the bottom 8 of 12 transformer layers plus the embedding layer. Train only the top 4 layers and the three classification heads. Use learning rate **1e-4** for the classification heads. This prevents the randomly initialized heads from corrupting pretrained representations.

**Phase 2 — Full fine-tuning (5–10 epochs):** Unfreeze all parameters. Use a lower learning rate (**2e-5**) with linear warmup over 10% of steps. Apply **discriminative learning rates**—lower layers get smaller LR (e.g., 1e-5) than upper layers (2e-5).

**Key hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 16–32 (gradient accumulation if needed) | Standard for encoder fine-tuning |
| Optimizer | AdamW | Default, well-tested |
| Weight decay | 0.01 | Standard regularization |
| Max sequence length | 128 tokens | Sufficient for tweets |
| Early stopping | On validation macro-F1, patience=3 | Prevents overfitting on small data |
| Warmup | 10% of total steps | Stabilizes early training |

**MPS-specific training configuration:**
```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=10,
    bf16=True,                    # Use bf16, NOT fp16 on Apple Silicon
    fp16=False,                   # fp16 causes issues on MPS
    dataloader_num_workers=0,     # MUST be 0 — multiprocessing breaks MPS
    dataloader_pin_memory=False,  # Prevents memory conflicts
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
)
```

**Critical MPS notes:** Upgrade to **macOS 15+** and **PyTorch 2.4+** to fix a silent correctness bug in `addcmul_` and `addcdiv_` that caused model weights to freeze during training. The first epoch will be slower due to Metal shader compilation—this is normal. MPS provides **2–5× speedup over CPU** for transformer training with batch size ≥ 4.

### Alternative fast path: SetFit

If you want a working classifier in under an hour, **SetFit** is remarkably effective with small datasets. It uses contrastive fine-tuning of a sentence transformer followed by a lightweight classifier head. With just **8 labeled examples per class**, SetFit approaches full-data fine-tuning performance. With your 1K–5K examples, it will be highly competitive.

```python
from setfit import SetFitModel, SetFitTrainer
model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
# Train separate SetFit models for each task, or use multi-label mode
```

SetFit trains in minutes even on CPU, uses standard PyTorch (MPS-compatible), and requires no prompt engineering. Consider it as a rapid prototyping baseline before investing in the full multi-task architecture.

---

## Data augmentation can double your effective dataset size

With 1K–5K labeled examples, augmentation yields the highest marginal returns. Back-translation alone produces **4–8% accuracy gains** with 50–500 training samples. Here is the recommended pipeline, ordered by expected impact:

**Back-translation (highest priority):** Translate each English tweet to Spanish and back (EN→ES→EN), and vice versa (ES→EN→ES), using Helsinki-NLP MarianMT models available on Hugging Face. Generate 2–3 back-translations per sample. This naturally creates cross-lingual paraphrases and implicitly builds cross-lingual training signal. With EN↔ES being relatively close languages, translation quality is high.

**LLM synthetic generation for minority classes:** Use Qwen 2.5 7B (running locally via MLX at 4-bit quantization) to generate diverse examples for underrepresented target categories like Disability or LGBTQ. Provide 5–10 real examples as few-shot context and request variations. Recent research shows **3–26% F1 improvements** when augmenting 100 real samples with 100 synthetic samples, though gains diminish with larger base datasets. Use synthetic data as a supplement—never a substitute.

**Oversampling underrepresented classes:** Apply class-weighted loss functions or SMOTE-style oversampling to address inevitable class imbalance in target categories. OffensEval shared task analyses identified class imbalance as the central challenge, with categories like "Other" severely underrepresented.

**Code-switching augmentation:** If your real data contains EN/ES code-switching (common in bilingual communities), randomly replace words in English tweets with Spanish equivalents using bilingual dictionaries. This mirrors natural usage patterns and has shown improvements in code-mixed benchmarks.

---

## Evaluation must go beyond accuracy to capture what matters

**Macro-F1 should be your primary metric.** It is the official ranking metric for SemEval HatEval, OffensEval, and HASOC shared tasks because it treats all classes equally regardless of prevalence. A model predicting "not hateful" for everything achieves 60–70% accuracy on typical hate speech datasets but scores near zero on macro-F1 for the hate class.

**Full evaluation suite:**

- **Macro-F1** (primary ranking metric) — equal treatment of all classes
- **Per-class F1** — essential for identifying which target categories the model struggles with (expect Disability and LGBTQ to be hardest due to fewer examples)
- **Per-language macro-F1** — track the EN/ES performance gap; a gap >5% signals the model needs more Spanish data or augmentation
- **Exact Match Ratio** — the fraction of samples where all three labels are predicted correctly; this is the strictest multi-label metric
- **Precision-recall curves** — critical for deployment decisions: optimize for recall in human-review pipelines, precision in automated systems

**Use stratified 5-fold cross-validation** with iterative stratification (Sechidis et al., 2011) to handle the multi-label structure. Report mean ± standard deviation. With only 1K–5K samples, a single train/test split produces unreliable estimates.

**Diagnostic testing:** Use **Multilingual HateCheck** (Röttger et al., 2022)—3,728 functional test cases across 34 functionalities in 10 languages including Spanish—to probe specific model weaknesses like handling of negation, counter-speech, slurs, and spelling variations.

---

## Existing models and datasets provide strong starting points

Several pretrained models and datasets align closely with your task and can dramatically reduce the cold-start problem.

**Directly relevant pretrained models:** The `pysentimiento/robertuito-hate-speech` model was trained on HatEval Subtask B with labels for hateful, targeted, and aggressive—a three-label annotation scheme closely mirroring your supportive/non-supportive, individual/group, and category structure. The `cardiffnlp/twitter-roberta-base-hate-multiclass-latest` model classifies by target category (sexism, racism, disability, sexual orientation, religion) with labels that overlap substantially with your target categories. Starting from these checkpoints and continuing fine-tuning on your data (a technique called **further fine-tuning** or **domain-adaptive fine-tuning**) can save significant training time and improve performance.

**Key datasets for pretraining or augmentation:**

- **HatEval (SemEval-2019 Task 5):** 19,600 tweets in English and Spanish, annotated for hate speech, target type (individual/group), and aggression—the closest existing dataset to your annotation scheme. Available on Hugging Face as `valeriobasile/HatEval`.
- **Spanish Hate Speech Superset** (`manueltonneau/spanish-hate-speech-superset`): 29,855 Spanish posts merged from all publicly available Spanish hate speech datasets.
- **EXIST (2021–2025):** Bilingual EN/ES annotated tweets for sexism identification with multi-label annotations across source intention and categorization.
- **SuperTweetEval:** Heterogeneous benchmark with multiclass hate speech by target (gender, race, sexuality, religion, origin, disability, age).

**Recent competition insights from EXIST 2025:** The winning system (GrootWatch) used a multi-task headed BERT model—validating the architecture recommended in this report. XLM-RoBERTa fine-tuned with additional social media datasets performed best for binary classification. These results confirm that **encoder-based multi-task architectures remain state of the art** for bilingual social media classification, even as LLMs continue to advance.

---

## Conclusion

The optimal system for this task pairs a tweet-domain multilingual encoder (`twitter-xlm-roberta-base`) with a multi-task architecture and careful fine-tuning on Apple Silicon. Three key insights that should guide implementation:

**Encoders, not decoders.** The research consensus is unambiguous for classification with labeled data. An encoder model is faster (20–50×), more memory-efficient (278M vs. 8B parameters), and more accurate when fine-tuned on 1K–5K examples. Reserve decoder models for data augmentation.

**Multi-task learning is not optional at this data scale.** With limited labeled examples, the regularization benefit of joint training across three related tasks materially improves generalization. The EXIST 2025 winning approach and multiple 2024 studies confirm that shared encoder + task-specific heads outperforms independent classifiers for hate speech subtasks.

**The Apple Silicon ecosystem has matured enough.** PyTorch MPS support for encoder fine-tuning is stable with the right configuration (`bf16=True`, `dataloader_num_workers=0`, macOS 15+). XLM-RoBERTa-base requires only ~3–5 GB for training—comfortable on even the base M4. For any decoder model work (augmentation, baselines), MLX is the purpose-built framework and consistently outperforms PyTorch MPS for LLM workloads on Apple Silicon. Avoid Unsloth and Axolotl, which lack proper MPS support.