"""Dataset analysis script: distribution, lexical (bigrams/trigrams), and text length analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from collections import Counter
from pathlib import Path

# --- Config ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "Train_Data_SSD26"
FIG_DIR = Path(__file__).resolve().parent.parent / "CEURART" / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# --- Load data ---
en = pd.read_csv(DATA_DIR / "train-english.csv")
es = pd.read_csv(DATA_DIR / "train-spanish.csv")

# Normalize Spanish labels
es["task1"] = es["task1"].map({"Support": "Supportive", "Non Support": "Non-Supportive"})
es["task2"] = es["task2"].map({"Group": "Group", "Individual": "Individual", "No": "No"})

# Add word count columns
en["word_count"] = en["text"].astype(str).str.split().str.len()
es["word_count"] = es["comment"].astype(str).str.split().str.len()

# ============================================================
# 1. LABEL DISTRIBUTION — Combined EN/ES bar chart per task
# ============================================================

def plot_combined_distribution(en_col, es_col, title, fname, order=None):
    en_counts = en[en_col].value_counts()
    es_counts = es[es_col].value_counts()
    if order:
        en_counts = en_counts.reindex(order, fill_value=0)
        es_counts = es_counts.reindex(order, fill_value=0)
    labels = en_counts.index.tolist()
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.9), 3))
    bars_en = ax.bar(x - w/2, en_counts.values, w, label="English", color="#4C72B0")
    bars_es = ax.bar(x + w/2, es_counts.values, w, label="Spanish", color="#DD8452")

    for bars in [bars_en, bars_es]:
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + max(en_counts.max(), es_counts.max()) * 0.01,
                        f"{int(h)}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(FIG_DIR / fname, format="png")
    plt.close(fig)
    print(f"  Saved {fname}")

plot_combined_distribution("task1", "task1", "Subtask 1: Support Detection",
                           "dist_task1.png", ["Non-Supportive", "Supportive"])
plot_combined_distribution("task2", "task2", "Subtask 2: Target Type (Supportive only)",
                           "dist_task2.png", ["Group", "Individual", "No"])
plot_combined_distribution("task3", "task3", "Subtask 3: Community (Group only)",
                           "dist_task3.png",
                           ["Nation", "Other", "LGBTQ", "Black Community", "Women", "Religion", "No"])

# --- Combined 3-panel distribution figure ---
def plot_all_distributions_combined():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    configs = [
        ("task1", "task1", "Subtask 1", ["Non-Supportive", "Supportive"]),
        ("task2", "task2", "Subtask 2", ["Group", "Individual"]),
        ("task3", "task3", "Subtask 3", ["Nation", "Other", "LGBTQ", "Black\nCommunity", "Women", "Religion"]),
    ]
    # For task2/task3 we need filtered data
    datasets = [
        (en, es),
        (en[en["task1"] == "Supportive"], es[es["task1"] == "Supportive"]),
        (en[(en["task1"] == "Supportive") & (en["task2"] == "Group")],
         es[(es["task1"] == "Supportive") & (es["task2"] == "Group")]),
    ]
    # Label orders matching the actual data values (without newlines)
    orders_data = [
        ["Non-Supportive", "Supportive"],
        ["Group", "Individual"],
        ["Nation", "Other", "LGBTQ", "Black Community", "Women", "Religion"],
    ]

    for ax, (en_col, es_col, title, labels), (en_df, es_df), order in zip(
            axes, configs, datasets, orders_data):
        en_counts = en_df[en_col].value_counts().reindex(order, fill_value=0)
        es_counts = es_df[es_col].value_counts().reindex(order, fill_value=0)
        x = np.arange(len(labels))
        w = 0.35
        bars_en = ax.bar(x - w/2, en_counts.values, w, label="English", color="#4C72B0")
        bars_es = ax.bar(x + w/2, es_counts.values, w, label="Spanish", color="#DD8452")
        for bars in [bars_en, bars_es]:
            for b in bars:
                h = b.get_height()
                if h > 0:
                    ax.text(b.get_x() + b.get_width()/2, h + max(en_counts.max(), es_counts.max()) * 0.02,
                            f"{int(h)}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dist_combined.png", format="png")
    plt.close(fig)
    print("  Saved dist_combined.png")

plot_all_distributions_combined()

# ============================================================
# 2. TEXT LENGTH DISTRIBUTION by Task 1 label (EN & ES)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False)

for ax, df, text_col, lang in [(axes[0], en, "text", "English"), (axes[1], es, "comment", "Spanish")]:
    for label, color in [("Non-Supportive", "#4C72B0"), ("Supportive", "#DD8452")]:
        subset = df[df["task1"] == label]["word_count"]
        ax.hist(subset, bins=40, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel("Word count")
    ax.set_ylabel("Density")
    ax.set_title(f"{lang}")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Text Length Distribution by Support Label", fontsize=11, y=1.02)
fig.savefig(FIG_DIR / "text_length_dist.png", format="png")
plt.close(fig)
print("  Saved text_length_dist.png")

# ============================================================
# 3. BIGRAM / TRIGRAM ANALYSIS (English)
# ============================================================

import re
from sklearn.feature_extraction.text import CountVectorizer

# Emoji-to-text artifacts to filter out before n-gram extraction
_EMOJI_WORDS = re.compile(
    r'\b(?:face|smiling|grinning|crying|tears|joy|heart|red|heartred|joyface|'
    r'facecrying|faceloudly|loudly|open|mouth|winking|beaming|eyes|hands|'
    r'clapping|folded|raised|fist|fire|sparkles|star|struck|thumbs|'
    r'sun|moon|rainbow|wave|prayer|skull|rolling|sweat|weary|sob|'
    r'pleading|thinking|exploding|head|backhand|index|pointing|'
    r'sign|horns|love|revolving|hearts|growing|beating|broken|'
    r'exclamation|mark|question|hundred|points|collision|droplet|'
    r'flexed|biceps|victory|hand|palms|person|bowing|raising|'
    r'folding|crossed|fingers|oncoming|two|light|skin|tone|medium|'
    r'handsfolded|kissface|blowing|kiss|smiley|sticking|cheeky|playful|'
    r'raspberry|laughingrolling|floor|laughing|annoyed|undecided|uneasy|'
    r'skeptical|httpsskeptical|heartbroken|happy)\b',
    re.IGNORECASE
)

def clean_emoji_artifacts(text):
    """Remove emoji-to-text tokens so n-grams reflect actual words."""
    return _EMOJI_WORDS.sub("", text).strip()

def get_top_ngrams(corpus, n=2, top_k=15, stop_words="english"):
    cleaned = [clean_emoji_artifacts(t) for t in corpus]
    vec = CountVectorizer(ngram_range=(n, n), stop_words=stop_words,
                          max_features=10000, min_df=2)
    X = vec.fit_transform(cleaned)
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    top_idx = freqs.argsort()[-top_k:][::-1]
    return [(vocab[i], freqs[i]) for i in top_idx]

def plot_ngrams_by_class(df, text_col, label_col, classes, n, title, fname, colors=None):
    ncols = len(classes)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = [axes]
    if colors is None:
        cmap = matplotlib.colormaps["tab10"]
        colors = [cmap(i) for i in range(ncols)]

    for ax, cls, color in zip(axes, classes, colors):
        corpus = df[df[label_col] == cls][text_col].dropna().astype(str).tolist()
        if len(corpus) < 5:
            ax.set_title(f"{cls}\n(too few samples)")
            continue
        ngrams = get_top_ngrams(corpus, n=n, top_k=15)
        labels_ng = [g[0] for g in ngrams][::-1]
        counts = [g[1] for g in ngrams][::-1]
        ax.barh(range(len(labels_ng)), counts, color=color, alpha=0.8)
        ax.set_yticks(range(len(labels_ng)))
        ax.set_yticklabels(labels_ng, fontsize=7)
        ax.set_xlabel("Frequency")
        ax.set_title(cls, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, format="png")
    plt.close(fig)
    print(f"  Saved {fname}")

# Bigrams for Task 1 (EN)
plot_ngrams_by_class(en, "text", "task1", ["Non-Supportive", "Supportive"],
                     n=2, title="Top 15 Bigrams by Support Label (English)",
                     fname="bigrams_task1_en.png",
                     colors=["#4C72B0", "#DD8452"])

# Trigrams for Task 1 (EN)
plot_ngrams_by_class(en, "text", "task1", ["Non-Supportive", "Supportive"],
                     n=3, title="Top 15 Trigrams by Support Label (English)",
                     fname="trigrams_task1_en.png",
                     colors=["#4C72B0", "#DD8452"])

# Bigrams for Task 3 communities (EN, top 4 classes only)
en_group = en[en["task2"] == "Group"]
task3_classes = ["Nation", "Other", "LGBTQ", "Black Community"]
plot_ngrams_by_class(en_group, "text", "task3", task3_classes,
                     n=2, title="Top 15 Bigrams by Community (English, Group-supportive)",
                     fname="bigrams_task3_en.png")

# ============================================================
# 4. BIGRAM / TRIGRAM ANALYSIS (Spanish)
# ============================================================

# Bigrams for Task 1 (ES)
plot_ngrams_by_class(es, "comment", "task1", ["Non-Supportive", "Supportive"],
                     n=2, title="Top 15 Bigrams by Support Label (Spanish)",
                     fname="bigrams_task1_es.png",
                     colors=["#4C72B0", "#DD8452"])

# Trigrams for Task 1 (ES)
plot_ngrams_by_class(es, "comment", "task1", ["Non-Supportive", "Supportive"],
                     n=3, title="Top 15 Trigrams by Support Label (Spanish)",
                     fname="trigrams_task1_es.png",
                     colors=["#4C72B0", "#DD8452"])

# ============================================================
# 5. TEXT LENGTH STATISTICS TABLE (print for paper)
# ============================================================
print("\n=== Text Length Statistics ===")
for lang, df, text_col in [("English", en, "text"), ("Spanish", es, "comment")]:
    print(f"\n{lang}:")
    for label in ["Non-Supportive", "Supportive"]:
        sub = df[df["task1"] == label]["word_count"]
        print(f"  {label}: mean={sub.mean():.1f}, median={sub.median():.1f}, std={sub.std():.1f}, "
              f"min={sub.min()}, max={sub.max()}")

print("\nAll figures saved to:", FIG_DIR)
