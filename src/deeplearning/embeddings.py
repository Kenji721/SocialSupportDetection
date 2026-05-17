"""
Pretrained word embedding loaders for deep learning models.
"""

import os
from collections import Counter

import numpy as np
from tqdm import tqdm


def build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    """Build a word→index vocabulary from a list of texts.

    Index 0 is reserved for <PAD>, index 1 for <UNK>.
    """
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, freq in counter.most_common():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab


def texts_to_indices(texts: list[str], vocab: dict[str, int], max_len: int = 128) -> np.ndarray:
    """Convert texts to padded index sequences."""
    unk_idx = vocab.get("<UNK>", 1)
    pad_idx = vocab.get("<PAD>", 0)

    result = np.full((len(texts), max_len), pad_idx, dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = text.lower().split()[:max_len]
        for j, token in enumerate(tokens):
            result[i, j] = vocab.get(token, unk_idx)
    return result


def load_fasttext(path: str = None, dim: int = 300) -> dict[str, np.ndarray]:
    """Load FastText embeddings from a .vec text file.

    Args:
        path: Path to .vec file. If None, looks for cc.en.300.vec in data/.
        dim: Embedding dimension.

    Returns:
        dict mapping word → numpy vector.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "cc.en.300.vec",
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FastText embeddings not found at {path}. "
            "Download from https://fasttext.cc/docs/en/crawl-vectors.html "
            "and place the .vec file in data/"
        )

    embeddings = {}
    print(f"Loading FastText embeddings from {path}...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        # Skip header line
        header = f.readline()
        for line in tqdm(f, desc="Loading embeddings"):
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    print(f"Loaded {len(embeddings)} word vectors.")
    return embeddings


def load_glove(path: str = None, dim: int = 300) -> dict[str, np.ndarray]:
    """Load GloVe embeddings from a text file.

    Args:
        path: Path to GloVe file. If None, looks for glove.6B.300d.txt in data/.
        dim: Embedding dimension.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "glove.6B.300d.txt",
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GloVe embeddings not found at {path}. "
            "Download from https://nlp.stanford.edu/projects/glove/"
        )

    embeddings = {}
    print(f"Loading GloVe embeddings from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading embeddings"):
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
    print(f"Loaded {len(embeddings)} word vectors.")
    return embeddings


def build_embedding_matrix(
    vocab: dict[str, int],
    embeddings: dict[str, np.ndarray],
    dim: int = 300,
) -> np.ndarray:
    """Build an embedding matrix from a vocab and pretrained embeddings.

    Returns:
        numpy array of shape (vocab_size, dim). Words not found in embeddings
        are initialized with small random values.
    """
    vocab_size = len(vocab)
    matrix = np.random.uniform(-0.05, 0.05, (vocab_size, dim)).astype(np.float32)
    matrix[0] = 0.0  # <PAD> is zeros

    found = 0
    for word, idx in vocab.items():
        if word in embeddings:
            matrix[idx] = embeddings[word]
            found += 1

    coverage = found / vocab_size * 100
    print(f"Embedding coverage: {found}/{vocab_size} ({coverage:.1f}%)")
    return matrix
