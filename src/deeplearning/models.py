"""
Deep learning models for text classification: TextCNN, BiLSTM, BiLSTM+Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Kim (2014) CNN for text classification.

    Multiple parallel convolutional filters with different kernel sizes,
    followed by max-over-time pooling and a fully connected classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        num_classes: int = 2,
        filter_sizes: tuple = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch, emb_dim, seq_len)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))  # (batch, num_filters, seq_len - fs + 1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outs.append(c)

        out = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        out = self.dropout(out)
        return self.fc(out)


class BiLSTM(nn.Module):
    """Bidirectional LSTM for text classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Concatenate last hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim * 2)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class BiLSTMAttention(nn.Module):
    """BiLSTM with self-attention for text classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 1,
        dropout: float = 0.3,
        pretrained_embeddings: torch.Tensor = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        mask = (x != 0).float()  # (batch, seq_len)

        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim * 2)

        # Attention weights
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim * 2)
        context = self.dropout(context)
        return self.fc(context)
