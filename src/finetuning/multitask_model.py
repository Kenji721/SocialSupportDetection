"""
Multi-task classification model with shared XLM-T encoder and 3 task-specific heads.

Architecture (from research doc):
  - Shared encoder: cardiffnlp/twitter-xlm-roberta-base (278M params)
  - Head 1: Support (binary) from [CLS] embedding
  - Heads 2 & 3: Receive concat([CLS], Head1_probs) — soft label dependency
  - Each head: Linear → LayerNorm → GELU → Dropout → Linear
  - Combined loss: L = λ₁·L_support + λ₂·L_target + λ₃·L_category
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TaskHead(nn.Module):
    """MLP classification head: Linear → LayerNorm → GELU → Dropout → Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


class MultiTaskSSDModel(nn.Module):
    """Multi-task model with shared encoder and soft label dependencies.

    Head 1 predicts support from [CLS].
    Heads 2 and 3 receive concat([CLS], softmax(Head1_logits)) to capture
    the logical dependency that target/category are conditioned on support.
    """

    def __init__(
        self,
        encoder_name: str = "cardiffnlp/twitter-xlm-roberta-base",
        num_task1: int = 2,
        num_task2: int = 2,
        num_task3: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        enc_dim = self.encoder.config.hidden_size  # 768 for XLM-R base

        # Head 1: [CLS] (768) → binary support
        self.head1 = TaskHead(enc_dim, hidden_dim, num_task1, dropout)

        # Heads 2 & 3: [CLS] + Head1 probs (768 + num_task1) → task2/task3
        head23_input = enc_dim + num_task1
        self.head2 = TaskHead(head23_input, hidden_dim, num_task2, dropout)
        self.head3 = TaskHead(head23_input, hidden_dim, num_task3, dropout)

        self.num_task1 = num_task1

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits1, logits2, logits3)."""
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        logits1 = self.head1(cls)

        # Soft label dependency: concat [CLS] with Head 1 probabilities
        h1_probs = F.softmax(logits1, dim=-1).detach()  # detach to avoid gradient flow back through probs
        cls_augmented = torch.cat([cls, h1_probs], dim=-1)

        logits2 = self.head2(cls_augmented)
        logits3 = self.head3(cls_augmented)

        return logits1, logits2, logits3

    def freeze_encoder_layers(self, num_freeze: int = 8):
        """Freeze the bottom N transformer layers + embeddings (Phase 1)."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze bottom layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        print(f"Frozen {frozen}/{total} parameter groups "
              f"(bottom {num_freeze} layers + embeddings)")

    def unfreeze_all(self):
        """Unfreeze all parameters (Phase 2)."""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")

    def get_layer_groups(self) -> list[dict]:
        """Return parameter groups for discriminative learning rates.

        Groups: embeddings, layer 0-3, layer 4-7, layer 8-11, heads.
        """
        groups = []

        # Embeddings (lowest LR)
        groups.append({
            "params": list(self.encoder.embeddings.parameters()),
            "name": "embeddings",
        })

        # Transformer layers in chunks of 4
        layers = list(self.encoder.encoder.layer)
        for start in range(0, len(layers), 4):
            end = min(start + 4, len(layers))
            params = []
            for layer in layers[start:end]:
                params.extend(layer.parameters())
            groups.append({
                "params": params,
                "name": f"layers_{start}-{end-1}",
            })

        # Classification heads (highest LR)
        head_params = (
            list(self.head1.parameters())
            + list(self.head2.parameters())
            + list(self.head3.parameters())
        )
        groups.append({
            "params": head_params,
            "name": "heads",
        })

        return groups


def multitask_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    logits3: torch.Tensor,
    labels1: torch.Tensor,
    labels2: torch.Tensor,
    labels3: torch.Tensor,
    weights1: torch.Tensor | None = None,
    weights2: torch.Tensor | None = None,
    weights3: torch.Tensor | None = None,
    lambdas: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined multi-task loss, masking invalid labels with -100.

    Returns (total_loss, loss1, loss2, loss3).
    """
    device = logits1.device

    # Task 1: all samples have labels
    ce1 = nn.CrossEntropyLoss(weight=weights1.to(device) if weights1 is not None else None)
    loss1 = ce1(logits1, labels1)

    # Task 2: only Supportive samples (labels2 != -100)
    mask2 = labels2 != -100
    if mask2.any():
        ce2 = nn.CrossEntropyLoss(weight=weights2.to(device) if weights2 is not None else None)
        loss2 = ce2(logits2[mask2], labels2[mask2])
    else:
        loss2 = torch.tensor(0.0, device=device)

    # Task 3: only Group samples (labels3 != -100)
    mask3 = labels3 != -100
    if mask3.any():
        ce3 = nn.CrossEntropyLoss(weight=weights3.to(device) if weights3 is not None else None)
        loss3 = ce3(logits3[mask3], labels3[mask3])
    else:
        loss3 = torch.tensor(0.0, device=device)

    total = lambdas[0] * loss1 + lambdas[1] * loss2 + lambdas[2] * loss3
    return total, loss1, loss2, loss3
