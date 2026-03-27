"""Projection head and embedding extractor for contrastive fine-tuning.

Architecture (Phase B):
  Frozen MERT backbone → trainable projection head → L2-normalized embedding
  Loss: TripletMarginLoss(d(a,p) - d(a,n) + margin)

The projection head is intentionally small to avoid overfitting on a
100-piece MAESTRO subset. For Phase D (partial/full MERT unfreezing),
the backbone freeze logic lives here too.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP that maps MERT hidden states → compact embedding for retrieval.

    Default: 768 → 256 with a single hidden layer + BN + ReLU.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) → (B, output_dim), L2-normalized."""
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class MERTEmbedder(nn.Module):
    """Frozen MERT + trainable projection head.

    Extracts MERT last-hidden-state, pools over time, then projects.
    """

    def __init__(self, mert_model, projection_head: ProjectionHead, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = mert_model
        self.projection = projection_head

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_top_layers(self, n_layers: int = 2):
        """Unfreeze the last n transformer encoder layers (Phase D)."""
        encoder_layers = self.backbone.encoder.layers
        for layer in encoder_layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """input_values: (B, T) raw waveform at 24kHz → (B, output_dim) embedding."""
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            outputs = self.backbone(input_values, output_hidden_states=True)
        pooled = outputs.last_hidden_state.mean(dim=1)  # (B, 768)
        return self.projection(pooled)
