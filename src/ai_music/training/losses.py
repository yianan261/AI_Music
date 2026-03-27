"""Loss functions for contrastive training.

Phase B: TripletMarginLoss (simple, proven baseline)
Phase C: InfoNCE / CLEWS-style contrastive loss (future)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Standard triplet margin loss with L2 distance.

    L = max(d(a,p) - d(a,n) + margin, 0)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(anchor, positive, negative)


class HardTripletLoss(nn.Module):
    """In-batch hard triplet mining (à la CoverHunter).

    For each anchor, finds the hardest positive (farthest same-class)
    and hardest negative (closest different-class) within the batch.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """embeddings: (B, D), labels: (B,) integer piece IDs."""
        dist = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

        same_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff_mask = ~same_mask
        eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)

        pos_dist = dist.clone()
        pos_dist[~same_mask | eye] = 0.0
        hardest_pos, _ = pos_dist.max(dim=1)

        neg_dist = dist.clone()
        neg_dist[~diff_mask] = float("inf")
        hardest_neg, _ = neg_dist.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()
