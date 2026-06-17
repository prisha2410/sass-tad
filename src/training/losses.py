"""
Multi-task losses for SASS-TAD training.

Three loss components:
    1. Boundary BCE — main classification loss
    2. Contrastive loss — style embedding quality
    3. Topic adversarial loss — topic invariance via GRL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BoundaryLoss(nn.Module):
    """Weighted BCE for boundary detection (handles class imbalance)."""
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, logits, targets, mask=None):
        """
        logits: (B, N-1)
        targets: (B, N-1) float
        mask: (B, N-1) bool — True for valid positions
        """
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp(min=1)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss on style embeddings.

    For each boundary position, the two adjacent sentence embeddings
    should be CLOSE if same-author (label=0) and FAR if diff-author (label=1).
    Uses cosine similarity with a margin.
    """
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, style_emb, boundary_labels, mask=None):
        """
        style_emb: (B, N, D) — L2-normalized style embeddings
        boundary_labels: (B, N-1) — 0=same author, 1=different author
        mask: (B, N-1) — valid positions
        """
        left = style_emb[:, :-1, :]   # (B, N-1, D)
        right = style_emb[:, 1:, :]   # (B, N-1, D)

        cos_sim = F.cosine_similarity(left, right, dim=-1)  # (B, N-1), range [-1, 1]

        # Same author (label=0): maximize similarity → minimize (1 - cos_sim)
        # Diff author (label=1): minimize similarity → minimize max(0, cos_sim - margin)
        same_loss = (1 - boundary_labels) * (1 - cos_sim)
        diff_loss = boundary_labels * F.relu(cos_sim - self.margin)

        loss = same_loss + diff_loss

        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp(min=1)
        return loss.mean()


class TopicAdversarialLoss(nn.Module):
    """Cross-entropy for topic discrimination (used with GRL)."""
    def __init__(self):
        super().__init__()

    def forward(self, topic_logits, topic_labels, mask=None):
        """
        topic_logits: (B, N, n_topics)
        topic_labels: (B, N) long — topic cluster IDs
        mask: (B, N) bool
        """
        B, N, C = topic_logits.shape
        logits_flat = topic_logits.reshape(-1, C)
        labels_flat = topic_labels.reshape(-1)

        loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")
        loss = loss.reshape(B, N)

        if mask is not None:
            loss = loss * mask.float()
            return loss.sum() / mask.float().sum().clamp(min=1)
        return loss.mean()


class SASTADLoss(nn.Module):
    """
    Combined multi-task loss for SASS-TAD.

    L = α * L_boundary + β * L_contrastive + γ * L_topic

    Default weights follow curriculum schedule:
        - Start: heavy on boundary + contrastive, low adversarial
        - Ramp adversarial weight over epochs
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        pos_weight: float = 1.0,
        contrastive_margin: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.boundary_loss = BoundaryLoss(pos_weight)
        self.contrastive_loss = ContrastiveLoss(contrastive_margin)
        self.topic_loss = TopicAdversarialLoss()

    def forward(
        self,
        boundary_logits,
        boundary_labels,
        style_embeddings,
        topic_logits=None,
        topic_labels=None,
        boundary_mask=None,
        sentence_mask=None,
    ):
        losses = {}

        l_boundary = self.boundary_loss(boundary_logits, boundary_labels, boundary_mask)
        losses["boundary"] = l_boundary

        l_contrastive = self.contrastive_loss(style_embeddings, boundary_labels, boundary_mask)
        losses["contrastive"] = l_contrastive

        total = self.alpha * l_boundary + self.beta * l_contrastive

        if topic_logits is not None and topic_labels is not None:
            l_topic = self.topic_loss(topic_logits, topic_labels, sentence_mask)
            losses["topic"] = l_topic
            total = total + self.gamma * l_topic

        losses["total"] = total
        return losses
