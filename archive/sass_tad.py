"""
SASS-TAD: Sequence-Aware Stylometric-Semantic Fusion for
Topic-Controlled Authorship Change Detection.

Architecture:
    [SBERT(384) | Stylometric(~182)] → FusionMLP → ContrastiveEncoder → BiLSTM → BoundaryHead

Training uses three losses:
    1. BCE on boundary predictions (main task)
    2. Contrastive loss on style embeddings (same vs diff author)
    3. Adversarial topic loss via GRL (force topic-invariance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, Optional, Tuple


class GradientReversalFunction(Function):
    """Gradient Reversal Layer — flips gradient sign during backprop."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class FusionLayer(nn.Module):
    """Projects concatenated [SBERT, stylometric] into a shared space."""
    def __init__(self, sbert_dim=384, style_dim=182, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sbert_dim + style_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sbert_emb, style_vec):
        # sbert_emb: (B, N, 384), style_vec: (B, N, style_dim)
        x = torch.cat([sbert_emb, style_vec], dim=-1)
        return self.proj(x)  # (B, N, fusion_dim)


class ContrastiveStyleEncoder(nn.Module):
    """
    Projects fused representations into a style-only embedding space.
    Trained with contrastive loss to separate same-author from diff-author pairs.
    """
    def __init__(self, input_dim=256, style_embed_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, style_embed_dim),
            nn.LayerNorm(style_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(style_embed_dim, style_embed_dim),
        )

    def forward(self, x):
        # x: (B, N, input_dim)
        return F.normalize(self.encoder(x), p=2, dim=-1)  # L2-normalized


class TopicDiscriminator(nn.Module):
    """
    Adversarial topic classifier — sits behind GRL.
    Predicts topic cluster from style embeddings.
    If the style encoder is doing its job, this should fail.
    """
    def __init__(self, input_dim=128, n_topics=50, dropout=0.3):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_topics),
        )

    def forward(self, style_emb, lambda_=None):
        if lambda_ is not None:
            self.grl.set_lambda(lambda_)
        x = self.grl(style_emb)
        return self.classifier(x)  # (B, N, n_topics)


class BoundaryBiLSTM(nn.Module):
    """
    Sequence-aware boundary detector.
    Takes style embeddings for all sentences in a document,
    outputs per-boundary (between consecutive sentences) predictions.
    """
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # For boundary i (between sentence i and i+1), concat hidden states
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # *4 because bidirectional + two positions
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, style_emb, lengths=None):
        """
        style_emb: (B, N, input_dim) — per-sentence style embeddings
        lengths: (B,) — actual sequence lengths (for packing)

        Returns: (B, N-1) logits for each boundary
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                style_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(style_emb)

        # Boundary features: concat adjacent hidden states
        left = lstm_out[:, :-1, :]   # (B, N-1, 2*hidden)
        right = lstm_out[:, 1:, :]   # (B, N-1, 2*hidden)
        boundary_repr = torch.cat([left, right], dim=-1)  # (B, N-1, 4*hidden)

        logits = self.boundary_head(boundary_repr).squeeze(-1)  # (B, N-1)
        return logits


class SASSTAD(nn.Module):
    """
    Full SASS-TAD model.

    Forward pass returns a dict of outputs for multi-task training:
        - boundary_logits: (B, N-1) — main task
        - style_embeddings: (B, N, style_dim) — for contrastive loss
        - topic_logits: (B, N, n_topics) — for adversarial loss (optional)
    """
    def __init__(
        self,
        sbert_dim: int = 384,
        style_dim: int = 182,
        fusion_dim: int = 256,
        style_embed_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_topics: int = 50,
        dropout: float = 0.3,
        use_adversarial: bool = True,
    ):
        super().__init__()
        self.use_adversarial = use_adversarial

        self.fusion = FusionLayer(sbert_dim, style_dim, fusion_dim, dropout)
        self.style_encoder = ContrastiveStyleEncoder(fusion_dim, style_embed_dim, dropout)
        self.bilstm = BoundaryBiLSTM(style_embed_dim, lstm_hidden, lstm_layers, dropout)

        if use_adversarial:
            self.topic_disc = TopicDiscriminator(style_embed_dim, n_topics, dropout)

    def forward(
        self,
        sbert_emb: torch.Tensor,
        style_vec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        fused = self.fusion(sbert_emb, style_vec)
        style_emb = self.style_encoder(fused)
        boundary_logits = self.bilstm(style_emb, lengths)

        outputs = {
            "boundary_logits": boundary_logits,
            "style_embeddings": style_emb,
        }

        if self.use_adversarial:
            topic_logits = self.topic_disc(style_emb, lambda_=grl_lambda)
            outputs["topic_logits"] = topic_logits

        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
