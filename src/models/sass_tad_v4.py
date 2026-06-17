"""
SASS-TAD v4: v2 architecture + BatchNorm + CRF boundary layer + NER masking.

Changes from v3:
    1. Reverts to v2's simpler boundary head (no richer repr that caused overfitting)
    2. Keeps BatchNorm on stylometric features from v3
    3. Adds CRF layer for structured boundary prediction
    4. Higher dropout (0.4) and weight decay for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, Optional

try:
    from torchcrf import CRF
    HAS_CRF = True
except ImportError:
    HAS_CRF = False


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, val):
        self.lambda_ = val


class FusionLayer(nn.Module):
    def __init__(self, sbert_dim=384, style_dim=193, fusion_dim=192, dropout=0.4):
        super().__init__()
        self.style_norm = nn.BatchNorm1d(style_dim)
        input_dim = sbert_dim + style_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sbert_emb, style_vec):
        B, N, D = style_vec.shape
        style_normed = self.style_norm(style_vec.reshape(B * N, D)).reshape(B, N, D)
        x = torch.cat([sbert_emb, style_normed], dim=-1)
        return self.proj(x)


class ContrastiveStyleEncoder(nn.Module):
    def __init__(self, input_dim=192, style_embed_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, style_embed_dim),
            nn.LayerNorm(style_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(style_embed_dim, style_embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=-1)


class TopicDiscriminator(nn.Module):
    def __init__(self, input_dim=128, n_topics=50, dropout=0.4):
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
        return self.classifier(self.grl(style_emb))


class BoundaryBiLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, dropout=0.4, use_crf=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_crf = use_crf and HAS_CRF
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        boundary_input_dim = hidden_dim * 4
        self.boundary_head = nn.Sequential(
            nn.Linear(boundary_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 if self.use_crf else 1),
        )
        if self.use_crf:
            self.crf = CRF(num_tags=2, batch_first=True)

    def forward(self, style_emb, lengths=None, sbert_emb=None, boundary_labels=None, boundary_mask=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                style_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(style_emb)

        left = lstm_out[:, :-1, :]
        right = lstm_out[:, 1:, :]
        boundary_repr = torch.cat([left, right], dim=-1)
        emissions = self.boundary_head(boundary_repr)

        if not self.use_crf:
            return emissions.squeeze(-1)

        B, T, _ = emissions.shape
        if boundary_mask is not None:
            crf_mask = boundary_mask[:, :T].bool()
        elif lengths is not None:
            crf_mask = torch.zeros(B, T, dtype=torch.bool, device=emissions.device)
            for i in range(B):
                n = min(lengths[i].item() - 1, T)
                crf_mask[i, :n] = True
        else:
            crf_mask = torch.ones(B, T, dtype=torch.bool, device=emissions.device)

        if boundary_labels is not None:
            labels_clamped = boundary_labels[:, :T].long().clamp(0, 1)
            labels_clamped = labels_clamped * crf_mask.long()
            crf_loss = -self.crf(emissions, labels_clamped, mask=crf_mask, reduction='mean')
            return emissions, crf_loss, crf_mask
        else:
            return emissions, crf_mask


class SASSTAD(nn.Module):
    def __init__(
        self,
        sbert_dim: int = 384,
        style_dim: int = 193,
        fusion_dim: int = 192,
        style_embed_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_topics: int = 50,
        dropout: float = 0.4,
        use_adversarial: bool = True,
        use_crf: bool = True,
    ):
        super().__init__()
        self.use_adversarial = use_adversarial
        self.use_crf = use_crf and HAS_CRF

        self.fusion = FusionLayer(sbert_dim, style_dim, fusion_dim, dropout)
        self.style_encoder = ContrastiveStyleEncoder(fusion_dim, style_embed_dim, dropout)
        self.bilstm = BoundaryBiLSTM(style_embed_dim, lstm_hidden, lstm_layers, dropout, use_crf)

        if use_adversarial:
            self.topic_disc = TopicDiscriminator(style_embed_dim, n_topics, dropout)

    def forward(
        self,
        sbert_emb: torch.Tensor,
        style_vec: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0,
        boundary_labels: Optional[torch.Tensor] = None,
        boundary_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        fused = self.fusion(sbert_emb, style_vec)
        style_emb = self.style_encoder(fused)

        if self.use_crf:
            result = self.bilstm(style_emb, lengths, boundary_labels=boundary_labels, boundary_mask=boundary_mask)
            if boundary_labels is not None:
                emissions, crf_loss, crf_mask = result
                outputs = {
                    "emissions": emissions,
                    "crf_loss": crf_loss,
                    "crf_mask": crf_mask,
                    "style_embeddings": style_emb,
                }
            else:
                emissions, crf_mask = result
                outputs = {
                    "emissions": emissions,
                    "crf_mask": crf_mask,
                    "style_embeddings": style_emb,
                }
        else:
            boundary_logits = self.bilstm(style_emb, lengths)
            outputs = {
                "boundary_logits": boundary_logits,
                "style_embeddings": style_emb,
            }

        if self.use_adversarial:
            outputs["topic_logits"] = self.topic_disc(style_emb, lambda_=grl_lambda)

        return outputs

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor):
        if self.use_crf:
            return self.bilstm.crf.decode(emissions, mask=mask.bool())
        else:
            return (torch.sigmoid(emissions) > 0.5).long().tolist()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)