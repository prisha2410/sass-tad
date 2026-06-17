"""
SASS-TAD v3: Improved architecture.

Changes from v2:
    1. Input normalization — BatchNorm on stylometric features
    2. Richer boundary representation — diff, product, cosine sim + concat
    3. Direct SBERT cosine shortcut to boundary head
    4. Residual connection in fusion layer
    5. Optimal threshold tuning on validation set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, Optional


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
    def __init__(self, sbert_dim=384, style_dim=193, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.style_norm = nn.BatchNorm1d(style_dim)
        input_dim = sbert_dim + style_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sbert_emb, style_vec):
        B, N, D = style_vec.shape
        style_normed = self.style_norm(style_vec.reshape(B * N, D)).reshape(B, N, D)
        x = torch.cat([sbert_emb, style_normed], dim=-1)
        h = self.proj(x)
        return h + self.residual(h)


class ContrastiveStyleEncoder(nn.Module):
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
        return F.normalize(self.encoder(x), p=2, dim=-1)


class TopicDiscriminator(nn.Module):
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
        return self.classifier(self.grl(style_emb))


class BoundaryBiLSTM(nn.Module):
    """
    Improved boundary detector with richer boundary representations:
    - Concat of left and right hidden states
    - Element-wise difference
    - Element-wise product
    - Cosine similarity scalar
    - Direct SBERT cosine similarity shortcut
    """
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # boundary repr: concat(4H) + diff(2H) + product(2H) + cos_sim(1) + sbert_cos(1)
        boundary_dim = hidden_dim * 8 + 2
        self.boundary_head = nn.Sequential(
            nn.Linear(boundary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, style_emb, lengths=None, sbert_emb=None):
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

        features = [
            left,
            right,
            left - right,
            left * right,
        ]

        cos_sim = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
        features.append(cos_sim)

        if sbert_emb is not None:
            sbert_left = sbert_emb[:, :-1, :]
            sbert_right = sbert_emb[:, 1:, :]
            sbert_cos = F.cosine_similarity(sbert_left, sbert_right, dim=-1).unsqueeze(-1)
            features.append(sbert_cos)
        else:
            features.append(torch.zeros_like(cos_sim))

        boundary_repr = torch.cat(features, dim=-1)
        return self.boundary_head(boundary_repr).squeeze(-1)


class SASSTAD(nn.Module):
    def __init__(
        self,
        sbert_dim: int = 384,
        style_dim: int = 193,
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
        boundary_logits = self.bilstm(style_emb, lengths, sbert_emb=sbert_emb)

        outputs = {
            "boundary_logits": boundary_logits,
            "style_embeddings": style_emb,
        }

        if self.use_adversarial:
            outputs["topic_logits"] = self.topic_disc(style_emb, lambda_=grl_lambda)

        return outputs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
