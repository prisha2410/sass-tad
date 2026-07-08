"""
SSPC Baseline: BiLSTM over frozen SBERT sentence embeddings.

Deliberately minimal — no CRF, no contrastive, no adversarial, no stylometric features.
Isolates whether sequential BiLSTM modeling alone helps on this dataset.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class SSPCBaseline(nn.Module):
    def __init__(
        self,
        sbert_dim: int = 384,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            sbert_dim, lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        boundary_input_dim = lstm_hidden * 4
        self.boundary_head = nn.Sequential(
            nn.Linear(boundary_input_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        sbert_emb: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                sbert_emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(sbert_emb)

        left = lstm_out[:, :-1, :]
        right = lstm_out[:, 1:, :]
        boundary_repr = torch.cat([left, right], dim=-1)
        logits = self.boundary_head(boundary_repr).squeeze(-1)
        return {"boundary_logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)