"""
DeBERTa + Stylometric Fusion for Style Change Detection.

Modes:
  1. Pair-level (--finetune): Independent pairwise classification
  2. Document-level (--finetune --bilstm): BiLSTM over boundary sequence
  3. Document-level + adversarial (--finetune --bilstm --adversarial):
     adds a pseudo-topic discriminator with gradient reversal to
     disentangle topic signal from the learned boundary representation.

Architecture (document-level):
    For each document:
      - Encode all sentence pairs through DeBERTa → [CLS] hidden states
      - Attention-weighted fusion with stylometric diffs
      - Project to lower dimension
      - BiLSTM over the boundary sequence → document context
      - MLP head → per-boundary logits
      - (optional) Gradient-reversed topic classifier on the BiLSTM output
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
#  Attention-Weighted Stylometric Fusion
# --------------------------------------------------------------------------- #

class StyleAttentionFusion(nn.Module):
    def __init__(self, deberta_dim: int, style_dim: int):
        super().__init__()
        self.style_norm = nn.LayerNorm(style_dim)
        self.gate = nn.Sequential(
            nn.Linear(deberta_dim, style_dim),
            nn.Tanh(),
            nn.Linear(style_dim, style_dim),
            nn.Sigmoid(),
        )

    def forward(self, cls_hidden: torch.Tensor, style_diff: torch.Tensor) -> torch.Tensor:
        style_normed = self.style_norm(style_diff)
        attn_weights = self.gate(cls_hidden)
        weighted_style = style_normed * attn_weights
        return torch.cat([cls_hidden, weighted_style], dim=-1)


# --------------------------------------------------------------------------- #
#  Gradient Reversal Layer (for pseudo-topic adversarial training)
# --------------------------------------------------------------------------- #

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class TopicAdversary(nn.Module):
    """Small MLP that tries to predict pseudo-topic from the boundary
    representation. Gradient reversal forces the encoder to NOT carry
    topic information forward, only style information."""

    def __init__(self, input_dim: int, n_topics: int = 20, hidden: int = 64):
        super().__init__()
        self.n_topics = n_topics
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_topics),
        )

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        reversed_x = GradientReversal.apply(x, lambda_)
        return self.classifier(reversed_x)


# --------------------------------------------------------------------------- #
#  End-to-end fine-tuning model
# --------------------------------------------------------------------------- #

class DeBERTaFusionE2E(nn.Module):
    DEBERTA_MODEL = "microsoft/deberta-v3-small"
    DEBERTA_DIM = 768

    def __init__(
        self,
        style_dim: int = 193,
        unfreeze_layers: int = 2,
        use_style: bool = True,
        use_attention_fusion: bool = False,
        use_bilstm: bool = False,
        bilstm_hidden: int = 128,
        dropout: float = 0.3,
        use_adversarial: bool = False,
        n_topics: int = 20,
    ):
        super().__init__()
        self.use_style = use_style
        self.use_attention_fusion = use_attention_fusion
        self.use_bilstm = use_bilstm
        self.unfreeze_layers = unfreeze_layers
        self.use_adversarial = use_adversarial
        self.n_topics = n_topics

        from transformers import DebertaV2Model
        self.deberta = DebertaV2Model.from_pretrained(self.DEBERTA_MODEL)

        for param in self.deberta.parameters():
            param.requires_grad = False

        n_layers = len(self.deberta.encoder.layer)
        for layer in self.deberta.encoder.layer[n_layers - unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(self.deberta, 'pooler') and self.deberta.pooler is not None:
            for param in self.deberta.pooler.parameters():
                param.requires_grad = True

        # Fusion
        if use_style and use_attention_fusion:
            self.style_fusion = StyleAttentionFusion(self.DEBERTA_DIM, style_dim)
            fused_dim = self.DEBERTA_DIM + style_dim
        elif use_style:
            fused_dim = self.DEBERTA_DIM + style_dim
        else:
            fused_dim = self.DEBERTA_DIM

        self.fused_dim = fused_dim

        # BiLSTM path
        if use_bilstm:
            self.proj = nn.Sequential(
                nn.Linear(fused_dim, bilstm_hidden * 2),
                nn.LayerNorm(bilstm_hidden * 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.bilstm = nn.LSTM(
                bilstm_hidden * 2, bilstm_hidden,
                num_layers=1, batch_first=True,
                bidirectional=True, dropout=0,
            )
            head_input = bilstm_hidden * 2  # 256
        else:
            head_input = fused_dim

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(head_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Topic adversarial branch — operates on the same representation
        # that feeds the classification head (post-BiLSTM if BiLSTM is on,
        # otherwise the fused pair representation).
        if use_adversarial:
            self.topic_adversary = TopicAdversary(head_input, n_topics)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"DeBERTa Fusion E2E: {trainable:,} trainable / {total:,} total")
        print(f"Fine-tuning last {unfreeze_layers} DeBERTa layers")
        print(f"Stylometric features: {'ON' if use_style else 'OFF'}")
        print(f"Attention fusion: {'ON' if use_attention_fusion else 'OFF (concat)'}")
        print(f"BiLSTM: {'ON' if use_bilstm else 'OFF'}")
        print(f"Topic adversarial: {'ON (' + str(n_topics) + ' pseudo-topics)' if use_adversarial else 'OFF'}")

    def encode_pairs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_diff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode sentence pairs through DeBERTa + fusion. Returns fused representations."""
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]

        if self.use_style and style_diff is not None:
            if self.use_attention_fusion:
                fused = self.style_fusion(cls_hidden, style_diff)
            else:
                fused = torch.cat([cls_hidden, style_diff], dim=-1)
        else:
            fused = cls_hidden

        return fused

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        style_diff: Optional[torch.Tensor] = None,
        topic_lambda: Optional[float] = None,
    ):
        """Pair-level forward (no BiLSTM). Returns logits (B,), or
        (logits, topic_logits) if adversarial mode is on and topic_lambda
        is provided."""
        fused = self.encode_pairs(input_ids, attention_mask, style_diff)
        logits = self.head(fused).squeeze(-1)

        if self.use_adversarial and topic_lambda is not None:
            topic_logits = self.topic_adversary(fused, lambda_=topic_lambda)
            return logits, topic_logits

        return logits

    def forward_document(
        self,
        pair_reprs: torch.Tensor,
        topic_lambda: Optional[float] = None,
    ):
        """
        Document-level forward with BiLSTM.
        pair_reprs: (1, n_boundaries, fused_dim) — pre-encoded pair representations
        Returns: logits (n_boundaries,), or (logits, topic_logits) if
        adversarial mode is on and topic_lambda is provided.
        """
        x = self.proj(pair_reprs)          # (1, n_boundaries, hidden*2)
        lstm_out, _ = self.bilstm(x)       # (1, n_boundaries, hidden*2)
        logits = self.head(lstm_out)       # (1, n_boundaries, 1)
        logits = logits.squeeze(-1).squeeze(0)  # (n_boundaries,)

        if self.use_adversarial and topic_lambda is not None:
            topic_logits = self.topic_adversary(lstm_out.squeeze(0), lambda_=topic_lambda)
            return logits, topic_logits

        return logits

    def get_deberta_params(self):
        params = []
        for name, param in self.deberta.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def get_head_params(self):
        head_params = list(self.head.parameters())
        if self.use_style and self.use_attention_fusion:
            head_params += list(self.style_fusion.parameters())
        if self.use_bilstm:
            head_params += list(self.proj.parameters())
            head_params += list(self.bilstm.parameters())
        return head_params

    def get_adversary_params(self):
        if self.use_adversarial:
            return list(self.topic_adversary.parameters())
        return []


# --------------------------------------------------------------------------- #
#  Dataset for E2E fine-tuning (pair-level)
# --------------------------------------------------------------------------- #

class E2EDataset(Dataset):
    def __init__(
        self,
        sentences_list: List[List[str]],
        labels_list: List[List[int]],
        style_diffs_list: Optional[List[np.ndarray]] = None,
    ):
        self.pairs = []
        self.labels = []
        self.styles = []

        for doc_idx, (sents, labels) in enumerate(zip(sentences_list, labels_list)):
            n_boundaries = len(sents) - 1
            for i in range(min(n_boundaries, len(labels))):
                self.pairs.append((sents[i], sents[i + 1]))
                self.labels.append(labels[i])
                if style_diffs_list is not None:
                    self.styles.append(style_diffs_list[doc_idx][i])
                else:
                    self.styles.append(None)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {
            "sent_a": self.pairs[idx][0],
            "sent_b": self.pairs[idx][1],
            "label": self.labels[idx],
            "style_diff": self.styles[idx],
        }


def e2e_collate_fn(batch, tokenizer, max_length=256):
    sent_a = [item["sent_a"] for item in batch]
    sent_b = [item["sent_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    encoded = tokenizer(
        sent_a, sent_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    result = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }

    if batch[0]["style_diff"] is not None:
        style_diffs = torch.tensor(
            np.stack([item["style_diff"] for item in batch]),
            dtype=torch.float32,
        )
        result["style_diff"] = style_diffs

    return result


# --------------------------------------------------------------------------- #
#  Dataset for document-level training (BiLSTM)
# --------------------------------------------------------------------------- #

class DocLevelDataset(Dataset):
    """Each item is one document: list of sentence pairs + labels + style diffs.

    doc_id is carried through so the adversarial branch can look up the
    precomputed pseudo-topic labels for each boundary of this document.
    """

    def __init__(
        self,
        sentences_list: List[List[str]],
        labels_list: List[List[int]],
        style_diffs_list: Optional[List[np.ndarray]] = None,
        doc_ids: Optional[List[str]] = None,
    ):
        self.docs = []
        for doc_idx, (sents, labels) in enumerate(zip(sentences_list, labels_list)):
            n = min(len(sents) - 1, len(labels))
            if n <= 0:
                continue
            pairs = [(sents[i], sents[i + 1]) for i in range(n)]
            doc_labels = labels[:n]
            if style_diffs_list is not None:
                doc_styles = style_diffs_list[doc_idx][:n]
            else:
                doc_styles = None
            doc_id = doc_ids[doc_idx] if doc_ids is not None else str(doc_idx)
            self.docs.append({
                "doc_id": doc_id,
                "pairs": pairs,
                "labels": doc_labels,
                "style_diffs": doc_styles,
            })

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]


# --------------------------------------------------------------------------- #
#  Frozen DeBERTa encoder (for precompute mode)
# --------------------------------------------------------------------------- #

class DeBERTaEncoder:
    DEBERTA_MODEL = "microsoft/deberta-v3-small"

    def __init__(self, device: str = "cpu", batch_size: int = 16):
        from transformers import DebertaV2Model, DebertaV2Tokenizer
        self.device = device
        self.batch_size = batch_size

        print(f"[DeBERTaEncoder] Loading {self.DEBERTA_MODEL} on {device}...")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.DEBERTA_MODEL)
        self.model = DebertaV2Model.from_pretrained(self.DEBERTA_MODEL).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        print("[DeBERTaEncoder] Ready.")

    @torch.no_grad()
    def encode_pairs(self, sentences: List[str]) -> torch.Tensor:
        if len(sentences) < 2:
            return torch.zeros(0, 768)

        all_cls = []
        pairs_a = sentences[:-1]
        pairs_b = sentences[1:]

        for start in range(0, len(pairs_a), self.batch_size):
            end = min(start + self.batch_size, len(pairs_a))
            batch_a = pairs_a[start:end]
            batch_b = pairs_b[start:end]

            encoded = self.tokenizer(
                batch_a, batch_b,
                padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**encoded)
            cls_vec = out.last_hidden_state[:, 0, :]
            all_cls.append(cls_vec.cpu())

        return torch.cat(all_cls, dim=0)


# --------------------------------------------------------------------------- #
#  Frozen mode model + dataset
# --------------------------------------------------------------------------- #

class DeBERTaFusionModel(nn.Module):
    def __init__(self, deberta_dim=768, style_dim=193, use_style=True, dropout=0.3):
        super().__init__()
        self.use_style = use_style
        input_dim = deberta_dim + (style_dim if use_style else 0)

        self.bilstm = nn.LSTM(
            input_dim, 128, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.bilstm(x)
        logits = self.head(lstm_out).squeeze(-1)
        return logits


class DeBERTaFusionDataset(Dataset):
    def __init__(self, deberta_embs, style_diffs, labels):
        self.items = []
        for emb, sd, lbl in zip(deberta_embs, style_diffs, labels):
            if emb.shape[0] == 0:
                continue
            self.items.append((emb, sd, torch.tensor(lbl, dtype=torch.float32)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def deberta_fusion_collate_fn(batch):
    embs, styles, labels = zip(*batch)

    max_len = max(e.shape[0] for e in embs)
    B = len(batch)
    feat_dim = embs[0].shape[1] + styles[0].shape[1]

    x_pad = torch.zeros(B, max_len, feat_dim)
    y_pad = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (e, s, l) in enumerate(batch):
        n = e.shape[0]
        x_pad[i, :n] = torch.cat([e, s], dim=-1)
        y_pad[i, :n] = l
        mask[i, :n] = True

    return x_pad, y_pad, mask