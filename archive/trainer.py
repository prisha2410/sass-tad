"""
SASS-TAD Trainer — handles the full training loop with:
    - Multi-task loss (boundary + contrastive + adversarial)
    - GRL lambda scheduling (curriculum)
    - Mixed precision (AMP) for 4GB VRAM
    - Early stopping on validation WindowDiff
    - Gradient accumulation for effective larger batch sizes
"""

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.models.sass_tad import SASSTAD
from src.training.losses import SASTADLoss
from src.training.dataset import collate_fn
from src.evaluation.metrics import evaluate_predictions


class SASTADTrainer:
    def __init__(
        self,
        model: SASSTAD,
        loss_fn: SASTADLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        max_epochs: int = 30,
        patience: int = 7,
        grad_accum_steps: int = 4,
        use_amp: bool = True,
        grl_warmup_epochs: int = 5,
        grl_max_lambda: float = 1.0,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and device == "cuda"
        self.grl_warmup_epochs = grl_warmup_epochs
        self.grl_max_lambda = grl_max_lambda
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_wd = float("inf")
        self.epochs_no_improve = 0
        self.history = []

    def _grl_lambda(self, epoch: int) -> float:
        if epoch < self.grl_warmup_epochs:
            return self.grl_max_lambda * (epoch / self.grl_warmup_epochs)
        return self.grl_max_lambda

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        grl_lambda = self._grl_lambda(epoch)

        total_losses = {}
        n_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            sbert = batch["sbert"].to(self.device)
            style = batch["style"].to(self.device)
            boundary_labels = batch["boundary_labels"].to(self.device)
            boundary_mask = batch["boundary_mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            sentence_mask = batch["sentence_mask"].to(self.device)

            topic_labels = batch.get("topic_labels")
            if topic_labels is not None:
                topic_labels = topic_labels.to(self.device)

            with autocast(enabled=self.use_amp):
                outputs = self.model(sbert, style, lengths, grl_lambda=grl_lambda)

                losses = self.loss_fn(
                    boundary_logits=outputs["boundary_logits"],
                    boundary_labels=boundary_labels,
                    style_embeddings=outputs["style_embeddings"],
                    topic_logits=outputs.get("topic_logits"),
                    topic_labels=topic_labels,
                    boundary_mask=boundary_mask,
                    sentence_mask=sentence_mask,
                )

            loss = losses["total"] / self.grad_accum_steps
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            n_batches += 1

        # Handle remaining gradients
        if n_batches % self.grad_accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_true, all_pred = [], []
        total_losses = {}
        n_batches = 0

        for batch in dataloader:
            sbert = batch["sbert"].to(self.device)
            style = batch["style"].to(self.device)
            boundary_labels = batch["boundary_labels"].to(self.device)
            boundary_mask = batch["boundary_mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            sentence_mask = batch["sentence_mask"].to(self.device)

            topic_labels = batch.get("topic_labels")
            if topic_labels is not None:
                topic_labels = topic_labels.to(self.device)

            outputs = self.model(sbert, style, lengths)

            losses = self.loss_fn(
                boundary_logits=outputs["boundary_logits"],
                boundary_labels=boundary_labels,
                style_embeddings=outputs["style_embeddings"],
                topic_logits=outputs.get("topic_logits"),
                topic_labels=topic_labels,
                boundary_mask=boundary_mask,
                sentence_mask=sentence_mask,
            )

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            n_batches += 1

            preds = (torch.sigmoid(outputs["boundary_logits"]) > 0.5).long()

            for i in range(len(batch["doc_ids"])):
                n = lengths[i].item() - 1
                if n > 0:
                    all_true.append(boundary_labels[i, :n].cpu().long().tolist())
                    all_pred.append(preds[i, :n].cpu().tolist())

        avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        metrics = evaluate_predictions(all_true, all_pred)
        metrics.update({f"loss_{k}": v for k, v in avg_losses.items()})
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Training SASS-TAD | {self.model.count_parameters():,} parameters")
        print(f"Device: {self.device} | AMP: {self.use_amp} | Patience: {self.patience}")
        print("=" * 70)

        for epoch in range(self.max_epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_metrics.get("window_diff", val_metrics.get("loss_total", 0)))

            elapsed = time.time() - t0
            grl_l = self._grl_lambda(epoch)

            record = {
                "epoch": epoch + 1,
                "train_loss": train_losses.get("total", 0),
                "val_f1": val_metrics.get("f1", 0),
                "val_wd": val_metrics.get("window_diff", 1),
                "val_loss": val_metrics.get("loss_total", 0),
                "grl_lambda": grl_l,
                "elapsed": elapsed,
            }
            self.history.append(record)

            print(
                f"Epoch {epoch+1:3d}/{self.max_epochs} | "
                f"train_loss={train_losses.get('total', 0):.4f} | "
                f"val_F1={val_metrics.get('f1', 0):.4f} | "
                f"val_WD={val_metrics.get('window_diff', 1):.4f} | "
                f"GRL_λ={grl_l:.2f} | "
                f"{elapsed:.1f}s"
            )

            wd = val_metrics.get("window_diff", 1.0)
            if wd < self.best_wd:
                self.best_wd = wd
                self.epochs_no_improve = 0
                ckpt_path = self.checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_wd": self.best_wd,
                    "val_metrics": val_metrics,
                }, ckpt_path)
                print(f"  → New best! Saved to {ckpt_path}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  → Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                    break

        print("=" * 70)
        print(f"Training complete. Best WindowDiff: {self.best_wd:.4f}")
        return self.history
