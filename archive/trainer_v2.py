"""
SASS-TAD Trainer v2 — with curriculum training.

Key fix: trains boundary-only for warmup epochs, then gradually
introduces contrastive and adversarial losses. This prevents the
multi-task losses from destabilizing early training.

Schedule:
    Epochs 1-5:   boundary loss only (alpha=1, beta=0, gamma=0)
    Epochs 6-10:  + contrastive (beta ramps 0→0.3)
    Epochs 11+:   + adversarial (gamma ramps 0→0.1, GRL lambda ramps)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
        max_epochs: int = 50,
        patience: int = 10,
        grad_accum_steps: int = 4,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        # Curriculum schedule
        boundary_only_epochs: int = 5,
        contrastive_ramp_epochs: int = 5,
        adversarial_ramp_epochs: int = 5,
        max_beta: float = 0.3,
        max_gamma: float = 0.1,
        max_grl_lambda: float = 1.0,
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
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.boundary_only_epochs = boundary_only_epochs
        self.contrastive_ramp_epochs = contrastive_ramp_epochs
        self.adversarial_ramp_epochs = adversarial_ramp_epochs
        self.max_beta = max_beta
        self.max_gamma = max_gamma
        self.max_grl_lambda = max_grl_lambda

        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.best_wd = float("inf")
        self.epochs_no_improve = 0
        self.history: List[Dict] = []

    def _curriculum_weights(self, epoch: int):
        """Returns (beta, gamma, grl_lambda) for current epoch."""
        # Phase 1: boundary only
        if epoch < self.boundary_only_epochs:
            return 0.0, 0.0, 0.0

        # Phase 2: ramp contrastive
        contrastive_start = self.boundary_only_epochs
        contrastive_end = contrastive_start + self.contrastive_ramp_epochs
        if epoch < contrastive_end:
            progress = (epoch - contrastive_start) / self.contrastive_ramp_epochs
            return self.max_beta * progress, 0.0, 0.0

        # Phase 3: ramp adversarial
        adv_start = contrastive_end
        adv_end = adv_start + self.adversarial_ramp_epochs
        if epoch < adv_end:
            progress = (epoch - adv_start) / self.adversarial_ramp_epochs
            return self.max_beta, self.max_gamma * progress, self.max_grl_lambda * progress

        # Phase 4: all losses at full strength
        return self.max_beta, self.max_gamma, self.max_grl_lambda

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        beta, gamma, grl_lambda = self._curriculum_weights(epoch)

        # Temporarily set loss weights
        self.loss_fn.beta = beta
        self.loss_fn.gamma = gamma

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

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(sbert, style, lengths, grl_lambda=grl_lambda)
                    losses = self.loss_fn(
                        boundary_logits=outputs["boundary_logits"],
                        boundary_labels=boundary_labels,
                        style_embeddings=outputs["style_embeddings"],
                        topic_logits=outputs.get("topic_logits") if gamma > 0 else None,
                        topic_labels=topic_labels if gamma > 0 else None,
                        boundary_mask=boundary_mask,
                        sentence_mask=sentence_mask,
                    )
                loss = losses["total"] / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(sbert, style, lengths, grl_lambda=grl_lambda)
                losses = self.loss_fn(
                    boundary_logits=outputs["boundary_logits"],
                    boundary_labels=boundary_labels,
                    style_embeddings=outputs["style_embeddings"],
                    topic_logits=outputs.get("topic_logits") if gamma > 0 else None,
                    topic_labels=topic_labels if gamma > 0 else None,
                    boundary_mask=boundary_mask,
                    sentence_mask=sentence_mask,
                )
                loss = losses["total"] / self.grad_accum_steps
                loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            n_batches += 1

        # Flush remaining gradients
        if n_batches % self.grad_accum_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
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

            # Evaluate with boundary loss only (stable metric)
            old_beta, old_gamma = self.loss_fn.beta, self.loss_fn.gamma
            self.loss_fn.beta = 0.0
            self.loss_fn.gamma = 0.0
            losses = self.loss_fn(
                boundary_logits=outputs["boundary_logits"],
                boundary_labels=boundary_labels,
                style_embeddings=outputs["style_embeddings"],
                boundary_mask=boundary_mask,
                sentence_mask=sentence_mask,
            )
            self.loss_fn.beta = old_beta
            self.loss_fn.gamma = old_gamma

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
        print(f"Curriculum: boundary-only={self.boundary_only_epochs}ep, "
              f"contrastive-ramp={self.contrastive_ramp_epochs}ep, "
              f"adversarial-ramp={self.adversarial_ramp_epochs}ep")
        print("=" * 70)

        for epoch in range(self.max_epochs):
            t0 = time.time()
            beta, gamma, grl_l = self._curriculum_weights(epoch)

            train_losses = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_metrics.get("window_diff", val_metrics.get("loss_total", 0)))

            elapsed = time.time() - t0

            # Determine current phase
            if epoch < self.boundary_only_epochs:
                phase = "boundary"
            elif epoch < self.boundary_only_epochs + self.contrastive_ramp_epochs:
                phase = "contrastive"
            else:
                phase = "full"

            record = {
                "epoch": epoch + 1,
                "train_loss": train_losses.get("total", 0),
                "val_f1": val_metrics.get("f1", 0),
                "val_wd": val_metrics.get("window_diff", 1),
                "val_loss": val_metrics.get("loss_total", 0),
                "beta": beta,
                "gamma": gamma,
                "grl_lambda": grl_l,
                "phase": phase,
                "elapsed": elapsed,
            }
            self.history.append(record)

            print(
                f"Epoch {epoch+1:3d}/{self.max_epochs} [{phase:11s}] | "
                f"loss={train_losses.get('total', 0):.4f} | "
                f"F1={val_metrics.get('f1', 0):.4f} | "
                f"WD={val_metrics.get('window_diff', 1):.4f} | "
                f"b={beta:.2f} g={gamma:.2f} l={grl_l:.2f} | "
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
                    "history": self.history,
                }, ckpt_path)
                print(f"  -> New best! Saved to {ckpt_path}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch+1} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        print("=" * 70)
        print(f"Training complete. Best WindowDiff: {self.best_wd:.4f}")
        return self.history
