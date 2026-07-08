"""
SASS-TAD Trainer v3 — curriculum training + optimal threshold tuning.

Changes from v2:
    1. Threshold tuning on validation set (searches 0.3-0.7)
    2. Uses best threshold for final predictions
    3. Reports both fixed-0.5 and tuned-threshold metrics
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.sass_tad_v3 import SASSTAD
from src.training.losses import SASTADLoss
from src.training.dataset import collate_fn
from src.evaluation.metrics import evaluate_predictions


class SASTADTrainer:
    def __init__(
        self,
        model: SASSTAD,
        loss_fn: SASTADLoss,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        max_epochs: int = 50,
        patience: int = 10,
        grad_accum_steps: int = 4,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
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

        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.best_wd = float("inf")
        self.best_threshold = 0.5
        self.epochs_no_improve = 0
        self.history: List[Dict] = []

    def _curriculum_weights(self, epoch: int):
        if epoch < self.boundary_only_epochs:
            return 0.0, 0.0, 0.0
        contrastive_end = self.boundary_only_epochs + self.contrastive_ramp_epochs
        if epoch < contrastive_end:
            p = (epoch - self.boundary_only_epochs) / self.contrastive_ramp_epochs
            return self.max_beta * p, 0.0, 0.0
        adv_end = contrastive_end + self.adversarial_ramp_epochs
        if epoch < adv_end:
            p = (epoch - contrastive_end) / self.adversarial_ramp_epochs
            return self.max_beta, self.max_gamma * p, self.max_grl_lambda * p
        return self.max_beta, self.max_gamma, self.max_grl_lambda

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        beta, gamma, grl_lambda = self._curriculum_weights(epoch)
        self.loss_fn.beta = beta
        self.loss_fn.gamma = gamma

        total_losses = {}
        n_batches = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            sbert = batch["sbert"].to(self.device)
            style = batch["style"].to(self.device)
            bl = batch["boundary_labels"].to(self.device)
            bm = batch["boundary_mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            sm = batch["sentence_mask"].to(self.device)
            tl = batch.get("topic_labels")
            if tl is not None:
                tl = tl.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    out = self.model(sbert, style, lengths, grl_lambda=grl_lambda)
                    losses = self.loss_fn(
                        out["boundary_logits"], bl, out["style_embeddings"],
                        out.get("topic_logits") if gamma > 0 else None,
                        tl if gamma > 0 else None, bm, sm,
                    )
                self.scaler.scale(losses["total"] / self.grad_accum_steps).backward()
            else:
                out = self.model(sbert, style, lengths, grl_lambda=grl_lambda)
                losses = self.loss_fn(
                    out["boundary_logits"], bl, out["style_embeddings"],
                    out.get("topic_logits") if gamma > 0 else None,
                    tl if gamma > 0 else None, bm, sm,
                )
                (losses["total"] / self.grad_accum_steps).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v.item()
            n_batches += 1

        if n_batches % self.grad_accum_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

        return {k: v / n_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def _collect_predictions(self, dataloader: DataLoader):
        """Collect raw probabilities and labels for threshold tuning."""
        self.model.eval()
        all_true, all_probs = [], []
        doc_lengths = []

        for batch in dataloader:
            sbert = batch["sbert"].to(self.device)
            style = batch["style"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            bl = batch["boundary_labels"].to(self.device)

            out = self.model(sbert, style, lengths)
            probs = torch.sigmoid(out["boundary_logits"])

            for i in range(len(batch["doc_ids"])):
                n = lengths[i].item() - 1
                if n > 0:
                    all_true.append(bl[i, :n].cpu().long().tolist())
                    all_probs.append(probs[i, :n].cpu().tolist())
                    doc_lengths.append(n)

        return all_true, all_probs, doc_lengths

    def _find_best_threshold(self, all_true, all_probs):
        """Search for optimal threshold on validation set."""
        best_wd = float("inf")
        best_t = 0.5
        for t in np.arange(0.30, 0.71, 0.02):
            preds = [[1 if p > t else 0 for p in doc] for doc in all_probs]
            metrics = evaluate_predictions(all_true, preds)
            if metrics["window_diff"] < best_wd:
                best_wd = metrics["window_diff"]
                best_t = t
        return best_t, best_wd

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        all_true, all_probs, _ = self._collect_predictions(dataloader)
        all_pred = [[1 if p > threshold else 0 for p in doc] for doc in all_probs]
        return evaluate_predictions(all_true, all_pred)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Training SASS-TAD v3 | {self.model.count_parameters():,} parameters")
        print(f"Device: {self.device} | AMP: {self.use_amp} | Patience: {self.patience}")
        print(f"Curriculum: boundary={self.boundary_only_epochs}ep, "
              f"contrastive={self.contrastive_ramp_epochs}ep, "
              f"adversarial={self.adversarial_ramp_epochs}ep")
        print("=" * 70)

        for epoch in range(self.max_epochs):
            t0 = time.time()
            beta, gamma, grl_l = self._curriculum_weights(epoch)

            train_losses = self.train_epoch(train_loader, epoch)

            # Collect val predictions and tune threshold
            all_true, all_probs, _ = self._collect_predictions(val_loader)
            best_t, best_wd_tuned = self._find_best_threshold(all_true, all_probs)

            # Also compute fixed-0.5 metrics for comparison
            preds_fixed = [[1 if p > 0.5 else 0 for p in doc] for doc in all_probs]
            metrics_fixed = evaluate_predictions(all_true, preds_fixed)

            preds_tuned = [[1 if p > best_t else 0 for p in doc] for doc in all_probs]
            metrics_tuned = evaluate_predictions(all_true, preds_tuned)

            if self.scheduler is not None:
                self.scheduler.step(best_wd_tuned)

            elapsed = time.time() - t0

            if epoch < self.boundary_only_epochs:
                phase = "boundary"
            elif epoch < self.boundary_only_epochs + self.contrastive_ramp_epochs:
                phase = "contrastive"
            else:
                phase = "full"

            self.history.append({
                "epoch": epoch + 1, "phase": phase,
                "train_loss": train_losses.get("total", 0),
                "val_f1_fixed": metrics_fixed["f1"],
                "val_wd_fixed": metrics_fixed["window_diff"],
                "val_f1_tuned": metrics_tuned["f1"],
                "val_wd_tuned": metrics_tuned["window_diff"],
                "threshold": best_t,
                "beta": beta, "gamma": gamma, "grl_lambda": grl_l,
                "elapsed": elapsed,
            })

            print(
                f"Epoch {epoch+1:3d}/{self.max_epochs} [{phase:11s}] | "
                f"loss={train_losses.get('total', 0):.4f} | "
                f"F1={metrics_tuned['f1']:.4f} | "
                f"WD={metrics_tuned['window_diff']:.4f} | "
                f"t={best_t:.2f} | "
                f"b={beta:.2f} g={gamma:.2f} | "
                f"{elapsed:.1f}s"
            )

            if best_wd_tuned < self.best_wd:
                self.best_wd = best_wd_tuned
                self.best_threshold = best_t
                self.epochs_no_improve = 0
                ckpt_path = self.checkpoint_dir / "best_model_v3.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_wd": self.best_wd,
                    "best_threshold": best_t,
                    "val_metrics": metrics_tuned,
                    "history": self.history,
                }, ckpt_path)
                print(f"  -> New best! WD={best_wd_tuned:.4f} t={best_t:.2f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

        print("=" * 70)
        print(f"Best: WD={self.best_wd:.4f} threshold={self.best_threshold:.2f}")
        return self.history
