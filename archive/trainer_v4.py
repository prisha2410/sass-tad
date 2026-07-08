"""
SASS-TAD Trainer v4 — CRF + NER masking + threshold tuning.

Changes from v3:
    1. Supports CRF-based boundary loss (replaces BCE when CRF is available)
    2. Falls back to v3 behavior if torchcrf not installed
    3. Threshold tuning still used for non-CRF mode
"""

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.sass_tad_v4 import SASSTAD
from src.training.losses import SASTADLoss
from src.evaluation.metrics import evaluate_predictions

try:
    from torchcrf import CRF
    HAS_CRF = True
except ImportError:
    HAS_CRF = False


class SASTADTrainer:
    def __init__(
        self,
        model: SASSTAD,
        loss_fn: SASTADLoss,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cuda",
        max_epochs: int = 50,
        patience: int = 12,
        grad_accum_steps: int = 4,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        boundary_only_epochs: int = 10,
        contrastive_ramp_epochs: int = 5,
        adversarial_ramp_epochs: int = 5,
        max_beta: float = 0.3,
        max_gamma: float = 0.1,
        max_grl_lambda: float = 1.0,
        disable_contrastive: bool = False,
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
        self.disable_contrastive = disable_contrastive

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
            beta = 0.0 if self.disable_contrastive else self.max_beta * p
            return beta, 0.0, 0.0
        adv_end = contrastive_end + self.adversarial_ramp_epochs
        if epoch < adv_end:
            p = (epoch - contrastive_end) / self.adversarial_ramp_epochs
            beta = 0.0 if self.disable_contrastive else self.max_beta
            return beta, self.max_gamma * p, self.max_grl_lambda * p
        beta = 0.0 if self.disable_contrastive else self.max_beta
        return beta, self.max_gamma, self.max_grl_lambda

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
                    out = self.model(sbert, style, lengths, grl_lambda=grl_lambda,
                                     boundary_labels=bl, boundary_mask=bm)
                    loss = self._compute_loss(out, bl, bm, sm, tl, gamma)
                self.scaler.scale(loss / self.grad_accum_steps).backward()
            else:
                out = self.model(sbert, style, lengths, grl_lambda=grl_lambda,
                                 boundary_labels=bl, boundary_mask=bm)
                loss = self._compute_loss(out, bl, bm, sm, tl, gamma)
                (loss / self.grad_accum_steps).backward()

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

            total_losses["total"] = total_losses.get("total", 0) + loss.item()
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

    def _compute_loss(self, out, bl, bm, sm, tl, gamma):
        if self.model.use_crf and "crf_loss" in out:
            total = out["crf_loss"]
            if self.loss_fn.beta > 0:
                from src.training.losses import ContrastiveLoss
                cl = ContrastiveLoss()
                cl = cl.to(self.device)
                total = total + self.loss_fn.beta * cl(out["style_embeddings"], bl, sm[:, :-1])
            if gamma > 0 and "topic_logits" in out and tl is not None:
                from src.training.losses import TopicAdversarialLoss
                tal = TopicAdversarialLoss()
                tal = tal.to(self.device)
                total = total + gamma * tal(out["topic_logits"], tl, sm)
            return total
        else:
            losses = self.loss_fn(
                out["boundary_logits"], bl, out["style_embeddings"],
                out.get("topic_logits") if gamma > 0 else None,
                tl if gamma > 0 else None, bm, sm,
            )
            return losses["total"]

    @torch.no_grad()
    def _collect_predictions(self, dataloader: DataLoader):
        self.model.eval()
        all_true, all_preds = [], []

        for batch in dataloader:
            sbert = batch["sbert"].to(self.device)
            style = batch["style"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            bl = batch["boundary_labels"].to(self.device)

            out = self.model(sbert, style, lengths)

            if self.model.use_crf and "emissions" in out:
                decoded = self.model.decode(out["emissions"], out["crf_mask"])
                for i in range(len(batch["doc_ids"])):
                    n = lengths[i].item() - 1
                    if n > 0:
                        pred = decoded[i][:n]
                        all_preds.append(pred)
                        all_true.append(bl[i, :n].cpu().long().tolist())
            else:
                probs = torch.sigmoid(out["boundary_logits"])
                for i in range(len(batch["doc_ids"])):
                    n = lengths[i].item() - 1
                    if n > 0:
                        all_true.append(bl[i, :n].cpu().long().tolist())
                        all_preds.append(probs[i, :n].cpu().tolist())

        return all_true, all_preds

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        all_true, all_preds = self._collect_predictions(dataloader)

        if self.model.use_crf:
            return evaluate_predictions(all_true, all_preds)
        else:
            all_pred_binary = [[1 if p > threshold else 0 for p in doc] for doc in all_preds]
            return evaluate_predictions(all_true, all_pred_binary)

    def _find_best_threshold(self, all_true, all_probs):
        if self.model.use_crf:
            metrics = evaluate_predictions(all_true, all_probs)
            return 0.5, metrics["window_diff"]
        best_wd = float("inf")
        best_t = 0.5
        for t in np.arange(0.30, 0.71, 0.02):
            preds = [[1 if p > t else 0 for p in doc] for doc in all_probs]
            metrics = evaluate_predictions(all_true, preds)
            if metrics["window_diff"] < best_wd:
                best_wd = metrics["window_diff"]
                best_t = t
        return best_t, best_wd

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        crf_status = "CRF" if self.model.use_crf else "sigmoid"
        print(f"Training SASS-TAD v4 ({crf_status}) | {self.model.count_parameters():,} parameters")
        print(f"Device: {self.device} | AMP: {self.use_amp} | Patience: {self.patience}")
        print(f"Curriculum: boundary={self.boundary_only_epochs}ep, "
              f"contrastive={self.contrastive_ramp_epochs}ep, "
              f"adversarial={self.adversarial_ramp_epochs}ep")
        if self.disable_contrastive:
            print("Contrastive loss: DISABLED (beta forced to 0.0)")
        print("=" * 70)

        for epoch in range(self.max_epochs):
            t0 = time.time()
            beta, gamma, grl_l = self._curriculum_weights(epoch)

            train_losses = self.train_epoch(train_loader, epoch)

            all_true, all_preds = self._collect_predictions(val_loader)
            best_t, best_wd = self._find_best_threshold(all_true, all_preds)

            if self.model.use_crf:
                metrics = evaluate_predictions(all_true, all_preds)
            else:
                preds_t = [[1 if p > best_t else 0 for p in doc] for doc in all_preds]
                metrics = evaluate_predictions(all_true, preds_t)

            if self.scheduler is not None:
                self.scheduler.step(best_wd)

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
                "val_f1": metrics["f1"],
                "val_wd": metrics["window_diff"],
                "threshold": best_t,
                "beta": beta, "gamma": gamma, "grl_lambda": grl_l,
                "elapsed": elapsed,
            })

            print(
                f"Epoch {epoch+1:3d}/{self.max_epochs} [{phase:11s}] | "
                f"loss={train_losses.get('total', 0):.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"WD={metrics['window_diff']:.4f} | "
                f"t={best_t:.2f} | "
                f"b={beta:.2f} g={gamma:.2f} | "
                f"{elapsed:.1f}s"
            )

            if best_wd < self.best_wd:
                self.best_wd = best_wd
                self.best_threshold = best_t
                self.epochs_no_improve = 0
                ckpt_path = self.checkpoint_dir / "best_model_v4.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_wd": self.best_wd,
                    "best_threshold": best_t,
                    "val_metrics": metrics,
                    "history": self.history,
                }, ckpt_path)
                print(f"  -> New best! WD={best_wd:.4f} t={best_t:.2f}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

        print("=" * 70)
        print(f"Best: WD={self.best_wd:.4f} threshold={self.best_threshold:.2f}")
        return self.history