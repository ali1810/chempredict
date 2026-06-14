"""
trainer.py — Training Loop for Retrosynthesis Transformer
=========================================================
Author: Dr. Mushtaq Ali · KIT
"""
import os, math, time, argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional

from src.model.transformer import RetrosynthesisTransformer
from src.train.config import ModelConfig, TrainConfig


class WarmupNoamScheduler(LambdaLR):
    """Noam learning rate schedule from 'Attention is All You Need'."""
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.d_model = d_model
        self.warmup = warmup_steps
        super().__init__(optimizer, self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        step = max(step, 1)
        return (self.d_model ** -0.5) * min(
            step ** -0.5, step * self.warmup ** -1.5
        )


class Trainer:
    def __init__(self, model: RetrosynthesisTransformer,
                 train_loader, val_loader,
                 cfg: TrainConfig, vocab_src: dict, vocab_tgt: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.998))
        self.scheduler = WarmupNoamScheduler(self.optimizer, model.d_model, cfg.warmup_steps)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab_tgt.get("<PAD>", 0),
            label_smoothing=cfg.label_smoothing
        )

        os.makedirs(cfg.output_dir, exist_ok=True)
        self.best_val_loss = float("inf")
        self.step = 0

    def train_step(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        self.model.train()
        src, tgt = src.to(self.device), tgt.to(self.device)
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        self.optimizer.zero_grad()
        logits = self.model(src, tgt_in)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for src, tgt in self.val_loader:
            src, tgt = src.to(self.device), tgt.to(self.device)
            logits = self.model(src, tgt[:, :-1])
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, name: str):
        path = os.path.join(self.cfg.output_dir, f"{name}.pt")
        self.model.save(path, self.vocab_src, self.vocab_tgt, {
            "src_vocab_size": self.model.src_embed.num_embeddings,
            "tgt_vocab_size": self.model.output_proj.out_features,
            "d_model": self.model.d_model,
            "nhead": 8, "num_encoder_layers": 6, "num_decoder_layers": 6,
            "dim_feedforward": 2048, "dropout": 0.1,
        })
        print(f"Saved: {path}")

    def train(self):
        print(f"Training on {self.device}")
        print(f"Target: {self.cfg.max_iterations:,} steps")

        for epoch in range(self.cfg.max_epochs):
            epoch_loss = 0.0
            for batch_idx, (src, tgt) in enumerate(self.train_loader):
                loss = self.train_step(src, tgt)
                epoch_loss += loss
                self.step += 1

                if self.step % 100 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {self.step:,} | Loss: {loss:.4f} | LR: {lr:.2e}")

                if self.step % self.cfg.eval_every == 0:
                    val_loss = self.validate()
                    print(f"  Val Loss: {val_loss:.4f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("checkpoint_best")

                if self.step % self.cfg.save_every == 0:
                    self.save_checkpoint(f"checkpoint_{self.step}")

                if self.step >= self.cfg.max_iterations:
                    print(f"Reached {self.cfg.max_iterations:,} steps. Training complete.")
                    self.save_checkpoint("checkpoint_final")
                    return

            avg = epoch_loss / max(len(self.train_loader), 1)
            print(f"Epoch {epoch+1} | Avg Loss: {avg:.4f}")
