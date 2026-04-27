#!/usr/bin/env python3
"""
examples/01_single_gpu_baseline.py
────────────────────────────────────────────────────────────────────────────────
Example 1 of 4 — Single GPU Baseline

The simplest possible training loop. Start here before moving to distributed.
Understanding this code is a prerequisite for understanding DDP, FSDP, and
hybrid parallelism — all of which wrap this core loop.

Run:
    python examples/01_single_gpu_baseline.py

No torchrun, no SLURM, no config files needed.
────────────────────────────────────────────────────────────────────────────────
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Minimal dataset ───────────────────────────────────────────────────────────

class TinyTextDataset(Dataset):
    """Synthetic token ID dataset for demonstration."""
    def __init__(self, num_samples: int = 2000, seq_len: int = 128, vocab: int = 30000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        ids = torch.randint(0, self.vocab, (self.seq_len,))
        return {"input_ids": ids, "labels": ids.clone()}


# ── Minimal transformer model ─────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    """Single-layer transformer for baseline demonstration."""
    def __init__(self, vocab_size: int = 30000, d_model: int = 256,
                 nhead: int = 8, seq_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=1024, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        return loss, logits


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Model
    model = TinyTransformer().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Data
    dataset = TinyTextDataset()
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader))

    # Training
    model.train()
    for epoch in range(3):
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, labels)
            loss.backward()

            # Gradient clipping — important habit even for small models
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % 20 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} step {step}/{len(loader)} "
                      f"loss={loss.item():.4f} lr={lr:.2e}")

        elapsed = time.time() - t0
        print(f"Epoch {epoch}: avg_loss={epoch_loss/len(loader):.4f} "
              f"time={elapsed:.1f}s")

    print("\nDone. Next step: see examples/02_ddp_single_node.py")


if __name__ == "__main__":
    train()
