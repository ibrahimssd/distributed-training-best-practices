#!/usr/bin/env python3
"""
examples/02_ddp_single_node.py
────────────────────────────────────────────────────────────────────────────────
Example 2 of 4 — DDP Single Node

Converts the single-GPU baseline to Data Parallel training on one machine.
Minimal changes from Example 1 are marked with "# DDP CHANGE".

Run (4 GPUs):
    torchrun --standalone --nproc_per_node=4 examples/02_ddp_single_node.py

What changes from single-GPU:
  1. dist.init_process_group()   — initialise communication backend
  2. DistributedSampler          — each GPU sees a different data shard
  3. model wrapped with DDP      — gradients are all-reduced automatically
  4. Only rank 0 prints/saves    — avoid duplicated output
────────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist                              # DDP CHANGE
from torch.nn.parallel import DistributedDataParallel as DDP # DDP CHANGE
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler  # DDP CHANGE


# ── Same dataset and model as Example 1 ──────────────────────────────────────

class TinyTextDataset(Dataset):
    def __init__(self, num_samples=8000, seq_len=128, vocab=30000):
        self.n = num_samples; self.s = seq_len; self.v = vocab
    def __len__(self): return self.n
    def __getitem__(self, idx):
        ids = torch.randint(0, self.v, (self.s,))
        return {"input_ids": ids, "labels": ids.clone()}


class TinyTransformer(nn.Module):
    def __init__(self, vocab=30000, d=256, h=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, h, 1024, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(d, vocab)
    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)
        logits = self.head(self.enc(x))
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        ) if labels is not None else None
        return loss, logits


# ── DDP training loop ─────────────────────────────────────────────────────────

def train():
    # ── DDP CHANGE 1: read rank information from environment ──────────────────
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # ── DDP CHANGE 2: initialise process group ────────────────────────────────
    dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Only rank 0 prints — avoids N copies of every log line
    def log(msg):
        if rank == 0: print(msg)

    log(f"Training on {world_size} GPUs")

    # Model
    model = TinyTransformer().to(device)

    # ── DDP CHANGE 3: wrap model with DDP ─────────────────────────────────────
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── DDP CHANGE 4: use DistributedSampler ─────────────────────────────────
    dataset = TinyTextDataset(num_samples=8000)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    loader = DataLoader(
        dataset, batch_size=16, sampler=sampler,
        num_workers=2, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(3):
        # ── DDP CHANGE 5: tell sampler the current epoch ──────────────────────
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(loader):
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(ids, labels)
            loss.backward()
            # DDP all-reduces gradients automatically during backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if step % 20 == 0:
                log(f"  Epoch {epoch} step {step}/{len(loader)} "
                    f"loss={loss.item():.4f}")

        log(f"Epoch {epoch}: avg_loss={epoch_loss/len(loader):.4f} "
            f"time={time.time()-t0:.1f}s")

    # ── DDP CHANGE 6: always clean up ─────────────────────────────────────────
    dist.destroy_process_group()
    log("\nDone. Next step: see examples/03_fsdp_large_model.py")


if __name__ == "__main__":
    train()
