#!/usr/bin/env python3
"""
examples/03_fsdp_large_model.py
────────────────────────────────────────────────────────────────────────────────
Example 3 of 4 — FSDP for Large Models

When a model is too big to fit on a single GPU (even with gradient checkpointing),
FSDP shards the parameters across all ranks. Each rank stores and computes only
1/world_size of the model at any given time.

Run (4 GPUs, each seeing a "large" model):
    torchrun --standalone --nproc_per_node=4 examples/03_fsdp_large_model.py

Key differences from DDP (Example 2):
  - model is wrapped with FSDP instead of DDP
  - Mixed precision is configured via FSDPMixedPrecision (not GradScaler)
  - Gradient clipping uses model.clip_grad_norm_() (FSDP-aware version)
  - Checkpoint saving requires FSDP.state_dict_type context manager
────────────────────────────────────────────────────────────────────────────────
"""

import os
import time
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# ── "Large" model (scaled up to demonstrate memory savings) ──────────────────

class TransformerBlock(nn.Module):
    """Single transformer block — the FSDP wrapping unit."""
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class LargerTransformer(nn.Module):
    """12-layer transformer (~85 M parameters at d=512)."""
    def __init__(self, vocab=30000, d=512, nhead=8, num_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(
            [TransformerBlock(d, nhead) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return loss, logits


# ── Dataset ───────────────────────────────────────────────────────────────────

class TinyDataset(Dataset):
    def __init__(self, n=4000, s=256): self.n = n; self.s = s
    def __len__(self): return self.n
    def __getitem__(self, i):
        ids = torch.randint(0, 30000, (self.s,))
        return {"input_ids": ids, "labels": ids.clone()}


# ── FSDP training ─────────────────────────────────────────────────────────────

def train():
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    def log(msg):
        if rank == 0: print(msg)

    log(f"FSDP training on {world_size} GPUs")

    # ── FSDP CHANGE 1: define mixed precision policy ──────────────────────────
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # ── FSDP CHANGE 2: define wrapping policy (shard at block level) ──────────
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Create model on CPU first — FSDP moves parameters to GPU on first use
    model = LargerTransformer()
    total = sum(p.numel() for p in model.parameters())
    log(f"Model: {total:,} parameters")

    # ── FSDP CHANGE 3: wrap with FSDP instead of DDP ─────────────────────────
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,
    )

    # Optimizer MUST be created after FSDP wrapping
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    dataset = TinyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size,
                                  rank=rank, shuffle=True, drop_last=True)
    loader  = DataLoader(dataset, batch_size=8, sampler=sampler,
                         num_workers=2, pin_memory=True)

    for epoch in range(2):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(loader):
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(ids, labels)
            loss.backward()

            # ── FSDP CHANGE 4: use FSDP-aware gradient clipping ───────────────
            model.clip_grad_norm_(max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

            if step % 10 == 0:
                mem = torch.cuda.memory_allocated(device) / 1024**3
                log(f"  Epoch {epoch} step {step} loss={loss.item():.4f} "
                    f"mem={mem:.2f}GB/GPU")

        log(f"Epoch {epoch}: avg_loss={epoch_loss/len(loader):.4f} "
            f"time={time.time()-t0:.1f}s")

    # ── FSDP CHANGE 5: save checkpoint — must use FSDP state dict API ─────────
    if rank == 0:
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
            state = model.state_dict()
        torch.save(state, "/tmp/fsdp_example_checkpoint.pt")
        log("Checkpoint saved to /tmp/fsdp_example_checkpoint.pt")

    dist.barrier()
    dist.destroy_process_group()
    log("\nDone. Next step: see examples/04_profiling_and_debugging.py")


if __name__ == "__main__":
    train()
