#!/usr/bin/env python3
"""
examples/04_profiling_and_debugging.py
────────────────────────────────────────────────────────────────────────────────
Example 4 of 4 — Profiling and Debugging

Demonstrates the production debugging and profiling utilities built into
this repository. Shows how to:
  1. Profile GPU kernels with GPUProfiler
  2. Track memory to diagnose OOM issues
  3. Detect NaN/Inf gradients early
  4. Validate NCCL configuration before long runs
  5. Read Prometheus metrics in a running job

Run (single GPU — profiling works on 1 GPU):
    python examples/04_profiling_and_debugging.py

Run (multi GPU):
    torchrun --standalone --nproc_per_node=2 examples/04_profiling_and_debugging.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

# Add parent to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.profiling import GPUProfiler, MemoryTracker
from utils.debugging import (
    check_nccl_config,
    compute_gradient_norm,
    check_gradients,
    print_environment_summary,
    log_node_info,
)
from utils.monitoring import setup_prometheus_metrics, update_training_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Minimal model ─────────────────────────────────────────────────────────────

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.GELU(),
            nn.Linear(2048, 512),
        )
    def forward(self, x):
        return self.layers(x)


# ── Demo 1: Memory tracking ───────────────────────────────────────────────────

def demo_memory_tracking():
    if not torch.cuda.is_available():
        logger.info("[SKIP] Memory tracking demo requires CUDA.")
        return

    logger.info("=== Demo 1: Memory Tracking ===")
    device = "cuda:0"
    tracker = MemoryTracker(device=device)
    tracker.reset()

    tracker.snapshot("before_model_load")
    model = SmallModel().to(device)
    tracker.snapshot("after_model_load")

    batch = torch.randn(64, 512, device=device)
    out = model(batch)
    tracker.snapshot("after_forward")

    loss = out.mean()
    loss.backward()
    tracker.snapshot("after_backward")

    tracker.report()
    logger.info("Memory tracking complete.\n")


# ── Demo 2: GPU Profiler ──────────────────────────────────────────────────────

def demo_gpu_profiler():
    if not torch.cuda.is_available():
        logger.info("[SKIP] GPU profiler demo requires CUDA.")
        return

    logger.info("=== Demo 2: GPU Profiler ===")
    device = "cuda:0"
    model = SmallModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    profiler = GPUProfiler(
        output_dir="/tmp/profiler_demo",
        rank=0,
        wait_steps=1,
        warmup_steps=1,
        active_steps=3,
    )
    profiler.start()

    for step in range(6):
        batch = torch.randn(32, 512, device=device)
        loss = model(batch).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        profiler.step()
        logger.info(f"  Profiler step {step}/5 loss={loss.item():.4f}")

    profiler.stop()
    logger.info("Profiling complete. TensorBoard traces in /tmp/profiler_demo\n")
    logger.info("View with: tensorboard --logdir /tmp/profiler_demo\n")


# ── Demo 3: Gradient health checks ───────────────────────────────────────────

def demo_gradient_checks():
    logger.info("=== Demo 3: Gradient Health Checks ===")
    model = SmallModel()
    if torch.cuda.is_available():
        model = model.cuda()

    device = next(model.parameters()).device
    batch = torch.randn(16, 512, device=device)
    loss = model(batch).mean()
    loss.backward()

    # Normal gradients
    norm = compute_gradient_norm(model)
    healthy = check_gradients(model, rank=0)
    logger.info(f"  Normal gradients — norm={norm:.4f} healthy={healthy}")

    # Inject NaN to simulate FP16 overflow
    for p in model.parameters():
        if p.grad is not None:
            p.grad[0] = float("nan")
            break

    healthy = check_gradients(model, rank=0)
    logger.info(f"  After NaN injection — healthy={healthy} (expected: False)\n")


# ── Demo 4: NCCL config check ─────────────────────────────────────────────────

def demo_nccl_check():
    logger.info("=== Demo 4: NCCL Configuration Check ===")
    # Simulate missing config
    old = os.environ.pop("NCCL_SOCKET_IFNAME", None)
    check_nccl_config()     # should warn about missing IFNAME
    if old:
        os.environ["NCCL_SOCKET_IFNAME"] = old

    # Simulate correct config
    os.environ["NCCL_SOCKET_IFNAME"] = "ib0"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_DEBUG"] = "WARN"
    check_nccl_config()     # should be clean
    logger.info("")


# ── Demo 5: Prometheus metrics ────────────────────────────────────────────────

def demo_prometheus():
    logger.info("=== Demo 5: Prometheus Metrics ===")
    setup_prometheus_metrics(port=8099)
    logger.info("  Metrics server started on http://localhost:8099/metrics")

    for step in range(5):
        fake_loss = 2.0 / (step + 1)
        fake_lr   = 3e-4 * (0.9 ** step)
        fake_tps  = 50000 + step * 1000
        update_training_metrics(
            loss=fake_loss, lr=fake_lr,
            tokens_per_sec=fake_tps, grad_norm=1.0 - step * 0.1,
        )
        logger.info(f"  Step {step}: loss={fake_loss:.4f} tps={fake_tps:,}")
        time.sleep(0.3)

    logger.info("  Metrics updated. Scrape http://localhost:8099/metrics to verify.\n")


# ── Demo 6: Environment summary ───────────────────────────────────────────────

def demo_environment():
    logger.info("=== Demo 6: Environment Summary ===")
    print_environment_summary(rank=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("Distributed Training — Profiling & Debugging Demo")
    logger.info("=" * 60)

    demo_environment()
    demo_nccl_check()
    demo_memory_tracking()
    demo_gpu_profiler()
    demo_gradient_checks()
    demo_prometheus()

    logger.info("All demos complete.")
    logger.info("You are now ready to use the full train_ddp.py / train_fsdp.py pipelines.")


if __name__ == "__main__":
    main()
