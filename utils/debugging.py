#!/usr/bin/env python3
"""
utils/debugging.py - Distributed Training Debugging Utilities

Provides runtime checks and diagnostics for NCCL, DDP setup,
and GPU health. Based on real debugging sessions on the AImotion
HPC cluster managing 80+ researchers.

Author: Ibrahim Siddig
"""

import logging
import os
import socket
import subprocess
from typing import List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ── NCCL checks ───────────────────────────────────────────────────────────────

def check_nccl_config() -> None:
    """
    Warn if common NCCL misconfigurations are detected.

    Covers the most frequent root causes of hangs and slowdowns
    seen on InfiniBand-equipped HPC clusters:
      - Missing NCCL_SOCKET_IFNAME (falls back to slow TCP)
      - IB disabled while the hardware is present
      - Overly verbose NCCL debug output degrading throughput
    """
    issues: List[str] = []

    ifname = os.environ.get("NCCL_SOCKET_IFNAME", "")
    if not ifname:
        issues.append(
            "NCCL_SOCKET_IFNAME is not set. "
            "NCCL may fall back to TCP instead of InfiniBand. "
            "Set NCCL_SOCKET_IFNAME=ib0 (or your IB interface name)."
        )

    ib_disabled = os.environ.get("NCCL_IB_DISABLE", "0")
    if ib_disabled == "1":
        issues.append(
            "NCCL_IB_DISABLE=1 — InfiniBand is disabled. "
            "Set NCCL_IB_DISABLE=0 to enable high-speed interconnect."
        )

    nccl_debug = os.environ.get("NCCL_DEBUG", "")
    if nccl_debug == "INFO":
        issues.append(
            "NCCL_DEBUG=INFO produces very verbose output and can slow "
            "down training. Use NCCL_DEBUG=WARN in production."
        )

    if issues:
        logger.warning("=== NCCL Configuration Warnings ===")
        for issue in issues:
            logger.warning(f"  ⚠  {issue}")
        logger.warning("====================================")
    else:
        logger.info("NCCL configuration looks healthy.")


def check_infiniband_available() -> bool:
    """Return True if InfiniBand devices are visible on this node."""
    try:
        result = subprocess.run(
            ["ibstat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        available = result.returncode == 0
        if available:
            logger.info("InfiniBand detected on this node.")
        else:
            logger.warning("ibstat returned non-zero — IB may not be available.")
        return available
    except FileNotFoundError:
        logger.warning("ibstat not found. Cannot verify InfiniBand availability.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("ibstat timed out.")
        return False


# ── Distributed setup validation ─────────────────────────────────────────────

def validate_distributed_setup(world_size: int, local_rank: int) -> None:
    """
    Run a lightweight all-reduce smoke-test to confirm that all
    processes can communicate before starting expensive training.

    Raises RuntimeError if the test fails.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "validate_distributed_setup called before dist.init_process_group()."
        )

    # Smoke-test: each rank contributes its rank value; sum should equal
    # world_size * (world_size - 1) / 2
    rank = dist.get_rank()
    test_tensor = torch.tensor(float(rank), device=f"cuda:{local_rank}")
    dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)

    expected = world_size * (world_size - 1) / 2
    if abs(test_tensor.item() - expected) > 1e-3:
        raise RuntimeError(
            f"Distributed all-reduce smoke-test FAILED. "
            f"Expected sum {expected}, got {test_tensor.item()}."
        )

    if rank == 0:
        logger.info(
            f"Distributed setup validated: {world_size} processes, "
            f"all-reduce smoke-test passed."
        )


def log_node_info(rank: int) -> None:
    """Log hostname and GPU info for each participating node."""
    hostname = socket.gethostname()
    num_gpus = torch.cuda.device_count()
    gpu_names = [
        torch.cuda.get_device_name(i) for i in range(num_gpus)
    ]
    logger.info(
        f"[Rank {rank}] Host: {hostname} | "
        f"GPUs: {num_gpus} x {gpu_names[0] if gpu_names else 'N/A'}"
    )


# ── Gradient health checks ────────────────────────────────────────────────────

def check_gradients(
    model: torch.nn.Module,
    rank: int = 0,
    nan_threshold: float = 0.0,
) -> bool:
    """
    Iterate over model parameters and warn about NaN/Inf gradients.

    Returns True if all gradients are finite, False otherwise.
    Useful to call after backward() before optimizer.step().
    """
    has_issues = False
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            logger.error(
                f"[Rank {rank}] Non-finite gradient detected in '{name}'. "
                "This will corrupt model weights. Check loss scaling."
            )
            has_issues = True
    if not has_issues and rank == 0:
        logger.debug("Gradient health check passed — all gradients are finite.")
    return not has_issues


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the global L2 norm of all parameter gradients."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ── Environment diagnostics ───────────────────────────────────────────────────

def print_environment_summary(rank: int = 0) -> None:
    """Print a diagnostic summary of the training environment (rank 0 only)."""
    if rank != 0:
        return

    logger.info("=== Environment Summary ===")
    logger.info(f"  PyTorch version : {torch.__version__}")
    logger.info(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version    : {torch.version.cuda}")
        logger.info(f"  cuDNN version   : {torch.backends.cudnn.version()}")
        logger.info(f"  GPU count       : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"  GPU {i}: {props.name} | "
                f"{props.total_memory / 1024**3:.1f} GB | "
                f"CC {props.major}.{props.minor}"
            )
    for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK",
                "LOCAL_RANK", "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE"]:
        logger.info(f"  {key:25s}: {os.environ.get(key, '<not set>')}")
    logger.info("===========================")
