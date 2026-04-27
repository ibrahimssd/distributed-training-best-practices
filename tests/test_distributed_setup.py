#!/usr/bin/env python3
"""
tests/test_distributed_setup.py
Unit and integration tests for distributed training utilities.

Run single-process tests (no GPU required):
    pytest tests/test_distributed_setup.py -v

Run distributed smoke-test (requires 2 GPUs):
    torchrun --standalone --nproc_per_node=2 \
        -m pytest tests/test_distributed_setup.py -v -k distributed

Author: Ibrahim Siddig
"""

import os
import logging
import unittest
from unittest.mock import patch, MagicMock

import torch
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gpu_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0

def _multi_gpu_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


# ── NCCL config checks ────────────────────────────────────────────────────────

class TestCheckNcclConfig(unittest.TestCase):

    def test_warns_when_ifname_missing(self):
        """Should log a warning when NCCL_SOCKET_IFNAME is not set."""
        env = {k: v for k, v in os.environ.items()
               if k != "NCCL_SOCKET_IFNAME"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertLogs(level="WARNING") as log_ctx:
                from utils.debugging import check_nccl_config
                check_nccl_config()
        self.assertTrue(
            any("NCCL_SOCKET_IFNAME" in msg for msg in log_ctx.output),
            "Expected warning about missing NCCL_SOCKET_IFNAME",
        )

    def test_warns_when_ib_disabled(self):
        """Should warn when InfiniBand is explicitly disabled."""
        env_patch = {"NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "ib0"}
        with patch.dict(os.environ, env_patch):
            with self.assertLogs(level="WARNING") as log_ctx:
                from utils.debugging import check_nccl_config
                check_nccl_config()
        self.assertTrue(
            any("NCCL_IB_DISABLE" in msg for msg in log_ctx.output),
        )

    def test_no_warning_when_correctly_configured(self):
        """Should not warn when NCCL env vars are properly set."""
        env_patch = {
            "NCCL_SOCKET_IFNAME": "ib0",
            "NCCL_IB_DISABLE": "0",
            "NCCL_DEBUG": "WARN",
        }
        with patch.dict(os.environ, env_patch):
            # check_nccl_config should log INFO only
            with self.assertLogs(level="INFO") as log_ctx:
                from utils.debugging import check_nccl_config
                check_nccl_config()
        warnings = [m for m in log_ctx.output if "WARNING" in m]
        self.assertEqual(len(warnings), 0, f"Unexpected warnings: {warnings}")


# ── Gradient health checks ────────────────────────────────────────────────────

class TestGradientChecks(unittest.TestCase):

    def _simple_model(self) -> torch.nn.Module:
        return torch.nn.Linear(4, 2)

    def test_check_gradients_passes_for_finite_grads(self):
        from utils.debugging import check_gradients
        model = self._simple_model()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        result = check_gradients(model, rank=0)
        self.assertTrue(result)

    def test_check_gradients_fails_for_nan_grads(self):
        from utils.debugging import check_gradients
        model = self._simple_model()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        # Manually corrupt gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.fill_(float("nan"))
        result = check_gradients(model, rank=0)
        self.assertFalse(result)

    def test_compute_gradient_norm(self):
        from utils.debugging import compute_gradient_norm
        model = self._simple_model()
        x = torch.randn(2, 4)
        loss = model(x).sum()
        loss.backward()
        norm = compute_gradient_norm(model)
        self.assertGreater(norm, 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(norm)))


# ── Memory tracker ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _gpu_available(), reason="Requires CUDA GPU")
class TestMemoryTracker(unittest.TestCase):

    def test_snapshot_and_report(self):
        from utils.profiling import MemoryTracker
        tracker = MemoryTracker(device="cuda:0")
        # Allocate some memory
        t = torch.zeros(1024, 1024, device="cuda:0")
        tracker.snapshot("after_alloc")
        del t
        tracker.snapshot("after_del")
        # Should not raise
        tracker.report()

    def test_reset_clears_snapshots(self):
        from utils.profiling import MemoryTracker
        tracker = MemoryTracker(device="cuda:0")
        tracker.snapshot("step1")
        self.assertIn("step1", tracker.snapshots)
        tracker.reset()
        self.assertEqual(len(tracker.snapshots), 0)


# ── Config loading ────────────────────────────────────────────────────────────

class TestLoadConfig(unittest.TestCase):

    def test_loads_bert_config(self):
        """bert_base_4gpu.yaml should load without error."""
        import yaml
        config_path = "configs/bert_base_4gpu.yaml"
        if not os.path.exists(config_path):
            self.skipTest(f"{config_path} not found")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("model", cfg)
        self.assertIn("training", cfg)
        self.assertIn("data", cfg)
        self.assertEqual(cfg["model"]["max_seq_length"], 512)

    def test_loads_llama_config(self):
        """llama_7b_32gpu.yaml should load without error."""
        import yaml
        config_path = "configs/llama_7b_32gpu.yaml"
        if not os.path.exists(config_path):
            self.skipTest(f"{config_path} not found")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg["model"]["max_seq_length"], 4096)
        self.assertEqual(cfg["distributed"]["nccl_socket_ifname"], "ib0")


# ── Distributed smoke test (requires 2 GPUs + torchrun) ─────────────────────

@pytest.mark.distributed
@pytest.mark.skipif(not _multi_gpu_available(), reason="Requires 2+ GPUs")
class TestDistributedSmoke(unittest.TestCase):

    def test_all_reduce_smoke(self):
        """
        Minimal all-reduce sanity check.
        Run with: torchrun --standalone --nproc_per_node=2 -m pytest ...
        """
        import torch.distributed as dist
        if not dist.is_initialized():
            self.skipTest("dist not initialized — run with torchrun")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")

        tensor = torch.tensor(float(rank), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        expected = world_size * (world_size - 1) / 2
        self.assertAlmostEqual(tensor.item(), expected, places=3)


if __name__ == "__main__":
    unittest.main()
