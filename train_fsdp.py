#!/usr/bin/env python3
"""FSDP training entrypoint for 7B-13B scale models."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import yaml

sys.path.append(str(Path(__file__).parent))
from utils.debugging import check_nccl_config, validate_distributed_setup
from utils.gradient_checkpointing import enable_gradient_checkpointing
from utils.config_validation import validate_training_config
from utils.base_trainer import BaseTrainer


SHARDING_MAP = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
}


class FSDPTrainer(BaseTrainer):
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        super().__init__(config)
        self.args = args
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        log_level = logging.INFO if self.rank == 0 else logging.WARNING
        logging.basicConfig(level=log_level, format=f"[Rank {self.rank}] %(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scaler = None

    def setup_distributed(self) -> None:
        os.environ.setdefault("NCCL_SOCKET_IFNAME", self.config.get("distributed", {}).get("nccl_socket_ifname", "ib0"))
        os.environ.setdefault("NCCL_IB_DISABLE", str(self.config.get("distributed", {}).get("nccl_ib_disable", 0)))
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

        dist.init_process_group(backend="nccl", init_method="env://", rank=self.rank, world_size=self.world_size)
        if self.rank == 0:
            check_nccl_config()
            validate_distributed_setup(self.world_size, self.local_rank)

    def setup_model(self) -> None:
        model_name = self.config["model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = AutoConfig.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=torch.bfloat16 if self.config["training"].get("mixed_precision", "bf16") == "bf16" else torch.float32,
        )

        if self.config["training"].get("gradient_checkpointing", True):
            base_model = enable_gradient_checkpointing(base_model)

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ) if self.config["training"].get("mixed_precision", "bf16") == "bf16" else None

        strategy = SHARDING_MAP.get(self.args.sharding_strategy, ShardingStrategy.FULL_SHARD)

        self.model = FSDP(
            base_model,
            device_id=self.local_rank,
            sharding_strategy=strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            mixed_precision=mp_policy,
            use_orig_params=True,
            limit_all_gathers=True,
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.config["training"]["learning_rate"]))
        self.scaler = GradScaler(enabled=self.config["training"].get("mixed_precision", "bf16") == "fp16")

        if self.rank == 0:
            self.logger.info("FSDP model initialized")

    def _dummy_dataset(self):
        config = self.config

        class Dummy:
            def __len__(self):
                return int(config["data"].get("dataset_size", 2048))

            def __getitem__(self, idx):
                seq_len = int(config["model"].get("max_seq_length", 2048))
                x = torch.randint(0, 32000, (seq_len,))
                return {"input_ids": x, "attention_mask": torch.ones(seq_len), "labels": x.clone()}

        return Dummy()

    def train(self) -> None:
        dataset = self._dummy_dataset()
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=int(self.config["training"].get("batch_size", 1)), sampler=sampler)

        self.model.train()
        for epoch in range(int(self.config["training"].get("num_epochs", 1))):
            sampler.set_epoch(epoch)
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.config["training"].get("mixed_precision", "bf16") != "fp32"):
                    loss = self.model(**batch).loss
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.rank == 0 and step % int(self.config["training"].get("log_interval", 10)) == 0:
                    self.logger.info(f"epoch={epoch} step={step} loss={loss.item():.4f}")

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("training", {})
    cfg["training"].setdefault("mixed_precision", "bf16")
    cfg["training"].setdefault("gradient_checkpointing", True)
    cfg.setdefault("data", {})
    cfg["data"].setdefault("dataset_size", 2048)
    validate_training_config(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSDP training")
    p.add_argument("--config", required=True)
    p.add_argument("--sharding-strategy", default="FULL_SHARD", choices=list(SHARDING_MAP.keys()))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    trainer = FSDPTrainer(cfg, args)
    try:
        trainer.setup_distributed()
        trainer.setup_model()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
