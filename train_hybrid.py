#!/usr/bin/env python3
"""Hybrid parallel training (TP + DP/FSDP + pipeline stubs) for 70B+ models."""

import argparse
import logging
import os
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, AutoModelForCausalLM
import yaml

from utils.gradient_checkpointing import enable_gradient_checkpointing
from utils.config_validation import validate_training_config
from utils.base_trainer import BaseTrainer


class HybridTrainer(BaseTrainer):
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

        self.tp_size = args.tp_size
        self.pp_size = args.pp_size
        self.dp_size = args.dp_size
        self.tp_group = None
        self.model = None
        self.optimizer = None

    def setup_distributed(self) -> None:
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        dist.init_process_group(backend="nccl", init_method="env://", rank=self.rank, world_size=self.world_size)

        if self.tp_size > 1:
            tp_group_id = self.rank // self.tp_size
            tp_ranks = list(range(tp_group_id * self.tp_size, (tp_group_id + 1) * self.tp_size))
            self.tp_group = dist.new_group(tp_ranks)

        if self.rank == 0:
            self.logger.info(
                f"Hybrid setup initialized (tp={self.tp_size}, pp={self.pp_size}, dp={self.dp_size}, world={self.world_size})"
            )

    def setup_model(self) -> None:
        model_name = self.config["model"]["name"]
        model_cfg = AutoConfig.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_cfg,
            torch_dtype=torch.bfloat16 if self.config["training"].get("mixed_precision", "bf16") == "bf16" else torch.float32,
        )

        if self.config["training"].get("gradient_checkpointing", True):
            base_model = enable_gradient_checkpointing(base_model)

        # Pipeline parallelism stage partitioning stub.
        # This placeholder keeps the script structure ready for stage-wise model partitioning.
        base_model = self._pipeline_partition_stub(base_model)

        self.model = FSDP(base_model, device_id=self.local_rank, use_orig_params=False, limit_all_gathers=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.config["training"]["learning_rate"]))

    def _pipeline_partition_stub(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def _dummy_dataset(self):
        config = self.config

        class Dummy:
            def __len__(self):
                return int(config["data"].get("dataset_size", 1024))

            def __getitem__(self, idx):
                seq_len = int(config["model"].get("max_seq_length", 2048))
                x = torch.randint(0, 32000, (seq_len,))
                return {"input_ids": x, "attention_mask": torch.ones(seq_len), "labels": x.clone()}

        return Dummy()

    def train(self) -> None:
        dataset = self._dummy_dataset()
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=int(self.config["training"].get("batch_size", 1)))

        self.model.train()
        for epoch in range(int(self.config["training"].get("num_epochs", 1))):
            sampler.set_epoch(epoch)
            for step, batch in enumerate(dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.config["training"].get("mixed_precision", "bf16") != "fp32"):
                    loss = self.model(**batch).loss
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
    cfg["data"].setdefault("dataset_size", 1024)
    validate_training_config(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid TP+PP+DP training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    trainer = HybridTrainer(cfg, args)
    try:
        trainer.setup_distributed()
        trainer.setup_model()
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
