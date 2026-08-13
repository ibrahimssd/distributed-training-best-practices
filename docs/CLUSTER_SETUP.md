# Cluster Setup Guide

## Prerequisites
- CUDA 12.x driver/toolkit
- NCCL 2.18+
- PyTorch 2.2+
- SLURM with `gres/gpu`
- InfiniBand or RoCE network

## Node baseline
1. Set hostnames and passwordless SSH.
2. Validate GPU + NCCL:
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.device_count())"
   ```
3. Export NCCL defaults:
   ```bash
   export NCCL_SOCKET_IFNAME=ib0
   export NCCL_IB_DISABLE=0
   export NCCL_ASYNC_ERROR_HANDLING=1
   ```

## SLURM essentials
- Use consistent CUDA + Python modules on all nodes.
- Pin one process per GPU (`--ntasks-per-node=$GPUS`).
- Set `MASTER_ADDR` to rank-0 node hostname.

## Launch examples
- DDP: `slurm/multi_node_32gpu.sh`
- FSDP: `slurm/fsdp_8gpu.sh`

## Health checks
- Run all-reduce smoke test before training.
- Verify network interface in NCCL logs.
- Fail fast if rank count mismatches expected world size.
