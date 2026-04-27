#!/bin/bash
# slurm/multi_node_32gpu.sh
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job script for 32-GPU multi-node DDP training (4 nodes x 8 GPUs)
# Tested on AImotion Bavaria HPC cluster (SLURM + InfiniBand + A100s)
#
# Usage:
#   sbatch slurm/multi_node_32gpu.sh
#
# Author: Ibrahim Siddig
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp_llama_7b
#SBATCH --partition=gpu_large            # adjust to your cluster partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8             # one task per GPU
#SBATCH --cpus-per-task=8               # CPU cores for dataloader workers
#SBATCH --gres=gpu:a100:8               # 8 x A100 per node
#SBATCH --mem=480G                       # ~60 GB per GPU task
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out         # stdout: jobname_jobid.out
#SBATCH --error=logs/%x_%j.err          # stderr: jobname_jobid.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@institution.edu

# ── Environment setup ─────────────────────────────────────────────────────────
set -euo pipefail

# Load required modules (adjust module names to your cluster)
module purge
module load python/3.11
module load cuda/12.1
module load nccl/2.18
module load openmpi/4.1

# Activate virtual environment
source /path/to/your/venv/bin/activate

# ── Distributed training environment variables ────────────────────────────────
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export NODE_RANK=$SLURM_NODEID

# ── NCCL tuning for InfiniBand HDR (200 Gbps) ────────────────────────────────
export NCCL_SOCKET_IFNAME=ib0           # InfiniBand interface (verify: ip a)
export NCCL_IB_DISABLE=0               # enable InfiniBand
export NCCL_IB_HCA=mlx5_0:1           # HCA device (verify: ibstat)
export NCCL_NET_GDR_LEVEL=5           # GPU Direct RDMA
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600               # 60-minute timeout for large models

# ── PyTorch / CUDA settings ──────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ── Logging ──────────────────────────────────────────────────────────────────
mkdir -p logs outputs
echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Job name     : $SLURM_JOB_NAME"
echo "Nodes        : $SLURM_NNODES"
echo "Tasks/node   : $SLURM_NTASKS_PER_NODE"
echo "World size   : $WORLD_SIZE"
echo "Master addr  : $MASTER_ADDR:$MASTER_PORT"
echo "Start time   : $(date)"
echo "============================================================"

# ── Launch with torchrun (replaces deprecated torch.distributed.launch) ───────
srun torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_NTASKS_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    --max_restarts=3 \
    train_ddp.py \
        --config configs/llama_7b_32gpu.yaml \
        --output-dir "./outputs/llama_7b_$SLURM_JOB_ID"

echo "============================================================"
echo "Job completed: $(date)"
echo "============================================================"
