#!/bin/bash
# slurm/single_node_4gpu.sh
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job script for 4-GPU single-node DDP training (BERT-scale models)
#
# Usage:
#   sbatch slurm/single_node_4gpu.sh
#
# Author: Ibrahim Siddig
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ddp_bert_base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load python/3.11
module load cuda/12.1

source /path/to/your/venv/bin/activate

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4

# Single-node: use loopback, no InfiniBand needed
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs outputs

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_ddp.py \
        --config configs/bert_base_4gpu.yaml \
        --output-dir "./outputs/bert_base_$SLURM_JOB_ID"

echo "Job $SLURM_JOB_ID finished at $(date)"
