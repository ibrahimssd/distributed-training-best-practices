#!/bin/bash
# slurm/fsdp_8gpu.sh
# SLURM job script for FSDP training on 8 x A100 80 GB (single node)
#
# Usage:
#   sbatch slurm/fsdp_8gpu.sh

#SBATCH --job-name=fsdp_llama_7b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:8
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load python/3.11 cuda/12.1

source /path/to/your/venv/bin/activate

export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=8

# Single node — no InfiniBand needed
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs outputs

echo "FSDP Job $SLURM_JOB_ID started on $(hostname) at $(date)"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    train_fsdp.py \
        --config configs/llama_7b_fsdp.yaml \
        --output-dir "./outputs/fsdp_$SLURM_JOB_ID"

echo "Job $SLURM_JOB_ID finished at $(date)"
