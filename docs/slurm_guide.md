# SLURM Guide for Distributed Training

Practical reference for submitting and managing distributed training jobs
on HPC clusters using SLURM.

---

## Job Submission

```bash
# Submit a job
sbatch slurm/multi_node_32gpu.sh

# Submit with overrides
sbatch --time=24:00:00 --partition=gpu_debug slurm/multi_node_32gpu.sh

# Submit an array of hyperparameter sweep jobs
sbatch --array=0-7 slurm/multi_node_32gpu.sh
```

---

## Monitoring Running Jobs

```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job $JOBID

# Watch job output in real time
tail -f logs/ddp_llama_7b_${JOBID}.out

# Check GPU utilization on all nodes
srun --jobid=$JOBID nvidia-smi

# Interactive session on a compute node (for debugging)
srun --partition=gpu --gres=gpu:a100:1 --pty bash
```

---

## Environment Variables Set by SLURM

SLURM populates these automatically — your scripts read them:

| Variable | Meaning |
|---|---|
| `$SLURM_JOB_ID` | Unique job identifier |
| `$SLURM_NNODES` | Number of nodes allocated |
| `$SLURM_NTASKS_PER_NODE` | Tasks (processes) per node |
| `$SLURM_NODELIST` | List of allocated node hostnames |
| `$SLURM_NODEID` | Index of current node (0-based) |
| `$SLURM_PROCID` | Global rank of current task |
| `$SLURM_LOCALID` | Local rank within current node |
| `$SLURM_CPUS_PER_TASK` | CPU cores allocated per task |

---

## Getting MASTER_ADDR Correctly

```bash
# Always derive MASTER_ADDR from the allocated nodelist — never hardcode
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500  # any free port; 29500 is convention
```

---

## torchrun vs srun

**Use `srun torchrun` for multi-node jobs** (recommended):
```bash
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id=$SLURM_JOB_ID \
    train_ddp.py --config configs/llama_7b_32gpu.yaml
```

**Use `torchrun --standalone` for single-node jobs** (simpler):
```bash
torchrun \
    --standalone \
    --nproc_per_node=4 \
    train_ddp.py --config configs/bert_base_4gpu.yaml
```

---

## Checkpointing for Long Jobs

SLURM may preempt jobs. Always checkpoint frequently and resume cleanly:

```bash
# In your SLURM script, pass the checkpoint path
torchrun ... train_ddp.py \
    --config configs/llama_7b_32gpu.yaml \
    --checkpoint-path ./outputs/llama_7b_${SLURM_JOB_ID}/checkpoint_last.pt
```

For jobs expected to exceed the time limit, use job arrays with dependencies:
```bash
# Submit first segment
JOB1=$(sbatch --parsable slurm/multi_node_32gpu.sh)

# Submit continuation — starts only after JOB1 finishes or is preempted
sbatch --dependency=afterany:$JOB1 slurm/multi_node_32gpu.sh
```

---

## Common SLURM Errors

| Error | Cause | Fix |
|---|---|---|
| `srun: error: Unable to connect` | MASTER_ADDR wrong or firewall | Check `ping $MASTER_ADDR` between nodes |
| `CANCELLED due to time limit` | Job exceeded `--time` | Increase `--time` or add checkpointing |
| `OOM killed` | `--mem` too low | Increase `--mem-per-node` |
| `Requested node configuration is not available` | Wrong GPU model in `--gres` | Check `sinfo -o "%n %G"` |
| `Job step creation temporarily disabled` | Cluster under load | Wait and resubmit |
