# NCCL Troubleshooting Guide

Common distributed training failures and their fixes,
collected from real debugging sessions on the AImotion Bavaria HPC cluster.

---

## Symptom: Job hangs indefinitely at startup

**Cause:** Process group initialisation (`dist.init_process_group`) never completes.
All ranks are waiting to rendezvous but cannot reach each other.

**Checklist:**
```bash
# 1. Verify MASTER_ADDR is reachable from all nodes
ping $MASTER_ADDR

# 2. Check MASTER_PORT is not in use
ss -tlnp | grep 29500

# 3. Confirm all ranks agree on WORLD_SIZE
echo "WORLD_SIZE=$WORLD_SIZE RANK=$RANK LOCAL_RANK=$LOCAL_RANK"

# 4. Check firewall rules — NCCL needs TCP and UDP on MASTER_PORT
sudo iptables -L INPUT | grep REJECT
```

**Fix:**
```bash
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
# Add a longer timeout for large models
dist.init_process_group(..., timeout=datetime.timedelta(minutes=60))
```

---

## Symptom: Training hangs mid-run (not at startup)

**Cause:** One rank crashed or finished early; others wait indefinitely at a barrier or all-reduce.

**Diagnosis:**
```bash
# Check if all SLURM tasks are still alive
squeue -j $SLURM_JOB_ID

# Enable async error handling so the job fails fast instead of hanging
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Get a Python stack trace from a hung process
kill -USR1 $PID  # sends SIGUSR1 which PyTorch handles to print stacks
```

**Fix:**
```python
# Set a timeout on all-reduce operations
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(minutes=30)
)
```

---

## Symptom: Slow training — much lower throughput than expected

**Cause A — TCP fallback instead of InfiniBand:**
```bash
# Check which interface NCCL is using
export NCCL_DEBUG=INFO
# Look for lines like: NCCL INFO NET/Socket : Using [0]eth0
# If you see eth0 or lo instead of ib0, NCCL is using slow TCP
```

**Fix:**
```bash
export NCCL_SOCKET_IFNAME=ib0   # force InfiniBand interface
export NCCL_IB_DISABLE=0        # make sure IB is not disabled
export NCCL_IB_HCA=mlx5_0:1    # specify exact HCA (check with: ibstat)
```

**Cause B — Small NCCL bucket size:**
```python
# Default bucket_cap_mb=25 is conservative; try 100 MB for large models
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=100)
```

**Cause C — DataLoader is the bottleneck:**
```python
# Profile to confirm
# If GPU utilization is < 80% and dataloader is slow:
DataLoader(dataset, num_workers=8, pin_memory=True,
           prefetch_factor=2, persistent_workers=True)
```

---

## Symptom: `NCCL error: unhandled cuda error` or `CUDA error: no kernel image`

**Cause:** NCCL version is incompatible with the installed CUDA version.

**Fix:**
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.nccl.version())"
nvcc --version

# Reinstall PyTorch with matching CUDA version
pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## Symptom: `RuntimeError: Expected all tensors to be on the same device`

**Cause:** Model parameters are on different devices after DDP/FSDP wrapping.

**Fix:**
```python
# Always set device BEFORE wrapping
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
model = model.to(device)          # move BEFORE wrapping
model = DDP(model, device_ids=[local_rank])
```

---

## Symptom: `NCCL WARN Timeout` after exactly 30 minutes

**Cause:** Default NCCL timeout (1800 seconds) hit on large model checkpointing or slow all-gather.

**Fix:**
```python
import datetime
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(hours=2)  # increase for large models
)
```

---

## Symptom: Loss is NaN after first gradient update

**Cause A — FP16 overflow:**
```python
# Switch from fp16 to bf16 (A100/H100 only — more numerically stable)
# In FSDP:
MixedPrecision(param_dtype=torch.bfloat16, ...)
# In DDP with GradScaler:
scaler = GradScaler(init_scale=2**10)  # start with smaller scale
```

**Cause B — Learning rate too high:**
```python
# Linear scaling rule: if base_lr=2e-5 for 1 GPU, use 2e-5 * world_size
# But cap at ~1e-4 regardless — gradient noise grows with batch size
scaled_lr = min(base_lr * world_size, 1e-4)
```

**Cause C — Gradient explosion from non-finite input:**
```python
# Add this check in your training loop
from utils.debugging import check_gradients
if not check_gradients(model, rank=rank):
    logger.error("NaN/Inf gradients detected — skipping this batch")
    optimizer.zero_grad()
    continue
```

---

## Useful NCCL Environment Variables

| Variable | Recommended Value | Purpose |
|---|---|---|
| `NCCL_SOCKET_IFNAME` | `ib0` | Force InfiniBand interface |
| `NCCL_IB_DISABLE` | `0` | Enable InfiniBand |
| `NCCL_IB_HCA` | `mlx5_0:1` | Specify HCA device |
| `NCCL_NET_GDR_LEVEL` | `5` | Enable GPU Direct RDMA |
| `NCCL_DEBUG` | `WARN` | Logging verbosity (INFO is very noisy) |
| `NCCL_ASYNC_ERROR_HANDLING` | `1` | Fail fast instead of hang |
| `NCCL_TIMEOUT` | `3600` | All-reduce timeout in seconds |
| `NCCL_BUFFSIZE` | `2097152` | Communication buffer size (2 MB) |

---

## Diagnostic One-Liner

Run this on all nodes before starting a job to confirm NCCL is healthy:

```bash
python -c "
import torch, torch.distributed as dist, os
dist.init_process_group('nccl', init_method='env://')
t = torch.ones(1).cuda()
dist.all_reduce(t)
print(f'Rank {dist.get_rank()}/{dist.get_world_size()} OK, sum={t.item()}')
dist.destroy_process_group()
"
```

Expected output (4 GPUs):
```
Rank 0/4 OK, sum=4.0
Rank 1/4 OK, sum=4.0
Rank 2/4 OK, sum=4.0
Rank 3/4 OK, sum=4.0
```
