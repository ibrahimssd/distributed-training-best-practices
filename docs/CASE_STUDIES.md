# Case Studies

Real-world problem solving from managing HPC infrastructure supporting 80+ researchers.

## Case Study 1: Scaling Efficiency Optimization

### Problem
- **Model**: 3B parameter German language model
- **Setup**: 4 nodes × 8 A100 GPUs (32 total)
- **Issue**: Only achieving 52% scaling efficiency
- **Impact**: 3.7× longer training time than expected

### Investigation

```bash
# Step 1: Check NCCL communication
export NCCL_DEBUG=INFO
torchrun --nnodes=4 --nproc_per_node=8 train_ddp.py

# Found in logs: "Using network TCP" instead of "Using network IB"
# Root cause: NCCL falling back to TCP instead of InfiniBand
```

### Root Cause Analysis

1. **Network Interface Detection**: NCCL couldn't detect InfiniBand interface
2. **TCP Fallback**: Communication over 1Gbps Ethernet instead of 100Gbps InfiniBand
3. **Gradient Sync Bottleneck**: 95% of step time spent on gradient synchronization

### Solution Implementation

```bash
# Force InfiniBand interface
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5

# Verify InfiniBand is active
ibstat  # Should show active ports
ibv_devinfo  # Show InfiniBand devices
```

### Results

| Metric | Before | After | Improvement |
|--------|---------|-------|--------------|
| Scaling Efficiency | 52% | 92% | +77% |
| Tokens/sec | 2,200 | 8,100 | +268% |
| Step Time | 12.5s | 3.4s | -73% |
| Training Time | 8.2 days | 2.2 days | -73% |

**Key Learning**: Always verify NCCL is using the fastest available network interface.

---

## Case Study 2: Memory Optimization for Large Models

### Problem
- **Model**: LLaMA-13B (13 billion parameters)
- **Hardware**: 8× A100 40GB GPUs
- **Issue**: OOM (Out of Memory) errors during training
- **Memory Requirement**: ~26GB per GPU for model weights alone

### Initial Analysis

```python
# Memory breakdown for 13B model
model_memory = 13e9 * 2  # BF16 weights: 26GB
gradient_memory = 13e9 * 2  # Gradients: 26GB  
optimizer_memory = 13e9 * 8  # AdamW states: 104GB
activation_memory = batch_size * seq_len * hidden_size * layers * 2  # ~8GB

# Total per GPU: 164GB (exceeds 40GB available)
```

### Solution: FSDP Implementation

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# Mixed precision policy
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16
)

# FSDP configuration
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=bf16_policy,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,
    cpu_offload=None,  # Keep everything on GPU for performance
)
```

### Memory Optimization Techniques

1. **Parameter Sharding**: Distribute model weights across GPUs
2. **Gradient Checkpointing**: Trade computation for memory
3. **Mixed Precision**: Use BF16 instead of FP32

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=1,  # Micro batch per GPU
    num_workers=4,  # Reduced from 8 to save memory
    pin_memory=False,  # Disable for large models
)
```

### Results

| Configuration | Memory/GPU | Status |
|---------------|------------|--------|
| Standard DDP | 164GB | ❌ OOM |
| FSDP + BF16 | 32GB | ✅ Success |
| FSDP + Checkpointing | 28GB | ✅ Success |

**Key Learning**: FSDP enables training models that don't fit in single GPU memory.

---

## Case Study 3: Queue Time Reduction

### Problem
- **Environment**: SLURM cluster with 80+ active researchers
- **Issue**: Average job queue time: 4+ hours
- **Impact**: Researchers losing productivity, GPU utilization dropping

### Analysis

```bash
# Investigate queue patterns
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --sort=-S

# Found issues:
# 1. Jobs requesting more resources than needed
# 2. Long-running jobs blocking queue
# 3. No priority system for different user types
```

### Resource Usage Patterns

```python
# Analysis of job submissions over 1 month
job_data = {
    'requested_gpus': [8, 8, 8, 4, 16, 8, ...],  # Often overprovisioned
    'actual_utilization': [65, 72, 58, 89, 43, ...],  # Low efficiency
    'duration_hours': [24, 72, 48, 12, 96, ...]  # Highly variable
}

# Key findings:
# - 60% of jobs used <80% of requested GPUs
# - 25% of jobs ran longer than initially estimated
# - Peak queue times during 9 AM - 5 PM
```

### Solution: Priority Scheduling System

```bash
# SLURM configuration changes
# /etc/slurm/slurm.conf

# Priority weights
PriorityWeightAge=1000
PriorityWeightFairshare=10000
PriorityWeightJobSize=1000
PriorityWeightPartition=1000
PriorityWeightQOS=2000

# Quality of Service levels
# Normal: Default for regular research
# High: For time-sensitive experiments
# Low: For long-running, low-priority jobs
```

```python
# Automatic resource optimization
#!/bin/bash
# Smart job submission script

# Analyze previous jobs to suggest optimal resources
def suggest_resources(job_script):
    model_size = extract_model_size(job_script)
    
    if model_size < 1e9:  # <1B parameters
        return {"nodes": 1, "gpus_per_node": 4}
    elif model_size < 7e9:  # 1B-7B parameters  
        return {"nodes": 2, "gpus_per_node": 8}
    else:  # >7B parameters
        return {"nodes": 4, "gpus_per_node": 8}
```

### GPU Sharing Implementation

```bash
# Enable GPU sharing for small jobs
# SLURM partition configuration
PartitionName=shared Nodes=gpu[001-020] Default=YES MaxTime=4:00:00 \
    DefMemPerCPU=8000 MaxMemPerNode=500000 \
    OverSubscribe=YES:4  # Allow 4 jobs per node

# GPU sharing with MPS (Multi-Process Service)
echo "export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" >> ~/.bashrc
echo "export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log" >> ~/.bashrc
```

### Results

| Metric | Before | After | Improvement |
|--------|---------|-------|--------------|
| Avg Queue Time | 4.2 hours | 2.7 hours | -35% |
| GPU Utilization | 71% | 89% | +25% |
| Job Completion Rate | 83% | 94% | +13% |
| User Satisfaction | 6.2/10 | 8.7/10 | +40% |

**Key Learning**: Resource optimization and priority scheduling dramatically improve cluster efficiency.

---

## Case Study 4: NCCL Timeout Issues

### Problem
- **Setup**: 8 nodes × 8 GPUs training LLaMA-7B
- **Issue**: Random NCCL timeout errors causing job failures
- **Frequency**: 30% of long-running jobs (>12 hours) failed

### Error Investigation

```bash
# Typical error message
[RANK 0]: Watchdog caught collective operation timeout: WorkNCCL(
    seq=123, opType=ALLREDUCE, timeout=1800000ms, 
    state=TIMEOUT_CLEANUP
) op was initiated 1800001ms ago
```

### Root Cause Analysis

1. **Network Congestion**: Multiple large jobs competing for bandwidth
2. **NCCL Default Timeout**: 30 minutes too short for large models
3. **Memory Pressure**: Occasional GC pauses causing synchronization delays

### Solution Implementation

```bash
# Increase NCCL timeout for large models
export NCCL_TIMEOUT=3600000  # 1 hour (in milliseconds)
export NCCL_ASYNC_ERROR_HANDLING=1  # Better error reporting

# Network optimization
export NCCL_IB_TIMEOUT=22  # Increase InfiniBand timeout
export NCCL_IB_RETRY_CNT=7  # More retry attempts

# Tree topology for large clusters
export NCCL_TOPO_FILE=/etc/nccl/topology.xml
```

```python
# Code changes for better error handling
try:
    dist.all_reduce(tensor)
except RuntimeError as e:
    if "timeout" in str(e).lower():
        logger.warning(f"NCCL timeout on rank {rank}, attempting recovery")
        # Attempt recovery or graceful shutdown
        save_checkpoint(emergency=True)
        raise
```

### Results

| Metric | Before | After |
|--------|---------|-------|
| Job Failure Rate | 30% | 3% |
| Average Job Runtime | 18.2 hours | 16.8 hours |
| Successful Completions | 70% | 97% |

**Key Learning**: Proper NCCL timeout configuration is critical for large-scale training reliability.

---

## Lessons Learned Summary

### Technical Best Practices

1. **Always verify network configuration** - NCCL defaults may not be optimal
2. **Monitor memory patterns** - Use FSDP for models >7B parameters  
3. **Implement proper timeouts** - Default NCCL timeouts too short for large models
4. **Profile before scaling** - Understand single-GPU performance first

### Operational Best Practices

1. **Resource right-sizing** - Most jobs over-provision by 20-40%
2. **Priority systems work** - Essential for multi-user environments
3. **Monitoring is critical** - Real-time metrics prevent issues
4. **Documentation saves time** - Clear guides reduce support burden

### Performance Targets Achieved

- **Scaling Efficiency**: 92% (target: >85%)
- **GPU Utilization**: 90%+ (up from 70%)
- **Queue Time Reduction**: 35% improvement
- **Job Success Rate**: 97% (up from 70%)

These case studies represent real solutions to production problems encountered while supporting distributed training workloads at scale.