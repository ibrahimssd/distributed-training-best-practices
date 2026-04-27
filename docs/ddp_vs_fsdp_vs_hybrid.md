# DDP vs FSDP vs Hybrid Parallelism — Decision Guide

A practical reference for choosing the right parallelism strategy
based on model size, hardware, and training objectives.

---

## Quick Decision Table

| Model Size | GPUs Available | Recommended Strategy | Script |
|---|---|---|---|
| < 1 B | 1 | Single GPU | `examples/01_single_gpu_baseline.py` |
| < 1 B | 2–32 | DDP | `train_ddp.py` |
| 1 B – 13 B | 4–64 | FSDP | `train_fsdp.py` |
| 13 B – 70 B | 16–128 | FSDP + gradient checkpointing | `train_fsdp.py` |
| 70 B+ | 64–512 | Hybrid (TP + PP + DP) | `train_hybrid.py` |

---

## Memory Estimation

For a model with P parameters, the memory required per GPU is approximately:

| Strategy | Parameters per GPU | Gradients per GPU | Optimizer States per GPU |
|---|---|---|---|
| Single GPU | P | P | 2P (AdamW) |
| DDP (N GPUs) | P | P | 2P |
| FSDP full shard | P/N | P/N | 2P/N |
| FSDP + CPU offload | P/N | P/N | 0 (on CPU) |
| Hybrid TP×PP×DP | P/(TP×PP×DP) | P/(TP×PP×DP) | 2P/(TP×PP×DP) |

**Example — LLaMA-7B (7 B params, fp16):**

| Strategy | GPUs | Memory/GPU |
|---|---|---|
| DDP | 8 × A100 80 GB | ~112 GB ❌ (does not fit) |
| FSDP | 8 × A100 80 GB | ~14 GB ✅ |
| FSDP + grad ckpt | 8 × A100 40 GB | ~11 GB ✅ |

---

## DDP — Data Parallel

**How it works:**
Each GPU holds a full copy of the model. Each step:
1. Each GPU processes its own data shard
2. All-reduce synchronises gradients across all GPUs
3. Each GPU updates its local copy identically

**When to use:**
- Model fits on a single GPU
- You want the simplest implementation
- Fast inter-GPU connectivity (NVLink or InfiniBand)

**Key settings:**
```python
# bucket_cap_mb: controls all-reduce granularity
# larger = fewer syncs but more memory; 25 MB is a good default
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)
```

**Bottlenecks:**
- Memory: model replicated N times
- Network: gradient all-reduce proportional to parameter count

---

## FSDP — Fully Sharded Data Parallel

**How it works:**
Parameters, gradients, and optimizer states are sharded across all GPUs.
Before each layer's forward/backward pass, an all-gather reconstructs the
full parameters. After the backward pass, a reduce-scatter accumulates
sharded gradients. No GPU ever holds the full model simultaneously.

**When to use:**
- Model does NOT fit in single GPU memory
- You need to scale from 7B to ~65B parameters
- You have NVLink or InfiniBand for fast all-gathers

**Sharding strategies:**

| Strategy | Parameters | Gradients | Optimizer States | Use Case |
|---|---|---|---|---|
| `FULL_SHARD` | Sharded | Sharded | Sharded | Maximum memory savings |
| `SHARD_GRAD_OP` | Sharded during fwd/bwd | Not sharded | Not sharded | Faster, less savings |
| `NO_SHARD` | Not sharded | Not sharded | Not sharded | Equivalent to DDP |
| `HYBRID_SHARD` | Full shard within node | Replicated across nodes | Sharded | Multi-node efficiency |

**Key settings:**
```python
# Wrap at transformer layer level — coarser wrapping = fewer all-gathers
wrap_policy = partial(transformer_auto_wrap_policy,
                      transformer_layer_cls={LlamaDecoderLayer})

# use_orig_params=True required for some optimizers (e.g., bitsandbytes)
model = FSDP(model, auto_wrap_policy=wrap_policy, use_orig_params=True)
```

**Common mistakes:**
- Creating optimizer BEFORE FSDP wrapping (parameters don't exist yet)
- Using `torch.nn.utils.clip_grad_norm_` instead of `model.clip_grad_norm_()`
- Saving checkpoint without FSDP state dict context manager

---

## Hybrid Parallelism (TP + PP + DP)

**How it works:**
Three orthogonal parallelism dimensions combined:

```
Tensor Parallel (TP):
  Splits individual weight matrices across GPUs within a node.
  Each GPU computes partial matrix multiplications.
  Best for: attention Q/K/V projections, MLP layers

Pipeline Parallel (PP):
  Assigns different transformer layers to different GPUs/nodes.
  GPUs execute in pipeline fashion (1F1B schedule).
  Best for: distributing model depth across nodes

Data Parallel (DP):
  Replicates the entire TP×PP pipeline and trains on different data.
  Best for: scaling throughput beyond a single pipeline
```

**When to use:**
- 70B+ parameters
- 64+ GPUs across multiple nodes
- High-bandwidth interconnect required for TP (NVLink within node, IB between nodes)

**Parallelism sizing rules:**

1. **TP size = GPUs per node** (NVLink bandwidth keeps all-reduces fast)
2. **PP size = total nodes / 2** (pipeline depth; more stages = more bubble overhead)
3. **DP = total GPUs / (TP × PP)** (data parallel replicas)

**Example for 64 GPUs (8 nodes × 8 GPUs):**
```yaml
tensor_parallel_size: 8   # within each node (NVLink)
pipeline_parallel_size: 4 # 4 pipeline stages across nodes (IB)
data_parallel_size: 2     # 2 independent replicas
# 8 * 4 * 2 = 64 GPUs
```

---

## Performance Tuning Checklist

### DDP
- [ ] Set `NCCL_SOCKET_IFNAME=ib0` to force InfiniBand
- [ ] Tune `bucket_cap_mb` (try 25, 50, 100 MB)
- [ ] Enable `gradient_as_bucket_view=True`
- [ ] Use `persistent_workers=True` in DataLoader
- [ ] Pin memory with `pin_memory=True`

### FSDP
- [ ] Use `FULL_SHARD` for memory-constrained cases
- [ ] Use `HYBRID_SHARD` for multi-node (avoids inter-node all-gathers)
- [ ] Enable `limit_all_gathers=True` to prevent memory spikes
- [ ] Enable gradient checkpointing for sequences > 2048 tokens
- [ ] Use `BackwardPrefetch.BACKWARD_PRE` for overlapped communication

### Hybrid
- [ ] Verify TP = GPUs per node (NVLink boundary)
- [ ] Set `NCCL_IB_HCA` to correct InfiniBand HCA device
- [ ] Enable GPU Direct RDMA (`NCCL_NET_GDR_LEVEL=5`)
- [ ] Profile pipeline bubble ratio — aim for < 5%

---

## Scaling Efficiency Targets

| Cluster Size | Expected DDP Efficiency | Expected FSDP Efficiency |
|---|---|---|
| 2–4 GPUs (single node) | 95–98% | 90–95% |
| 8 GPUs (single node) | 92–96% | 88–93% |
| 16–32 GPUs (2–4 nodes) | 88–94% | 82–90% |
| 64–128 GPUs (8–16 nodes) | 82–90% | 75–85% |

Efficiency below these ranges indicates a network, NCCL, or I/O bottleneck.
Use `utils/profiling.py` and the Grafana dashboard to diagnose.
