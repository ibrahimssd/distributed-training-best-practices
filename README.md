# Distributed Training Best Practices

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready reference implementation demonstrating distributed training best practices, based on real experience managing HPC infrastructure and supporting 80+ researchers with distributed training workloads.

## 🎯 Purpose

This repository serves as both:
1. **Portfolio piece** showcasing hands-on distributed training expertise
2. **Educational resource** for ML engineers scaling from single-GPU to multi-node training

## 👨‍💻 About the Author

**Ibrahim Siddig** - NLP Research Engineer & HPC Engineer  
📧 [LinkedIn](https://www.linkedin.com/in/ibrahimsiddig) | 🐙 [GitHub](https://github.com/ibrahimssd)

**Experience:**
- 1.5 years + months managing SLURM-based HPC clusters with A100/H100 GPUs for 80+ researchers
- Optimized GPU utilization from 70% → 90%+ across production workloads
- Reduced job queue times by 35% through priority scheduling optimizations
- Helped researchers scale from single-GPU to multi-node distributed training (4-8 nodes, 32-64 GPUs)

This repository is based on my experience at **Technische Hochschule Ingolstadt**, where I managed HPC infrastructure and solved real distributed training challenges at scale.

## 🚀 Quick Start

### Single GPU (Development)
```bash
python examples/quickstart_single_gpu.py --model bert-base --batch-size 32
```

### Multi-GPU (Single Node)
```bash
torchrun --nproc_per_node=4 examples/quickstart_ddp_4gpu.py --model bert-base
```

### Multi-Node (Production)
```bash
sbatch slurm/multi_node_32gpu.sh  # SLURM cluster
```

## 📁 Repository Structure

```
distributed-training-best-practices/
├── train_ddp.py              # Data Parallel training (multi-node ready)
├── train_fsdp.py             # FSDP for large models (7B-13B params)
├── train_hybrid.py           # Hybrid parallelism (70B+ params)
├── utils/
│   ├── profiling.py          # GPU profiling & performance metrics
│   ├── monitoring.py         # Prometheus metrics export
│   └── debugging.py          # NCCL debugging utilities
├── configs/                  # Training configurations
├── slurm/                    # SLURM job scripts
├── docker/                   # Containerization
├── monitoring/               # Observability stack
├── tests/                    # Unit & integration tests
├── examples/                 # Progressive complexity demos
└── docs/                     # Comprehensive guides
```

## 🔥 Key Features

### What Makes This Different

- **🏭 Production-Ready**: Real error handling, checkpointing, monitoring
- **🖥️ Multi-Node Focus**: Not just single-node 8 GPU examples
- **⚡ Performance Optimization**: NCCL tuning, memory optimization, scaling efficiency
- **🔧 HPC Integration**: SLURM scripts, Apptainer containers, InfiniBand config
- **📊 Observability**: Prometheus/Grafana monitoring stack
- **🐛 Real Debugging**: Solutions to actual issues encountered in production

### Supported Training Paradigms

| Paradigm | Model Size | Nodes | GPUs | Use Case |
|----------|------------|-------|------|-----------|
| **DDP** | ≤3B params | 1-4 | 8-32 | Language models, classification |
| **FSDP** | 7B-13B params | 4-8 | 32-64 | Large language models |
| **Hybrid** | 70B+ params | 8+ | 64+ | Frontier models |

## 📈 Performance Benchmarks

Based on real production workloads:

### Scaling Efficiency
- **Target**: >85% scaling efficiency (industry standard)
- **Achieved**: 92% on 4-node setup (vs 52% before optimization)

### Throughput Examples
| Model | Hardware | Tokens/sec | GPU Utilization |
|-------|----------|------------|------------------|
| BERT-Base | 4x A100 40GB | 12,800 | 94% |
| LLaMA-7B | 32x A100 40GB | 8,400 | 91% |
| LLaMA-13B | 64x A100 40GB | 4,200 | 89% |

## 🛠️ Installation

### Option 1: Local Development
```bash
git clone https://github.com/ibrahimssd/distributed-training-best-practices.git
cd distributed-training-best-practices
pip install -r requirements.txt
```

### Option 2: Docker
```bash
docker-compose up -d
docker exec -it distributed-training bash
```

### Option 3: HPC Cluster (Apptainer)
```bash
apptainer build training.sif docker/Singularity.def
sbatch slurm/multi_node_32gpu.sh
```

## 📚 Training Scripts

### 1. Data Parallel Training (`train_ddp.py`)
Perfect for models ≤3B parameters across 1-4 nodes.

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py \\
    --config configs/bert_base_4gpu.yaml \\
    --output-dir ./outputs

# Multi-node, 32 GPUs (4 nodes × 8 GPUs)
torchrun --nnodes=4 --nproc_per_node=8 \\
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
    train_ddp.py --config configs/llama_7b_32gpu.yaml
```

**Key Features:**
- ✅ Distributed sampler (no data duplication)
- ✅ Gradient synchronization
- ✅ NCCL optimization for InfiniBand
- ✅ Checkpoint recovery
- ✅ Mixed precision (BF16/FP16)

### 2. Fully Sharded Data Parallel (`train_fsdp.py`)
For large models (7B-13B parameters) that don't fit in single GPU memory.

```bash
# 13B model on 8×40GB A100s (fits with 8GB headroom)
torchrun --nproc_per_node=8 train_fsdp.py \\
    --config configs/llama_13b_64gpu.yaml \\
    --sharding-strategy FULL_SHARD \\
    --mixed-precision bf16
```

**Memory Optimization:**
- Model parameters sharded across GPUs
- Gradient checkpointing enabled
- Mixed precision reduces memory by 50%

### 3. Hybrid Parallelism (`train_hybrid.py`)
For frontier models (70B+ parameters) using tensor + data parallelism.

```bash
# 70B model: Tensor Parallel within nodes, FSDP across nodes
torchrun --nnodes=8 --nproc_per_node=8 \\
    train_hybrid.py --tp-size=8 --pp-size=1 --dp-size=8
```

## 🔍 Monitoring & Debugging

### GPU Profiling
```bash
python utils/profiling.py --profile-run \\
    --script train_ddp.py --config configs/bert_base_4gpu.yaml
```

### Performance Metrics
- **Throughput**: Tokens/second, samples/second
- **Efficiency**: Scaling efficiency, GPU utilization
- **Memory**: Peak memory usage, gradient accumulation

### NCCL Debugging
```bash
# Enable detailed NCCL logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Test NCCL communication
python utils/debugging.py --test-nccl --nodes=4 --gpus-per-node=8
```

## 🏗️ Real-World Case Studies

### Case Study 1: Scaling Efficiency Optimization
**Problem**: 3B parameter German language model achieving only 52% scaling efficiency on 4 nodes  
**Root Cause**: NCCL falling back to TCP instead of InfiniBand  
**Solution**: Configured `NCCL_SOCKET_IFNAME=ib0` and optimized data loading  
**Result**: **92% scaling efficiency** - 3.7× speedup improvement

### Case Study 2: Memory Optimization
**Problem**: 13B parameter model wouldn't fit on 8× A100 40GB GPUs  
**Root Cause**: Insufficient memory management  
**Solution**: FSDP with `FULL_SHARD` + gradient checkpointing  
**Result**: Model fits in **32GB per GPU** with 8GB headroom

### Case Study 3: Queue Time Reduction
**Problem**: Researchers waiting 4+ hours for job scheduling  
**Root Cause**: Inefficient SLURM configuration and resource allocation  
**Solution**: Implemented priority scheduling and GPU sharing policies  
**Result**: **35% reduction** in average queue times

*See [docs/CASE_STUDIES.md](docs/CASE_STUDIES.md) for detailed technical analysis.*

## 🔧 Configuration Examples

### Small Model (BERT-Base, 110M params)
```yaml
# configs/bert_base_4gpu.yaml
model:
  name: "bert-base-uncased"
  max_seq_length: 512

training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 3
  mixed_precision: "bf16"

distributed:
  backend: "nccl"
  gradient_accumulation_steps: 1
```

### Large Model (LLaMA-13B)
```yaml
# configs/llama_13b_64gpu.yaml
model:
  name: "meta-llama/Llama-2-13b-hf"
  max_seq_length: 2048

training:
  batch_size: 1  # Per GPU
  gradient_accumulation_steps: 32
  learning_rate: 1e-4
  mixed_precision: "bf16"

fsdp:
  sharding_strategy: "FULL_SHARD"
  backward_prefetch: "BACKWARD_PRE"
  forward_prefetch: true
  use_orig_params: false
```

## 🐛 Common Issues & Solutions

### Issue: Low GPU Utilization
```python
# Bad: CPU bottleneck
dataloader = DataLoader(dataset, batch_size=32, num_workers=2)

# Good: Sufficient workers + pinned memory
dataloader = DataLoader(dataset, batch_size=32, 
                       num_workers=8, pin_memory=True)
```

### Issue: NCCL Timeout
```bash
# Solution: Increase timeout for large models
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_ASYNC_ERROR_HANDLING=1
```

### Issue: Out of Memory
```python
# Solution: Gradient accumulation
effective_batch_size = batch_size * num_gpus * gradient_accumulation_steps
```

## 📊 Monitoring Dashboard

Monitor training in real-time with Grafana:

```bash
# Start monitoring stack
docker-compose -f monitoring/docker-compose.yml up -d

# Access Grafana: http://localhost:3000
# Username: admin, Password: admin
```

**Key Metrics:**
- 🔥 GPU utilization per node
- 📈 Training throughput (tokens/sec)
- 💾 Memory usage patterns  
- 🌐 Network bandwidth utilization
- ⚡ NCCL communication efficiency

## 🧪 Testing

Run the test suite to verify your setup:

```bash
# Unit tests
pytest tests/ -v

# Integration test (requires 4 GPUs)
python tests/test_ddp.py --test-scaling-efficiency

# NCCL communication test
python tests/test_nccl.py --nodes=2 --gpus-per-node=4
```

## 📖 Documentation

- **[Distributed Training Guide](docs/DISTRIBUTED_TRAINING_GUIDE.md)** - Comprehensive concepts and best practices
- **[Cluster Setup](docs/CLUSTER_SETUP.md)** - How to configure HPC infrastructure  
- **[Case Studies](docs/CASE_STUDIES.md)** - Real-world problem solving examples
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Debug common issues

## 🤝 Contributing

This repository represents production patterns I've developed through real HPC management experience. While primarily a portfolio piece, contributions that enhance educational value are welcome.

## 📝 License

MIT License - feel free to use this as a learning resource or starting point for your own distributed training projects.

## 🔗 Additional Resources

- [PyTorch Distributed Training Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Best Practices](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/best_practices.html)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)

---

**⭐ If this repository helped you scale your training, please give it a star!**

*This repository demonstrates real production experience from managing HPC infrastructure at Technische Hochschule Ingolstadt, where I supported 80+ researchers with distributed training workloads. The examples and best practices are derived from actual production scenarios.*
