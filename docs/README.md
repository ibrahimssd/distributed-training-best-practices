# Documentation

Comprehensive guides for distributed training best practices.

## Guides

### Getting Started
- [Installation Guide](INSTALLATION.md) - Setup and dependencies
- [Quick Start](QUICK_START.md) - First distributed training job
- [Configuration](CONFIGURATION.md) - YAML configuration reference

### Distributed Training
- [DDP Guide](DISTRIBUTED_TRAINING_GUIDE.md) - Data Parallel concepts
- [FSDP Guide](FSDP_GUIDE.md) - Fully Sharded Data Parallel
- [Hybrid Parallelism](HYBRID_PARALLELISM.md) - Multi-dimensional scaling

### HPC & Infrastructure
- [Cluster Setup](CLUSTER_SETUP.md) - HPC infrastructure configuration
- [SLURM Integration](SLURM_GUIDE.md) - Job scheduling and management
- [NCCL Optimization](NCCL_OPTIMIZATION.md) - Network optimization

### Monitoring & Debugging
- [Performance Monitoring](MONITORING.md) - Metrics and dashboards
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [Debugging Guide](DEBUGGING.md) - Advanced debugging techniques

### Real-World Experience
- [Case Studies](CASE_STUDIES.md) - Production problem solving
- [Performance Benchmarks](BENCHMARKS.md) - Scaling efficiency results
- [Best Practices](BEST_PRACTICES.md) - Lessons learned from managing 80+ researchers

## Quick Reference

### Commands
```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py --config configs/bert_base_4gpu.yaml

# Multi-node, 32 GPUs
sbatch slurm/multi_node_32gpu.sh
```

### Performance Targets
- Scaling efficiency: >85% (achieved 92% in production)
- GPU utilization: >90%
- Queue time reduction: 35% improvement achieved