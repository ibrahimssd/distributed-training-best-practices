# Documentation

Comprehensive guides for distributed training best practices.

## Guides

### Core Guides
- [Distributed Training Guide](DISTRIBUTED_TRAINING_GUIDE.md) - Data parallel and scaling concepts
- [Cluster Setup](CLUSTER_SETUP.md) - HPC infrastructure configuration
- [Case Studies](CASE_STUDIES.md) - Production problem solving
- [Troubleshooting](TROUBLESHOOTING.md) - Common NCCL/training issues
- [Benchmarks](BENCHMARKS.md) - Benchmarking methodology and templates
- [Best Practices](BEST_PRACTICES.md) - Lessons learned from production usage

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