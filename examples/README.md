# Training Examples

Progressive complexity examples for distributed training.

## Quick Start Examples

### 1. Single GPU Training
```bash
python quickstart_single_gpu.py --model bert-base --batch-size 32
```

### 2. Multi-GPU (Single Node)
```bash
torchrun --nproc_per_node=4 quickstart_ddp_4gpu.py --model bert-base
```

### 3. Multi-Node Training
```bash
sbatch ../slurm/multi_node_32gpu.sh
```

## Example Structure

- `quickstart_single_gpu.py` - Single GPU baseline
- `quickstart_ddp_4gpu.py` - 4 GPU DDP example
- `bert_classification.py` - BERT fine-tuning example
- `llama_pretraining.py` - LLaMA pre-training example
- `scaling_benchmark.py` - Scaling efficiency testing

## Performance Targets

| Model | GPUs | Target Tokens/sec | Scaling Efficiency |
|-------|------|------------------|--------------------|
| BERT-Base | 4 | 12,800 | >90% |
| LLaMA-7B | 32 | 8,400 | >85% |
| LLaMA-13B | 64 | 4,200 | >80% |