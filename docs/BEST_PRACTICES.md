# Distributed Training Best Practices

## Configuration
- Keep config-driven runs (model/training/distributed sections).
- Version control every config used in benchmarks.

## Stability
- Prefer BF16 for large models.
- Enable gradient checkpointing for 7B+ models.
- Add periodic validation, not train-loss only.

## Performance
- Use distributed sampler to avoid duplicated data.
- Scale learning rate with world size (linear rule as baseline).
- Profile dataloader throughput before optimizing model code.

## Reliability
- Save `latest` and `best` checkpoints.
- Enable async NCCL error handling.
- Run distributed smoke checks before long runs.

## Observability
- Emit structured JSON logs in production.
- Track throughput, memory, and scaling efficiency over time.
