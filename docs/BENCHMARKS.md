# Benchmarking Methodology

## Goals
- Throughput (tokens/sec)
- GPU utilization
- Scaling efficiency
- Peak memory per GPU

## Procedure
1. Warm up 50-100 steps.
2. Measure over fixed window (e.g., 500 steps).
3. Record per-rank and aggregated metrics.
4. Compare 1 GPU baseline to N-GPU throughput:
   `efficiency = throughput_N / (throughput_1 * N)`

## Recommended matrix
- Models: 110M, 1B, 3B, 7B, 13B
- GPU counts: 4, 8, 16, 32, 64
- Strategies: DDP, FSDP, Hybrid

## Reporting
Include:
- Config file path
- Hardware topology
- Effective batch size
- Final throughput, memory, efficiency
