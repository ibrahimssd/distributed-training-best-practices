# Monitoring Stack

Real-time monitoring and observability for distributed training workloads.

## Components

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards
- **Node Exporter**: System metrics
- **GPU Exporter**: NVIDIA GPU metrics

## Quick Start

```bash
# Start monitoring stack
docker-compose up -d

# Access Grafana dashboard
open http://localhost:3000
```

## Key Metrics

- GPU utilization per node
- Training throughput (tokens/sec)
- Memory usage patterns
- Network bandwidth
- NCCL communication efficiency