# Troubleshooting

## NCCL timeout / hangs
- Ensure `NCCL_SOCKET_IFNAME` points to IB interface.
- Increase timeout for large models:
  ```bash
  export NCCL_TIMEOUT=1800
  ```
- Check rank-to-GPU mapping (`LOCAL_RANK`).

## CUDA OOM
- Lower per-GPU batch size.
- Increase `gradient_accumulation_steps`.
- Enable `gradient_checkpointing`.
- Use BF16 mixed precision.

## Low scaling efficiency
- Increase DataLoader workers and pin memory.
- Tune DDP bucket size for larger models.
- Confirm no TCP fallback in NCCL transport.

## Unstable loss / NaN gradients
- Reduce learning rate.
- Enable gradient clipping.
- Validate mixed precision mode and loss scaling.
