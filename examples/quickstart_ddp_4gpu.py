#!/usr/bin/env python3
"""
Quick start example for 4 GPU DDP training.
Demonstrates basic distributed setup.
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel

def setup():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    
def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--batch-size', type=int, default=8)  # Per GPU
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    
    # Setup distributed training
    setup()
    
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Training on {world_size} GPUs")
    
    # Load model and wrap with DDP
    model = AutoModel.from_pretrained(args.model).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5 * world_size)
    
    # Dummy data
    input_ids = torch.randint(0, 30000, (args.batch_size, args.seq_length)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for step in range(args.steps):
        step_start = time.time()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.last_hidden_state.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step_time = time.time() - step_start
        tokens_per_sec = args.batch_size * args.seq_length * world_size / step_time
        
        if step % 10 == 0 and rank == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, "
                  f"Step_time={step_time:.3f}s, "
                  f"Tokens/sec={tokens_per_sec:,.0f}")
    
    if rank == 0:
        total_time = time.time() - start_time
        total_tokens = args.steps * args.batch_size * args.seq_length * world_size
        avg_tokens_per_sec = total_tokens / total_time
        
        print(f"\nTraining completed:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average tokens/sec: {avg_tokens_per_sec:,.0f}")
        print(f"Scaling efficiency: {avg_tokens_per_sec / (12800 * world_size / 4):.2%}")
    
    cleanup()

if __name__ == '__main__':
    main()