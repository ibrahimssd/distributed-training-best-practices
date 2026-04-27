#!/usr/bin/env python3
"""
Quick start example for single GPU training.
Baseline for comparing distributed performance.
"""

import argparse
import time
import torch
from transformers import AutoModel, AutoConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = AutoModel.from_pretrained(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
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
        tokens_per_sec = args.batch_size * args.seq_length / step_time
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, "
                  f"Step_time={step_time:.3f}s, "
                  f"Tokens/sec={tokens_per_sec:,.0f}")
    
    total_time = time.time() - start_time
    total_tokens = args.steps * args.batch_size * args.seq_length
    avg_tokens_per_sec = total_tokens / total_time
    
    print(f"\nTraining completed:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average tokens/sec: {avg_tokens_per_sec:,.0f}")
    print(f"GPU memory used: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

if __name__ == '__main__':
    main()