#!/usr/bin/env python3
"""
Simple PyTorch Distributed Hello World Example

This script demonstrates basic torch.distributed functionality.
Each process prints its rank and size information.

Usage:
    # Single node, 4 processes
    torchrun --nproc_per_node=4 hello_distributed.py
    
    # Or using torch.distributed.launch (older method)
    python -m torch.distributed.launch --nproc_per_node=4 hello_distributed.py
    
    # Multi-node (run on each node)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 hello_distributed.py
"""

import torch
import torch.distributed as dist
import os
import socket


def main():
    # Initialize the distributed environment
    dist.init_process_group(backend='gloo')  # Use 'nccl' for GPU, 'gloo' for CPU
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()
    
    # Print hello message from each process
    print(f"Hello from rank {rank}/{world_size} on host {hostname}")
    
    # Demonstrate a simple collective operation (all_reduce)
    tensor = torch.tensor([rank], dtype=torch.float32)
    print(f"Rank {rank}: Before all_reduce, tensor = {tensor.item()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: After all_reduce (sum), tensor = {tensor.item()}")
    
    # Barrier to synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print(f"\nAll {world_size} processes completed successfully!")
        print(f"Expected sum: {sum(range(world_size))}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

