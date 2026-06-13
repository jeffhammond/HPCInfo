#!/usr/bin/env python3
"""
PyTorch Distributed Collective Operations Demo

Demonstrates various collective operations:
- all_reduce
- broadcast
- gather
- scatter
- all_gather
- reduce

Usage:
    torchrun --nproc_per_node=4 collectives_demo.py
"""

import torch
import torch.distributed as dist
import os


def demo_all_reduce(rank, world_size):
    """Demonstrate all_reduce operation"""
    print(f"\n{'='*60}")
    print(f"ALL_REDUCE Demo (Rank {rank})")
    print(f"{'='*60}")
    
    tensor = torch.tensor([rank + 1.0])
    print(f"Rank {rank}: Initial tensor = {tensor.item()}")
    
    dist.barrier()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Rank {rank}: After all_reduce (SUM) = {tensor.item()}")
    dist.barrier()


def demo_broadcast(rank, world_size):
    """Demonstrate broadcast operation"""
    print(f"\n{'='*60}")
    print(f"BROADCAST Demo (Rank {rank})")
    print(f"{'='*60}")
    
    if rank == 0:
        tensor = torch.tensor([42.0, 3.14, 2.71])
        print(f"Rank {rank} (source): Broadcasting tensor = {tensor}")
    else:
        tensor = torch.zeros(3)
        print(f"Rank {rank}: Initial tensor = {tensor}")
    
    dist.barrier()
    dist.broadcast(tensor, src=0)
    
    print(f"Rank {rank}: After broadcast = {tensor}")
    dist.barrier()


def demo_gather(rank, world_size):
    """Demonstrate gather operation"""
    print(f"\n{'='*60}")
    print(f"GATHER Demo (Rank {rank})")
    print(f"{'='*60}")
    
    tensor = torch.tensor([rank * 10.0])
    print(f"Rank {rank}: Sending tensor = {tensor.item()}")
    
    if rank == 0:
        gather_list = [torch.zeros(1) for _ in range(world_size)]
    else:
        gather_list = None
    
    dist.barrier()
    dist.gather(tensor, gather_list, dst=0)
    
    if rank == 0:
        print(f"Rank {rank}: Gathered tensors = {[t.item() for t in gather_list]}")
    dist.barrier()


def demo_all_gather(rank, world_size):
    """Demonstrate all_gather operation"""
    print(f"\n{'='*60}")
    print(f"ALL_GATHER Demo (Rank {rank})")
    print(f"{'='*60}")
    
    tensor = torch.tensor([rank * 100.0])
    print(f"Rank {rank}: Sending tensor = {tensor.item()}")
    
    gather_list = [torch.zeros(1) for _ in range(world_size)]
    
    dist.barrier()
    dist.all_gather(gather_list, tensor)
    
    print(f"Rank {rank}: All gathered = {[t.item() for t in gather_list]}")
    dist.barrier()


def demo_scatter(rank, world_size):
    """Demonstrate scatter operation"""
    print(f"\n{'='*60}")
    print(f"SCATTER Demo (Rank {rank})")
    print(f"{'='*60}")
    
    if rank == 0:
        scatter_list = [torch.tensor([i * 2.0]) for i in range(world_size)]
        print(f"Rank {rank} (source): Scattering list = {[t.item() for t in scatter_list]}")
    else:
        scatter_list = None
    
    output = torch.zeros(1)
    
    dist.barrier()
    dist.scatter(output, scatter_list, src=0)
    
    print(f"Rank {rank}: Received tensor = {output.item()}")
    dist.barrier()


def demo_reduce(rank, world_size):
    """Demonstrate reduce operation"""
    print(f"\n{'='*60}")
    print(f"REDUCE Demo (Rank {rank})")
    print(f"{'='*60}")
    
    tensor = torch.tensor([rank + 1.0])
    print(f"Rank {rank}: Initial tensor = {tensor.item()}")
    
    dist.barrier()
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print(f"Rank {rank}: After reduce (SUM) = {tensor.item()}")
    else:
        print(f"Rank {rank}: (unchanged, reduce only affects dst)")
    dist.barrier()


def main():
    # Initialize process group
    dist.init_process_group(backend='gloo')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"\nRunning collective operations demo with {world_size} processes\n")
    
    # Run demos
    demo_all_reduce(rank, world_size)
    demo_broadcast(rank, world_size)
    demo_gather(rank, world_size)
    demo_all_gather(rank, world_size)
    demo_scatter(rank, world_size)
    demo_reduce(rank, world_size)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("All collective operations completed successfully!")
        print(f"{'='*60}\n")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

