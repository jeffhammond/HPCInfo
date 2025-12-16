#!/usr/bin/env python3
"""
PyTorch Distributed Data Parallel (DDP) Training Example

This script demonstrates a simple neural network training with DDP.

Usage:
    # Single node, 2 GPUs (or 2 CPU processes)
    torchrun --nproc_per_node=2 ddp_example.py
    
    # Multi-node with GPUs
    # Node 0:
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 ddp_example.py
    # Node 1:
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=<MASTER_IP> --master_port=29500 ddp_example.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os


class SimpleDataset(Dataset):
    """Simple synthetic dataset for demonstration"""
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.input_dim = input_dim
        # Generate random data
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    """Simple neural network for binary classification"""
    def __init__(self, input_dim=10, hidden_dim=20):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def setup():
    """Initialize distributed training"""
    dist.init_process_group(backend='gloo')  # Use 'nccl' for GPU
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train(rank, world_size):
    """Training function"""
    print(f"Running DDP training on rank {rank}/{world_size}")
    
    # Create model and move to device
    model = SimpleModel(input_dim=10, hidden_dim=20)
    
    # For GPU training, uncomment:
    # device = torch.device(f'cuda:{rank}')
    # model = model.to(device)
    
    # Wrap model with DDP
    ddp_model = DDP(model)
    
    # Create dataset and distributed sampler
    dataset = SimpleDataset(size=1000, input_dim=10)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle data differently at each epoch
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            # For GPU training, uncomment:
            # data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Only print from rank 0
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    if rank == 0:
        print("\nTraining completed successfully!")


def main():
    rank, world_size = setup()
    train(rank, world_size)
    cleanup()


if __name__ == "__main__":
    main()

