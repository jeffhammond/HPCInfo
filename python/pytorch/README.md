# PyTorch Distributed Examples

This directory contains examples demonstrating PyTorch's distributed training capabilities.

## Examples

### 1. Hello Distributed (`hello_distributed.py`)
A simple "Hello World" example showing basic distributed process initialization and a simple all_reduce operation.

**Run:**
```bash
# 4 processes on single node
torchrun --nproc_per_node=4 hello_distributed.py

# Or using older launch method
python -m torch.distributed.launch --nproc_per_node=4 hello_distributed.py
```

**Expected Output:**
```
Hello from rank 0/4 on host <hostname>
Hello from rank 1/4 on host <hostname>
Hello from rank 2/4 on host <hostname>
Hello from rank 3/4 on host <hostname>
Rank 0: Before all_reduce, tensor = 0.0
Rank 1: Before all_reduce, tensor = 1.0
Rank 2: Before all_reduce, tensor = 2.0
Rank 3: Before all_reduce, tensor = 3.0
Rank 0: After all_reduce (sum), tensor = 6.0
Rank 1: After all_reduce (sum), tensor = 6.0
Rank 2: After all_reduce (sum), tensor = 6.0
Rank 3: After all_reduce (sum), tensor = 6.0

All 4 processes completed successfully!
Expected sum: 6
```

### 2. Distributed Data Parallel (`ddp_example.py`)
A complete example showing DDP training with a simple neural network, including:
- Model wrapping with DDP
- Distributed data loading with DistributedSampler
- Training loop with gradient synchronization

**Run:**
```bash
# CPU training with 2 processes
torchrun --nproc_per_node=2 ddp_example.py

# GPU training (requires GPUs and NCCL)
# Edit the script to uncomment GPU-specific lines
torchrun --nproc_per_node=2 ddp_example.py
```

### 3. Collective Operations Demo (`collectives_demo.py`)
Demonstrates all major collective operations:
- `all_reduce`: Reduce and broadcast result to all processes
- `broadcast`: Send data from one process to all others
- `gather`: Collect data from all processes to one
- `all_gather`: Collect data from all processes to all processes
- `scatter`: Distribute data from one process to all others
- `reduce`: Reduce data from all processes to one

**Run:**
```bash
torchrun --nproc_per_node=4 collectives_demo.py
```

## Requirements

```bash
pip install torch torchvision
```

For GPU support with NCCL backend:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

## Backends

PyTorch supports multiple distributed backends:

- **NCCL**: Best for GPU training (NVIDIA GPUs only)
- **Gloo**: Works on both CPU and GPU, good for CPU-only training
- **MPI**: Requires MPI installation, good for HPC environments

To change backend, modify the `backend` parameter in `dist.init_process_group()`:
```python
dist.init_process_group(backend='nccl')  # For GPU
dist.init_process_group(backend='gloo')  # For CPU
dist.init_process_group(backend='mpi')   # For MPI
```

## Multi-Node Training

For multi-node training, run the same command on each node with appropriate parameters:

**Node 0 (master):**
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    hello_distributed.py
```

**Node 1:**
```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<MASTER_IP> \
    --master_port=29500 \
    hello_distributed.py
```

Replace `<MASTER_IP>` with the IP address of the master node.

## Environment Variables

PyTorch distributed can also be configured via environment variables:

```bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0  # Different for each process

python hello_distributed.py
```

## Slurm Integration

Example Slurm batch script:

```bash
#!/bin/bash
#SBATCH --job-name=pytorch-dist
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch training
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ddp_example.py
```

## Troubleshooting

### Common Issues

1. **"Address already in use"**: Change `MASTER_PORT` to a different port number
2. **Timeout errors**: Check network connectivity between nodes
3. **NCCL errors**: Make sure CUDA and NCCL are properly installed
4. **Hanging**: Ensure all processes are launched correctly and can communicate

### Debug Mode

Enable debug logging:
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --nproc_per_node=4 hello_distributed.py
```

## References

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Getting Started with DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [Distributed Communication Package](https://pytorch.org/docs/stable/distributed.html)

