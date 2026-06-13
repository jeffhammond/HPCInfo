.PHONY: help hello ddp collectives clean

NPROC ?= 4

help:
	@echo "PyTorch Distributed Examples"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  make hello      - Run hello_distributed.py with $(NPROC) processes"
	@echo "  make ddp        - Run ddp_example.py with $(NPROC) processes"
	@echo "  make collectives - Run collectives_demo.py with $(NPROC) processes"
	@echo "  make clean      - Remove Python cache files"
	@echo ""
	@echo "Options:"
	@echo "  NPROC=N        - Number of processes (default: 4)"
	@echo ""
	@echo "Examples:"
	@echo "  make hello NPROC=2"
	@echo "  make ddp NPROC=4"

hello:
	@echo "Running hello_distributed.py with $(NPROC) processes..."
	torchrun --nproc_per_node=$(NPROC) hello_distributed.py

ddp:
	@echo "Running ddp_example.py with $(NPROC) processes..."
	torchrun --nproc_per_node=$(NPROC) ddp_example.py

collectives:
	@echo "Running collectives_demo.py with $(NPROC) processes..."
	torchrun --nproc_per_node=$(NPROC) collectives_demo.py

clean:
	rm -rf __pycache__
	rm -f *.pyc

