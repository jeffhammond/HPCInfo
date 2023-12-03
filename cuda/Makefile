NVCC = nvcc
CXX  = nvc++ -Wno-unused

CC   = gcc
CFLAGS = -I/usr/local/cuda-12.1/targets/x86_64-linux/include
LDFLAGS = -L/usr/local/cuda-12.1/targets/x86_64-linux/lib -lcudart
LDFLAGS += -L/usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs -lcuda

CUTFLAG = -cudalib=cutensor
CUTFLAG += -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/math_libs/12.2/targets/x86_64-linux/lib -lcutensorMg

.PHONY: clean

all: gpu-detect getptrattr getptrattr2 mps-detect

%: %.c
	$(CC) $< $(CFLAGS) $(LDFLAGS) -o $@

%: %.cu
	$(NVCC) $< -o $@

%: %.cc
	$(CXX) $< $(CUTFLAG) -o $@

clean:
	-rm -f gpu-detect getptrattr getptrattr2
