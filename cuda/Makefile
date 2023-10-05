NVCC = nvcc
CXX  = nvc++ -Wno-unused

CUTFLAG = -cudalib=cutensor
CUTFLAG += -L/opt/nvidia/hpc_sdk/Linux_x86_64/22.1/math_libs/11.5/targets/x86_64-linux/lib -lcutensorMg

.PHONY: clean

all: gpu-detect getptrattr

%: %.c
	$(NVCC) $< -o $@

%: %.cu
	$(NVCC) $< -o $@

%: %.cc
	$(CXX) $< $(CUTFLAG) -o $@

clean:
	-rm -f gpu-detect
