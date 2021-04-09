NVCC = /usr/local/cuda-10.0/bin/nvcc

.PHONY: clean

all: gpu-detect

gpu-detect: gpu-detect.cu
	$(NVCC) $< -o $@

clean:
	-rm -f gpu-detect
