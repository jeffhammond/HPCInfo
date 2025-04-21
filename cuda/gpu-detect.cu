#include <stdio.h>
#include <stdlib.h>

#ifndef __NVCC__
#warning Please compile CUDA code with CC=nvcc.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#endif

static const int print_errors = 1;

static inline int cuda_check(cudaError_t rc)
{
    if (rc!=cudaSuccess && print_errors) {
        printf("CUDA error: %s - %s\n", cudaGetErrorName(rc), cudaGetErrorString(rc));
    }
    return rc;
}

typedef struct
{
    short fp64;
    short fp64t;
    short fp32;
    short fp32t;
    short tf32;
    short tf32t;
    short fp16;
    short fp16t;
    short bf16;
    short bf16t;
} cuda_flops_per_sm_s;

cuda_flops_per_sm_s cuda_flops_per_sm(int major, int minor)
{
    cuda_flops_per_sm_s r = {0,0,0,0,0,0,0,0,0,0};

    // https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    switch (major) {
        // Fermi
        // https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf
        // "The Fermi architecture has been specifically designed to offer unprecedented performance in double precision;
        //  up to 16 double precision fused multiply-add operations can be performed per SM, per clock..."
        case 2:
            r.fp64 =  16;
            r.fp32 =  32;
            break;
        // Kepler
        // https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf
        // "Each of the Kepler GK110/210 SMX units feature 192 single-precision CUDA cores,
        //  and each core has fully pipelined floating-point and integer arithmetic logic units."
        //  "the GK110 and GK210 Kepler-based GPUs are capable of performing double precision calculations at a rate of up to 1/3 of single precision compute performance."
        case 3:
            switch (minor) {
                // K20
                case 0:
                // K40
                case 5:
                // K80
                case 7:
                    r.fp64 =  64;
                    r.fp32 = 192;
                    break;
            }
            break;
        // Maxwell
        // https://developer.nvidia.com/blog/5-things-you-should-know-about-new-maxwell-gpu-architecture/
        // 128 comes from 640 "cuda cores" for 5 SMs
        // https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/NVIDIA-Kepler-GK110-GK210-Architecture-Whitepaper.pdf
        // "While the Maxwell architecture performs double precision calculations at rate of 1/32 that of single precision calculations..."
        case 5:
            switch (minor) {
                // Quadro M
                case 0:
                // Quadro etc
                case 1:
                // Jetson Nano
                case 2:
                    r.fp64 =   4;
                    r.fp32 = 128;
                    break;
            }
            break;
        // Pascal
        // GP100 = 6.0 https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf Table 1
        // GP102 = 6.1 https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf Table 1
        // https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
        // "Like Maxwell, each GP104 SM provides four warp schedulers managing a total of 128 single-precision (FP32) and four double-precision (FP64) cores."
        case 6:
            switch (minor) {
                // GP100
                case 0:
                    r.fp64 =  32;
                    r.fp32 =  64;
                    break;
                // GP102, GP104
                case 1:
                // Jetson TX2
                case 2:
                    r.fp64 =   4;
                    r.fp32 = 128;
                    break;
            }
            break;
        // Volta and Turing
        // https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
        // "Each SM has: 64 FP32 cores; 64 INT32 cores; 32 FP64 cores; 8 Tensor Cores"
        // https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
        // "The TU102 GPU also features 144 FP64 units (two per SM)... The FP64 TFLOP rate is 1/32nd the TFLOP rate of FP32 operation"
        case 7:
            switch (minor) {
                // GV100
                case 0:
                    r.fp64 =  32;
                    r.fp32 =  64;
                    break;
                // Xavier AGX (Volta)
                // I cannot find docs but I measure the peaks as ~1.4 TF/s FP32 and ~43 GF/s FP64
                case 2:
                // TU102
                case 5:
                    r.fp64 =   2;
                    r.fp32 =  64;
                    break;
            }
            break;
        // Ampere and Ada
        // https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf
        // "The GA102 GPU also features 168 FP64 units (two per SM), which are not depicted in this diagram.
        //  The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations."
        // https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
        // "The AD102 GPU also includes 288 FP64 Cores (2 per SM) which are not depicted in the above diagram.
        //  The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations."
        case 8:
            switch (minor) {
                // GA100
                case 0:
                    r.fp64 =    32;
                    r.fp32 =    64;
                    r.fp16  =  256;
                    r.bf16  =  128;
                    r.fp64t =   64;
                    r.tf32t =  512;
                    r.fp16t = 1024;
                    r.bf16t = 1024;
                    break;
                // GA102
                case 6:
                    r.fp64 =   2;
                    r.fp32 = 128;
                    break;
                // Orin GA10b (?)
                case 7:
                    r.fp64 =   2;
                    r.fp32 = 128;
                    break;
                // AD102
                case 9:
                    r.fp64  =    2;
                    r.fp32  =  128;
                    r.fp16  =  128;
                    r.bf16  =  128;
                    r.tf32t =  128;
                    r.fp16t =  512; // FP16 accumulator
                    r.bf16t =  256; // FP32 accumulator
                    break;
            }
            break;
        // Hopper
        // https://nvdam.widen.net/content/hj0uek1pxq/original/NVIDIA_H100_Tensor_Core_GPU_Architecture_Whitepaper_V1.03.pdf
        // 128 FP32 CUDA Cores per SM
        // 4 Fourth-Generation Tensor Cores per SM
        // NVIDIA       H100 SXM5    H100 PCIe
        // Peak FP64    33.5 TFLOPS  25.6 TFLOPS
        // Peak FP64 TC 66.9 TFLOPS  51.2 TFLOPS
        // Peak FP32    66.9 TFLOPS  51.2 TFLOPS
        // Peak FP16   133.8 TFLOPS 102.4 TFLOPS
        case 9:
            switch (minor) {
                // GH100
                case 0:
                    r.fp64  =   64; // GPU Boost Clock for FP64 TC Ops, FP32 and FP64 nonTensor Core Ops
                    r.fp32  =  128; // 1980 MHz; 1755 MHz
                    r.fp16  =  256; // (H100 SXM5; H100 PCIe)
                    r.bf16  =  256;
                    r.fp64t =  128;
                    r.tf32t = 1024; // GPU Boost Clock for FP8, FP16, BF16, TF32 TC Ops
                    r.fp16t = 2048; // 1830 MHz; 1620 MHz
                    r.bf16t = 2048; // (H100 SXM5; H100 PCIe)
                    break;
            }

            break;
        default:
            break;
    }
    return r;
}


void find_nvgpu(void)
{
    int nd;
    cuda_check( cudaGetDeviceCount(&nd) );

    for (int i=0; i<nd; ++i) {

        printf("============= GPU number %d =============\n", i);

        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, i);

        printf("GPU name                                = %s.\n", dp.name);

        int major = dp.major;
        int minor = dp.minor;
        printf("Compute Capability (CC)                 = %d.%d\n", major, minor);

        // memory bandwidth
        int memoryClock = dp.memoryClockRate;
        int memoryBusWidth = dp.memoryBusWidth;
        // Xavier AGX override
        if (major==7 && minor==2) {
            memoryClock = 2133000; // 2.133 GHz in Khz
            printf("memoryClockRate (Xavier AGX LPDDR4x)    = %.3f GHz\n",  memoryClock*1.e-6);
        }
        // Orin AGX override
        if (major==8 && minor==7) {
            // https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf
            memoryClock = 3200000; // 3.2 GHz in Khz
            memoryBusWidth = 256;
            printf("memoryClockRate (Orin AGX LPDDR5x)      = %.3f GHz\n",  memoryClock*1.e-6);
            printf("memoryBusWidth (Orin AGX LPDDR5x)       = %d bits\n",  memoryBusWidth);
        }
        // 2 for Dual Data Rate (https://en.wikipedia.org/wiki/Double_data_rate)
        // 1/8 = 0.125 for bit to byte
        printf("memoryClockRate (CUDA device query)     = %.3f GHz\n",  dp.memoryClockRate*1.e-6);
        printf("memoryBusWidth (CUDA device query)      = %d bits\n",   dp.memoryBusWidth );
        printf("peak bandwidth                          = %.1f GB/s\n", 2 * memoryClock*1.e-6 * memoryBusWidth * 0.125);

        // memory capacity
        //printf("totalGlobalMem                          = %zu bytes\n", dp.totalGlobalMem);
        printf("totalGlobalMem                          = %zu MiB\n",   dp.totalGlobalMem/(1<<20));
        //printf("totalGlobalMem                          = %zu GiB\n",   dp.totalGlobalMem/(1<<30));

        // compute throughput
        int nSMs = dp.multiProcessorCount;
        printf("multiProcessorCount                     = %d\n",       nSMs);
        printf("warpSize                                = %d\n",       dp.warpSize);
        int clockRate = dp.clockRate; // in KHz
        if (major==8 && minor==7) {
            clockRate = 2201600; // 2.2 GHz in KHz
            printf("clockRate (Orin AGX)                    = %.3f GHz\n", clockRate*1.e-6);
        }
        printf("clockRate (CUDA device query)           = %.3f GHz\n", dp.clockRate*1.e-6);

        cuda_flops_per_sm_s r = cuda_flops_per_sm(major,minor);

        printf("non-TensorCore throughput:\n");

        printf("FP64 FMA/clock per SM                   = %d\n", r.fp64);
        printf("FP32 FMA/clock per SM                   = %d\n", r.fp32);
        if (r.fp16 > 0) {
            printf("FP16 FMA/clock per SM                   = %d\n", r.fp16);
        }
        if (r.bf16 > 0) {
            printf("BF16 FMA/clock per SM                   = %d\n", r.bf16);
        }
                                                                 // FMA=2 * ops/clock/SM * SMs * GHz
        printf("GigaFP64/second per GPU                 = %.1f\n", 2 * r.fp64 * nSMs * 1.e-6*clockRate);
        printf("GigaFP32/second per GPU                 = %.1f\n", 2 * r.fp32 * nSMs * 1.e-6*clockRate);
        if (r.fp16 > 0) {
            printf("GigaFP16/second per GPU                 = %.1f\n", 2 * r.fp16 * nSMs * 1.e-6*clockRate);
        }
        if (r.bf16 > 0) {
            printf("GigaBF16/second per GPU                 = %.1f\n", 2 * r.bf16 * nSMs * 1.e-6*clockRate);
        }

        if (major >= 7) {
            int clockRate = dp.clockRate; // in KHz
            if (major==9 && minor==0) {
                // H100 SXM5
                if (nSMs == 132) {
                    clockRate = 1830000; // 1.83 GHz in KHz
                }
                // H100 PCIe
                if (nSMs == 114) {
                    clockRate = 1620000; // 1.62 GHz in KHz
                }
                printf("clockRate (H100 Tensor Core)            = %.3f GHz\n", clockRate*1.e-6);
            }
            printf("TensorCore throughput:\n");

            if (r.fp64t > 0) {
                printf("FP64 FMA/clock per SM                   = %d\n", r.fp64t);
            }
            if (r.fp32t > 0) {
                printf("FP32 FMA/clock per SM                   = %d\n", r.fp32t);
            }
            if (r.fp16t > 0) {
                printf("FP16 FMA/clock per SM                   = %d\n", r.fp16t);
            }
            if (r.bf16t > 0) {
                printf("BF16 FMA/clock per SM                   = %d\n", r.bf16t);
            }
                                                                     // FMA=2 * ops/clock/SM * SMs * GHz
            if (r.fp64t > 0) {
                printf("GigaFP64/second per GPU                 = %.1f\n", 2 * r.fp64t * nSMs * 1.e-6*clockRate);
            }
            if (r.fp32t > 0) {
                printf("GigaFP32/second per GPU                 = %.1f\n", 2 * r.fp32t * nSMs * 1.e-6*clockRate);
            }
            if (r.fp16t > 0) {
                printf("GigaFP16/second per GPU                 = %.1f\n", 2 * r.fp16t * nSMs * 1.e-6*clockRate);
            }
            if (r.bf16t > 0) {
                printf("GigaBF16/second per GPU                 = %.1f\n", 2 * r.bf16t * nSMs * 1.e-6*clockRate);
            }

        }

        // memory sharing characteristics
        printf("unifiedAddressing                       = %d\n", dp.unifiedAddressing);
        printf("managedMemory                           = %d\n", dp.managedMemory);
        printf("pageableMemoryAccess                    = %d\n", dp.pageableMemoryAccess);
        printf("pageableMemoryAccessUsesHostPageTables  = %d\n", dp.pageableMemoryAccessUsesHostPageTables);
        printf("concurrentManagedAccess                 = %d\n", dp.concurrentManagedAccess);
        printf("canMapHostMemory                        = %d\n", dp.canMapHostMemory);
    }
}

int main(void)
{
    find_nvgpu();
    return 0;
}

