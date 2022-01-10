#include <cstdio>
#include <cstdlib> // atoi, getenv
#include <cstdint>
#include <cfloat>  // FLT_MIN
#include <climits>
#include <cmath>

#include <string>
#include <iostream>
#include <iomanip> // std::setprecision
#include <exception>
#include <list>
#include <vector>

#include <chrono>
#include <typeinfo>
#include <array>
#include <numeric>
#include <algorithm>
#include <thread> // std::thread::hardware_concurrency

#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>

namespace prk
{
    inline double wtime(void)
    {
        using t = std::chrono::high_resolution_clock;
        auto c = t::now().time_since_epoch().count();
        auto n = t::period::num;
        auto d = t::period::den;
        double r = static_cast<double>(c)/static_cast<double>(d)*static_cast<double>(n);
        return r;
    }

    template <class T1, class T2>
    inline auto divceil(T1 numerator, T2 denominator) -> decltype(numerator / denominator) {
        return ( numerator / denominator + (numerator % denominator > 0) );
    }

    namespace CUDA
    {
        void check(cudaError_t rc)
        {
            if (rc==cudaSuccess) {
                return;
            } else {
                std::cerr << "PRK CUDA error: " << cudaGetErrorString(rc) << std::endl;
                std::abort();
            }
        }

        void check(cublasStatus_t rc)
        {
            if (rc==CUBLAS_STATUS_SUCCESS) {
                return;
            } else {
                std::cerr << "PRK CUBLAS error: " << rc << std::endl;
                std::abort();
            }
        }

        class info {

            private:
                int nDevices;
                std::vector<cudaDeviceProp> vDevices;

            public:
                int maxThreadsPerBlock;
                std::array<unsigned,3> maxThreadsDim;
                std::array<unsigned,3> maxGridSize;

                info() {
                    prk::CUDA::check( cudaGetDeviceCount(&nDevices) );
                    vDevices.resize(nDevices);
                    for (int i=0; i<nDevices; ++i) {
                        cudaGetDeviceProperties(&(vDevices[i]), i);
                        if (i==0) {
                            maxThreadsPerBlock = vDevices[i].maxThreadsPerBlock;
                            for (int j=0; j<3; ++j) {
                                maxThreadsDim[j]   = vDevices[i].maxThreadsDim[j];
                                maxGridSize[j]     = vDevices[i].maxGridSize[j];
                            }
                        }
                    }
                }

                // do not use cached value as a hedge against weird stuff happening
                int num_gpus() {
                    int g;
                    prk::CUDA::check( cudaGetDeviceCount(&g) );
                    return g;
                }

                int get_gpu() {
                    int g;
                    prk::CUDA::check( cudaGetDevice(&g) );
                    return g;
                }

                void set_gpu(int g) {
                    prk::CUDA::check( cudaSetDevice(g) );
                }

                void print() {
                    for (int i=0; i<nDevices; ++i) {
                        std::cout << "device name: " << vDevices[i].name << "\n";
                        std::cout << "total global memory:     " << vDevices[i].totalGlobalMem << "\n";
                        std::cout << "max threads per block:   " << vDevices[i].maxThreadsPerBlock << "\n";
                        std::cout << "max threads dim:         " << vDevices[i].maxThreadsDim[0] << ","
                                                                 << vDevices[i].maxThreadsDim[1] << ","
                                                                 << vDevices[i].maxThreadsDim[2] << "\n";
                        std::cout << "max grid size:           " << vDevices[i].maxGridSize[0] << ","
                                                                 << vDevices[i].maxGridSize[1] << ","
                                                                 << vDevices[i].maxGridSize[2] << "\n";
                        std::cout << "memory clock rate (KHz): " << vDevices[i].memoryClockRate << "\n";
                        std::cout << "memory bus width (bits): " << vDevices[i].memoryBusWidth << "\n";
                    }
                }

                bool checkDims(dim3 dimBlock, dim3 dimGrid) {
                    if (dimBlock.x > maxThreadsDim[0]) {
                        std::cout << "dimBlock.x too large" << std::endl;
                        return false;
                    }
                    if (dimBlock.y > maxThreadsDim[1]) {
                        std::cout << "dimBlock.y too large" << std::endl;
                        return false;
                    }
                    if (dimBlock.z > maxThreadsDim[2]) {
                        std::cout << "dimBlock.z too large" << std::endl;
                        return false;
                    }
                    if (dimGrid.x  > maxGridSize[0])   {
                        std::cout << "dimGrid.x  too large" << std::endl;
                        return false;
                    }
                    if (dimGrid.y  > maxGridSize[1]) {
                        std::cout << "dimGrid.y  too large" << std::endl;
                        return false;
                    }
                    if (dimGrid.z  > maxGridSize[2]) {
                        std::cout << "dimGrid.z  too large" << std::endl;
                        return false;
                    }
                    return true;
                }
        };

        template <typename T>
        T * malloc(size_t n) {
            T * ptr;
            size_t bytes = n * sizeof(T);
            prk::CUDA::check( cudaMalloc((void**)&ptr, bytes) );
            return ptr;
        }

        template <typename T>
        T * malloc_host(size_t n) {
            T * ptr;
            size_t bytes = n * sizeof(T);
            prk::CUDA::check( cudaMallocHost((void**)&ptr, bytes) );
            return ptr;
        }

        template <typename T>
        T * malloc_managed(size_t n) {
            T * ptr;
            size_t bytes = n * sizeof(T);
            prk::CUDA::check( cudaMallocManaged((void**)&ptr, bytes) );
            return ptr;
        }

        template <typename T>
        void free(T * ptr) {
            prk::CUDA::check( cudaFree((void*)ptr) );
        }

        template <typename T>
        void free_host(T * ptr) {
            prk::CUDA::check( cudaFreeHost((void*)ptr) );
        }

        template <typename T>
        void copyD2H(T * output, T * const input, size_t n) {
            size_t bytes = n * sizeof(T);
            prk::CUDA::check( cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToHost) );
        }

        template <typename T>
        void copyH2D(T * output, T * const input, size_t n) {
            size_t bytes = n * sizeof(T);
            prk::CUDA::check( cudaMemcpy(output, input, bytes, cudaMemcpyHostToDevice) );
        }

        template <typename T>
        void copyD2Hasync(T * output, T * const input, size_t n) {
            size_t bytes = n * sizeof(T);
            prk::CUDA::check( cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToHost) );
        }

        template <typename T>
        void copyH2Dasync(T * output, T * const input, size_t n) {
            size_t bytes = n * sizeof(T);
            prk::CUDA::check( cudaMemcpyAsync(output, input, bytes, cudaMemcpyHostToDevice) );
        }

        template <typename T>
        void prefetch(T * ptr, size_t n, int device = 0) {
            size_t bytes = n * sizeof(T);
            //std::cout << "device=" << device << "\n";
            prk::CUDA::check( cudaMemPrefetchAsync(ptr, bytes, device) );
        }

        void sync(void) {
            prk::CUDA::check( cudaDeviceSynchronize() );
        }

        void set_device(int i) {
            prk::CUDA::check( cudaSetDevice(i) );
        }

    } // CUDA namespace

} // prk namespace


__global__ void foo(int n, double * A)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i]++;
    }
}

int main(int argc, char * argv[])
{
  const int n = 1000;
  const int s = 1000;
  const int reps = 100000;

  const int b = 32;
  dim3 dimBlock(b, 1, 1);
  dim3 dimGrid(prk::divceil(s,b), 1, 1);
  std::cout << "Grid= " << dimGrid.x << "," << dimGrid.y << "," << dimGrid.z << std::endl;
  std::cout << "Block=" << dimBlock.x << "," << dimBlock.y << "," << dimBlock.z << std::endl;

  auto X = prk::CUDA::malloc_host<double>(n);
  for (int i=0; i<n; i++) X[i] = i;
  auto Y = prk::CUDA::malloc<double>(n);

  prk::CUDA::copyH2D(Y, X, n);

  auto t0 = prk::wtime();
  foo<<<dimGrid, dimBlock>>>(1, Y);
  auto t1 = prk::wtime();
  std::cout << "first call latency dt=" << t1-t0 << std::endl;

  for (int i=0; i<reps; i++) {
      foo<<<dimGrid, dimBlock>>>(1, Y);
  }

  auto t2 = prk::wtime();
  std::cout << "repeated calls dt=" << (t2-t1)/reps << std::endl;

  prk::CUDA::copyD2H(X, Y, n);

  prk::CUDA::free<double>(Y);
  prk::CUDA::free_host<double>(X);

  return 0;
}


