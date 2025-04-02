#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <typeinfo>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#include <mpi.h>
#include <nccl.h>

int me, np;
ncclComm_t NCCL_COMM_WORLD;
cublasHandle_t cublas_handle;

void check(cudaError_t rc)
{   
    if (rc!=cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorName(rc) << "=" << cudaGetErrorString(rc) << std::endl;
        std::abort();
    }
}

void check(curandStatus_t rc)
{   
    if (rc!=CURAND_STATUS_SUCCESS) {
        std::cerr << "CURAND error: " << rc << std::endl;
        std::abort();
    }
}
void check(ncclResult_t rc)
{
    if (rc != ncclSuccess) {
        std::cerr << "NCCL error: " << ncclGetErrorString(rc) << std::endl;
        std::abort();
    }
}

void check(cublasStatus_t rc)
{
    if (rc!=CUBLAS_STATUS_SUCCESS) {
#if defined(CUBLAS_VERSION) && (CUBLAS_VERSION >= (11*10000+4*100+2))
        std::cerr << "CUBLAS error: " << cublasGetStatusName(rc) << "=" << cublasGetStatusString(rc) << std::endl;
#else
        std::cerr << "CUBLAS error: " << rc << std::endl;
#endif
        std::abort();
    }
}

template <typename T>
ncclDataType_t get_NCCL_Datatype(T t) {
    std::cerr << "NCCL datatype resolution failed for type " << typeid(T).name() << std::endl;
    std::abort();
}

template <>
constexpr ncclDataType_t get_NCCL_Datatype(double d) { return ncclFloat64; }
template <>
constexpr ncclDataType_t get_NCCL_Datatype(float d) { return ncclFloat32; }
template <>
constexpr ncclDataType_t get_NCCL_Datatype(half d) { return ncclFloat16; }
template <>
constexpr ncclDataType_t get_NCCL_Datatype(nv_bfloat16 d) { return ncclBfloat16; }
template <>
constexpr ncclDataType_t get_NCCL_Datatype(int i) { return ncclInt32; }


#ifdef __NVCC__

template <typename T>
__global__
void cast_from_double(T * __restrict__ out, const double * __restrict__ in, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;         
    if (i < n) {
        out[i] = (T)in[i];
    }
}

template <typename T>
__global__
void cast_to_double(double * __restrict__ out, const T * __restrict__ in, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;         
    if (i < n) {
        out[i] = (double)in[i];
    }
}

template <typename T>
__global__
void scale(T * __restrict__ out, int s, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;         
    if (i < n) {
        out[i] *= s;
    }
}

template <typename T>
__global__
void diff(double * __restrict__ out, const T * __restrict__ in, const double * __restrict__ ref, unsigned n)
{
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;         
    if (i < n) {
        out[i] = ref[i] - (double)in[i];
    }
}

#endif

template <typename T> 
void print_norm(const T * x, int n, const std::string & name)
{}

template<>
void print_norm(const float * x, int n, const std::string & name)
{
    float result;
    check( cublasSnrm2(cublas_handle, n, x, 1, &result) );
    std::cout << me << ": " << "the 2-norm of " << name << " is " << result << std::endl;
}

template<>
void print_norm(const double * x, int n, const std::string & name)
{
    double result;
    check( cublasDnrm2(cublas_handle, n, x, 1, &result) );
    std::cout << me << ": " << "the 2-norm of " << name << " is " << result << std::endl;
}

template <typename T>
void reduce_test(int count)
{
    const size_t bytes = count * sizeof(T);

    const unsigned threads_per_block = 256;
    const unsigned blocks_per_grid = (count + threads_per_block - 1) / threads_per_block;

    curandGenerator_t gen;
    check( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    check( curandSetPseudoRandomGeneratorSeed(gen, me * 1234ULL) );
    check( cudaDeviceSynchronize() );

    double * ref = nullptr;
    check( cudaMalloc((void**)&ref, count * sizeof(double)) );
    check( curandGenerateUniformDouble(gen, ref, count) );
    scale<<<blocks_per_grid, threads_per_block>>>(ref, 10, count);
    check( cudaDeviceSynchronize() );
    print_norm(ref, count, "ref");

    double * res = nullptr;
    check( cudaMalloc((void**)&res, count * sizeof(double)) );
    check( cudaMemset((void*)res, 0, count * sizeof(double)) );
    check( cudaDeviceSynchronize() );
    //print_norm(res, count, "res");

    {
        T * in  = nullptr;
        check( cudaMalloc((void**)&in,  bytes) );
        //check( cudaMemset((void*)in, 0xFFFFFFFF, bytes) );
        cast_from_double<<<blocks_per_grid, threads_per_block>>>(in, ref, count);
        check( cudaDeviceSynchronize() );
        print_norm(in, count, "in");

        T * out = nullptr;
        check( cudaMalloc((void**)&out, bytes) );
        check( cudaMemset((void*)out, 0, bytes) );
        check( cudaDeviceSynchronize() );

        check( ncclAllReduce(in, out, count, get_NCCL_Datatype(*in), ncclSum, NCCL_COMM_WORLD, 0 /* default stream */) );
        check( cudaDeviceSynchronize() );
        if (me == 0) print_norm(out, count, "out");

        check( ncclAllReduce(ref, ref, count, ncclDouble, ncclSum, NCCL_COMM_WORLD, 0 /* default stream */) );
        check( cudaDeviceSynchronize() );
        if (me == 0) print_norm(ref, (int)count, "ref (after ncclAllReduce)");

        diff<<<blocks_per_grid, threads_per_block>>>(res, out, ref, count);

        double result;
        check( cublasDnrm2(cublas_handle, (int)count, res, 1, &result) );
        if (me == 0) {
            std::cout << me << ": difference between " << typeid(T).name() <<" and double is " << result << std::endl;
        }

        check( cudaFree((void*)out) );
        check( cudaFree((void*)in) );
    }

    check( cudaFree((void*)res) );
    check( cudaFree((void*)ref) );
    check( curandDestroyGenerator(gen) );
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    const int count = (argc > 1) ? std::atoi(argv[1]) : 1024*1024;
    if (me == 0) {
        std::cout << "count = " << count << std::endl;
    }

    int num_gpus;
    check( cudaGetDeviceCount(&num_gpus) );
    if (np > num_gpus) {
        std::cerr << "run with no more MPI processes than GPUs" << std::endl;
        MPI_Abort(MPI_COMM_WORLD,num_gpus);
    }
    check( cudaSetDevice(me % num_gpus) );
    MPI_Barrier(MPI_COMM_WORLD);

    check( cublasCreate(&cublas_handle) );

    ncclUniqueId uniqueId;
    if (me == 0) {
        check( ncclGetUniqueId(&uniqueId) );
    }
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    check( ncclGroupStart() );
    check( ncclCommInitRank(&NCCL_COMM_WORLD, np, uniqueId, me) );
    check( ncclGroupEnd() );
    MPI_Barrier(MPI_COMM_WORLD);

    reduce_test<float>(count);
    MPI_Barrier(MPI_COMM_WORLD);

    reduce_test<half>(count);
    MPI_Barrier(MPI_COMM_WORLD);

    reduce_test<nv_bfloat16>(count);
    MPI_Barrier(MPI_COMM_WORLD);

    check( ncclCommDestroy(NCCL_COMM_WORLD) );

    if (me == 0) std::cout << "FINISHED" << std::endl;

    MPI_Finalize();
    return 0;
}
