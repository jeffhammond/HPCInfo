#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>
#include <cutensor.h>

[[maybe_unused]] const bool debug = false;

static inline void check(const cutensorStatus_t s, const char * info = "")
{
    if ( s != CUTENSOR_STATUS_SUCCESS ) {
        std::cerr << "Error: " << cutensorGetErrorString(s) << " (" << info << ")" << std::endl;
        std::exit(s);
    }
}

static inline void check(const cudaError_t s, const char * info = "")
{
    if ( s != cudaSuccess ) {
        std::cerr << "Error: " << cudaGetErrorName(s) << " : " << cudaGetErrorString(s) << " (" << info << ")" << std::endl;
        std::exit(s);
    }
}

template <typename T>
static inline void init(const T r, const T c, float * const m)
{
    for (T i=0; i<r; ++i) {
        for (T j=0; j<c; ++j) {
            m[i*c+j] = i*c+j;
        }
    }
}

template <typename T>
static inline void mset(const T r, const T c, float * const m, const float value)
{
    for (T i=0; i<r; ++i) {
        for (T j=0; j<c; ++j) {
            m[i*c+j] = value;
        }
    }
}

template <typename T>
static inline void mprint(const T r, const T c, float * const m, const char * label)
{
    if (debug) {
        std::cout << label << "\n";
        for (T i=0; i<r; ++i) {
            for (T j=0; j<c; ++j) {
                std::cout << i << "," << j << "=" << m[i*c+j] << "\n";
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "This program takes 3 integer arguments" << std::endl;
        return argc;
    }

    const int m = std::atoi(argv[1]);
    const int n = std::atoi(argv[2]);
    const int k = std::atoi(argv[3]);

    std::cout << "dims=" << m << "," << n << "," << k << std::endl;

    cudaError_t s2;

    cudaStream_t stream;
    s2 = cudaStreamCreate(&stream);
    check(s2);

    cutensorStatus_t s;
    cutensorHandle_t h;

    s = cutensorInit(&h);
    check(s,"cutensorInit");

    cutensorContractionFind_t f;
    s = cutensorInitContractionFind(&h, &f, CUTENSOR_ALGO_DEFAULT);
    check(s,"cutensorInitContractionFind");

    int64_t eA[2]={m,k};
    int64_t eB[2]={k,n};
    int64_t eC[2]={m,n};

    float *pA, *pB, *pC;

    s2 = cudaMallocManaged(&pA, eA[0]*eA[1]*sizeof(float) );
    check(s2,"cudaMallocManaged");
    init(eA[0],eA[1],pA);
    mprint(eA[0],eA[1],pA,"A before");

    s2 = cudaMallocManaged(&pB, eB[0]*eB[1]*sizeof(float) );
    check(s2,"cudaMallocManaged");
    init(eB[0],eB[1],pB);
    mprint(eB[0],eB[1],pB,"B before");

    s2 = cudaMallocManaged(&pC, eC[0]*eC[1]*sizeof(float) );
    check(s2,"cudaMallocManaged");
    mset(eC[0],eC[1],pC,0);
    mprint(eC[0],eC[1],pC,"C before");

    cutensorTensorDescriptor_t dA, dB, dC;

    s = cutensorInitTensorDescriptor(&h, &dA, 2, eA, NULL, CUDA_R_32F, CUTENSOR_OP_IDENTITY);
    check(s,"cutensorInitTensorDescriptor");

    s = cutensorInitTensorDescriptor(&h, &dB, 2, eB, NULL, CUDA_R_32F, CUTENSOR_OP_IDENTITY);
    check(s,"cutensorInitTensorDescriptor");

    s = cutensorInitTensorDescriptor(&h, &dC, 2, eC, NULL, CUDA_R_32F, CUTENSOR_OP_IDENTITY);
    check(s,"cutensorInitTensorDescriptor");

    uint32_t aA, aB, aC;

    s = cutensorGetAlignmentRequirement(&h, pA, &dA, &aA);
    check(s);

    s = cutensorGetAlignmentRequirement(&h, pB, &dB, &aB);
    check(s);

    s = cutensorGetAlignmentRequirement(&h, pC, &dC, &aC);
    check(s);

    cutensorContractionDescriptor_t dX;
    int32_t mA[2]={'i','k'};
    int32_t mB[2]={'k','j'};
    int32_t mC[2]={'i','j'};
    s = cutensorInitContractionDescriptor(&h, &dX, &dA, mA, aA, &dB, mB, aB, &dC, mC, aC, &dC, mC, aC, CUTENSOR_R_MIN_32F);
    check(s,"cutensorInitContractionDescriptor");

    size_t worksize = 0;
    s = cutensorContractionGetWorkspace(&h, &dX, &f, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize);
    check(s,"cutensorContractionGetWorkspace");

    void * pW = nullptr;
    s2 = cudaMalloc(&pW, worksize);
    check(s2,"cudaMalloc");

    cutensorContractionPlan_t p;
    s = cutensorInitContractionPlan(&h, &p, &dX, &f, worksize);
    check(s,"cutensorInitContractionPlan");

    float alpha = 1, beta = 0;
    s = cutensorContraction(&h, &p, (void*) &alpha, pA, pB, (void*) &beta, pC, pC, pW, worksize, stream);
    check(s,"cutensorContraction");

    s2 = cudaStreamSynchronize(stream);
    check(s2,"cudaStreamSynchronize");

    mprint(eC[0],eC[1],pC,"C after");

    s2 = cudaFree(pW);
    check(s2,"cudaFree");

    s2 = cudaFree(pA);
    check(s2,"cudaFree");

    s2 = cudaFree(pB);
    check(s2,"cudaFree");

    s2 = cudaFree(pC);
    check(s2,"cudaFree");

    s2 = cudaStreamDestroy(stream);
    check(s2);

    std::cout << "THE END" << std::endl;

    return 0;
}
