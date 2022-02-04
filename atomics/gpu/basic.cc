#include <iostream>

#if 0

#include <cuda/std/atomic>
cuda::std::atomic<int> ai{0};
cuda::std::atomic<float> af{0};
cuda::std::atomic<double> ad{0};
int i{0};
float f{0};
double d{0};
cuda::std::atomic_ref<int> ri{&i};
cuda::std::atomic_ref<float> rf{&f};
cuda::std::atomic_ref<double> rd{&d};

#else

#include <atomic>
std::atomic<int> ai{0};
std::atomic<float> af{0};
std::atomic<double> ad{0};
int i{0};
float f{0};
double d{0};
std::atomic_ref<int> ri{i};
std::atomic_ref<float> rf{f};
std::atomic_ref<double> rd{d};

#endif

#define ALSO_NO 1

int main(void)
{
    const int n{10000};
#ifdef _OPENMP
    #pragma omp parallel for
#else
    #pragma acc parallel loop
#endif
    for (int i=0; i<n; ++i) {
        ai++;
        af.fetch_add(1.0f);
        ad.fetch_add(1.0);
        ri++;
        rf.fetch_add(1.0f);
        rd.fetch_add(1.0);
    }
    std::cout << ai << std::endl;
    std::cout << af << std::endl;
    std::cout << ad << std::endl;
    std::cout << ri << std::endl;
    std::cout << rf << std::endl;
    std::cout << rf << std::endl;

    return 0;
}
