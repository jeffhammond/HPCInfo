#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <atomic>

#include <chrono>

#ifdef _OPENMP
# include <omp.h>
#else
# error No OpenMP support!
#endif

#ifdef SEQUENTIAL_CONSISTENCY
auto update_model = std::memory_order_seq_cst;
#else
auto update_model = std::memory_order_relaxed;
#endif

template <class T>
static inline T atomic_fetch_sum(std::atomic<T> * obj, T arg)
{
    T original, desired;
    do {
      original = *obj;
      desired  = original + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &original, desired));
    return original;
}

template <class T>
static inline T atomic_fetch_sum(volatile std::atomic<T> * obj, T arg)
{
    T original, desired;
    do {
      original = *obj;
      desired  = original + arg;
    } while (!std::atomic_compare_exchange_weak(obj, &original, desired));
    return original;
}

int main(int argc, char * argv[])
{
    int iterations = (argc>1) ? atoi(argv[1]) : 10000000;

    std::cout << "thread counter benchmark\n";
    std::cout << "num threads  = " << omp_get_max_threads() << "\n";
    std::cout << "iterations   = " << iterations << "\n";
#ifdef SEQUENTIAL_CONSISTENCY
    std::cout << "memory model = " << "seq_cst";
#else
    std::cout << "memory model = " << "relaxed";
#endif
    std::cout << std::endl;

    std::cout << "1) atomic_fetch_sum(&counter, one)\n";

    std::atomic<double> counter = {0.0};
    const double one = 1.0;

    #pragma omp parallel
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            //counter += one;
            atomic_fetch_sum(&counter, one);
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << static_cast<int>(counter) << std::endl;
        }
    }

    std::cout << "2) output = atomic_fetch_sum(&counter, one)\n";

    counter = 0.0;

    #pragma omp parallel
    {
        std::atomic<double> output = {-1.0};

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            //output = counter += one;
            output = atomic_fetch_sum(&counter, one);
        }

        /// STOP TIME
        #pragma omp barrier
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        #pragma omp critical
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << static_cast<int>(counter) << std::endl;
            std::cout << static_cast<int>(output) << std::endl;
        }
    }

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
