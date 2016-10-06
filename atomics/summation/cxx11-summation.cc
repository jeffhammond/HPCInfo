#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <chrono>

#include <atomic>
#include "myatomics.h"

#ifdef SEQUENTIAL_CONSISTENCY
auto update_model = std::memory_order_seq_cst;
#else
auto update_model = std::memory_order_relaxed;
#endif

#ifdef _OPENMP
# include <omp.h>
# define OMP_PARALLEL            _Pragma("omp parallel")
# define OMP_BARRIER             _Pragma("omp barrier")
# define OMP_CRITICAL            _Pragma("omp critical")
# ifdef SEQUENTIAL_CONSISTENCY
#  define OMP_ATOMIC              _Pragma("omp atomic seq_cst")
#  define OMP_ATOMIC_CAPTURE      _Pragma("omp atomic capture seq_cst")
# else
#  define OMP_ATOMIC              _Pragma("omp atomic")
#  define OMP_ATOMIC_CAPTURE      _Pragma("omp atomic capture")
# endif
#else
# error No OpenMP support!
#endif

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

    OMP_PARALLEL
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            //counter += one;
            atomic_fetch_sum_explicit(&counter, one, update_model);
        }

        /// STOP TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << static_cast<int>(counter) << std::endl;
        }
    }

    std::cout << "2) output = atomic_fetch_sum(&counter, one)\n";

    counter = 0.0;

    OMP_PARALLEL
    {
        std::atomic<double> output = {-1.0};

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            //output = counter += one;
            output = atomic_fetch_sum_explicit(&counter, one, update_model);
        }

        /// STOP TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << static_cast<int>(counter) << std::endl;
            std::cout << static_cast<int>(output) << std::endl;
        }
    }

    std::cout << "3) #pragma omp atomic\n";

    double omp_counter = 0.0;

    OMP_PARALLEL
    {
        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            OMP_ATOMIC
            omp_counter += 1.0;
        }

        /// STOP TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << static_cast<int>(omp_counter) << std::endl;
        }
    }

    std::cout << "4) #pragma omp atomic capture\n";

    omp_counter = 0.0;

    OMP_PARALLEL
    {
        double output = -1.0;

        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        for (int i=0; i<iterations; ++i) {
            OMP_ATOMIC_CAPTURE
            output = omp_counter += 1.0;
        }

        /// STOP TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        /// PRINT TIME
        std::chrono::duration<double> dt = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);
        OMP_CRITICAL
        {
            std::cout << "total time elapsed = " << dt.count() << "\n";
            std::cout << "time per iteration = " << dt.count()/iterations  << "\n";
            std::cout << static_cast<int>(omp_counter) << std::endl;
            std::cout << static_cast<int>(output) << std::endl;
        }
    }

    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
