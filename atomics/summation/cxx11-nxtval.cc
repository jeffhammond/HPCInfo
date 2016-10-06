#if defined(__cplusplus) && (__cplusplus >= 201103L)

#include <iostream>
#include <iomanip>

#include <chrono>

#include <atomic>

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

    const int num_threads = omp_get_max_threads();

    std::cout << "thread counter benchmark\n";
    std::cout << "num threads  = " << num_threads << "\n";
    std::cout << "iterations   = " << iterations << "\n";
#ifdef SEQUENTIAL_CONSISTENCY
    std::cout << "memory model = " << "seq_cst";
#else
    std::cout << "memory model = " << "relaxed";
#endif
    std::cout << std::endl;

    std::cout << "1) std::atomic_fetch_add(&counter, 1)\n";

    std::atomic<int> counter = {0};

    OMP_PARALLEL
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);

        int work  = 0;

        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        int count = 0;
        int next  = std::atomic_fetch_add(&counter, 1);
        for (int i=0; i<iterations; ++i) {
            if (next==count) {
                ++work;
                next = std::atomic_fetch_add(&counter, 1);
            }
            ++count;
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
            std::cout << work << std::endl;
        }
    }

    std::cout << "2) #pragma omp atomic acquire\n";

    int omp_counter = 0;

    OMP_PARALLEL
    {
        int work  = 0;

        /// START TIME
        OMP_BARRIER
        std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();

        int count = 0;
        int next;
        OMP_ATOMIC_CAPTURE
        next = omp_counter++;
        for (int i=0; i<iterations; ++i) {
            if (next==count) {
                ++work;
                OMP_ATOMIC_CAPTURE
                next = omp_counter++;
            }
            ++count;
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
            std::cout << work << std::endl;
        }
    }


    return 0;
}

#else  // C++11
#error You need C++11 for this test!
#endif // C++11
